"""Actual recovery producers and real SessionDB round-trips, without provider calls."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent.context_compressor import (
    ContextCompressor, SUMMARY_PREFIX, _SUMMARY_END_MARKER, _INFLIGHT_TASK_REPLAY_HEADER,
    _MERGED_PRIOR_CONTEXT_HEADER, _MERGED_SUMMARY_DELIMITER,
)
from agent.message_metadata import without_persistence_fields
from agent.turn_truncation import continue_codex_incomplete, _continue_text, _Trunc
from agent.turn_retry_state import TurnRetryState


def _compressor():
    c = ContextCompressor.__new__(ContextCompressor)
    c.quiet_mode = True
    return c


def _carrier(role='assistant', prior=None):
    text = SUMMARY_PREFIX + '\nPRIVATE_SUMMARY\n' + _SUMMARY_END_MARKER
    if prior:
        text = _MERGED_PRIOR_CONTEXT_HEADER + '\n' + prior + '\n\n' + _MERGED_SUMMARY_DELIMITER + '\n\n' + text
    return {'role': role, 'content': text, '_compressed_summary': True}


def _origin(content='ORIGINAL HUMAN REQUEST'):
    return {'role': 'user', 'content': content, 'message_uid': 'human-uid',
            'platform_message_id': 'platform-human', 'timestamp': 1234.5,
            'display_metadata': {'reactions': [{'emoji': 'ok'}]}}


def _codex_nudge():
    rows = [{'role': 'assistant', 'content': ''}]
    a = SimpleNamespace(_codex_incomplete_retries=0, _codex_reasoning_only_streak=0,
        _build_assistant_message=lambda *a: {'role': 'assistant', 'content': ''},
        quiet_mode=True, _emit_diagnostic_wait=lambda *a: None)
    assert continue_codex_incomplete(a, SimpleNamespace(tool_calls=None), 'incomplete',
        messages=rows, conversation_history=[], api_call_count=1) is None
    return rows[-1]


def _length_nudge(stub=False, dropped=None):
    a = MagicMock()
    a._build_assistant_message.return_value = {'role': 'assistant', 'content': 'partial'}
    from hermes_constants import PARTIAL_STREAM_STUB_ID
    st = _Trunc(agent=a, response=SimpleNamespace(id=PARTIAL_STREAM_STUB_ID if stub else 'normal',
            _dropped_tool_names=dropped), finish_reason='length', conversation_history=[],
        api_call_count=1, effective_task_id=None, current_turn_user_idx=0,
        messages=[_origin()], length_continue_retries=0, truncated_response_parts=[],
        truncated_tool_call_retries=0, retry_count=0, compression_attempts=0)
    result = _continue_text(st, TurnRetryState(), SimpleNamespace(content='partial'))
    assert result.action == 'break'
    return result.messages[-1]


@pytest.mark.parametrize('producer,make', [
    ('codex_incomplete', _codex_nudge),
    ('length_continuation', _length_nudge),
    ('length_continuation', lambda: _length_nudge(True)),
    ('length_continuation', lambda: _length_nudge(True, ['terminal'])),
])
def test_actual_recovery_producers_are_explicit_hidden(producer, make):
    row = make()
    assert row.get('display_kind') == 'hidden'
    assert row['display_metadata']['hermes_provenance'] == {
        'version': 1, 'kind': 'recovery_nudge', 'producer': producer}
    assert without_persistence_fields(row) == {
        k: v for k, v in row.items() if k not in ('timestamp', 'display_kind', 'display_metadata')}


@pytest.mark.parametrize('role,prior', [('assistant', None), ('user', None), ('user', 'AUTHENTIC PRIOR REPLY')])
def test_replay_preserves_origin_safe_projection_and_provider_bytes(role, prior):
    origin = _origin()
    before = deepcopy(origin)
    compressed = [_carrier(role, prior)]
    out = _compressor()._reappend_inflight_user_task(compressed, origin)
    target = out[-1]
    p = target.get('display_metadata', {}).get('hermes_provenance')
    assert p is not None
    assert p == {'version': 1, 'kind': 'active_request_replay', 'producer': 'context_compressor',
        'placement': 'standalone' if role == 'assistant' else 'summary_suffix',
        'origin_message_uid': 'human-uid', 'origin_platform_message_id': 'platform-human',
        'origin_timestamp': 1234.5, 'display_content': 'ORIGINAL HUMAN REQUEST'}
    assert origin == before  # never mutate/share the source's display metadata
    assert _INFLIGHT_TASK_REPLAY_HEADER + '\nORIGINAL HUMAN REQUEST' in target['content']
    from agent.recovery_provenance import project_active_request_for_display
    human = project_active_request_for_display(target)
    assert human['role'] == 'user'
    assert human['content'] == origin['content']
    assert human['message_uid'] == origin['message_uid']
    assert human['timestamp'] == origin['timestamp']
    if prior:
        from agent.compaction_display import project_compaction_message_for_display
        assert project_compaction_message_for_display(target)['content'] == prior


def test_repeated_compaction_from_merged_carrier_keeps_original_provenance():
    c = _compressor()
    merged = c._reappend_inflight_user_task([_carrier('user')], _origin())[-1]
    replay = c._reappend_inflight_user_task([_carrier()], merged)[-1]
    p = replay['display_metadata']['hermes_provenance']
    assert p['origin_message_uid'] == 'human-uid'
    assert p['origin_platform_message_id'] == 'platform-human'
    assert p['origin_timestamp'] == 1234.5
    assert p['display_content'] == 'ORIGINAL HUMAN REQUEST'
    assert replay['content'].count(_INFLIGHT_TASK_REPLAY_HEADER) == 1


def test_literal_human_headers_are_not_used_for_display_classification():
    # The execution path has legacy header stripping; display must preserve the exact source.
    literal = _INFLIGHT_TASK_REPLAY_HEADER + '\nLITERAL HUMAN HEADER EXAMPLE'
    from agent.recovery_provenance import project_active_request_for_display
    human = _origin(literal)
    assert project_active_request_for_display(human) is None
    replay = _compressor()._reappend_inflight_user_task([_carrier()], human)[-1]
    assert project_active_request_for_display(replay)['content'] == literal
    assert project_active_request_for_display({'role': 'user', 'content': _codex_nudge()['content']}) is None


def test_actual_flush_and_cold_sessiondb_reload_preserve_provenance(tmp_path):
    from hermes_state import SessionDB
    from agent.session_persistence import _db_flush_collect, _db_flush_write
    path = tmp_path / 'state.db'
    db = SessionDB(path)
    sid = 'provenance-test'
    db.create_session(session_id=sid, source='test')
    rows = [_codex_nudge(), _length_nudge(),
        _compressor()._reappend_inflight_user_task([_carrier()], _origin())[-1],
        _compressor()._reappend_inflight_user_task([_carrier('user')], _origin())[-1]]
    # Different logical origins; identical content is deliberately NOT an identity.
    rows[-1]['display_metadata']['hermes_provenance']['origin_message_uid'] = 'other-human-uid'
    rows.append({'role': 'user', 'content': rows[0]['content']})
    a = SimpleNamespace(_session_db=db, session_id=sid, _last_flushed_db_idx=0)
    batch, live = _db_flush_collect(a, rows, [])
    _db_flush_write(a, batch, live, rows)
    expected = [deepcopy(row.get('display_metadata')) for row in rows]
    db.close()
    db = SessionDB(path)
    raw = db.get_messages(sid)
    conversation = db.get_messages_as_conversation(sid)
    assert len(raw) == len(conversation) == len(rows)
    for stored, replayed, metadata in zip(raw, conversation, expected):
        assert stored.get('display_metadata') == replayed.get('display_metadata') == metadata
    assert raw[0]['display_kind'] == raw[1]['display_kind'] == 'hidden'
    assert not raw[-1].get('display_kind')
    assert raw[0]['content'] == raw[-1]['content']
    from agent.recovery_provenance import project_active_request_for_display
    assert project_active_request_for_display(raw[2])['content'] == 'ORIGINAL HUMAN REQUEST'
    assert project_active_request_for_display(raw[3])['message_uid'] == 'other-human-uid'
    assert project_active_request_for_display(raw[-1]) is None
    db.close()


@pytest.mark.parametrize('kind', ['ack', 'stall', 'degenerate', 'dropped'])
def test_actual_text_continuation_branches_are_hidden(monkeypatch, kind):
    import agent.agent_runtime_helpers as helpers
    from agent.turn_final_response import finish_text_response
    a = MagicMock()
    a._intent_ack_continuation = True
    a._dropped_toolcall_retries = 0
    a._strip_think_blocks.side_effect = lambda s: s
    a._has_content_after_think_block.return_value = True
    a.valid_tool_names = {'terminal'}
    a._build_assistant_message.return_value = {'role': 'assistant', 'content': 'partial'}
    a._looks_like_codex_intermediate_ack.return_value = kind == 'ack'
    monkeypatch.setattr(helpers, 'trailing_continue_intent', lambda text: kind == 'stall')
    monkeypatch.setattr(helpers, 'tool_results_this_turn', lambda rows: 1 if kind == 'degenerate' else 0)
    monkeypatch.setattr(helpers, 'looks_like_degenerate_final', lambda *a, **kw: kind == 'degenerate')
    rows = [_origin()]
    verdict = finish_text_response(a, assistant_message=SimpleNamespace(content='partial', tool_calls=None),
        response=None, finish_reason='tool_calls' if kind == 'dropped' else 'stop', messages=rows, api_messages=[], conversation_history=[],
        api_call_count=1, user_message='request', active_system_prompt='', final_response=None,
        _turn_exit_reason=None, _preflight_compression_blocked=False, codex_ack_continuations=0,
        truncated_response_parts=[], length_continue_retries=0, _pending_verification_response=None,
        _pending_verification_response_previewed=False, effective_task_id=None)
    assert verdict.action == 'continue'
    assert verdict.codex_ack_continuations == (0 if kind == 'dropped' else 1)
    assert rows[-1].get('display_kind') == 'hidden'
    assert rows[-1]['display_metadata']['hermes_provenance']['producer'] == (
        'dropped_toolcall' if kind == 'dropped' else 'degenerate_final' if kind == 'degenerate' else 'ack_continuation')


def test_max_iterations_nudge_is_hidden_without_changing_summary_call(monkeypatch):
    import agent.chat_completion_helpers as helpers
    a = MagicMock(api_mode='fixture_mode', max_iterations=1, suppress_status_output=True)
    observed = []
    monkeypatch.setattr(helpers, '_iteration_summary_api_messages', lambda a, rows: observed.extend(deepcopy(rows)) or rows)
    monkeypatch.setitem(helpers._SUMMARY_ATTEMPT_BUILDERS, 'fixture_mode', lambda *a: lambda retry: 'FINAL SUMMARY')
    rows = [_origin()]
    assert helpers.handle_max_iterations(a, rows, 1) == 'FINAL SUMMARY'
    assert rows[-2].get('display_kind') == 'hidden'
    assert rows[-2]['display_metadata']['hermes_provenance']['producer'] == 'max_iterations'
    assert observed[-1]['content'] == rows[-2]['content']


def test_recovery_and_replay_metadata_never_changes_actual_provider_wire():
    from agent.turn_context import build_api_messages
    from agent.model_metadata import estimate_messages_tokens_rough
    rows = [_codex_nudge(), _length_nudge(),
        _compressor()._reappend_inflight_user_task([_carrier()], _origin())[-1],
        _compressor()._reappend_inflight_user_task([_carrier('user')], _origin())[-1]]
    baseline = [{k: deepcopy(v) for k, v in r.items() if k not in {'display_kind', 'display_metadata'}} for r in rows]
    a = SimpleNamespace(_current_turn_timestamp=1, _copy_reasoning_content_for_api=lambda *a: None,
        _should_sanitize_tool_calls=lambda: False, ephemeral_system_prompt=None, api_mode='chat_completions')
    def wire(messages):
        return build_api_messages(a, messages, current_turn_user_idx=None, ext_prefetch_cache=None,
            plugin_user_context=None, moa_config=None, active_system_prompt='UNCHANGED SYSTEM')
    assert wire(rows) == wire(baseline)
    assert estimate_messages_tokens_rough(rows) == estimate_messages_tokens_rough(baseline)
    assert all('display_metadata' not in row and 'display_kind' not in row for row in wire(rows)[0])


def test_actual_in_place_compaction_and_cold_replay_keep_source_identity(tmp_path):
    from hermes_state import SessionDB
    from agent.recovery_provenance import project_active_request_for_display
    path = tmp_path / 'compacted.db'
    db = SessionDB(path)
    db.create_session('compacted', 'test')
    origin = _origin()
    db.append_messages_batch('compacted', [origin])
    source = db.get_messages_as_conversation('compacted')[0]
    first = _compressor()._reappend_inflight_user_task([_carrier('user', 'AUTHENTIC PRIOR')], source)
    db.archive_and_compact('compacted', first)
    db.close()
    db = SessionDB(path)
    restored = db.get_messages_as_conversation('compacted')
    projected = project_active_request_for_display(restored[-1])
    assert projected is not None
    assert projected['message_uid'] == 'human-uid'
    assert projected['timestamp'] == 1234.5
    second = _compressor()._reappend_inflight_user_task([_carrier()], restored[-1])
    db.archive_and_compact('compacted', second)
    db.close()
    db = SessionDB(path)
    rows = db.get_messages_as_conversation('compacted')
    replays = [p for row in rows if (p := project_active_request_for_display(row)) is not None]
    assert replays == [projected]
    db.close()


@pytest.mark.parametrize('gate', ['verification', 'pre_verify', 'kanban'])
def test_stop_gate_recovery_is_hidden_even_before_ephemeral_filter(monkeypatch, gate):
    import agent.turn_stop_gates as gates
    a = MagicMock()
    a._verification_stop_nudges = a._pre_verify_nudges = a._kanban_stop_nudges = 0
    monkeypatch.setattr(gates, '_verify_on_stop_nudge', lambda a: 'NUDGE' if gate == 'verification' else None)
    monkeypatch.setattr(gates, '_pre_verify_nudge', lambda *a: 'NUDGE' if gate == 'pre_verify' else None)
    monkeypatch.setattr(gates, '_kanban_stop_nudge', lambda *a: 'NUDGE' if gate == 'kanban' else None)
    rows = [_origin()]
    verdict = gates.apply_stop_gates(a, {'role': 'assistant', 'content': 'real interim'},
        final_response='real interim', messages=rows, conversation_history=[],
        pending_verification_response=None, pending_verification_response_previewed=False)
    assert verdict.continue_turn
    assert rows[-1].get('display_kind') == 'hidden'
    from agent.session_persistence import _is_ephemeral_scaffolding
    assert _is_ephemeral_scaffolding(rows[-1])


def test_post_tool_empty_recovery_is_hidden_but_remains_ephemeral():
    from agent.turn_empty_response import recover_empty_response
    from agent.session_persistence import _is_ephemeral_scaffolding
    a = MagicMock()
    a._has_content_after_think_block.return_value = False
    a._last_content_with_tools = None
    a._post_tool_empty_retried = False
    a._build_assistant_message.return_value = {'role': 'assistant', 'content': ''}
    rows = [{'role': 'tool', 'content': 'result'}]
    verdict = recover_empty_response(a, SimpleNamespace(tool_calls=None), None, 'stop', final_response='',
        messages=rows, api_messages=[], conversation_history=[], active_system_prompt='', api_call_count=1,
        turn_exit_reason=None, preflight_compression_blocked=False)
    assert verdict.action == 'continue'
    assert rows[-1].get('display_kind') == 'hidden'
    assert _is_ephemeral_scaffolding(rows[-1])


def test_compression_fallback_placeholder_is_explicit_hidden():
    from agent.conversation_compression import _ensure_compressed_has_user_turn
    rows = []
    assert _ensure_compressed_has_user_turn([], rows) == 'placeholder_appended'
    assert rows[-1].get('display_kind') == 'hidden'
    assert rows[-1]['display_metadata']['hermes_provenance']['producer'] == 'compression_continuation'
