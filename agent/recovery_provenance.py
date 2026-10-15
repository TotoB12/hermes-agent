"""Display-only provenance for synthetic recovery and active-request restatements.

These fields live under ``display_metadata`` so all transports strip them before
provider replay. They are not text classifiers: an identical human-authored
string without the explicit metadata remains an ordinary message.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional

PROVENANCE_KEY = "hermes_provenance"


def mark_recovery_message(message: Dict[str, Any], producer: str) -> Dict[str, Any]:
    """Mark model-only scaffolding, preserving its role, bytes and lifecycle flags."""
    message["display_kind"] = "hidden"
    message["display_metadata"] = {
        **(message.get("display_metadata") or {}),
        PROVENANCE_KEY: {"version": 1, "kind": "recovery_nudge", "producer": producer},
    }
    return message


def active_request_provenance(message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The supported explicit replay contract, or None (never infer from content)."""
    metadata = message.get("display_metadata")
    provenance = metadata.get(PROVENANCE_KEY) if isinstance(metadata, dict) else None
    if (
        isinstance(provenance, dict)
        and provenance.get("version") == 1
        and provenance.get("kind") == "active_request_replay"
        and provenance.get("producer") == "context_compressor"
        and provenance.get("placement") in {"standalone", "summary_suffix"}
        and isinstance(provenance.get("display_content"), str)
    ):
        return provenance
    return None


def mark_active_request_replay(
    message: Dict[str, Any], origin: Dict[str, Any], *, placement: str,
    fallback_text: str, origin_is_carrier: bool = False,
) -> None:
    """Attach original request identity and safe display text without changing execution.

    Replays are meaningful user intent, not hidden recovery nudges. A merged
    carrier's own uid/time must never replace the request's identity on the next
    compaction. Legacy carriers with no provenance have no provable source uid.
    """
    previous = active_request_provenance(origin)
    if previous is not None:
        provenance = deepcopy(previous)
    else:
        content = origin.get("content")
        # Keep exact literal text, including examples of our own headers. The
        # existing model-facing restatement may strip those; display must not.
        if not origin_is_carrier and isinstance(content, str):
            display_content = content
        elif not origin_is_carrier and isinstance(content, list):
            display_content = "\n".join(
                part if isinstance(part, str) else part.get("text", "")
                for part in content
                if isinstance(part, str) or (
                    isinstance(part, dict) and part.get("type") in {"text", "input_text"}
                    and isinstance(part.get("text"), str)
                )
            )
        else:
            display_content = fallback_text
        provenance = {
            "version": 1, "kind": "active_request_replay", "producer": "context_compressor",
            "origin_message_uid": None if origin_is_carrier else origin.get("message_uid"),
            "origin_platform_message_id": None if origin_is_carrier else (
                origin.get("platform_message_id") or origin.get("message_id")),
            "origin_timestamp": None if origin_is_carrier else origin.get("timestamp"),
            "display_content": display_content,
        }
    provenance["placement"] = placement
    message["display_metadata"] = {
        **deepcopy(message.get("display_metadata") or {}), PROVENANCE_KEY: provenance,
    }


def project_active_request_for_display(message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """A separate human request projection, never the summary/recovery wrapper.

    For ``summary_suffix`` carriers, retain any independent prior-tail view from
    ``project_compaction_message_for_display`` as well. This projection uses the
    original uid/time for deduplication and retention, not the new carrier time.
    Missing legacy origin fields remain None; do not invent source identities.
    """
    provenance = active_request_provenance(message)
    if provenance is None:
        return None
    return {
        "role": "user", "content": provenance["display_content"],
        "message_uid": provenance.get("origin_message_uid"),
        "platform_message_id": provenance.get("origin_platform_message_id"),
        "timestamp": provenance.get("origin_timestamp"),
    }
