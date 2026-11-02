"""JSON schemas for Muse Spark structured outputs (object roots, no recursion, all keys required)."""
from __future__ import annotations

from ..constants import ACTIONS

ENGINE_BRIEFING_SCHEMA = {
    "type": "object",
    "properties": {
        "engine_id": {"type": "string"},
        "recommendation": {"type": "string", "enum": list(ACTIONS)},
        "headline": {"type": "string", "description": "At most 12 words."},
        "summary": {"type": "string", "description": "2-4 plain-language sentences for a maintenance engineer."},
        "key_numbers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "value": {"type": "number"},
                    "unit": {"type": "string"},
                    "source": {"type": "string", "description": "Tool the number was copied from."},
                },
                "required": ["label", "value", "unit", "source"],
                "additionalProperties": False,
            },
        },
        "sensor_drivers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "sensor": {"type": "string"},
                    "name": {"type": "string"},
                    "contribution_cycles": {"type": "number"},
                    "plain_language": {"type": "string"},
                },
                "required": ["sensor", "name", "contribution_cycles", "plain_language"],
                "additionalProperties": False,
            },
        },
        "abstain_reason": {"type": "string"},
        "next_steps": {"type": "array", "items": {"type": "string"}},
        "caveats": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["engine_id", "recommendation", "headline", "summary", "key_numbers", "sensor_drivers",
                 "abstain_reason", "next_steps", "caveats"],
    "additionalProperties": False,
}

SAFETY_REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["approve", "revise", "force_abstain"]},
        "issues": {"type": "array", "items": {"type": "string"}},
        "safety_concerns": {"type": "array", "items": {"type": "string"}},
        "rationale": {"type": "string"},
    },
    "required": ["verdict", "issues", "safety_concerns", "rationale"],
    "additionalProperties": False,
}

MAINTENANCE_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "work_orders": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "engine_id": {"type": "string"},
                    "day": {"type": "integer"},
                    "action": {"type": "string", "enum": ["INSPECT", "ENGINEERING_REVIEW"]},
                    "priority": {"type": "string", "enum": ["P1", "P2", "P3"]},
                    "reason": {"type": "string"},
                },
                "required": ["engine_id", "day", "action", "priority", "reason"],
                "additionalProperties": False,
            },
        },
        "capacity_warnings": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "work_orders", "capacity_warnings"],
    "additionalProperties": False,
}
