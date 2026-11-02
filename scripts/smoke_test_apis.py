"""Step 2 smoke test: Muse Spark chat, tool calling, structured output (no images).

    python scripts/smoke_test_apis.py

Reads MODEL_API_KEY from the environment or .env. Caching is off, so every call hits the API.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ehagent.agent.client import MetaClient, json_schema_format, parse_json_object  # noqa: E402
from ehagent.config import load_config  # noqa: E402


def main() -> None:
    cfg = load_config()
    client = MetaClient(cfg)
    client.use_cache = False
    if not client.available:
        print("MODEL_API_KEY is not set (or EHA_OFFLINE=1). Put it in .env first.")
        sys.exit(1)
    print(f"base_url={client.base_url}  model={client.model}")

    out = client.chat([{"role": "user", "content": "Reply with exactly: OK"}], use_cache=False)
    print("1. chat:", out["content"][:80])

    tools = [{"type": "function", "function": {
        "name": "get_engine_rul", "description": "Return the remaining useful life of an engine.",
        "parameters": {"type": "object", "properties": {"engine_id": {"type": "string"}}, "required": ["engine_id"]},
    }}]
    msgs: list[dict] = [{"role": "user", "content": "What is the remaining useful life of engine FD001-TE-001? Use the tool."}]
    out = client.chat(msgs, tools=tools, use_cache=False)
    print("2. tool calls:", out["tool_calls"])
    if out["tool_calls"]:
        msgs.append({"role": "assistant", "content": out["content"], "tool_calls": out["tool_calls"]})
        for call in out["tool_calls"]:
            msgs.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps({"engine_id": "FD001-TE-001", "rul_mean": 87.4})})
        out = client.chat(msgs, tools=tools, use_cache=False)
        print("   final answer:", out["content"][:200])

    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}, "note": {"type": "string"}},
              "required": ["ok", "note"], "additionalProperties": False}
    out = client.chat([{"role": "user", "content": "Return ok=true and a five-word note."}],
                      response_format=json_schema_format("smoke_test", schema), use_cache=False)
    print("3. structured output:", parse_json_object(out["content"]))
    print("usage:", client.usage)


if __name__ == "__main__":
    main()
