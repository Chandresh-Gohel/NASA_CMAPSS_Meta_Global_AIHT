"""Meta Model API client (OpenAI-compatible): disk cache, retries and offline mode.

Chat Completions for Muse Spark (tool calling, structured output). No image paths.
``tool_choice`` is never sent (only "auto" is supported); structured output uses
``response_format={"type": "json_schema", ...}``.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from typing import Any

from ..config import Config, ensure_dir, load_dotenv_if_present, path_for


class OfflineError(RuntimeError):
    """The API is unreachable (or offline mode is on) and the response isn't cached."""


class ApiError(RuntimeError):
    """A non-retryable API error, or a transient one that kept failing."""


def json_schema_format(name: str, schema: dict, strict: bool = False) -> dict:
    return {"type": "json_schema", "json_schema": {"name": name, "schema": schema, "strict": bool(strict)}}


def parse_json_object(text: str | None) -> dict:
    """Pull a JSON object out of model output (tolerates fences and leading prose)."""
    if not text:
        raise ValueError("empty model output")
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.S).strip()
    try:
        obj = json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", t, re.S)
        if not m:
            raise ValueError(f"no JSON object in model output: {t[:200]}")
        obj = json.loads(m.group(0))
    if not isinstance(obj, dict):
        raise ValueError("model output is not a JSON object")
    return obj


class MetaClient:
    def __init__(self, cfg: Config):
        load_dotenv_if_present()
        ac = cfg.agent
        self.cfg = cfg
        self.api_key = os.environ.get(str(ac.api_key_env), "") or os.environ.get("META_API_KEY", "")
        self.base_url = os.environ.get(str(ac.base_url_env), "") or str(ac.default_base_url)
        self.model = os.environ.get(str(ac.model_env), "") or str(ac.default_model)
        self.offline = os.environ.get(str(ac.offline_env), "0") == "1" or not self.api_key
        self.use_cache = bool(ac.cache)
        self.cache_dir = ensure_dir(path_for(cfg, "cache_dir", "llm"))
        self.timeout = float(ac.timeout_s)
        self.max_retries = int(ac.max_retries)
        self.reasoning_effort = ac.get("reasoning_effort")
        self.temperature = ac.get("temperature")
        self.usage = {"calls": 0, "cache_hits": 0, "prompt_tokens": 0, "completion_tokens": 0}
        self._client = None
        if not self.offline:
            from openai import OpenAI

            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.timeout, max_retries=0)

    @property
    def available(self) -> bool:
        return self._client is not None

    def _key(self, kind: str, payload: dict) -> str:
        blob = json.dumps({"kind": kind, **payload}, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _cache_get(self, key: str):
        p = self.cache_dir / f"{key}.json"
        if self.use_cache and p.exists():
            self.usage["cache_hits"] += 1
            return json.loads(p.read_text(encoding="utf-8"))
        return None

    def _cache_put(self, key: str, value) -> None:
        if not self.use_cache:
            return
        p = self.cache_dir / f"{key}.json"
        # Atomic write (temp file + rename) so a concurrent reader never sees half a file.
        tmp = self.cache_dir / f"{key}.{os.getpid()}.tmp"
        try:
            tmp.write_text(json.dumps(value), encoding="utf-8")
            os.replace(tmp, p)
        finally:
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass

    def _with_retries(self, fn):
        from openai import APIConnectionError, APIStatusError, APITimeoutError, InternalServerError, RateLimitError

        delay = 1.0
        for attempt in range(self.max_retries + 1):
            try:
                return fn()
            except (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError) as exc:
                if attempt >= self.max_retries:
                    raise ApiError(f"transient API failure after {attempt + 1} attempts: {exc}") from exc
                retry_after = getattr(exc, "retry_after", None)
                time.sleep(float(retry_after) if retry_after else delay)
                delay *= 2
            except APIStatusError as exc:
                raise ApiError(f"HTTP {exc.status_code}: {exc}") from exc
        raise ApiError("unreachable")

    def _track(self, usage: Any) -> None:
        self.usage["calls"] += 1
        if usage is None:
            return
        self.usage["prompt_tokens"] += int(getattr(usage, "prompt_tokens", 0) or getattr(usage, "input_tokens", 0) or 0)
        self.usage["completion_tokens"] += int(getattr(usage, "completion_tokens", 0) or getattr(usage, "output_tokens", 0) or 0)

    @staticmethod
    def _message_to_dict(resp) -> dict:
        choice = resp.choices[0]
        msg = choice.message
        calls = [{"id": tc.id, "type": "function",
                  "function": {"name": tc.function.name, "arguments": tc.function.arguments or "{}"}}
                 for tc in (msg.tool_calls or [])]
        return {"role": "assistant", "content": msg.content or "", "tool_calls": calls,
                "finish_reason": getattr(choice, "finish_reason", None)}

    def chat(self, messages: list[dict], tools: list[dict] | None = None, response_format: dict | None = None,
             use_cache: bool = True, model: str | None = None) -> dict:
        kwargs: dict[str, Any] = {"model": model or self.model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
        if response_format:
            kwargs["response_format"] = response_format
        if self.temperature is not None:
            kwargs["temperature"] = float(self.temperature)
        extra: dict[str, Any] = {}
        if self.reasoning_effort:
            extra["reasoning_effort"] = str(self.reasoning_effort)

        key = self._key("chat", {**kwargs, "extra": extra})
        if use_cache:
            cached = self._cache_get(key)
            if cached is not None:
                return cached
        if not self.available:
            raise OfflineError("Meta Model API is not configured (set MODEL_API_KEY) and this response is not cached.")

        for _ in range(3):
            try:
                resp = self._with_retries(lambda: self._client.chat.completions.create(**kwargs, extra_body=extra or None))
                break
            except ApiError as exc:
                text = str(exc).lower()
                if "reasoning" in text and extra:
                    extra = {}
                elif "temperature" in text and "temperature" in kwargs:
                    kwargs.pop("temperature")
                else:
                    raise
        else:
            raise ApiError("chat request failed after removing optional parameters")
        self._track(getattr(resp, "usage", None))
        out = self._message_to_dict(resp)
        self._cache_put(key, out)
        return out
