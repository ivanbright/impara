"""Provider-agnostic access to an OpenAI-compatible chat-completion server.

Design rules, in priority order:

1. The CLI must work with no model configured. Nothing in this module is
   required by the deterministic pipeline; configuration is read on demand
   and every failure is reported as data, never raised at import time.
2. No SDK dependency. ``urllib`` only, so the install footprint does not
   grow and offline tests never need a network stack.
3. Any OpenAI-compatible endpoint works - OpenAI, Anthropic's compat layer,
   Ollama, LM Studio, vLLM, a corporate proxy - because only the base URL,
   model name and optional key are configurable.
"""
import json
import os
import re
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional

DEFAULT_TIMEOUT = 60
DEFAULT_MAX_TOKENS = 4096
MAX_429_ATTEMPTS = 3
RATE_LIMIT_WAIT_CAP = 120.0

ENV_BASE_URL = "IMPARA_LLM_BASE_URL"
ENV_MODEL = "IMPARA_LLM_MODEL"
ENV_API_KEY = "IMPARA_LLM_API_KEY"
ENV_TIMEOUT = "IMPARA_LLM_TIMEOUT"
ENV_MAX_TOKENS = "IMPARA_LLM_MAX_TOKENS"

HINT = (
    "Set %s and %s to any OpenAI-compatible server.\n"
    "  Ollama:      set %s=http://localhost:11434/v1\n"
    "               set %s=llama3.1\n"
    "  LM Studio:   set %s=http://localhost:1234/v1\n"
    "  OpenAI:      set %s=https://api.openai.com/v1\n"
    "               set %s=gpt-4o-mini   and %s=<your key>"
    % (ENV_BASE_URL, ENV_MODEL, ENV_BASE_URL, ENV_MODEL,
       ENV_BASE_URL, ENV_BASE_URL, ENV_MODEL, ENV_API_KEY)
)


class LLMError(RuntimeError):
    """The model could not be reached, or did not return usable output."""


@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    model: str
    api_key: str = ""
    timeout: int = DEFAULT_TIMEOUT
    temperature: float = 0.0
    max_tokens: int = 0  # 0 = leave to the provider's default

    @property
    def endpoint(self) -> str:
        base = self.base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        return base + "/chat/completions"


# A transport takes (config, request payload) and returns the parsed JSON
# response body. Tests inject a fake instead of opening a socket.
Transport = Callable[[LLMConfig, Dict[str, Any]], Dict[str, Any]]


def _get(env: Mapping[str, str], *names: str) -> str:
    for name in names:
        value = (env.get(name) or "").strip()
        if value:
            return value
    return ""


def config_from_env(env: Optional[Mapping[str, str]] = None) -> Optional[LLMConfig]:
    """Return the configured model, or None when no model is available."""
    env = os.environ if env is None else env

    base_url = _get(env, ENV_BASE_URL, "OPENAI_BASE_URL")
    model = _get(env, ENV_MODEL)
    if not base_url or not model:
        return None

    timeout_raw = _get(env, ENV_TIMEOUT)
    try:
        timeout = int(timeout_raw) if timeout_raw else DEFAULT_TIMEOUT
    except ValueError:
        timeout = DEFAULT_TIMEOUT

    max_raw = _get(env, ENV_MAX_TOKENS)
    try:
        max_tokens = int(max_raw) if max_raw else DEFAULT_MAX_TOKENS
    except ValueError:
        max_tokens = DEFAULT_MAX_TOKENS
    if max_tokens < 0:
        max_tokens = 0

    return LLMConfig(
        base_url=base_url,
        model=model,
        api_key=_get(env, ENV_API_KEY, "OPENAI_API_KEY"),
        timeout=max(1, timeout),
        max_tokens=max_tokens,
    )


def available(env: Optional[Mapping[str, str]] = None) -> bool:
    return config_from_env(env) is not None


def unavailable_reason(env: Optional[Mapping[str, str]] = None) -> str:
    if config_from_env(env) is not None:
        return ""
    return HINT


def _retry_delay(exc: urllib.error.HTTPError, attempt: int) -> float:
    """Seconds to wait before retrying a rate-limited request."""
    raw = exc.headers.get("Retry-After")
    if raw:
        try:
            return max(0.0, min(float(raw), RATE_LIMIT_WAIT_CAP))
        except ValueError:
            pass
    try:
        detail = exc.read().decode("utf-8", "replace")
    except Exception:
        detail = ""
    m = re.search(r"try again in ([\d.]+)\s*s", detail, re.IGNORECASE)
    if m:
        return max(0.0, min(float(m.group(1)), RATE_LIMIT_WAIT_CAP))
    return min(30.0, 5.0 * 2 ** (attempt - 1))


def _urlopen_transport(config: LLMConfig, payload: Dict[str, Any]) -> Dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = "Bearer " + config.api_key
    from . import __version__
    headers["User-Agent"] = "impara/%s" % __version__

    request = urllib.request.Request(
        config.endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    attempts = 0
    while True:
        attempts += 1
        try:
            with urllib.request.urlopen(request, timeout=config.timeout) as response:
                raw = response.read().decode("utf-8", "replace")
            break
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace")[:300]
            except Exception:
                pass
            if exc.code == 429 and attempts < MAX_429_ATTEMPTS:
                time.sleep(_retry_delay(exc, attempts))
                continue
            raise LLMError("server returned HTTP %d for %s: %s" % (exc.code, config.endpoint, detail))
        except urllib.error.URLError as exc:
            raise LLMError("cannot reach %s: %s" % (config.endpoint, getattr(exc, "reason", exc)))
        except socket.timeout:
            raise LLMError("timed out after %ds talking to %s" % (config.timeout, config.endpoint))

    try:
        body = json.loads(raw)
    except ValueError:
        raise LLMError("server returned a body that is not JSON")
    if not isinstance(body, dict):
        raise LLMError("server returned an unexpected JSON shape")
    return body


def chat(
    config: LLMConfig,
    messages: List[Dict[str, str]],
    transport: Optional[Transport] = None,
) -> str:
    """One chat completion. Returns the assistant message as text."""
    payload: Dict[str, Any] = {
        "model": config.model,
        "messages": messages,
        "temperature": config.temperature,
    }
    if config.max_tokens > 0:
        payload["max_tokens"] = config.max_tokens
    send = _urlopen_transport if transport is None else transport
    body = send(config, payload)

    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise LLMError("response from %s did not contain choices[0].message.content" % config.endpoint)

    if not isinstance(content, str) or not content.strip():
        raise LLMError("model returned an empty message")
    return content


def extract_json(text: str) -> Any:
    """Pull the JSON object out of a model reply, tolerating code fences."""
    body = (text or "").strip()
    if body.startswith("```"):
        body = re.sub(r"^```[A-Za-z]*\s*", "", body)
        body = re.sub(r"\s*```$", "", body)
    start = body.find("{")
    end = body.rfind("}")
    if start < 0 or end <= start:
        raise LLMError("model reply contained no JSON object")
    try:
        return json.loads(body[start:end + 1])
    except ValueError as exc:
        tail = body.rstrip()
        truncated = not tail.endswith("}")
        message = "model reply was not valid JSON: %s" % exc
        if truncated:
            message += (
                "; the reply appears truncated - the model may have hit an "
                "output limit or the account's credit budget"
            )
        raise LLMError(message)


def chat_json(
    config: LLMConfig,
    messages: List[Dict[str, str]],
    transport: Optional[Transport] = None,
) -> Any:
    return extract_json(chat(config, messages, transport=transport))
