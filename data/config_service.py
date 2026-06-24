from __future__ import annotations

import json
import os
import socket
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
LEGACY_MODEL_KEYS = {
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_MODEL",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL",
    "ANTHROPIC_DEFAULT_OPUS_MODEL",
    "ANTHROPIC_DEFAULT_SONNET_MODEL",
    "MODEL_PROVIDER",
    "OPENAI_MODEL_PREDICTOR",
}

_OPENAI_REQUEST_LOCKS: dict[str, threading.Lock] = {}
_OPENAI_REQUEST_LOCKS_GUARD = threading.Lock()
_OPENAI_LAST_REQUEST_AT: dict[str, float] = {}

_REVIEW_DECISIONS = {"keep", "promote", "downgrade", "abstain"}
_REVIEW_ACTIONS = {"主推", "轻仓", "观望"}
_REVIEW_EVIDENCE_GRADES = {"strong", "adequate", "weak", "unsafe"}
DEFAULT_REVIEW_MAX_TOKENS = 900
MIN_REVIEW_MAX_TOKENS = 256
MAX_REVIEW_MAX_TOKENS = 4096


class ChatProtocolError(RuntimeError):
    def __init__(
        self,
        public_message: str,
        diagnostic: str = "",
        *,
        retryable: bool = False,
    ) -> None:
        super().__init__(public_message)
        self.public_message = public_message
        self.diagnostic = diagnostic or public_message
        self.retryable = retryable

    def __str__(self) -> str:
        return self.public_message


def load_env_config() -> dict[str, str]:
    """Load .env file and return as dict."""
    config: dict[str, str] = {}
    if not ENV_PATH.exists():
        return config
    with open(ENV_PATH, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            config[key.strip().lstrip("\ufeff")] = value.strip()
    return config


def save_env_config(config: dict[str, str]) -> None:
    """Save config dict to .env file."""
    lines = [f"{key}={value}\n" for key, value in config.items()]
    with open(ENV_PATH, "w", encoding="utf-8") as file:
        file.writelines(lines)


def _truncate_text(text: str, limit: int = 320) -> str:
    normalized = " ".join(str(text or "").split())
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[: limit - 1]}…"


def get_review_max_tokens(value: Any | None = None) -> int:
    raw_value = os.getenv("COLLECTION_REVIEW_MAX_TOKENS", "") if value is None else value
    text = str(raw_value or "").strip()
    if not text:
        return DEFAULT_REVIEW_MAX_TOKENS
    try:
        parsed = int(text)
    except ValueError:
        return DEFAULT_REVIEW_MAX_TOKENS
    return max(MIN_REVIEW_MAX_TOKENS, min(MAX_REVIEW_MAX_TOKENS, parsed))


def _candidate_chat_endpoints(base_url: str) -> list[str]:
    base = base_url.rstrip("/")
    candidates = [f"{base}/chat/completions"]
    if not base.lower().endswith("/v1"):
        candidates.append(f"{base}/v1/chat/completions")

    unique_candidates: list[str] = []
    seen = set()
    for url in candidates:
        if url in seen:
            continue
        seen.add(url)
        unique_candidates.append(url)
    return unique_candidates


def _extract_text_fragment(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        text_value = value.get("text")
        if isinstance(text_value, str):
            return text_value.strip()
        if isinstance(text_value, dict):
            nested = text_value.get("value")
            if isinstance(nested, str):
                return nested.strip()
        value_field = value.get("value")
        if isinstance(value_field, str):
            return value_field.strip()
        return ""
    if isinstance(value, list):
        parts = [_extract_text_fragment(item) for item in value]
        return "\n".join(part for part in parts if part).strip()
    return str(value).strip()


def _extract_text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                fragment = item.strip()
            elif isinstance(item, dict):
                if item.get("type") == "text":
                    fragment = _extract_text_fragment(item.get("text"))
                else:
                    fragment = _extract_text_fragment(item)
            else:
                fragment = ""
            if fragment:
                parts.append(fragment)
        return "\n".join(parts).strip()
    return _extract_text_fragment(content)


def _response_summary(
    *,
    endpoint: str,
    response_payload: Any | None = None,
    raw_text: str = "",
) -> str:
    parts = [f"endpoint={endpoint}"]
    finish_reason = ""
    if isinstance(response_payload, dict):
        choices = response_payload.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            finish_reason = str(choices[0].get("finish_reason", "") or "").strip()
    if finish_reason:
        parts.append(f"finish_reason={finish_reason}")

    response_excerpt = _truncate_text(raw_text)
    if not response_excerpt and response_payload is not None:
        try:
            response_excerpt = _truncate_text(json.dumps(response_payload, ensure_ascii=False))
        except TypeError:
            response_excerpt = _truncate_text(repr(response_payload))
    if response_excerpt:
        parts.append(f"response={response_excerpt}")
    return " | ".join(parts)


def _extract_chat_content(
    response_payload: dict[str, Any],
    *,
    endpoint: str,
    raw_text: str = "",
    require_non_empty_content: bool = False,
) -> str:
    choices = response_payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ChatProtocolError(
            "模型响应格式异常",
            _response_summary(endpoint=endpoint, response_payload=response_payload, raw_text=raw_text),
        )

    choice = choices[0]
    content = ""
    message = choice.get("message")
    if isinstance(message, dict):
        content = _extract_text_from_content(message.get("content", ""))
    if not content:
        content = _extract_text_from_content(choice.get("text", ""))
    # DeepSeek / 思维链模型：content 为空，内容在 reasoning_content 中
    if not content and isinstance(message, dict):
        reasoning = _extract_text_from_content(message.get("reasoning_content", ""))
        if reasoning:
            content = reasoning
    content = content.strip()
    if require_non_empty_content and not content:
        raise ChatProtocolError(
            "模型返回空响应",
            _response_summary(endpoint=endpoint, response_payload=response_payload, raw_text=raw_text),
            retryable=True,
        )
    return content


def _request_bucket(base_url: str) -> str:
    return base_url.rstrip("/").lower()


def _get_request_lock(bucket: str) -> threading.Lock:
    with _OPENAI_REQUEST_LOCKS_GUARD:
        lock = _OPENAI_REQUEST_LOCKS.get(bucket)
        if lock is None:
            lock = threading.Lock()
            _OPENAI_REQUEST_LOCKS[bucket] = lock
        return lock


def _wait_for_request_slot(bucket: str, min_interval_seconds: float) -> None:
    if min_interval_seconds <= 0:
        return
    last_request_at = _OPENAI_LAST_REQUEST_AT.get(bucket, 0.0)
    wait_seconds = min_interval_seconds - (time.monotonic() - last_request_at)
    if wait_seconds > 0:
        time.sleep(wait_seconds)


def _mark_request_finished(bucket: str) -> None:
    _OPENAI_LAST_REQUEST_AT[bucket] = time.monotonic()


def _retry_delay_seconds(
    exc: urllib.error.HTTPError,
    attempt: int,
    retry_backoff_seconds: float,
) -> float:
    retry_after = exc.headers.get("Retry-After") if exc.headers else ""
    if retry_after:
        try:
            return max(float(retry_after), 0.0)
        except ValueError:
            pass
    return max(retry_backoff_seconds, 0.5) * (2**attempt)


def _generic_retry_delay(attempt: int, retry_backoff_seconds: float) -> float:
    return max(retry_backoff_seconds, 0.5) * (2**attempt)


def _is_timeout_error(exc: BaseException) -> bool:
    if isinstance(exc, TimeoutError | socket.timeout):
        return True
    if isinstance(exc, urllib.error.URLError):
        reason = exc.reason
        if isinstance(reason, TimeoutError | socket.timeout):
            return True
        return "timed out" in str(reason).lower()
    return "timed out" in str(exc).lower()


def _format_http_error(exc: urllib.error.HTTPError, body: str) -> str:
    if body:
        return f"HTTP 错误 {exc.code}: {_truncate_text(body, 240)}"
    return f"HTTP 错误 {exc.code}: {exc.reason}"


def is_response_format_unsupported(exc: BaseException) -> bool:
    text = str(exc).lower()
    if "response_format" not in text and "json_object" not in text:
        return False
    markers = (
        "unsupported",
        "not support",
        "not supported",
        "unknown parameter",
        "invalid parameter",
        "invalid_request_error",
        "not allowed",
    )
    return any(marker in text for marker in markers)


def request_openai_compatible_chat(
    base_url: str,
    api_key: str,
    model: str,
    *,
    messages: list[dict[str, str]],
    temperature: float = 0.0,
    max_tokens: int = 128,
    timeout: int = 20,
    response_format: dict[str, Any] | None = None,
    max_retries: int = 0,
    retry_backoff_seconds: float = 4.0,
    min_interval_seconds: float = 0.0,
    serialize_requests: bool = False,
    require_non_empty_content: bool = False,
    extra_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Call an OpenAI-compatible /chat/completions endpoint."""
    if not base_url or not api_key or not model:
        raise ValueError("缺少必要参数：base_url, api_key, model")

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if response_format:
        payload["response_format"] = response_format
    if extra_payload:
        payload.update(extra_payload)

    last_error: Exception | None = None
    bucket = _request_bucket(base_url)
    request_lock = _get_request_lock(bucket)

    def _run_request() -> dict[str, Any]:
        nonlocal last_error
        for endpoint in _candidate_chat_endpoints(base_url):
            attempt = 0
            while True:
                _wait_for_request_slot(bucket, min_interval_seconds)
                request = urllib.request.Request(
                    endpoint,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {api_key}",
                    },
                    method="POST",
                )
                try:
                    with urllib.request.urlopen(request, timeout=timeout) as response:
                        raw_text = response.read().decode("utf-8", errors="replace")
                    response_payload = json.loads(raw_text)
                    content = _extract_chat_content(
                        response_payload,
                        endpoint=endpoint,
                        raw_text=raw_text,
                        require_non_empty_content=require_non_empty_content,
                    )
                    _mark_request_finished(bucket)
                    return {
                        "endpoint": endpoint,
                        "content": content,
                        "response": response_payload,
                        "raw_response": raw_text,
                    }
                except urllib.error.HTTPError as exc:
                    body = exc.read().decode("utf-8", errors="replace")
                    _mark_request_finished(bucket)
                    last_error = RuntimeError(_format_http_error(exc, body))
                    if exc.code in {404, 405}:
                        break
                    if exc.code == 429 and attempt < max_retries:
                        time.sleep(_retry_delay_seconds(exc, attempt, retry_backoff_seconds))
                        attempt += 1
                        continue
                    raise last_error from exc
                except ChatProtocolError as exc:
                    _mark_request_finished(bucket)
                    last_error = exc
                    if exc.retryable and attempt < max_retries:
                        time.sleep(_generic_retry_delay(attempt, retry_backoff_seconds))
                        attempt += 1
                        continue
                    raise
                except urllib.error.URLError as exc:
                    _mark_request_finished(bucket)
                    if _is_timeout_error(exc) and attempt < max_retries:
                        time.sleep(_generic_retry_delay(attempt, retry_backoff_seconds))
                        attempt += 1
                        continue
                    raise RuntimeError(f"网络错误: {exc.reason}") from exc
                except (TimeoutError, socket.timeout) as exc:
                    _mark_request_finished(bucket)
                    if attempt < max_retries:
                        time.sleep(_generic_retry_delay(attempt, retry_backoff_seconds))
                        attempt += 1
                        continue
                    raise RuntimeError(f"请求超时: {exc}") from exc
                except json.JSONDecodeError as exc:
                    _mark_request_finished(bucket)
                    raise RuntimeError(f"响应不是有效 JSON: {exc}") from exc

        if last_error is not None:
            raise last_error
        raise RuntimeError("未能请求聊天接口")

    if serialize_requests:
        with request_lock:
            return _run_request()
    return _run_request()


def _validate_review_test_payload(content: str) -> None:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError("复核模型返回非 JSON 内容") from exc

    if not isinstance(payload, dict):
        raise ValueError("复核模型返回非法 JSON 结构")

    decision = str(payload.get("decision", "")).strip().lower()
    target_action = str(payload.get("target_action", "")).strip()
    reason = str(payload.get("reason", "")).strip()
    risk_flags = payload.get("risk_flags")
    evidence_grade = str(payload.get("evidence_grade", "") or "").strip().lower()
    confidence_delta = payload.get("confidence_delta", 0.0)
    stake_multiplier = payload.get("stake_multiplier", 1.0)
    outcome_decision = str(payload.get("outcome_decision", "") or "confirm").strip().lower()
    target_outcome = str(payload.get("target_outcome", "") or "").strip().lower()

    if decision not in _REVIEW_DECISIONS:
        raise ValueError("复核模型返回非法 JSON 结构")
    if target_action not in _REVIEW_ACTIONS:
        raise ValueError("复核模型返回非法 JSON 结构")
    if not reason:
        raise ValueError("复核模型返回非法 JSON 结构")
    if not isinstance(risk_flags, list):
        raise ValueError("复核模型返回非法 JSON 结构")
    if evidence_grade and evidence_grade not in _REVIEW_EVIDENCE_GRADES:
        raise ValueError("复核模型返回非法 JSON 结构")
    try:
        confidence_delta_float = float(confidence_delta)
        stake_multiplier_float = float(stake_multiplier)
    except (TypeError, ValueError) as exc:
        raise ValueError("复核模型返回非法 JSON 结构") from exc
    if not -0.12 <= confidence_delta_float <= 0.08:
        raise ValueError("复核模型返回非法 JSON 结构")
    if not 0.0 <= stake_multiplier_float <= 1.0:
        raise ValueError("复核模型返回非法 JSON 结构")
    if outcome_decision not in {"confirm", "challenge", "veto_to_watch"}:
        raise ValueError("复核模型返回非法 JSON 结构")
    if outcome_decision in {"confirm", "challenge"} and target_outcome and target_outcome not in {"home", "draw", "away"}:
        raise ValueError("复核模型返回非法 JSON 结构")


def _review_test_messages() -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "你是足球下注系统的动作强度复核器。直接输出最终 JSON，不要输出推理过程、markdown 或解释。",
        },
        {
            "role": "user",
            "content": (
                "只返回严格 JSON，不要输出 markdown 或解释。\n"
                '固定结构：{"outcome_decision":"confirm|challenge|veto_to_watch",'
                '"target_outcome":"home|draw|away","outcome_reason":"一句中文赛果理由",'
                '"decision":"keep|promote|downgrade|abstain",'
                '"target_action":"主推|轻仓|观望","confidence_delta":0.0,'
                '"stake_multiplier":1.0,"evidence_grade":"strong|adequate|weak|unsafe",'
                '"reason":"一句中文理由","risk_flags":["flag1"]}\n\n'
                "比赛: 测试主队 vs 测试客队\n"
                "联赛: 测试联赛 | 时间: 2026-04-28 19:35\n"
                "推荐方向(不可改): 主胜\n"
                "算法初判动作: 轻仓\n"
                "算法风险等级: medium\n"
                "算法置信度: 0.648\n"
                "算法建议仓位: 1.25%\n"
                "数据质量: 0.812\n"
                '主胜/平局/客胜概率: {"home": 0.49, "draw": 0.27, "away": 0.24}\n'
                '市场隐含概率: {"home": 0.44, "draw": 0.29, "away": 0.27}\n'
                '市场偏差: {"home": 0.05, "draw": -0.02, "away": -0.03}\n'
                'EV: {"home": 0.12, "draw": -0.05, "away": -0.09}\n'
                "风险提示: 模型探活测试\n"
                "如果证据不足，优先 keep 或 abstain。\n"
                "confidence_delta 范围 -0.12 到 0.08，stake_multiplier 范围 0 到 1。\n"
                "最终只输出一行 JSON，reason 控制在 40 个中文字符以内，risk_flags 最多 3 项。"
            ),
        },
    ]


def _test_openai_compatible_api(base_url: str, api_key: str, model: str) -> dict[str, Any]:
    try:
        response = request_openai_compatible_chat(
            base_url,
            api_key,
            model,
            messages=[{"role": "user", "content": "请回复 OK"}],
            temperature=0.0,
            max_tokens=16,
            timeout=15,
            require_non_empty_content=True,
        )
        return {
            "success": True,
            "message": "连接成功，摘要模型可用",
            "response": response["content"][:240],
        }
    except ValueError as exc:
        return {"success": False, "message": str(exc)}
    except ChatProtocolError as exc:
        public_message = exc.public_message
        if public_message.startswith("模型"):
            public_message = public_message[2:]
        return {
            "success": False,
            "message": f"连接成功，但摘要模型{public_message}",
            "detail": exc.diagnostic,
        }
    except RuntimeError as exc:
        return {"success": False, "message": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"success": False, "message": f"未知错误: {exc}"}


def test_openai_api(base_url: str, api_key: str, model: str) -> dict[str, Any]:
    """Test summary model endpoint."""
    return _test_openai_compatible_api(base_url, api_key, model)


def test_collection_api(
    base_url: str,
    api_key: str,
    model: str,
    max_tokens: Any | None = None,
) -> dict[str, Any]:
    """Test review model endpoint with the same strict JSON contract as production."""
    request_kwargs = {
        "messages": _review_test_messages(),
        "temperature": 0.0,
        "max_tokens": get_review_max_tokens(max_tokens),
        "timeout": 20,
        "max_retries": 2,
        "retry_backoff_seconds": 2.0,
        "serialize_requests": True,
        "require_non_empty_content": True,
    }

    try:
        try:
            response = request_openai_compatible_chat(
                base_url,
                api_key,
                model,
                response_format={"type": "json_object"},
                **request_kwargs,
            )
        except Exception as exc:  # noqa: BLE001
            if not is_response_format_unsupported(exc):
                raise
            response = request_openai_compatible_chat(
                base_url,
                api_key,
                model,
                **request_kwargs,
            )

        _validate_review_test_payload(response["content"])
        return {
            "success": True,
            "message": "连接成功，复核模型可用",
            "response": _truncate_text(response["content"], 240),
        }
    except ValueError as exc:
        public_message = str(exc)
        if "非 JSON" in public_message:
            message = "连接成功，但复核模型返回非 JSON 内容，不适合作为复核模型"
        else:
            message = "连接成功，但复核模型返回非法 JSON 结构，不适合作为复核模型"
        return {
            "success": False,
            "message": message,
            "detail": public_message,
        }
    except ChatProtocolError as exc:
        if exc.public_message == "模型返回空响应":
            message = "连接成功，但复核模型返回空响应，不适合作为复核模型"
        elif exc.public_message == "模型响应格式异常":
            message = "连接成功，但复核模型返回非 JSON 内容，不适合作为复核模型"
        else:
            message = f"连接成功，但复核模型{exc.public_message}，不适合作为复核模型"
        return {
            "success": False,
            "message": message,
            "detail": exc.diagnostic,
        }
    except RuntimeError as exc:
        text = str(exc)
        if "响应不是有效 JSON" in text:
            return {
                "success": False,
                "message": "连接成功，但复核模型返回非 JSON 内容，不适合作为复核模型",
                "detail": text,
            }
        return {"success": False, "message": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"success": False, "message": f"未知错误: {exc}"}


def reload_env_to_os() -> None:
    """Reload .env config into os.environ."""
    config = load_env_config()
    for key in LEGACY_MODEL_KEYS:
        os.environ.pop(key, None)
    for key, value in config.items():
        os.environ[key] = value
