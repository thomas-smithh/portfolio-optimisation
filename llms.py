import json
from pathlib import Path
from langchain_openai import ChatOpenAI
import ssl
import warnings
from typing import Any, Optional, Iterable, List
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage

REASONING_LEVELS = ("low", "medium", "high", "xhigh")

with open("api_keys.json", "r", encoding="utf-8") as f:
    api_keys = json.load(f)["api_keys"][0]

try:
    from openai import (
        APIConnectionError,
        APITimeoutError,
        InternalServerError,
        RateLimitError,
        UsageLimitExceededError,
    )
    _OPENAI_RETRYABLE_EXCEPTIONS = (
        APIConnectionError,
        APITimeoutError,
        RateLimitError,
        UsageLimitExceededError,
        InternalServerError,
    )
except Exception:  # pragma: no cover
    _OPENAI_RETRYABLE_EXCEPTIONS = ()

try:  # httpx/httpcore can surface raw TLS/connect errors before OpenAI wraps them
    import httpx  # type: ignore
    _HTTPX_RETRYABLE_EXCEPTIONS = (
        httpx.ConnectError,
        httpx.ReadError,
        httpx.RemoteProtocolError,
    )
except Exception:  # pragma: no cover
    _HTTPX_RETRYABLE_EXCEPTIONS = ()

try:
    import httpcore  # type: ignore
    _HTTPCORE_RETRYABLE_EXCEPTIONS = (
        httpcore.ConnectError,
        httpcore.ReadError,
        httpcore.RemoteProtocolError,
    )
except Exception:  # pragma: no cover
    _HTTPCORE_RETRYABLE_EXCEPTIONS = ()

_SSL_RETRYABLE_EXCEPTIONS = (ssl.SSLError, ssl.SSLEOFError)

warnings.filterwarnings('ignore')

class ContentFilterRetryWrapper:
    def __init__(
        self,
        runnable: Any,
        temperature_schedule: Optional[Iterable[float]] = None,
        max_attempts: int = 5,
        api_retry_max_attempts: int = 8,
        api_retry_base_seconds: float = 1.0,
        api_retry_max_seconds: float = 300.0,
        api_retry_jitter: float = 0.1,
    ) -> None:
        self._runnable = runnable
        base_temp = getattr(runnable, "temperature", 0)
        schedule = list(temperature_schedule) if temperature_schedule else [base_temp, 0.005, 0.001, 0.002, 0.01]
        # preserve order while de-duplicating
        seen = set()
        self._temperature_schedule = [t for t in schedule if not (t in seen or seen.add(t))]
        self._max_attempts = max_attempts
        self._api_retry_max_attempts = api_retry_max_attempts
        self._api_retry_base_seconds = api_retry_base_seconds
        self._api_retry_max_seconds = api_retry_max_seconds
        self._api_retry_jitter = api_retry_jitter
        self._api_retryable_exceptions = (
            _OPENAI_RETRYABLE_EXCEPTIONS
            + _HTTPX_RETRYABLE_EXCEPTIONS
            + _HTTPCORE_RETRYABLE_EXCEPTIONS
            + _SSL_RETRYABLE_EXCEPTIONS
        )

    def _get_response_metadata(self, response: Any) -> dict:
        metadata = getattr(response, "response_metadata", None)
        return metadata if isinstance(metadata, dict) else {}

    def _get_incomplete_reason(self, response: Any) -> Optional[str]:
        metadata = self._get_response_metadata(response)
        incomplete = metadata.get("incomplete_details")
        if not isinstance(incomplete, dict):
            return None
        reason = incomplete.get("reason")
        return str(reason) if reason is not None else None

    def _is_max_output_tokens_incomplete(self, response: Any) -> bool:
        metadata = self._get_response_metadata(response)
        status = metadata.get("status")
        if status != "incomplete":
            return False
        return self._get_incomplete_reason(response) == "max_output_tokens"

    def _bind_reasoning_effort(self, runnable: Any, effort: str) -> Any:
        if hasattr(runnable, "bind"):
            return runnable.bind(reasoning={"effort": effort})
        return runnable

    def _get_reasoning_effort(self, runnable: Any) -> Optional[str]:
        reasoning = getattr(runnable, "reasoning", None)
        if isinstance(reasoning, dict):
            effort = reasoning.get("effort")
            if isinstance(effort, str):
                return effort

        model_kwargs = getattr(runnable, "model_kwargs", None)
        if isinstance(model_kwargs, dict):
            reasoning = model_kwargs.get("reasoning")
            if isinstance(reasoning, dict):
                effort = reasoning.get("effort")
                if isinstance(effort, str):
                    return effort

        return None

    def _get_lower_reasoning_efforts(self, runnable: Any) -> List[str]:
        current_effort = self._get_reasoning_effort(runnable)
        effort_order = ["xhigh", "high", "medium", "low", "minimal"]
        if current_effort not in effort_order:
            return []
        current_index = effort_order.index(current_effort)
        return effort_order[current_index + 1 :]

    def _is_content_filtered(self, response: Any) -> bool:
        try:
            metadata = self._get_response_metadata(response)
            incomplete = metadata.get("incomplete_details", {}) if isinstance(metadata, dict) else {}
            if incomplete.get("reason") == "content_filter":
                return True
            status = metadata.get("status") if isinstance(metadata, dict) else None
            if status == "incomplete" and incomplete.get("reason") == "content_filter":
                return True
        except Exception:
            pass
        return False

    def _bind_temperature(self, temperature: float) -> Any:
        if hasattr(self._runnable, "bind"):
            return self._runnable.bind(temperature=temperature)
        return self._runnable

    def _is_item_not_found_error(self, exc: Exception) -> bool:
        msg = str(exc)
        # Observed error shape from OpenAI Responses API via langchain_openai:
        # "Item with id 'rs_...' not found." (param: input)
        return (
            "item with id" in msg.lower()
            and "not found" in msg.lower()
            and "param" in msg.lower()
            and "input" in msg.lower()
        )

    def _is_context_length_exceeded_error(self, exc: Exception) -> bool:
        msg = str(exc).lower()
        return (
            "context_length_exceeded" in msg
            or "exceeds the context window" in msg
            or "your input exceeds the context window" in msg
        )

    def _is_invalid_encrypted_content_error(self, exc: Exception) -> bool:
        msg = str(exc).lower()
        return "invalid_encrypted_content" in msg or "encrypted content" in msg

    def _is_missing_reasoning_item_error(self, exc: Exception) -> bool:
        msg = str(exc).lower()
        return (
            "reasoning" in msg
            and "required" in msg
            and ("provided without" in msg or "without its required" in msg)
        )

    def _is_rate_limit_or_capacity_error(self, exc: Exception) -> bool:
        msg = str(exc).lower()
        if "usage limit" in msg or "usagelimitexceedederror" in msg:
            return True
        if "429" in msg:
            return True
        if "rate limit" in msg or "ratelimit" in msg:
            return True
        if "too_many_requests" in msg:
            return True
        if "no_capacity" in msg:
            return True
        if "high demand" in msg and "maximum usage size allowed" in msg:
            return True
        status_code = getattr(exc, "status_code", None)
        if status_code == 429:
            return True
        response = getattr(exc, "response", None)
        if getattr(response, "status_code", None) == 429:
            return True
        return False

    def _get_retry_after_seconds(self, exc: Exception) -> Optional[float]:
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None)
        if not headers:
            return None

        retry_after = headers.get("retry-after") or headers.get("Retry-After")
        if not retry_after:
            return None

        try:
            return max(0.0, float(retry_after))
        except (TypeError, ValueError):
            pass

        try:
            retry_at = parsedate_to_datetime(retry_after)
            return max(0.0, retry_at.timestamp() - time.time())
        except Exception:
            return None

    def _get_retry_delay_seconds(self, attempt: int, exc: Optional[Exception] = None) -> float:
        retry_after = self._get_retry_after_seconds(exc) if exc is not None else None
        if retry_after is not None:
            delay = retry_after
        else:
            delay = self._api_retry_base_seconds * (2 ** attempt)
        delay = min(self._api_retry_max_seconds, delay)
        if self._api_retry_jitter:
            jitter = random.uniform(-self._api_retry_jitter, self._api_retry_jitter)
            delay = max(0.0, delay * (1.0 + jitter))
        return delay

    def _get_context_window_tokens(self) -> int:
        """Best-effort guess of the model context window.

        AzureChatOpenAI does not reliably expose context limits, so we pick a conservative default.
        """

        model_name = getattr(self._runnable, "model", None) or getattr(self._runnable, "model_name", None)
        model_name = str(model_name).lower() if model_name else ""
        if "gpt-5.4" in model_name:
            return 1_000_000    
        elif "gpt-5" in model_name or "gpt5" in model_name:
            return 240_000
        elif "gpt-4" in model_name:
            return 1_000_000
        else:
            return 64_000

    def _get_encoding(self):
        if tiktoken is None:
            return None
        model_name = getattr(self._runnable, "model", None) or getattr(self._runnable, "model_name", None)
        if isinstance(model_name, str) and model_name:
            try:
                return tiktoken.encoding_for_model(model_name)
            except Exception:
                pass
        try:
            return tiktoken.get_encoding("o200k_base")
        except Exception:
            return tiktoken.get_encoding("cl100k_base")

    def _truncate_text_to_tokens_keep_tail(self, text: str, max_tokens: int) -> str:
        if not text:
            return text
        if max_tokens <= 0:
            return ""

        enc = self._get_encoding()
        if enc is None:
            # Fallback: crude approximation (~4 chars/token) with a safety margin.
            approx_chars = max(0, max_tokens * 4)
            return text[-approx_chars:]

        tokens = enc.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return enc.decode(tokens[-max_tokens:])

    def _messages_to_history_text(self, messages: List[BaseMessage]) -> str:
        parts: List[str] = []
        for m in messages:
            role = m.__class__.__name__.replace("Message", "").upper()
            parts.append(f"[{role}]\n{self._message_content_to_text(m)}")
        return "\n\n".join(parts)

    def _truncate_messages_payload(self, messages: Any, max_input_tokens: int) -> Any:
        """Convert message history to a single string and truncate (keep newest).

        Returns a message list suitable for runnable.invoke.
        """

        # Unwrap LC dict payloads.
        if isinstance(messages, dict) and "messages" in messages:
            payload = dict(messages)
            payload["messages"] = self._truncate_messages_payload(payload.get("messages"), max_input_tokens)
            return payload

        if not isinstance(messages, list) or not messages:
            return messages

        if not all(isinstance(m, BaseMessage) for m in messages):
            return messages

        safe_messages: List[BaseMessage] = self._sanitise_messages(messages)
        history_text = self._messages_to_history_text(safe_messages)

        truncated = self._truncate_text_to_tokens_keep_tail(history_text, max_input_tokens)
        return [HumanMessage(content=truncated)]

    def _message_content_to_text(self, msg: Any) -> str:
        content = getattr(msg, "content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                    continue
                if isinstance(block, dict):
                    text = block.get("text")
                    if isinstance(text, str) and text.strip():
                        parts.append(text)
            return "\n".join(parts)
        return str(content)

    def _sanitise_messages(self, messages: Any) -> Any:
        """Return a plain-text version of a message list/dict payload.

        This strips content-block IDs that can cause the Responses API to treat them as item references.
        We keep roles where possible, and fall back to HumanMessage for unsupported message types.
        """

        if isinstance(messages, dict) and "messages" in messages:
            payload = dict(messages)
            payload["messages"] = self._sanitise_messages(payload.get("messages"))
            return payload

        if not isinstance(messages, list) or not messages:
            return messages

        # Only sanitise lists that look like LangChain messages.
        if not all(isinstance(m, BaseMessage) for m in messages):
            return messages

        safe: List[BaseMessage] = []
        for m in messages:
            text = self._message_content_to_text(m)
            if isinstance(m, SystemMessage):
                safe.append(SystemMessage(content=text))
            elif isinstance(m, HumanMessage):
                safe.append(HumanMessage(content=text))
            elif isinstance(m, AIMessage):
                safe.append(AIMessage(content=text))
            else:
                # ToolMessage and other specialised types often require IDs; preserve content only.
                safe.append(HumanMessage(content=f"[context:{m.__class__.__name__}]\n{text}"))
        return safe

    def _remove_trailing_reasoning_items(self, messages: Any) -> Any:
        """Remove trailing/orphan reasoning content blocks from the input payload."""

        if isinstance(messages, dict):
            payload = dict(messages)
            if "messages" in payload:
                payload["messages"] = self._remove_trailing_reasoning_items(payload.get("messages"))
            if "input" in payload:
                payload["input"] = self._remove_trailing_reasoning_items(payload.get("input"))
            return payload

        if not isinstance(messages, list) or not messages:
            return messages

        if not all(isinstance(m, BaseMessage) for m in messages):
            return messages

        cleaned: List[BaseMessage] = []
        for m in messages:
            if not isinstance(m, AIMessage):
                cleaned.append(m)
                continue

            content = getattr(m, "content", None)
            if not isinstance(content, list):
                cleaned.append(m)
                continue

            # Drop trailing reasoning blocks
            new_content = list(content)
            while new_content and isinstance(new_content[-1], dict) and new_content[-1].get("type") == "reasoning":
                new_content.pop()

            if not new_content:
                # If this AIMessage is now empty (only reasoning), drop it entirely.
                continue

            if new_content == content:
                cleaned.append(m)
            else:
                cleaned.append(
                    AIMessage(
                        content=new_content,
                        additional_kwargs=getattr(m, "additional_kwargs", None),
                        response_metadata=getattr(m, "response_metadata", None),
                        id=getattr(m, "id", None),
                        tool_calls=getattr(m, "tool_calls", None),
                    )
                )

        return cleaned

    def _invoke_with_api_retries(self, runnable: Any, *args: Any, **kwargs: Any) -> Any:
        last_exc: Optional[Exception] = None
        for attempt in range(self._api_retry_max_attempts + 1):
            try:
                return runnable.invoke(*args, **kwargs)
            except self._api_retryable_exceptions as exc:  # type: ignore[misc]
                last_exc = exc
                if attempt >= self._api_retry_max_attempts:
                    raise
                delay = self._get_retry_delay_seconds(attempt, exc)
                time.sleep(delay)
            except Exception as exc:
                if not self._is_rate_limit_or_capacity_error(exc):
                    raise
                last_exc = exc
                if attempt >= self._api_retry_max_attempts:
                    raise
                delay = self._get_retry_delay_seconds(attempt, exc)
                time.sleep(delay)
        if last_exc:
            raise last_exc
        return runnable.invoke(*args, **kwargs)

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        last_exc: Optional[Exception] = None
        attempts = 0
        for temperature in self._temperature_schedule:
            attempts += 1
            if attempts > self._max_attempts:
                break
            runnable = self._bind_temperature(temperature)
            new_args = list(args)
            if new_args:
                new_args[0] = self._remove_trailing_reasoning_items(new_args[0])
            new_kwargs = dict(kwargs)
            if "input" in new_kwargs:
                new_kwargs["input"] = self._remove_trailing_reasoning_items(new_kwargs["input"])
            try:
                response = self._invoke_with_api_retries(runnable, *new_args, **new_kwargs)
                if self._is_content_filtered(response):
                    continue
                if self._is_max_output_tokens_incomplete(response):
                    fallback_efforts = self._get_lower_reasoning_efforts(runnable)
                    recovered = False
                    for fallback_effort in fallback_efforts:
                        try:
                            fallback_runnable = self._bind_reasoning_effort(runnable, fallback_effort)
                            fallback_response = self._invoke_with_api_retries(
                                fallback_runnable,
                                *new_args,
                                **new_kwargs,
                            )
                            if self._is_content_filtered(fallback_response):
                                continue
                            if self._is_max_output_tokens_incomplete(fallback_response):
                                response = fallback_response
                                continue
                            response = fallback_response
                            recovered = True
                            break
                        except Exception as retry_exc:
                            last_exc = retry_exc
                    if not recovered and self._is_max_output_tokens_incomplete(response):
                        raise RuntimeError(
                            "Model response was incomplete because max_output_tokens was reached before final output was emitted."
                        )
                return response
            except Exception as exc:
                last_exc = exc
                if "content_filter" in str(exc).lower():
                    continue

                # Recovery path for invalid encrypted content (model-mismatch / cross-run reuse):
                # retry once with sanitised plain-text message content.
                if self._is_invalid_encrypted_content_error(exc):
                    try:
                        new_args = list(new_args)
                        if new_args:
                            new_args[0] = self._sanitise_messages(new_args[0])
                        new_kwargs = dict(new_kwargs)
                        if "input" in new_kwargs:
                            new_kwargs["input"] = self._sanitise_messages(new_kwargs["input"])
                        response = self._invoke_with_api_retries(runnable, *new_args, **new_kwargs)
                        if self._is_content_filtered(response):
                            continue
                        return response
                    except Exception as retry_exc:
                        last_exc = retry_exc
                        raise

                # Recovery path for OpenAI Responses API item-reference errors:
                # retry once with sanitised plain-text message content.
                if self._is_item_not_found_error(exc):
                    try:
                        new_args = list(new_args)
                        if new_args:
                            new_args[0] = self._sanitise_messages(new_args[0])
                        new_kwargs = dict(new_kwargs)
                        if "input" in new_kwargs:
                            new_kwargs["input"] = self._sanitise_messages(new_kwargs["input"])
                        response = self._invoke_with_api_retries(runnable, *new_args, **new_kwargs)
                        if self._is_content_filtered(response):
                            continue
                        return response
                    except Exception as retry_exc:
                        last_exc = retry_exc
                        # Fall through to raise the retry exception for this temperature attempt.
                        raise

                # Recovery path for context window exceeded:
                # retry once with message history flattened + token-truncated (keeping newest content).
                if self._is_context_length_exceeded_error(exc):
                    try:
                        context_window = self._get_context_window_tokens()
                        # Reserve a small buffer for system overhead / tool schemas.
                        max_input_tokens = max(1_000, context_window - 8_000)

                        new_args = list(new_args)
                        if new_args:
                            new_args[0] = self._truncate_messages_payload(new_args[0], max_input_tokens)
                        new_kwargs = dict(new_kwargs)
                        if "input" in new_kwargs:
                            new_kwargs["input"] = self._truncate_messages_payload(new_kwargs["input"], max_input_tokens)

                        response = self._invoke_with_api_retries(runnable, *new_args, **new_kwargs)
                        if self._is_content_filtered(response):
                            continue
                        return response
                    except Exception as retry_exc:
                        last_exc = retry_exc
                        raise

                raise
        if last_exc:
            raise last_exc
        return self._runnable.invoke(*args, **kwargs)

    def with_structured_output(self, *args: Any, **kwargs: Any) -> "ContentFilterRetryWrapper":
        return ContentFilterRetryWrapper(
            self._runnable.with_structured_output(*args, **kwargs),
            temperature_schedule=self._temperature_schedule,
            max_attempts=self._max_attempts,
            api_retry_max_attempts=self._api_retry_max_attempts,
            api_retry_base_seconds=self._api_retry_base_seconds,
            api_retry_max_seconds=self._api_retry_max_seconds,
            api_retry_jitter=self._api_retry_jitter,
        )

    def bind_tools(self, *args: Any, **kwargs: Any) -> "ContentFilterRetryWrapper":
        return ContentFilterRetryWrapper(
            self._runnable.bind_tools(*args, **kwargs),
            temperature_schedule=self._temperature_schedule,
            max_attempts=self._max_attempts,
            api_retry_max_attempts=self._api_retry_max_attempts,
            api_retry_base_seconds=self._api_retry_base_seconds,
            api_retry_max_seconds=self._api_retry_max_seconds,
            api_retry_jitter=self._api_retry_jitter,
        )

    def with_retry(self, *args: Any, **kwargs: Any) -> "ContentFilterRetryWrapper":
        return ContentFilterRetryWrapper(
            self._runnable.with_retry(*args, **kwargs),
            temperature_schedule=self._temperature_schedule,
            max_attempts=self._max_attempts,
            api_retry_max_attempts=self._api_retry_max_attempts,
            api_retry_base_seconds=self._api_retry_base_seconds,
            api_retry_max_seconds=self._api_retry_max_seconds,
            api_retry_jitter=self._api_retry_jitter,
        )

    def with_config(self, *args: Any, **kwargs: Any) -> "ContentFilterRetryWrapper":
        return ContentFilterRetryWrapper(
            self._runnable.with_config(*args, **kwargs),
            temperature_schedule=self._temperature_schedule,
            max_attempts=self._max_attempts,
            api_retry_max_attempts=self._api_retry_max_attempts,
            api_retry_base_seconds=self._api_retry_base_seconds,
            api_retry_max_seconds=self._api_retry_max_seconds,
            api_retry_jitter=self._api_retry_jitter,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._runnable, name)

models = {
    reasoning_level: ContentFilterRetryWrapper(
        ChatOpenAI(
            model="gpt-5.4",
            api_key=api_keys["openai"],
            reasoning={"effort": reasoning_level},
        )
    ) for reasoning_level in REASONING_LEVELS
}