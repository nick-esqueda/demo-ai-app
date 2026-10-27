"""Provider-independent LLM errors, so the API doesn't need to know which SDK raised them."""

import anthropic
import openai


class LLMError(Exception):
    """The provider returned an error or couldn't be reached."""


class LLMRateLimitError(LLMError):
    """The provider rejected the request for rate limit or quota reasons (common on free tiers)."""


class LLMUnavailableError(LLMError):
    """The provider is overloaded, down, or too slow (HTTP 503/529, or a timeout)."""


ERROR_TYPES_BY_STATUS: dict[int, type[LLMError]] = {
    429: LLMRateLimitError,
    503: LLMUnavailableError,
    529: LLMUnavailableError,  # Anthropic's "overloaded" status
}
TIMEOUT_ERRORS = (openai.APITimeoutError, anthropic.APITimeoutError)

SDKError = openai.APIError | anthropic.APIError


def _read_error_body(exc: SDKError) -> dict | None:
    """Return the error JSON as a dict. Gemini wraps it in a one-item list."""
    body = getattr(exc, "body", None)
    if isinstance(body, list) and body:
        body = body[0]
    return body if isinstance(body, dict) else None


def extract_error_message(exc: SDKError) -> str:
    """Pull the provider's own error message out of an SDK exception.

    The SDKs' default message can be just "Error code: 503", so dig the message out of the body.
    """
    body = _read_error_body(exc) or {}
    error = body.get("error", body)
    has_message = isinstance(error, dict) and bool(error.get("message"))
    if not has_message:
        return str(exc)

    status = getattr(exc, "status_code", "?")
    return f"{error['message']} (status {status})"


def convert_to_llm_error(exc: SDKError) -> LLMError:
    message = extract_error_message(exc)
    if isinstance(exc, TIMEOUT_ERRORS):
        return LLMUnavailableError(f"No response from the LLM provider in time: {message}")

    status = getattr(exc, "status_code", None)
    error_type = ERROR_TYPES_BY_STATUS.get(status, LLMError)
    return error_type(message)
