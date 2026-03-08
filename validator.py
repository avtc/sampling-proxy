"""
Response validator module for garbage detection.
Uses a local or remote model via OpenAI-compatible or Anthropic-compatible API to validate responses.
"""

import json
import asyncio
from dataclasses import dataclass
from typing import Optional, Literal
from datetime import datetime
from pathlib import Path
import httpx


@dataclass
class ValidationResult:
    """Result of response validation."""
    is_valid: bool
    issue_type: Optional[str] = None  # "repetition" | "truncation" | "malformed_tools" | None
    confidence: float = 1.0
    error: Optional[str] = None  # If validator itself failed


VALIDATION_PROMPT = """Check if this AI response has problems.

PROBLEMS:
- repetition: same phrase repeated 3+ times in a row
- truncation: text cuts off mid-sentence (incomplete)

If no problems, set is_valid=true and issue_type=null.

Output ONLY JSON (no markdown):
{{"is_valid":true/false,"issue_type":"repetition"|null,"confidence":0.0-1.0}}

Response:
{content}"""


def get_cache_dir() -> Path:
    """Get cache directory for logs and failed responses."""
    cache_dir = Path.home() / ".cache" / "garbage-proxy"
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "logs").mkdir(exist_ok=True)
    (cache_dir / "failed").mkdir(exist_ok=True)
    return cache_dir


def is_openai_format(response: dict) -> bool:
    """Detect if response is in OpenAI format (has 'choices' key)."""
    return "choices" in response


def extract_content_from_anthropic(response: dict) -> str:
    """Extract text and tool content from Anthropic response format."""
    content_parts = []

    for block in response.get("content", []):
        block_type = block.get("type")

        if block_type == "text":
            content_parts.append(block.get("text", ""))
        elif block_type == "tool_use":
            tool_name = block.get("name", "unknown")
            tool_input = block.get("input", {})
            content_parts.append(f"[TOOL: {tool_name}]\n{json.dumps(tool_input, indent=2)}")

    return "\n\n".join(content_parts)


def extract_content_from_openai(response: dict) -> str:
    """Extract text and tool content from OpenAI response format."""
    content_parts = []

    choices = response.get("choices", [])
    if not choices:
        return ""

    message = choices[0].get("message", {})

    # Extract text content
    text = message.get("content", "")
    if text:
        content_parts.append(text)

    # Extract tool calls
    tool_calls = message.get("tool_calls", [])
    for tool_call in tool_calls:
        function = tool_call.get("function", {})
        tool_name = function.get("name", "unknown")
        arguments_str = function.get("arguments", "{}")
        try:
            arguments = json.loads(arguments_str)
        except json.JSONDecodeError:
            arguments = {"raw": arguments_str}  # Keep raw if invalid JSON
        content_parts.append(f"[TOOL: {tool_name}]\n{json.dumps(arguments, indent=2)}")

    return "\n\n".join(content_parts)


def extract_content_from_response(response: dict) -> str:
    """Extract text and tool content from response (auto-detects format)."""
    if is_openai_format(response):
        return extract_content_from_openai(response)
    else:
        return extract_content_from_anthropic(response)


async def call_validator_model(content: str, config: dict) -> dict:
    """
    Call validator model using configured API format.

    Args:
        content: Response content to validate
        config: Validation config with validator_url, validator_model, validator_capabilities

    Returns:
        Raw response from validator as dict
    """
    validator_url = config.get("validator_url", "http://127.0.0.1:1234")
    validator_model = config.get("validator_model", "qwen-3.5-0.8b")
    enable_logs = config.get("enable_validation_logs", False)

    # Get validator capabilities directly from config (defaults: supports_openai=true, supports_anthropic=false)
    supports_openai = config.get("supports_openai", True)
    supports_anthropic = config.get("supports_anthropic", False)

    connect_timeout = config.get("connect_timeout_seconds", 5.0)
    read_timeout = config.get("timeout_seconds", 300.0)

    prompt = VALIDATION_PROMPT.format(content=content)

    # Determine which format to use
    if supports_anthropic:
        api_format = "anthropic"
        endpoint = f"{validator_url}/v1/messages"
    elif supports_openai:
        api_format = "openai"
        endpoint = f"{validator_url}/v1/chat/completions"
    else:
        raise ValueError("Validator does not support OpenAI or Anthropic API formats")

    if enable_logs:
        print(f"VALIDATION: Calling validator at {endpoint} (format: {api_format})")
        print(f"VALIDATION: Content length: {len(content)} chars")

    timeout = httpx.Timeout(connect=connect_timeout, read=read_timeout, write=read_timeout, pool=connect_timeout)
    async with httpx.AsyncClient(timeout=timeout) as client:
        if supports_anthropic:
            # Anthropic-compatible API format
            response = await client.post(
                endpoint,
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": "validator",  # Most endpoints accept any non-empty key
                    "anthropic-version": "2023-06-01"
                },
                json={
                    "model": validator_model,
                    "max_tokens": 100,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ]
                }
            )
        else:
            # OpenAI-compatible API format
            response = await client.post(
                endpoint,
                headers={
                    "Content-Type": "application/json",
                },
                json={
                    "model": validator_model,
                    "max_tokens": 100,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ]
                }
            )

        if response.status_code != 200:
            if enable_logs:
                print(f"VALIDATION: Validator returned error {response.status_code}")
            raise httpx.HTTPStatusError(
                f"Validator returned {response.status_code}: {response.text}",
                request=None,
                response=response
            )

        return response.json()


def parse_validator_response(response: dict, config: dict) -> ValidationResult:
    """Parse validator model response into ValidationResult (handles both formats)."""
    enable_logs = config.get("enable_validation_logs", False)
    confidence_threshold = config.get("confidence_threshold", 0.85)

    try:
        # Extract text based on response format
        if is_openai_format(response):
            # OpenAI format
            choices = response.get("choices", [])
            if choices:
                text = choices[0].get("message", {}).get("content", "")
            else:
                text = ""
        else:
            # Anthropic format
            text = ""
            for block in response.get("content", []):
                if block.get("type") == "text":
                    text = block.get("text", "")
                    break

        if enable_logs:
            print(f"VALIDATION: Raw validator response text: {text[:500] if text else 'EMPTY'}")

        # Parse JSON from response
        # Handle potential markdown code blocks
        text = text.strip()
        if text.startswith("```"):
            # Remove markdown code block
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

        result = json.loads(text)

        raw_is_valid = result.get("is_valid", True)
        raw_issue_type = result.get("issue_type")
        raw_confidence = result.get("confidence", 1.0)

        # If validator says invalid but has no issue_type, treat as valid (contradictory response)
        if not raw_is_valid and raw_issue_type is None:
            if enable_logs:
                print(f"VALIDATION: Validator returned invalid with no issue_type, treating as valid")
            return ValidationResult(
                is_valid=True,
                issue_type=None,
                confidence=raw_confidence
            )

        # Apply confidence threshold - only reject if confidence is high enough
        # Low confidence predictions are unreliable, so we pass through
        if not raw_is_valid and raw_confidence < confidence_threshold:
            if enable_logs:
                print(f"VALIDATION: Low confidence ({raw_confidence}) below threshold ({confidence_threshold}), "
                      f"treating as valid (issue was: {raw_issue_type})")
            return ValidationResult(
                is_valid=True,  # Pass through due to low confidence
                issue_type=raw_issue_type,
                confidence=raw_confidence
            )

        validation_result = ValidationResult(
            is_valid=raw_is_valid,
            issue_type=raw_issue_type,
            confidence=raw_confidence
        )

        if enable_logs:
            print(f"VALIDATION: Result - is_valid={validation_result.is_valid}, "
                  f"issue_type={validation_result.issue_type}, confidence={validation_result.confidence}")

        return validation_result

    except json.JSONDecodeError as e:
        # If we can't parse, assume valid (fail-open for parser errors)
        print(f"VALIDATION ERROR: Failed to parse JSON - {e}")
        return ValidationResult(
            is_valid=True,
            error=f"Failed to parse validator response: {e}"
        )
    except Exception as e:
        print(f"VALIDATION ERROR: Unexpected error parsing response - {type(e).__name__}: {e}")
        return ValidationResult(
            is_valid=True,
            error=f"Validator parse error: {type(e).__name__}: {e}"
        )


async def validate_response(response: dict, config: dict) -> ValidationResult:
    """
    Validate a single model response for garbage output (supports both Anthropic and OpenAI formats).

    Note: Only validates the last model response, not the entire conversation history.
    The response dict contains just the model's output, not the conversation context.

    Args:
        response: Response dict in Anthropic or OpenAI format (single response, not conversation)
        config: Validation config

    Returns:
        ValidationResult with is_valid, issue_type, confidence
    """
    enable_logs = config.get("enable_validation_logs", False)

    if not config.get("enabled", False):
        return ValidationResult(is_valid=True)

    if enable_logs:
        print(f"VALIDATION: Starting validation for response id={response.get('id', 'unknown')}")

    try:
        content = extract_content_from_response(response)
        if not content.strip():
            # Empty content is valid
            if enable_logs:
                print("VALIDATION: Empty content, skipping validation")
            return ValidationResult(is_valid=True)

        raw_result = await call_validator_model(content, config)
        return parse_validator_response(raw_result, config)

    except httpx.HTTPStatusError as e:
        print(f"VALIDATION ERROR: HTTP error - {e}")
        return ValidationResult(
            is_valid=True,  # Fail-open
            error=f"Validator HTTP error: {e}"
        )
    except httpx.RequestError as e:
        print(f"VALIDATION ERROR: Connection error - {e}")
        return ValidationResult(
            is_valid=True,  # Fail-open
            error=f"Validator connection error: {e}"
        )
    except Exception as e:
        import traceback
        print(f"VALIDATION ERROR: Unexpected error - {type(e).__name__}: {e}")
        traceback.print_exc()
        return ValidationResult(
            is_valid=True,  # Fail-open
            error=f"Validator unexpected error: {type(e).__name__}: {e}"
        )


def save_failed_response(response: dict, validation_result: ValidationResult, attempt: int):
    """Save failed response to cache for later analysis."""
    cache_dir = get_cache_dir()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    failed_data = {
        "timestamp": timestamp,
        "attempt": attempt,
        "is_valid": validation_result.is_valid,
        "issue_type": validation_result.issue_type,
        "confidence": validation_result.confidence,
        "response": response
    }

    # Ensure directory exists
    failed_dir = cache_dir / "failed"
    failed_dir.mkdir(parents=True, exist_ok=True)

    failed_file = failed_dir / f"failed_{timestamp}.json"
    with open(failed_file, "w") as f:
        json.dump(failed_data, f, indent=2)

    print(f"VALIDATION FAILURE: {validation_result.issue_type} (confidence: {validation_result.confidence}, attempt: {attempt})")

    return str(failed_file)


def create_error_message(issue_type: Optional[str], saved_path: str) -> dict:
    """Create synthetic assistant message for validation failure."""
    issue_display = {
        "repetition": "Repetition loop",
        "truncation": "Truncated response",
        "malformed_tools": "Malformed tool calls"
    }.get(issue_type, "Unknown issue")

    error_text = f"""**Garbage Output Detected**

After multiple retries, the model continues to produce invalid output.

**Last issue detected:** {issue_display}

**Options:**
- Try rephrasing your prompt
- Use `/rewind` to undo and try again
- Switch to a different model temporarily

**Raw response saved to:** `{saved_path}`"""

    return {
        "id": "msg_validation_failed",
        "type": "message",
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": error_text
            }
        ],
        "stop_reason": "stop",
        "usage": {"input_tokens": 0, "output_tokens": 0}
    }


async def calculate_retry_delay(attempt: int, config: dict) -> float:
    """Calculate exponential backoff delay for retry."""
    if attempt <= 1:
        return 0.0

    base_delay = config.get("retry_base_delay_seconds", 1.0)
    multiplier = config.get("retry_multiplier", 2.0)

    # attempt 2 -> 1s, attempt 3 -> 2s
    delay = base_delay * (multiplier ** (attempt - 2))
    return delay
