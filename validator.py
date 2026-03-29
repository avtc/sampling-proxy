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
    issue_type: Optional[str] = None  # "repetition" | None
    confidence: float = 1.0
    error: Optional[str] = None  # If validator itself failed


VALIDATION_PROMPT = """Is this text stuck in an infinite loop?

A LOOP is when the SAME paragraph or sentence repeats word-for-word multiple times in a row.

NOT a loop (return is_valid=true):
- Structured content with similar patterns (lists, code comments, headings)
- Technical explanations that reference similar concepts
- Any content that progresses or adds new information

IS a loop (return is_valid=false):
- "The quick brown fox. The quick brown fox. The quick brown fox."
- Same sentence repeated verbatim 3+ times consecutively

Answer with ONLY JSON, no other text:
{{"is_valid":true,"issue_type":null,"confidence":1.0}}
or
{{"is_valid":false,"issue_type":"repetition","confidence":0.95}}

Text:
{content}"""


def get_cache_dir() -> Path:
    """Get directory for failed response logs."""
    logs_dir = Path.home() / ".sampling-proxy" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir


def count_words_in_text(text: str) -> int:
    """
    Count words in text using simple whitespace splitting.
    Works for both English and most languages.
    """
    if not text:
        return 0
    return len(text.split())


def extract_text_from_sse_chunks(chunks: list) -> str:
    """
    Extract accumulated text from SSE chunks.

    Parses both Anthropic and OpenAI SSE formats to extract text deltas.
    Returns concatenated text content.
    """
    text_parts = []

    for chunk in chunks:
        if isinstance(chunk, bytes):
            chunk_str = chunk.decode('utf-8', errors='ignore')
        else:
            chunk_str = str(chunk)

        # Parse SSE lines
        for line in chunk_str.split('\n'):
            line = line.strip()
            if line.startswith('data: '):
                data_str = line[6:]
                if data_str == '[DONE]':
                    continue
                try:
                    data = json.loads(data_str)

                    # Handle Anthropic content_block_delta
                    if data.get('type') == 'content_block_delta':
                        delta = data.get('delta', {})
                        if delta.get('type') == 'text_delta':
                            text_parts.append(delta.get('text', ''))

                    # Handle OpenAI choices[0].delta.content
                    elif 'choices' in data:
                        choices = data.get('choices', [])
                        if choices:
                            delta = choices[0].get('delta', {})
                            content = delta.get('content', '')
                            if content:
                                text_parts.append(content)

                except json.JSONDecodeError:
                    continue

    return ''.join(text_parts)


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

    # Silent - no log needed

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
                print(f"WARN: Validator returned {response.status_code}")
            raise httpx.HTTPStatusError(
                f"Validator returned {response.status_code}: {response.text}",
                request=None,
                response=response
            )

        return response.json()


def parse_validator_response(response: dict, config: dict) -> ValidationResult:
    """Parse validator model response into ValidationResult (handles both formats)."""
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

        # Parse JSON from response
        # Handle potential markdown code blocks
        text = text.strip()
        if text.startswith("```"):
            # Remove markdown code block
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

        # Extract JSON object - handle extra text before/after
        # Find first { and last }
        start_idx = text.find('{')
        if start_idx == -1:
            raise json.JSONDecodeError("No JSON object found", text, 0)

        # Find matching closing brace
        brace_count = 0
        end_idx = start_idx
        for i, char in enumerate(text[start_idx:], start_idx):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i + 1
                    break

        json_text = text[start_idx:end_idx]
        result = json.loads(json_text)

        raw_is_valid = result.get("is_valid", True)
        raw_issue_type = result.get("issue_type")
        raw_confidence = result.get("confidence", 1.0)

        # If validator says invalid but has no issue_type, treat as valid (contradictory response)
        if not raw_is_valid and raw_issue_type is None:
            return ValidationResult(
                is_valid=True,
                issue_type=None,
                confidence=raw_confidence
            )

        # Apply confidence threshold - only reject if confidence is high enough
        # Low confidence predictions are unreliable, so we pass through
        if not raw_is_valid and raw_confidence < confidence_threshold:
            return ValidationResult(
                is_valid=True,  # Pass through due to low confidence
                issue_type=raw_issue_type,
                confidence=raw_confidence
            )

        return ValidationResult(
            is_valid=raw_is_valid,
            issue_type=raw_issue_type,
            confidence=raw_confidence
        )

    except json.JSONDecodeError as e:
        # If we can't parse, assume valid (fail-open for parser errors)
        print(f"WARN: Validation parse error: {e}")
        return ValidationResult(
            is_valid=True,
            error=f"Failed to parse validator response: {e}"
        )
    except Exception as e:
        print(f"WARN: Validation error: {type(e).__name__}: {e}")
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
    if not config.get("enabled", False):
        return ValidationResult(is_valid=True)

    word_count = count_words_in_text(extract_content_from_response(response))
    print(f"INFO: Validation started ({word_count} words, final)")

    try:
        content = extract_content_from_response(response)
        if not content.strip():
            # Empty content is valid
            return ValidationResult(is_valid=True)

        raw_result = await call_validator_model(content, config)
        result = parse_validator_response(raw_result, config)

        if result.error:
            print(f"WARN: Validation error: {result.error}")
        elif result.is_valid:
            print(f"INFO: Validation passed (is_valid={result.is_valid}, issue_type={result.issue_type}, confidence={result.confidence:.2f})")

        return result

    except httpx.HTTPStatusError as e:
        print(f"WARN: Validation HTTP error: {e}")
        return ValidationResult(
            is_valid=True,  # Fail-open
            error=f"Validator HTTP error: {e}"
        )
    except httpx.RequestError as e:
        print(f"WARN: Validation connection error: {e}")
        return ValidationResult(
            is_valid=True,  # Fail-open
            error=f"Validator connection error: {e}"
        )
    except Exception as e:
        import traceback
        print(f"WARN: Validation error: {type(e).__name__}: {e}")
        traceback.print_exc()
        return ValidationResult(
            is_valid=True,  # Fail-open
            error=f"Validator unexpected error: {type(e).__name__}: {e}"
        )


async def validate_response_partial(content: str, config: dict) -> ValidationResult:
    """
    Validate response content for garbage detection (repetition loops).

    Args:
        content: Text content to validate
        config: Validation config

    Returns:
        ValidationResult with is_valid, issue_type, confidence
    """
    enable_logs = config.get("enable_validation_logs", False)

    if not config.get("mid_stream_validation_enabled", False):
        return ValidationResult(is_valid=True)

    # Silent - log is handled by start_validation in StreamingValidator.start_validation

    try:
        if not content.strip():
            return ValidationResult(is_valid=True)

        # Build prompt with partial content
        prompt = VALIDATION_PROMPT.format(content=content)

        # Get validator config
        validator_url = config.get("validator_url", "http://127.0.0.1:1234")
        validator_model = config.get("validator_model", "qwen-3.5-0.8b")
        supports_openai = config.get("supports_openai", True)
        supports_anthropic = config.get("supports_anthropic", False)
        connect_timeout = config.get("connect_timeout_seconds", 5.0)
        read_timeout = config.get("timeout_seconds", 300.0)

        # Determine endpoint
        if supports_anthropic:
            endpoint = f"{validator_url}/v1/messages"
        elif supports_openai:
            endpoint = f"{validator_url}/v1/chat/completions"
        else:
            return ValidationResult(is_valid=True, error="No supported API format")

        timeout = httpx.Timeout(connect=connect_timeout, read=read_timeout, write=read_timeout, pool=connect_timeout)
        async with httpx.AsyncClient(timeout=timeout) as client:
            if supports_anthropic:
                response = await client.post(
                    endpoint,
                    headers={
                        "Content-Type": "application/json",
                        "x-api-key": "validator",
                        "anthropic-version": "2023-06-01"
                    },
                    json={
                        "model": validator_model,
                        "max_tokens": 100,
                        "messages": [{"role": "user", "content": prompt}]
                    }
                )
            else:
                response = await client.post(
                    endpoint,
                    headers={"Content-Type": "application/json"},
                    json={
                        "model": validator_model,
                        "max_tokens": 100,
                        "messages": [{"role": "user", "content": prompt}]
                    }
                )

            if response.status_code != 200:
                print(f"WARN: Validator error {response.status_code}")
                return ValidationResult(is_valid=True, error=f"Validator HTTP {response.status_code}")

            raw_result = response.json()
            return parse_validator_response(raw_result, config)

    except httpx.HTTPStatusError as e:
        print(f"WARN: Validation HTTP error: {e}")
        return ValidationResult(is_valid=True, error=f"HTTP error: {e}")
    except httpx.RequestError as e:
        print(f"WARN: Validation connection error: {e}")
        return ValidationResult(is_valid=True, error=f"Connection error: {e}")
    except Exception as e:
        print(f"WARN: Validation error: {type(e).__name__}: {e}")
        return ValidationResult(is_valid=True, error=f"Unexpected error: {type(e).__name__}: {e}")


def save_failed_response(response: dict, validation_result: ValidationResult, attempt: int):
    """Save failed response to cache for later analysis."""
    logs_dir = get_cache_dir()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    # Extract text content from response
    text_content = extract_content_from_response(response)

    # Save as plain text with metadata header for easy reading
    failed_file = logs_dir / f"failed_{timestamp}.txt"

    with open(failed_file, "w", encoding="utf-8") as f:
        f.write(f"=== FINAL VALIDATION FAILURE ===\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Issue: {validation_result.issue_type}\n")
        f.write(f"Confidence: {validation_result.confidence}\n")
        f.write(f"Attempt: {attempt}\n")
        f.write(f"\n--- RESPONSE CONTENT ---\n\n")
        f.write(text_content[:50000] if text_content else "")  # Limit size but allow more for analysis

    return str(failed_file)


def save_mid_stream_failure(text_content: str, word_count: int, issue_type: str, attempt: int, raw_chunks: list = None):
    """Save mid-stream detected garbage for later analysis."""
    logs_dir = get_cache_dir()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    # Save as plain text with metadata header for easy reading
    failed_file = logs_dir / f"midstream_{timestamp}.txt"

    with open(failed_file, "w", encoding="utf-8") as f:
        f.write(f"=== MID-STREAM VALIDATION FAILURE ===\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Issue: {issue_type}\n")
        f.write(f"Word count: {word_count}\n")
        f.write(f"Attempt: {attempt}\n")
        if raw_chunks:
            f.write(f"Chunk count: {len(raw_chunks)}\n")
        f.write(f"\n--- RESPONSE CONTENT ---\n\n")
        f.write(text_content[:50000] if text_content else "")  # Limit size but allow more for analysis

    return str(failed_file)


def create_error_message(issue_type: Optional[str], saved_path: str) -> dict:
    """Create synthetic assistant message for validation failure."""
    issue_display = {
        "repetition": "Repetition loop"
    }.get(issue_type, "Unknown issue")

    error_text = f"""**Garbage Output Detected**

After multiple retries, the model continues to produce invalid output.

**Last issue detected:** {issue_display}

**Options:**
- For local models - try adjust repetition_penalty and presence_penalty
- For cloud models - clear context window and try again or switch to a different model temporarily

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


def build_anthropic_error_stream(error_response: dict):
    """
    Generator that yields Anthropic SSE error stream events.

    Args:
        error_response: Error response dict from create_error_message()

    Yields:
        Encoded SSE events for Anthropic-compatible error stream
    """
    # 1. message_start event
    message_start = {"type": "message_start", "message": error_response}
    yield f"event: message_start\ndata: {json.dumps(message_start)}\n\n".encode()

    # 2. content_block_start for the text block
    content_block_start = {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": ""}
    }
    yield f"event: content_block_start\ndata: {json.dumps(content_block_start)}\n\n".encode()

    # 3. content_block_delta with the error text
    error_text = error_response["content"][0]["text"]
    content_block_delta = {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": error_text}
    }
    yield f"event: content_block_delta\ndata: {json.dumps(content_block_delta)}\n\n".encode()

    # 4. content_block_stop
    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n".encode()

    # 5. message_delta with stop_reason
    message_delta = {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn"},
        "usage": {"output_tokens": 0}
    }
    yield f"event: message_delta\ndata: {json.dumps(message_delta)}\n\n".encode()

    # 6. message_stop
    yield b"event: message_stop\ndata: {}\n\n"


def build_anthropic_error_json(issue_type: Optional[str], saved_path: str) -> dict:
    """
    Build an Anthropic-format error JSON response for non-streaming requests.

    Returns a dict matching Anthropic's error response format.
    """
    issue_display = {
        "repetition": "Repetition loop"
    }.get(issue_type, "Unknown issue")

    return {
        "type": "error",
        "error": {
            "type": "api_error",
            "message": (
                f"Garbage output detected ({issue_display}). "
                f"After multiple retries, the model continues to produce invalid output. "
                f"Raw response saved to: {saved_path}"
            )
        }
    }


def build_openai_error_stream(error_response: dict):
    """
    Generator that yields OpenAI SSE error stream events.

    Args:
        error_response: Error response dict from create_error_message()

    Yields:
        Encoded SSE events for OpenAI-compatible error stream
    """
    error_text = error_response["content"][0]["text"]
    error_chunk = {"id": "error", "choices": [{"delta": {"content": error_text}}]}
    yield f"data: {json.dumps(error_chunk)}\n\n".encode()
    yield b"data: [DONE]\n\n"


def build_openai_error_json(issue_type: Optional[str], saved_path: str) -> dict:
    """
    Build an OpenAI-format error JSON response for non-streaming requests.

    Returns a dict matching OpenAI's error response format.
    """
    issue_display = {
        "repetition": "Repetition loop"
    }.get(issue_type, "Unknown issue")

    return {
        "error": {
            "message": (
                f"Garbage output detected ({issue_display}). "
                f"After multiple retries, the model continues to produce invalid output. "
                f"Raw response saved to: {saved_path}"
            ),
            "type": "server_error",
            "code": "validation_failed"
        }
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


class StreamingValidator:
    """
    Handles periodic validation during streaming response buffering.

    Usage:
        validator = StreamingValidator(config)
        async for chunk in response.aiter_bytes():
            chunks.append(chunk)
            word_count = count_words_from_chunks(chunks)

            if validator.should_validate(word_count):
                content = extract_text_from_sse_chunks(chunks)
                await validator.start_validation(content, config)

            if validator.is_garbage_detected():
                break  # Early exit for retry
    """

    def __init__(self, config: dict):
        self.interval_words = config.get("mid_stream_validation_interval_words", 300)
        self.enabled = config.get("mid_stream_validation_enabled", False)
        self.validator_task: Optional[asyncio.Task] = None
        self.garbage_detected = asyncio.Event()
        self.last_validated_word_count = 0
        self._current_word_count = 0
        self.detected_issue_type: Optional[str] = None
        self._detected_confidence: float = 1.0

    def should_validate(self, current_word_count: int) -> bool:
        """Check if we've reached the interval threshold."""
        if not self.enabled:
            return False

        # Only validate when we've passed another full interval
        if current_word_count - self.last_validated_word_count >= self.interval_words:
            # Update counter immediately to prevent repeated triggers
            # This must happen here, not in start_validation, to avoid race conditions
            self.last_validated_word_count = current_word_count
            self._current_word_count = current_word_count
            return True
        return False

    def start_validation(self, content: str, config: dict):
        """Start async validation task (non-blocking).

        If previous validation is still running, skip this interval (next one will validate).
        Garbage detection from previous validation will still trigger break.

        Note: last_validated_word_count is updated in should_validate(), not here.
        """
        if not self.enabled:
            return

        # If previous validation detected garbage, no need to start new one
        if self.garbage_detected.is_set():
            return

        # If previous task running, skip this interval
        if self.validator_task and not self.validator_task.done():
            return

        self.validator_task = asyncio.create_task(
            self._validate_partial(content, config)
        )

    async def _validate_partial(self, content: str, config: dict):
        """Validate partial content, signal if garbage detected."""
        word_count = count_words_in_text(content)
        print(f"INFO: Validation started ({word_count} words, mid-stream)")
        try:
            result = await validate_response_partial(content, config)
            if result.error:
                # Validation itself failed (e.g., connection error)
                print(f"WARN: Validation error: {result.error}")
            elif not result.is_valid:
                # Garbage detected - log is handled by caller
                self.detected_issue_type = result.issue_type
                self._detected_confidence = result.confidence
                self.garbage_detected.set()
            else:
                print(f"INFO: Validation passed (is_valid={result.is_valid}, issue_type={result.issue_type}, confidence={result.confidence:.2f})")
        except Exception as e:
            # Log but don't signal garbage on validator errors (fail-open)
            print(f"WARN: Validation error: {e}")

    def is_garbage_detected(self) -> bool:
        """Check if garbage was detected by any validator task."""
        return self.garbage_detected.is_set()

    async def wait_for_pending_validation(self):
        """Wait for any in-flight validation to complete."""
        if self.validator_task and not self.validator_task.done():
            try:
                await asyncio.wait_for(self.validator_task, timeout=30.0)
            except asyncio.TimeoutError:
                print("VALIDATION [mid-stream]: Validator task timeout")

    def get_detection_info(self) -> Optional[str]:
        """Get info about detection (for logging)."""
        if self.is_garbage_detected():
            return f"{self.detected_issue_type or 'unknown'} at ~{self.last_validated_word_count} words"
        return None

    def get_detection_confidence(self) -> float:
        """Get the confidence of the detection."""
        return self._detected_confidence

    def get_issue_type(self) -> Optional[str]:
        """Get the detected issue type."""
        return self.detected_issue_type


class StreamingValidationBuffer:
    """
    Unified buffer for streaming validation that handles:
    - Chunk accumulation
    - Mid-stream garbage detection
    - Final validation preparation

    Usage:
        buffer = StreamingValidationBuffer(config)
        async for chunk in response.aiter_bytes():
            if not await buffer.add_chunk(chunk):
                break  # Garbage detected

        if buffer.is_garbage_detected():
            # Handle early retry
            ...

        # Final validation
        result = await buffer.validate_final(parse_fn, config)
    """

    def __init__(self, config: dict, debug_logs: bool = False):
        self.chunks: list = []
        self.config = config
        self.debug_logs = debug_logs
        self.streaming_validator = StreamingValidator(config)
        self._early_break = False

    async def add_chunk(self, chunk: bytes) -> bool:
        """
        Add a chunk and run mid-stream validation if needed.

        Returns:
            True if streaming should continue, False if garbage detected
        """
        self.chunks.append(chunk)

        # Mid-stream validation check
        text_so_far = extract_text_from_sse_chunks(self.chunks)
        word_count = count_words_in_text(text_so_far)

        should_val = self.streaming_validator.should_validate(word_count)
        if should_val:
            # Non-blocking - start validation in background
            self.streaming_validator.start_validation(text_so_far, self.config)

        # Check if garbage detected (immediately, non-blocking)
        if self.streaming_validator.is_garbage_detected():
            self._early_break = True
            return False

        return True

    def is_garbage_detected(self) -> bool:
        """Check if garbage was detected during streaming."""
        return self.streaming_validator.is_garbage_detected()

    def get_garbage_event(self) -> asyncio.Event:
        """Get the event that is set when garbage is detected."""
        return self.streaming_validator.garbage_detected

    def was_early_break(self) -> bool:
        """Check if streaming was interrupted due to garbage detection."""
        return self._early_break

    async def wait_for_pending_validation(self):
        """Wait for any in-flight mid-stream validation to complete."""
        await self.streaming_validator.wait_for_pending_validation()

    def get_chunks(self) -> list:
        """Get all accumulated chunks."""
        return self.chunks

    def get_content(self) -> bytes:
        """Get accumulated content as bytes."""
        return b''.join(self.chunks)

    def get_text_content(self) -> str:
        """Extract text content from accumulated SSE chunks."""
        return extract_text_from_sse_chunks(self.chunks)

    def get_word_count(self) -> int:
        """Get current word count of accumulated text."""
        return count_words_in_text(self.get_text_content())

    def get_detection_info(self) -> Optional[str]:
        """Get info about garbage detection (for logging)."""
        return self.streaming_validator.get_detection_info()

    def get_issue_type(self) -> Optional[str]:
        """Get the detected issue type."""
        return self.streaming_validator.get_issue_type()

    def get_detection_confidence(self) -> float:
        """Get the confidence of the detection."""
        return self.streaming_validator.get_detection_confidence()

    async def validate_final(self, response_dict: dict, config: dict) -> 'ValidationResult':
        """
        Run final validation on the parsed response.

        Args:
            response_dict: Parsed response dictionary
            config: Validation config

        Returns:
            ValidationResult
        """
        return await validate_response(response_dict, config)

    def reset(self):
        """Reset buffer for retry."""
        self.chunks = []
        self.streaming_validator = StreamingValidator(self.config)
        self._early_break = False
