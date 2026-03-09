# Mid-Stream Garbage Detection Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:executing-plans to implement this plan task-by-task.

**Goal:** Detect garbage LLM responses during streaming to enable early retry and reduce wasted time.

**Architecture:** Async periodic validation - spawn validator tasks at word intervals while continuing to buffer, interrupt early if garbage detected.

**Tech Stack:** Python, asyncio, httpx (existing stack)

---

## Task 1: Add config options

**Files:**
- Modify: `config_sample.json`
- Modify: `sampling_proxy.py:49-60` (default config)

**Step 1: Add new options to config_sample.json**

Add to the `validation` section (after `"confidence_threshold": 0.85`):

```json
{
  "validation": {
    "enabled": false,
    "validator_url": "http://127.0.0.1:1234",
    "validator_model": "qwen-3.5-0.8b",
    "supports_openai": true,
    "supports_anthropic": false,
    "connect_timeout_seconds": 5.0,
    "timeout_seconds": 300.0,
    "max_retries": 3,
    "retry_base_delay_seconds": 1.0,
    "retry_multiplier": 2.0,
    "confidence_threshold": 0.85,
    "mid_stream_validation_enabled": false,
    "mid_stream_validation_interval_words": 300
  }
}
```

**Step 2: Update default config in sampling_proxy.py**

In `load_config()` function, update the default validation config (around line 49-60):

```python
"validation": {
    "enabled": False,
    "validator_url": "http://127.0.0.1:1234",
    "validator_model": "qwen-3.5-0.8b",
    "supports_openai": True,
    "supports_anthropic": False,
    "connect_timeout_seconds": 5.0,
    "timeout_seconds": 300.0,
    "max_retries": 3,
    "retry_base_delay_seconds": 1.0,
    "retry_multiplier": 2.0,
    "mid_stream_validation_enabled": False,
    "mid_stream_validation_interval_words": 300
}
```

**Step 3: Commit**

```bash
cd E:/sync/unique/work/git/sampling-proxy
git add config_sample.json sampling_proxy.py
git commit -m "feat: add mid-stream validation config options"
```

---

## Task 2: Add PARTIAL_VALIDATION_PROMPT to validator.py

**Files:**
- Modify: `validator.py:24-36` (after VALIDATION_PROMPT)

**Step 1: Add the partial validation prompt**

Add after the existing `VALIDATION_PROMPT` (around line 37):

```python


PARTIAL_VALIDATION_PROMPT = """Check if this PARTIAL AI response has problems.

PROBLEMS TO DETECT:
- repetition: same phrase repeated 3+ times in a row
- nonsense: gibberish, random tokens, or completely incoherent text

This is a PARTIAL response that may continue. Do NOT mark as invalid for:
- Incomplete sentences (normal for partial responses)
- Missing conclusion (response may continue)

If no problems detected, set is_valid=true and issue_type=null.

Output ONLY JSON (no markdown):
{{"is_valid":true/false,"issue_type":"repetition"|"nonsense"|null,"confidence":0.0-1.0}}

Partial response:
{content}"""
```

**Step 2: Commit**

```bash
cd E:/sync/unique/work/git/sampling-proxy
git add validator.py
git commit -m "feat: add PARTIAL_VALIDATION_PROMPT for mid-stream validation"
```

---

## Task 3: Add count_words_in_text helper function

**Files:**
- Modify: `validator.py` (add after `get_cache_dir` function, around line 46)

**Step 1: Add word counting helper**

Add after `get_cache_dir()` function:

```python


def count_words_in_text(text: str) -> int:
    """
    Count words in text using simple whitespace splitting.
    Works for both English and most languages.
    """
    if not text:
        return 0
    return len(text.split())
```

**Step 2: Commit**

```bash
cd E:/sync/unique/work/git/sampling-proxy
git add validator.py
git commit -m "feat: add count_words_in_text helper function"
```

---

## Task 4: Add validate_response_partial function

**Files:**
- Modify: `validator.py` (add after `validate_response` function, around line 334)

**Step 1: Add the partial validation function**

Add after `validate_response()` function:

```python


async def validate_response_partial(content: str, config: dict) -> ValidationResult:
    """
    Validate partial response content for mid-stream garbage detection.

    Uses PARTIAL_VALIDATION_PROMPT which focuses only on:
    - Repetition loops
    - Nonsense/gibberish

    Does NOT check for truncation (expected in partial responses).

    Args:
        content: Partial text content to validate
        config: Validation config

    Returns:
        ValidationResult with is_valid, issue_type, confidence
    """
    enable_logs = config.get("enable_validation_logs", False)

    if not config.get("mid_stream_validation_enabled", False):
        return ValidationResult(is_valid=True)

    if enable_logs:
        print(f"VALIDATION [mid-stream]: Starting partial validation ({len(content)} chars)")

    try:
        if not content.strip():
            return ValidationResult(is_valid=True)

        # Build prompt with partial content
        prompt = PARTIAL_VALIDATION_PROMPT.format(content=content)

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
                if enable_logs:
                    print(f"VALIDATION [mid-stream]: Validator error {response.status_code}")
                return ValidationResult(is_valid=True, error=f"Validator HTTP {response.status_code}")

            raw_result = response.json()
            return parse_validator_response(raw_result, config)

    except httpx.HTTPStatusError as e:
        if enable_logs:
            print(f"VALIDATION [mid-stream]: HTTP error - {e}")
        return ValidationResult(is_valid=True, error=f"HTTP error: {e}")
    except httpx.RequestError as e:
        if enable_logs:
            print(f"VALIDATION [mid-stream]: Connection error - {e}")
        return ValidationResult(is_valid=True, error=f"Connection error: {e}")
    except Exception as e:
        if enable_logs:
            print(f"VALIDATION [mid-stream]: Unexpected error - {type(e).__name__}: {e}")
        return ValidationResult(is_valid=True, error=f"Unexpected error: {type(e).__name__}: {e}")
```

**Step 2: Commit**

```bash
cd E:/sync/unique/work/git/sampling-proxy
git add validator.py
git commit -m "feat: add validate_response_partial for mid-stream validation"
```

---

## Task 5: Add StreamingValidator class

**Files:**
- Modify: `validator.py` (add after `calculate_retry_delay` function, around line 410)

**Step 1: Add the StreamingValidator class**

Add at the end of `validator.py`:

```python


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

    def should_validate(self, current_word_count: int) -> bool:
        """Check if we've reached the interval threshold."""
        if not self.enabled:
            return False
        self._current_word_count = current_word_count
        return current_word_count - self.last_validated_word_count >= self.interval_words

    async def start_validation(self, content: str, config: dict):
        """Spawn async validation task (non-blocking)."""
        if not self.enabled:
            return
        if self.validator_task and not self.validator_task.done():
            return  # Previous validation still running

        self.last_validated_word_count = self._current_word_count
        self.validator_task = asyncio.create_task(
            self._validate_partial(content, config)
        )

    async def _validate_partial(self, content: str, config: dict):
        """Validate partial content, signal if garbage detected."""
        try:
            result = await validate_response_partial(content, config)
            if not result.is_valid and not result.error:
                # Only signal garbage if validation succeeded and found issue
                self.garbage_detected.set()
        except Exception as e:
            # Log but don't signal garbage on validator errors (fail-open)
            print(f"VALIDATION [mid-stream]: Error during async validation: {e}")

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
            return f"garbage detected at ~{self.last_validated_word_count} words"
        return None
```

**Step 2: Commit**

```bash
cd E:/sync/unique/work/git/sampling-proxy
git add validator.py
git commit -m "feat: add StreamingValidator class for mid-stream validation"
```

---

## Task 6: Add extract_text_from_sse_chunks helper

**Files:**
- Modify: `validator.py` (add after `count_words_in_text` function)

**Step 1: Add SSE text extraction helper**

Add after `count_words_in_text()` function:

```python


def extract_text_from_sse_chunks(chunks: list) -> str:
    """
    Extract accumulated text from SSE chunks.

    Parses Anthropic SSE format to extract text deltas.
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
                    import json
                    data = json.loads(data_str)
                    # Handle Anthropic content_block_delta
                    if data.get('type') == 'content_block_delta':
                        delta = data.get('delta', {})
                        if delta.get('type') == 'text_delta':
                            text_parts.append(delta.get('text', ''))
                except json.JSONDecodeError:
                    continue

    return ''.join(text_parts)
```

**Step 2: Commit**

```bash
cd E:/sync/unique/work/git/sampling-proxy
git add validator.py
git commit -m "feat: add extract_text_from_sse_chunks helper"
```

---

## Task 7: Update imports in sampling_proxy.py

**Files:**
- Modify: `sampling_proxy.py:12-19` (imports)

**Step 1: Add new imports from validator**

Update the import statement to include new functions:

```python
# Import validator module for garbage detection
from validator import (
    validate_response,
    save_failed_response,
    create_error_message,
    calculate_retry_delay,
    ValidationResult,
    StreamingValidator,
    count_words_in_text,
    extract_text_from_sse_chunks
)
```

**Step 2: Commit**

```bash
cd E:/sync/unique/work/git/sampling-proxy
git add sampling_proxy.py
git commit -m "feat: import StreamingValidator and helpers in sampling_proxy"
```

---

## Task 8: Integrate mid-stream validation in buffered_stream_with_validation

**Files:**
- Modify: `sampling_proxy.py:1189-1315` (buffered_stream_with_validation function)

**Step 1: Replace the buffered_stream_with_validation function**

Replace the entire `async def buffered_stream_with_validation():` function (lines ~1189-1315) with:

```python
                    async def buffered_stream_with_validation():
                        nonlocal initial_response
                        max_retries = VALIDATION_CONFIG.get("max_retries", 3)
                        # Total attempts = 1 (initial) + max_retries
                        max_attempts = 1 + max_retries
                        attempt = 0
                        current_response = initial_response

                        while attempt < max_attempts:
                            attempt += 1

                            # Buffer all chunks with mid-stream validation
                            chunks = []
                            streaming_validator = StreamingValidator(VALIDATION_CONFIG)
                            early_break = False

                            async for chunk in current_response.aiter_bytes():
                                chunks.append(chunk)

                                # Mid-stream validation check
                                text_so_far = extract_text_from_sse_chunks(chunks)
                                word_count = count_words_in_text(text_so_far)

                                if streaming_validator.should_validate(word_count):
                                    if ENABLE_DEBUG_LOGS:
                                        print(f"DEBUG: Mid-stream validation at {word_count} words")
                                    await streaming_validator.start_validation(text_so_far, VALIDATION_CONFIG)

                                # Check if garbage detected by async validator
                                if streaming_validator.is_garbage_detected():
                                    print(f"VALIDATION [mid-stream]: Garbage detected at ~{word_count} words, interrupting stream")
                                    early_break = True
                                    break

                            # Wait for any pending validation before proceeding
                            await streaming_validator.wait_for_pending_validation()

                            await current_response.aclose()

                            # Check if mid-stream detection triggered
                            if early_break or streaming_validator.is_garbage_detected():
                                detection_info = streaming_validator.get_detection_info()
                                print(f"VALIDATION FAILED [mid-stream]: {detection_info} (attempt: {attempt})")

                                if attempt >= max_attempts:
                                    # Return error message as stream
                                    error_response = create_error_message("repetition", "mid-stream detection")
                                    message_start = {"type": "message_start", "message": error_response}
                                    yield f"event: message_start\ndata: {json.dumps(message_start)}\n\n".encode()

                                    content_block_start = {
                                        "type": "content_block_start",
                                        "index": 0,
                                        "content_block": {"type": "text", "text": ""}
                                    }
                                    yield f"event: content_block_start\ndata: {json.dumps(content_block_start)}\n\n".encode()

                                    error_text = error_response["content"][0]["text"]
                                    content_block_delta = {
                                        "type": "content_block_delta",
                                        "index": 0,
                                        "delta": {"type": "text_delta", "text": error_text}
                                    }
                                    yield f"event: content_block_delta\ndata: {json.dumps(content_block_delta)}\n\n".encode()

                                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n".encode()

                                    message_delta = {
                                        "type": "message_delta",
                                        "delta": {"stop_reason": "end_turn"},
                                        "usage": {"output_tokens": 0}
                                    }
                                    yield f"event: message_delta\ndata: {json.dumps(message_delta)}\n\n".encode()

                                    yield b"event: message_stop\ndata: {}\n\n"

                                    print(f"VALIDATION FAILED: Max attempts ({max_attempts}) reached")
                                    return

                                # Retry with backoff
                                delay = await calculate_retry_delay(attempt, VALIDATION_CONFIG)
                                if delay > 0:
                                    print(f"Retrying in {delay}s (attempt {attempt + 1}/{max_attempts})")
                                    await asyncio.sleep(delay)

                                # Make retry request
                                retry_response = await client.request(
                                    method="POST",
                                    url=target_url,
                                    headers=headers,
                                    content=request_content,
                                )

                                if retry_response.status_code == 200:
                                    current_response = retry_response
                                    continue
                                else:
                                    print(f"WARNING: Retry request failed with status {retry_response.status_code}")
                                    await retry_response.aclose()
                                    # Fall through to final validation with current chunks
                                    break

                            # Reconstruct response for final validation
                            full_content = b''.join(chunks)
                            try:
                                response_text = full_content.decode('utf-8')
                                # Parse SSE to get final response
                                response_dict = parse_sse_to_response(response_text)

                                if response_dict:
                                    print(f"VALIDATION: [Anthropic passthrough streaming] validating response id={response_dict.get('id', 'unknown')}")
                                    validation_result = await validate_response(response_dict, VALIDATION_CONFIG)

                                    if validation_result.error:
                                        print(f"WARNING: Validator error: {validation_result.error}")
                                        # Pass through on validator error
                                        for chunk in chunks:
                                            yield chunk
                                        return

                                    if validation_result.is_valid:
                                        if ENABLE_DEBUG_LOGS:
                                            print(f"DEBUG: Streaming response validated (attempt {attempt})")
                                        # Replay buffered chunks
                                        for chunk in chunks:
                                            yield chunk
                                        return

                                    # Invalid - will retry
                                    print(f"VALIDATION FAILED: {validation_result.issue_type} (confidence: {validation_result.confidence}, attempt: {attempt})")

                                    if attempt >= max_attempts:
                                        # Return error message as stream using proper Anthropic SSE format
                                        saved_path = save_failed_response(response_dict, validation_result, attempt)
                                        error_response = create_error_message(validation_result.issue_type, saved_path)

                                        # Build proper SSE stream with message_start, content blocks, and message_stop
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

                                        print(f"VALIDATION FAILED: Max attempts ({max_attempts}) reached")
                                        return

                                    # Retry with backoff
                                    delay = await calculate_retry_delay(attempt, VALIDATION_CONFIG)
                                    if delay > 0:
                                        print(f"Retrying in {delay}s (attempt {attempt + 1}/{max_attempts})")
                                        await asyncio.sleep(delay)

                                    # Make retry request
                                    retry_response = await client.request(
                                        method="POST",
                                        url=target_url,
                                        headers=headers,
                                        content=request_content,
                                    )

                                    if retry_response.status_code == 200:
                                        current_response = retry_response
                                        continue
                                    else:
                                        # Retry request failed, use last response
                                        print(f"WARNING: Retry request failed with status {retry_response.status_code}")
                                        await retry_response.aclose()
                                        for chunk in chunks:
                                            yield chunk
                                        return
                                else:
                                    # Can't parse SSE, pass through
                                    for chunk in chunks:
                                        yield chunk
                                    return

                            except Exception as e:
                                print(f"ERROR during streaming validation: {e}")
                                for chunk in chunks:
                                    yield chunk
                                return
```

**Step 2: Commit**

```bash
cd E:/sync/unique/work/git/sampling-proxy
git add sampling_proxy.py
git commit -m "feat: integrate mid-stream validation in buffered_stream_with_validation"
```

---

## Task 9: Update README with new options

**Files:**
- Modify: `README.md`

**Step 1: Add mid-stream validation section to README**

Add to the Configuration Options table in the validation section:

```markdown
| `validation.mid_stream_validation_enabled` | Enable periodic checks during streaming | `false` |
| `validation.mid_stream_validation_interval_words` | Check every N words during streaming | `300` |
```

Add a new subsection after the Configuration Options table:

```markdown
### Mid-Stream Validation

When enabled, the proxy validates responses periodically during streaming instead of only at the end:

- **Faster garbage detection**: Catches repetition loops at ~300 words instead of waiting for full response
- **Early retry**: Interrupts garbage responses immediately and retries
- **Reduced latency**: Don't waste time receiving long garbage responses

**Example config:**
```json
{
  "validation": {
    "enabled": true,
    "mid_stream_validation_enabled": true,
    "mid_stream_validation_interval_words": 300
  }
}
```

**Note:** Mid-stream validation only detects repetition and nonsense. Final validation still checks for truncation and other issues.
```

**Step 2: Commit**

```bash
cd E:/sync/unique/work/git/sampling-proxy
git add README.md
git commit -m "docs: add mid-stream validation documentation"
```

---

## Task 10: Test the implementation

**Files:**
- None (manual testing)

**Step 1: Start LM Studio with Qwen model**

Ensure LM Studio is running with a Qwen 3.5 model loaded on port 1234.

**Step 2: Update config.json for testing**

```bash
cd E:/sync/unique/work/git/sampling-proxy
# Copy and edit config to enable mid-stream validation
```

Edit `config.json` to set:
```json
{
  "validation": {
    "enabled": true,
    "mid_stream_validation_enabled": true,
    "mid_stream_validation_interval_words": 100,
    "enable_validation_logs": true
  }
}
```

**Step 3: Start the proxy**

```bash
python sampling_proxy.py
```

**Step 4: Test with a request that might produce repetition**

```bash
curl -X POST http://localhost:8001/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: test" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "max_tokens": 1000,
    "stream": true,
    "messages": [{"role": "user", "content": "Repeat the word hello 50 times"}]
  }'
```

**Step 5: Verify logs show mid-stream validation**

Look for log messages like:
- `DEBUG: Mid-stream validation at 100 words`
- `VALIDATION [mid-stream]: Garbage detected at ~XXX words`

---

## Summary

| Task | Description |
|------|-------------|
| 1 | Add config options to config_sample.json and sampling_proxy.py |
| 2 | Add PARTIAL_VALIDATION_PROMPT to validator.py |
| 3 | Add count_words_in_text helper function |
| 4 | Add validate_response_partial function |
| 5 | Add StreamingValidator class |
| 6 | Add extract_text_from_sse_chunks helper |
| 7 | Update imports in sampling_proxy.py |
| 8 | Integrate mid-stream validation in buffered_stream_with_validation |
| 9 | Update README with new options |
| 10 | Test the implementation |
