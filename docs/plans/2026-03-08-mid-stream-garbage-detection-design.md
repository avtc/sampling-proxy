# Mid-Stream Garbage Detection Design

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:executing-plans to implement this plan task-by-task.

**Goal:** Detect garbage LLM responses during streaming (not just after completion) to enable early retry and reduce wasted time.

**Approach:** Async periodic validation - spawn validator tasks at word intervals while continuing to buffer, interrupt early if garbage detected.

---

## Problem Statement

Current implementation:
- Buffers entire streaming response before validating
- Only detects garbage AFTER full response received
- Wastes time receiving long garbage responses before retrying

Desired behavior:
- Detect repetition loops and nonsense mid-stream
- Interrupt early and retry immediately
- Reduce latency wasted on garbage responses

---

## Configuration

Add new options to `validation` section in `config.json`:

```json
{
  "validation": {
    "enabled": true,
    "mid_stream_validation_enabled": true,
    "mid_stream_validation_interval_words": 300,
    "validator_url": "http://127.0.0.1:1234",
    "validator_model": "qwen-3.5-0.8b",
    "max_retries": 3,
    "retry_base_delay_seconds": 1.0,
    "retry_multiplier": 2.0
  }
}
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `mid_stream_validation_enabled` | bool | false | Enable mid-stream validation checks |
| `mid_stream_validation_interval_words` | int | 300 | Trigger validation every N words |

---

## Architecture

### New Component: StreamingValidator Class

Location: `validator.py`

```python
class StreamingValidator:
    """Handles periodic validation during streaming."""

    def __init__(self, config: dict):
        self.interval_words = config.get("mid_stream_validation_interval_words", 300)
        self.validator_task: Optional[asyncio.Task] = None
        self.garbage_detected = asyncio.Event()
        self.last_validated_word_count = 0

    def should_validate(self, current_word_count: int) -> bool:
        """Check if we've reached the interval threshold."""
        return current_word_count - self.last_validated_word_count >= self.interval_words

    async def start_validation(self, content: str, config: dict):
        """Spawn async validation task (non-blocking)."""
        if self.validator_task and not self.validator_task.done():
            return  # Previous validation still running
        self.last_validated_word_count = self._current_word_count
        self.validator_task = asyncio.create_task(
            self._validate_partial(content, config)
        )

    async def _validate_partial(self, content: str, config: dict):
        """Validate partial content, signal if garbage detected."""
        result = await validate_response_partial(content, config)
        if not result.is_valid:
            self.garbage_detected.set()

    def is_garbage_detected(self) -> bool:
        """Check if garbage was detected by any validator task."""
        return self.garbage_detected.is_set()

    async def wait_for_pending_validation(self):
        """Wait for any in-flight validation to complete."""
        if self.validator_task and not self.validator_task.done():
            await self.validator_task
```

### New Validation Prompt for Partial Responses

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
{"is_valid":true/false,"issue_type":"repetition"|"nonsense"|null,"confidence":0.0-1.0}

Partial response:
{content}"""
```

### New Function: validate_response_partial

```python
async def validate_response_partial(content: str, config: dict) -> ValidationResult:
    """
    Validate partial response content for mid-stream garbage detection.

    Uses PARTIAL_VALIDATION_PROMPT which focuses only on:
    - Repetition loops
    - Nonsense/gibberish

    Does NOT check for truncation (expected in partial responses).
    """
```

---

## Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    buffered_stream_with_validation()            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  attempt = 0                                                    │
│  while attempt < max_attempts:                                  │
│      ┌──────────────────────────────────────────────────────┐   │
│      │ chunks = []                                           │   │
│      │ validator = StreamingValidator(config)               │   │
│      │                                                       │   │
│      │ async for chunk in response.aiter_bytes():           │   │
│      │     chunks.append(chunk)                              │   │
│      │     word_count = count_words(chunks)                  │   │
│      │                                                       │   │
│      │     # Mid-stream validation                           │   │
│      │     if validator.should_validate(word_count):         │   │
│      │         content = extract_text(chunks)                │   │
│      │         await validator.start_validation(content)     │   │
│      │                                                       │   │
│      │     # Check for early garbage detection               │   │
│      │     if validator.is_garbage_detected():               │   │
│      │         break  # Exit chunk loop early                │   │
│      │                                                       │   │
│      │ # End of stream or early break                        │   │
│      │ await validator.wait_for_pending_validation()         │   │
│      │                                                       │   │
│      │ if validator.is_garbage_detected():                   │   │
│      │     # Garbage detected - retry                        │   │
│      │     attempt += 1                                      │   │
│      │     if attempt >= max_attempts:                       │   │
│      │         return error_message                          │   │
│      │     await sleep(backoff_delay)                        │   │
│      │     response = make_retry_request()                   │   │
│      │     continue  # New attempt                           │   │
│      │                                                       │   │
│      │ # Final validation (existing logic)                   │   │
│      │ result = await validate_response(full_response)       │   │
│      │ if result.is_valid:                                   │   │
│      │     for chunk in chunks: yield chunk                  │   │
│      │     return                                            │   │
│      │                                                       │   │
│      │ # Final validation failed - retry                     │   │
│      │ attempt += 1                                          │   │
│      │ ...                                                   │   │
│      └──────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Error Handling

| Scenario | Behavior |
|----------|----------|
| Validator connection fails | Log warning, continue buffering (fail-open) |
| Validator timeout | Log warning, continue buffering (fail-open) |
| Validator parse error | Log warning, continue buffering (fail-open) |
| Garbage detected mid-stream | Interrupt, retry with backoff |
| Max retries exceeded | Return error message to client |

---

## Implementation Tasks

### Task 1: Add config options
- Add `mid_stream_validation_enabled` to `config_sample.json`
- Add `mid_stream_validation_interval_words` to `config_sample.json`
- Update default config in `sampling_proxy.py`

### Task 2: Add StreamingValidator to validator.py
- Create `StreamingValidator` class
- Add `PARTIAL_VALIDATION_PROMPT`
- Add `validate_response_partial()` function
- Add helper function `count_words_in_text()`

### Task 3: Integrate StreamingValidator in sampling_proxy.py
- Modify `buffered_stream_with_validation()` to use `StreamingValidator`
- Add word counting during chunk accumulation
- Add early break on garbage detection
- Add pending validation wait before final check

### Task 4: Update README
- Document new config options
- Explain mid-stream validation behavior

### Task 5: Test implementation
- Test with normal responses (should pass through)
- Test with repetition loops (should detect early)
- Test with nonsense output (should detect)
- Verify retry behavior works correctly

---

## Benefits

1. **Early detection**: Catch garbage responses 3-5x faster (at 300 words vs 1000+ words)
2. **Reduced latency**: Don't waste time receiving long garbage responses
3. **Better UX**: Faster retry means user gets valid response sooner
4. **Backward compatible**: Disabled by default, existing behavior unchanged

## Limitations

1. **Partial context**: Mid-stream validator has less context than final validation
2. **Validator load**: More frequent validation calls increase validator CPU usage
3. **False positives possible**: Very short responses may trigger incorrectly (mitigated by 300 word default)
