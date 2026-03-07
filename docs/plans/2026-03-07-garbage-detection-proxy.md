# Garbage Detection Proxy Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:executing-plans to implement this plan task-by-task.

**Goal:** Extend sampling-proxy to validate AI responses with a local model and auto-retry on garbage output.

**Architecture:** Add passthrough mode for Anthropic-to-Anthropic proxying, integrate validator module using LM Studio's Anthropic-compatible API, buffer responses for validation, retry with exponential backoff on failure.

**Tech Stack:** Python, FastAPI, httpx, Anthropic SDK (for validator calls)

---

## Task 0: Update config.json with new options

**Files:**
- Modify: `config_sample.json`

**Step 1: Add backend_mode and validation config sections**

Add to `config_sample.json`:

```json
{
  "server": {
    "target_base_url": "http://127.0.0.1:8000/v1",
    "sampling_proxy_base_path": "",
    "sampling_proxy_host": "0.0.0.0",
    "sampling_proxy_port": 8001,
    "timeout_seconds": 1200.0
  },
  "backend_mode": "openai_convert",
  "logging": {
    "enable_debug_logs": false,
    "enable_override_logs": false
  },
  "default_sampling_params": {
    "top_p": null,
    "min_p": null,
    "top_k": null,
    "repetition_penalty": null,
    "temperature": null
  },
  "override": {
    "only_anthropic": false,
    "model_name": null,
    "sampling_params": {
      "top_p": null,
      "min_p": null,
      "top_k": null,
      "repetition_penalty": null,
      "temperature": null
    }
  },
  "model_sampling_params": {
    "sample_model_name": {
        "top_p": null,
        "min_p": null,
        "top_k": null,
        "repetition_penalty": null,
        "temperature": null
    }
  },
  "validation": {
    "enabled": false,
    "validator_url": "http://127.0.0.1:1234",
    "validator_model": "qwen-3.5-0.8b",
    "max_retries": 3,
    "retry_base_delay_seconds": 1.0,
    "retry_multiplier": 2.0
  }
}
```

**Step 2: Commit**

```bash
cd E:/sync/unique/work/git/sampling-proxy
git add config_sample.json
git commit -m "feat: add backend_mode and validation config options"
```

---

## Task 1: Create validator.py module

**Files:**
- Create: `validator.py`
- Modify: `requirements.txt`

**Step 1: Add anthropic dependency to requirements.txt**

```
fastapi>=0.104.0
uvicorn>=0.24.0
httpx>=0.25.0
anthropic>=0.40.0
```

**Step 2: Create validator.py with core validation logic**

```python
"""
Response validator module for garbage detection.
Uses local model via LM Studio (Anthropic-compatible API) to validate responses.
"""

import json
import asyncio
from dataclasses import dataclass
from typing import Optional
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


VALIDATION_PROMPT = """You are a response quality checker. Analyze the AI response below for these issues:

1. REPETITION: Same phrase/paragraph repeated 3+ times
2. TRUNCATION: Response cuts off mid-sentence or mid-code-block
3. MALFORMED TOOLS: Tool calls with invalid JSON or missing required fields

Respond with ONLY valid JSON, no markdown, no explanation:
{"is_valid": true/false, "issue_type": "repetition|truncation|malformed_tools|null", "confidence": 0.0-1.0}

---
Response to validate:
{content}"""


def get_cache_dir() -> Path:
    """Get cache directory for logs and failed responses."""
    cache_dir = Path.home() / ".cache" / "garbage-proxy"
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "logs").mkdir(exist_ok=True)
    (cache_dir / "failed").mkdir(exist_ok=True)
    return cache_dir


def extract_content_from_response(response: dict) -> str:
    """Extract text and tool content from Anthropic response for validation."""
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


async def call_validator_model(content: str, config: dict) -> dict:
    """
    Call LM Studio validator model using Anthropic-compatible API.

    Args:
        content: Response content to validate
        config: Validation config with validator_url, validator_model

    Returns:
        Raw response from validator as dict
    """
    validator_url = config.get("validator_url", "http://127.0.0.1:1234")
    validator_model = config.get("validator_model", "qwen-3.5-0.8b")

    prompt = VALIDATION_PROMPT.format(content=content)

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{validator_url}/v1/messages",
            headers={
                "Content-Type": "application/json",
                "x-api-key": "lmstudio",  # LM Studio accepts any key
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

        if response.status_code != 200:
            raise httpx.HTTPStatusError(
                f"Validator returned {response.status_code}: {response.text}",
                request=None,
                response=response
            )

        return response.json()


def parse_validator_response(response: dict) -> ValidationResult:
    """Parse validator model response into ValidationResult."""
    try:
        # Extract text from Anthropic response
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

        result = json.loads(text)

        return ValidationResult(
            is_valid=result.get("is_valid", True),
            issue_type=result.get("issue_type"),
            confidence=result.get("confidence", 1.0)
        )

    except (json.JSONDecodeError, KeyError) as e:
        # If we can't parse, assume valid (fail-open for parser errors)
        return ValidationResult(
            is_valid=True,
            error=f"Failed to parse validator response: {e}"
        )


async def validate_response(response: dict, config: dict) -> ValidationResult:
    """
    Validate an Anthropic response for garbage output.

    Args:
        response: Anthropic-format response dict
        config: Validation config

    Returns:
        ValidationResult with is_valid, issue_type, confidence
    """
    if not config.get("enabled", False):
        return ValidationResult(is_valid=True)

    try:
        content = extract_content_from_response(response)
        if not content.strip():
            # Empty content is valid
            return ValidationResult(is_valid=True)

        raw_result = await call_validator_model(content, config)
        return parse_validator_response(raw_result)

    except httpx.HTTPStatusError as e:
        return ValidationResult(
            is_valid=True,  # Fail-open
            error=f"Validator HTTP error: {e}"
        )
    except httpx.RequestError as e:
        return ValidationResult(
            is_valid=True,  # Fail-open
            error=f"Validator connection error: {e}"
        )
    except Exception as e:
        return ValidationResult(
            is_valid=True,  # Fail-open
            error=f"Validator unexpected error: {e}"
        )


def log_validation_failure(response: dict, validation_result: ValidationResult, attempt: int):
    """Log validation failure to cache directory."""
    cache_dir = get_cache_dir()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    log_entry = {
        "timestamp": timestamp,
        "attempt": attempt,
        "issue_type": validation_result.issue_type,
        "confidence": validation_result.confidence,
        "response_id": response.get("id"),
        "response_preview": extract_content_from_response(response)[:500]
    }

    log_file = cache_dir / "logs" / f"validation_{timestamp}.json"
    with open(log_file, "w") as f:
        json.dump(log_entry, f, indent=2)

    print(f"VALIDATION FAILURE: {validation_result.issue_type} (confidence: {validation_result.confidence}, attempt: {attempt})")


def save_failed_response(response: dict, validation_result: ValidationResult):
    """Save failed response to cache for later analysis."""
    cache_dir = get_cache_dir()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    failed_data = {
        "timestamp": timestamp,
        "issue_type": validation_result.issue_type,
        "confidence": validation_result.confidence,
        "response": response
    }

    failed_file = cache_dir / "failed" / f"failed_{timestamp}.json"
    with open(failed_file, "w") as f:
        json.dump(failed_data, f, indent=2)

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
```

**Step 3: Commit**

```bash
cd E:/sync/unique/work/git/sampling-proxy
git add validator.py requirements.txt
git commit -m "feat: add validator module for garbage detection"
```

---

## Task 2: Add anthropic_passthrough backend mode

**Files:**
- Modify: `sampling_proxy.py:11-92` (config loading)
- Modify: `sampling_proxy.py:383-395` (path routing)

**Step 1: Add backend_mode to default config and config loading**

In `load_config()` function, add to `default_config` dict (after line 35):

```python
    default_config = {
        # ... existing config ...
        "model_sampling_params": {},
        "backend_mode": "openai_convert",  # "openai_convert" | "anthropic_passthrough"
        "validation": {
            "enabled": False,
            "validator_url": "http://127.0.0.1:1234",
            "validator_model": "qwen-3.5-0.8b",
            "max_retries": 3,
            "retry_base_delay_seconds": 1.0,
            "retry_multiplier": 2.0
        }
    }
```

**Step 2: Add global config variables (after line ~160)**

```python
# Backend mode configuration
BACKEND_MODE = CONFIG.get("backend_mode", "openai_convert")
VALIDATION_CONFIG = CONFIG.get("validation", {"enabled": False})
```

**Step 3: Modify path routing for anthropic_passthrough mode**

Replace the path routing logic (around lines 383-395) with:

```python
    # Construct the target URL based on backend mode
    if is_anthropic_request:
        if BACKEND_MODE == "anthropic_passthrough":
            # Keep Anthropic path as-is, no conversion
            target_path = transform_path("/" + original_path, SAMPLING_PROXY_BASE_PATH, TARGET_BASE_PATH)
            if ENABLE_DEBUG_LOGS:
                print(f"DEBUG: Anthropic passthrough mode - keeping path: {target_path}")
        else:
            # Convert /v1/messages to /chat/completions for OpenAI backend
            transformed_path = transform_path("/" + original_path, SAMPLING_PROXY_BASE_PATH, TARGET_BASE_PATH)
            target_path = transformed_path.replace("/v1/messages", "/chat/completions", 1)
            if ENABLE_DEBUG_LOGS:
                print(f"DEBUG: Converting Anthropic request from {original_path} to {target_path}")
    else:
        # Apply base path transformation
        target_path = transform_path("/" + original_path, SAMPLING_PROXY_BASE_PATH, TARGET_BASE_PATH)
```

**Step 4: Commit**

```bash
cd E:/sync/unique/work/git/sampling-proxy
git add sampling_proxy.py
git commit -m "feat: add anthropic_passthrough backend mode"
```

---

## Task 3: Skip request conversion in passthrough mode

**Files:**
- Modify: `sampling_proxy.py:432-650` (request conversion)

**Step 1: Wrap request conversion in backend mode check**

Around line 432, wrap the Anthropic-to-OpenAI conversion:

```python
            # Handle Anthropic request based on backend mode
            if is_anthropic_request:
                if BACKEND_MODE == "anthropic_passthrough":
                    # Passthrough mode: keep request as-is
                    if ENABLE_DEBUG_LOGS:
                        print("DEBUG: Anthropic passthrough mode - keeping request format")
                    # Don't modify incoming_json_body
                else:
                    # Convert Anthropic to OpenAI format
                    if ENABLE_DEBUG_LOGS:
                        print("DEBUG: Converting Anthropic request to OpenAI format.")

                    try:
                        # ... existing conversion code ...
```

**Step 2: Update request body sending logic**

Around line 680-720, update the request body handling:

```python
    # Prepare request body
    if is_anthropic_request and BACKEND_MODE == "openai_convert":
        # Use converted OpenAI format
        request_content = json.dumps(outgoing_json_body).encode('utf-8')
    else:
        # Use original request body (passthrough mode or non-anthropic)
        request_content = raw_body_for_forwarding
```

**Step 3: Commit**

```bash
cd E:/sync/unique/work/git/sampling-proxy
git add sampling_proxy.py
git commit -m "feat: skip request conversion in anthropic_passthrough mode"
```

---

## Task 4: Skip response conversion in passthrough mode

**Files:**
- Modify: `sampling_proxy.py:810-930` (streaming conversion)
- Modify: `sampling_proxy.py:950-1020` (non-streaming conversion)

**Step 1: Skip streaming response conversion**

Around line 810, wrap the chunk conversion:

```python
                            # Convert OpenAI streaming response to Anthropic format only in convert mode
                            if is_anthropic_request and BACKEND_MODE == "openai_convert" and chunk:
                                # ... existing conversion code ...
                            # In passthrough mode, just pass through the chunk
```

**Step 2: Skip non-streaming response conversion**

Around line 950, wrap the response conversion:

```python
                if is_anthropic_request and BACKEND_MODE == "openai_convert" and target_response.status_code == 200:
                    # ... existing conversion code ...
                elif is_anthropic_request and BACKEND_MODE == "anthropic_passthrough":
                    if ENABLE_DEBUG_LOGS:
                        print("DEBUG: Anthropic passthrough mode - keeping response format")
                    # Keep response as-is
```

**Step 3: Commit**

```bash
cd E:/sync/unique/work/git/sampling-proxy
git commit -m "feat: skip response conversion in anthropic_passthrough mode"
```

---

## Task 5: Add validation and retry logic for non-streaming

**Files:**
- Modify: `sampling_proxy.py` (imports and response handling)
- Import from: `validator.py`

**Step 1: Add imports at top of file**

```python
import os
import json
import httpx
from fastapi import FastAPI, Request, Response, status
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
import uvicorn
import asyncio
import argparse

# Import validator module
from validator import (
    validate_response,
    log_validation_failure,
    save_failed_response,
    create_error_message,
    calculate_retry_delay,
    ValidationResult
)
```

**Step 2: Create helper function for making requests**

Add before `proxy_target_requests`:

```python
async def make_anthropic_request(
    client: httpx.AsyncClient,
    target_url: httpx.URL,
    headers: dict,
    request_content: bytes,
    stream: bool = False
) -> httpx.Response:
    """Make request to backend and return response."""
    return await client.request(
        method="POST",
        url=target_url,
        headers=headers,
        content=request_content,
    )
```

**Step 3: Add validation wrapper for non-streaming responses**

After line 1020, before returning response, add validation:

```python
                # Validation logic for anthropic_passthrough mode
                if (is_anthropic_request and
                    BACKEND_MODE == "anthropic_passthrough" and
                    VALIDATION_CONFIG.get("enabled", False) and
                    target_response.status_code == 200):

                    max_retries = VALIDATION_CONFIG.get("max_retries", 3)
                    attempt = 0

                    while attempt < max_retries:
                        attempt += 1

                        # Parse response for validation
                        try:
                            response_dict = json.loads(response_content.decode('utf-8'))
                        except json.JSONDecodeError:
                            # Can't validate non-JSON, pass through
                            break

                        # Validate response
                        validation_result = await validate_response(response_dict, VALIDATION_CONFIG)

                        if validation_result.error:
                            # Validator failed, log and pass through
                            print(f"WARNING: Validator error: {validation_result.error}")
                            break

                        if validation_result.is_valid:
                            # Valid response, proceed
                            if ENABLE_DEBUG_LOGS:
                                print(f"DEBUG: Response validated successfully (attempt {attempt})")
                            break

                        # Invalid response - log and retry
                        log_validation_failure(response_dict, validation_result, attempt)

                        if attempt >= max_retries:
                            # Max retries reached, return error message
                            saved_path = save_failed_response(response_dict, validation_result)
                            error_response = create_error_message(validation_result.issue_type, saved_path)
                            response_content = json.dumps(error_response).encode('utf-8')
                            print(f"VALIDATION FAILED: Max retries ({max_retries}) reached")
                            break

                        # Retry with backoff
                        delay = await calculate_retry_delay(attempt, VALIDATION_CONFIG)
                        if delay > 0:
                            print(f"Retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
                            await asyncio.sleep(delay)

                        # Make retry request
                        retry_response = await make_anthropic_request(
                            client, target_url, headers, request_content
                        )

                        if retry_response.status_code == 200:
                            response_content = retry_response.content
                            await retry_response.aclose()
                        else:
                            # Retry request failed, use last response
                            print(f"WARNING: Retry request failed with status {retry_response.status_code}")
                            await retry_response.aclose()
                            break
```

**Step 4: Commit**

```bash
cd E:/sync/unique/work/git/sampling-proxy
git add sampling_proxy.py
git commit -m "feat: add validation and retry logic for non-streaming responses"
```

---

## Task 6: Add validation for streaming responses

**Files:**
- Modify: `sampling_proxy.py:803-930` (streaming handler)

**Step 1: Create buffered streaming generator with validation**

Replace `stream_and_close_response` function with buffered validation version:

```python
                # For streaming with validation, buffer and validate first
                if (is_anthropic_request and
                    BACKEND_MODE == "anthropic_passthrough" and
                    VALIDATION_CONFIG.get("enabled", False)):

                    async def buffered_stream_with_validation():
                        max_retries = VALIDATION_CONFIG.get("max_retries", 3)
                        attempt = 0

                        while attempt < max_retries:
                            attempt += 1

                            # Buffer all chunks
                            chunks = []
                            async for chunk in target_response.aiter_bytes():
                                chunks.append(chunk)

                            await target_response.aclose()

                            # Reconstruct response for validation
                            full_content = b''.join(chunks)
                            try:
                                response_text = full_content.decode('utf-8')
                                # Parse SSE to get final response
                                response_dict = parse_sse_to_response(response_text)

                                if response_dict:
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

                                    # Invalid - log and retry
                                    log_validation_failure(response_dict, validation_result, attempt)

                                    if attempt >= max_retries:
                                        # Return error message as stream
                                        saved_path = save_failed_response(response_dict, validation_result)
                                        error_response = create_error_message(validation_result.issue_type, saved_path)
                                        yield f"event: message_start\ndata: {{\"type\":\"message_start\",\"message\":{json.dumps(error_response)}}}\n\n".encode()
                                        yield b"event: message_stop\ndata: {}\n\n"
                                        print(f"VALIDATION FAILED: Max retries ({max_retries}) reached")
                                        return

                                    # Retry with backoff
                                    delay = await calculate_retry_delay(attempt, VALIDATION_CONFIG)
                                    if delay > 0:
                                        print(f"Retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
                                        await asyncio.sleep(delay)

                                    # Make retry request
                                    target_response = await client.request(
                                        method="POST",
                                        url=target_url,
                                        headers=headers,
                                        content=request_content,
                                    )
                                    continue
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

                    return StreamingResponse(
                        buffered_stream_with_validation(),
                        status_code=target_response.status_code,
                        headers=response_headers,
                        media_type=response_headers.get("content-type"),
                    )

                # Standard streaming without validation
                async def stream_and_close_response():
                    # ... existing code ...
```

**Step 2: Add SSE parsing helper function**

Add before `proxy_target_requests`:

```python
def parse_sse_to_response(sse_text: str) -> Optional[dict]:
    """Parse SSE stream text to extract final response dict."""
    content_blocks = {}
    current_index = None
    message_data = None

    for line in sse_text.split('\n'):
        line = line.strip()
        if not line:
            continue

        if line.startswith('data: '):
            data_str = line[6:]
            if data_str == '[DONE]':
                continue

            try:
                data = json.loads(data_str)
                event_type = data.get('type')

                if event_type == 'message_start':
                    message_data = data.get('message', {})
                elif event_type == 'content_block_start':
                    index = data.get('index', 0)
                    content_blocks[index] = data.get('content_block', {})
                    current_index = index
                elif event_type == 'content_block_delta':
                    index = data.get('index', 0)
                    delta = data.get('delta', {})

                    if index not in content_blocks:
                        content_blocks[index] = {}

                    if delta.get('type') == 'text_delta':
                        existing_text = content_blocks[index].get('text', '')
                        content_blocks[index]['text'] = existing_text + delta.get('text', '')
                    elif delta.get('type') == 'input_json_delta':
                        existing_json = content_blocks[index].get('_partial_json', '')
                        content_blocks[index]['_partial_json'] = existing_json + delta.get('partial_json', '')
                elif event_type == 'message_stop':
                    # Build final response
                    if message_data:
                        content = []
                        for idx in sorted(content_blocks.keys()):
                            block = content_blocks[idx]
                            block_type = block.get('type', 'text')
                            if block_type == 'tool_use':
                                partial_json = block.pop('_partial_json', '')
                                if partial_json:
                                    try:
                                        block['input'] = json.loads(partial_json)
                                    except json.JSONDecodeError:
                                        block['input'] = {}
                                content.append(block)
                            else:
                                content.append(block)

                        message_data['content'] = content
                        return message_data

            except json.JSONDecodeError:
                continue

    return None
```

**Step 3: Commit**

```bash
cd E:/sync/unique/work/git/sampling-proxy
git add sampling_proxy.py
git commit -m "feat: add validation for streaming responses with buffering"
```

---

## Task 7: Create sample config for z.ai usage

**Files:**
- Create: `config_zai_sample.json`

**Step 1: Create z.ai-specific config sample**

```json
{
  "server": {
    "target_base_url": "https://api.z.ai/api/anthropic",
    "sampling_proxy_base_path": "",
    "sampling_proxy_host": "0.0.0.0",
    "sampling_proxy_port": 8001,
    "timeout_seconds": 1200.0
  },
  "backend_mode": "anthropic_passthrough",
  "logging": {
    "enable_debug_logs": false,
    "enable_override_logs": false
  },
  "default_sampling_params": {},
  "override": {
    "only_anthropic": false,
    "model_name": null,
    "sampling_params": {}
  },
  "model_sampling_params": {},
  "validation": {
    "enabled": true,
    "validator_url": "http://127.0.0.1:1234",
    "validator_model": "qwen-3.5-0.8b",
    "max_retries": 3,
    "retry_base_delay_seconds": 1.0,
    "retry_multiplier": 2.0
  }
}
```

**Step 2: Commit**

```bash
cd E:/sync/unique/work/git/sampling-proxy
git add config_zai_sample.json
git commit -m "docs: add sample config for z.ai with validation enabled"
```

---

## Task 8: Update README with usage instructions

**Files:**
- Modify: `README.md`

**Step 1: Add validation section to README**

Add after existing content:

```markdown
## Garbage Detection Mode

The proxy can validate AI responses using a local model (LM Studio) and automatically retry when garbage output is detected.

### Features

- **Repetition detection**: Catches loops where the same phrase is repeated 3+ times
- **Truncation detection**: Identifies responses that cut off mid-sentence
- **Malformed tool calls**: Detects invalid JSON in tool use blocks
- **Auto-retry**: Automatically retries with exponential backoff (1s, 2s delays)
- **Fail-open**: If validator is unavailable, responses pass through unmodified

### Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Start LM Studio with a small model loaded (e.g., Qwen 3.5 0.8B or 4B)

3. Copy the z.ai sample config:
   ```bash
   cp config_zai_sample.json config.json
   ```

4. Edit `config.json` if needed:
   - `validation.validator_url`: LM Studio URL (default: http://127.0.0.1:1234)
   - `validation.validator_model`: Model name in LM Studio
   - `validation.max_retries`: Max retry attempts (default: 3)

5. Start the proxy:
   ```bash
   python sampling_proxy.py
   ```

6. Configure Claude Code to use the proxy as your API endpoint:
   - Base URL: `http://localhost:8001`

### Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `backend_mode` | `openai_convert` or `anthropic_passthrough` | `openai_convert` |
| `validation.enabled` | Enable response validation | `false` |
| `validation.validator_url` | LM Studio endpoint | `http://127.0.0.1:1234` |
| `validation.validator_model` | Model for validation | `qwen-3.5-0.8b` |
| `validation.max_retries` | Max retry attempts | `3` |
| `validation.retry_base_delay_seconds` | Initial retry delay | `1.0` |
| `validation.retry_multiplier` | Backoff multiplier | `2.0` |

### Logs and Failed Responses

- Validation failures are logged to `~/.cache/garbage-proxy/logs/`
- Failed responses are saved to `~/.cache/garbage-proxy/failed/`
```

**Step 2: Commit**

```bash
cd E:/sync/unique/work/git/sampling-proxy
git add README.md
git commit -m "docs: add garbage detection mode documentation"
```

---

## Task 9: Test the implementation

**Files:**
- None (manual testing)

**Step 1: Start LM Studio with Qwen model**

Ensure LM Studio is running with a Qwen 3.5 model loaded on port 1234.

**Step 2: Start the proxy**

```bash
cd E:/sync/unique/work/git/sampling-proxy
cp config_zai_sample.json config.json
python sampling_proxy.py
```

**Step 3: Test with curl**

```bash
curl -X POST http://localhost:8001/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: test" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "max_tokens": 100,
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

**Step 4: Verify logs appear in ~/.cache/garbage-proxy/ when validation occurs**

---

## Summary

| Task | Description |
|------|-------------|
| 0 | Update config.json with new options |
| 1 | Create validator.py module |
| 2 | Add anthropic_passthrough backend mode |
| 3 | Skip request conversion in passthrough mode |
| 4 | Skip response conversion in passthrough mode |
| 5 | Add validation and retry logic for non-streaming |
| 6 | Add validation for streaming responses |
| 7 | Create sample config for z.ai usage |
| 8 | Update README with usage instructions |
| 9 | Test the implementation |
