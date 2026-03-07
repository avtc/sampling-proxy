# Garbage Detection Proxy Design

**Date:** 2026-03-07
**Status:** Approved
**Related Tasks:** #1, #2, #3, #4, #5

## Problem Statement

z.ai coding plan (using glm-5 model) occasionally produces garbage output:
- Repetition loops
- Incomplete/truncated responses
- Failed tool calls (malformed JSON)

These issues occur intermittently even with empty context windows. Rewind doesn't always help because user responses to model questions don't appear in rewind history.

## Solution

Extend the existing sampling-proxy to:
1. Support Anthropic-to-Anthropic passthrough mode (for z.ai)
2. Validate all responses with a local model (LM Studio + Qwen 3.5)
3. Auto-retry on garbage detection with exponential backoff
4. Notify user when all retries fail

## Architecture

```
┌─────────────┐      ┌─────────────────┐      ┌─────────────────┐
│ Claude Code │ ───► │  Garbage Proxy  │ ───► │    z.ai API     │
└─────────────┘      │   (FastAPI)     │      │ /api/anthropic  │
                     │                 │      └─────────────────┘
                     │ All Anthropic   │            │
                     │ format          │      ┌─────▼─────┐
                     │                 │      │   Retry   │
                     │  ┌───────────┐  │      │   Logic   │
                     └──│ Validator │──┘      └───────────┘
                        │ (LM Studio)│
                        │ Qwen 0.8B/4B
                        └───────────┘
```

**All communication uses Anthropic format - no conversions needed.**

## Configuration

```json
{
  "server": {
    "target_base_url": "https://api.z.ai/api/anthropic",
    "sampling_proxy_host": "0.0.0.0",
    "sampling_proxy_port": 8001,
    "timeout_seconds": 1200.0
  },
  "backend_mode": "anthropic_passthrough",

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

### Backend Modes

| Mode | Behavior |
|------|----------|
| `openai_convert` | Current behavior - Anthropic in, OpenAI out |
| `anthropic_passthrough` | Anthropic in, Anthropic out (for z.ai) |

### Validation Settings

- `validator_url` - LM Studio endpoint (default: `http://127.0.0.1:1234`)
- `validator_model` - Model name loaded in LM Studio
- `max_retries` - Maximum retry attempts before notifying user
- `retry_base_delay_seconds` - Initial delay between retries
- `retry_multiplier` - Multiplier for exponential backoff

## Validation Flow

```
Response received → Buffer full content → Send to validator → Decision
                                                                │
                    ┌───────────────────────────────────────────┼───────────────────────────────────┐
                    ▼                                           ▼                                   ▼
              VALID (is_valid=true)                    GARBAGE (is_valid=false)              ERROR (validator failed)
                    │                                           │                                   │
                    ▼                                           ▼                                   ▼
              Stream to client                          Retry with backoff                   Log + pass through
                                                              │
                                                              ▼
                                                    Max retries reached?
                                                      │           │
                                                     No          Yes
                                                      │           │
                                                      ▼           ▼
                                                   Retry      Return error message
                                                              as assistant response
```

### Retry Backoff

| Retry | Delay |
|-------|-------|
| 1 | immediate |
| 2 | 1 second |
| 3 | 2 seconds |

(Configurable via `retry_base_delay_seconds` and `retry_multiplier`)

### Validation Prompt

```
You are a response quality checker. Analyze the AI response below for these issues:

1. REPETITION: Same phrase/paragraph repeated 3+ times
2. TRUNCATION: Response cuts off mid-sentence or mid-code-block
3. MALFORMED TOOLS: Tool calls with invalid JSON or missing required fields

Respond with ONLY valid JSON:
{"is_valid": true/false, "issue_type": "repetition|truncation|malformed_tools|null", "confidence": 0.0-1.0}

---
Response to validate:
[response content here]
```

## Error Handling

### When All Retries Fail

Return a synthetic assistant message:

```json
{
  "id": "msg_validation_failed",
  "type": "message",
  "role": "assistant",
  "content": [
    {
      "type": "text",
      "text": "⚠️ **Garbage Output Detected**\n\nAfter 3 retries, the model continues to produce invalid output.\n\n**Last issue detected:** Repetition loop\n\n**Options:**\n- Try rephrasing your prompt\n- Use `/rewind` to undo and try again\n- Switch to a different model temporarily\n\n**Raw response saved to:** `~/.cache/garbage-proxy/failed_2026-03-07_143052.json`"
    }
  ],
  "stop_reason": "stop"
}
```

### When Validator Fails (LM Studio Down)

- Log warning
- Pass response through unvalidated (fail-open)
- Don't retry validator failures

### Logging

- Each validation failure logged to `cache/garbage-proxy/logs/`
- Failed responses saved with timestamp to `cache/garbage-proxy/failed/`

## File Structure

```
sampling-proxy/
├── sampling_proxy.py      # Existing - add routing mode + validation hooks
├── config.json            # Extended with new options
├── validator.py           # NEW - validation logic (Anthropic format)
└── cache/                 # NEW - failed response logs
    └── garbage-proxy/
        ├── logs/
        └── failed/
```

## Implementation Tasks

1. **Task #1:** Add anthropic_passthrough backend mode
2. **Task #2:** Implement response validator module
3. **Task #3:** Add validation + retry logic to proxy
4. **Task #4:** Handle streaming response validation
5. **Task #5:** Update config.json with new options

## Dependencies

```
#5 (config) ──► #1 (backend mode)
             ──► #2 (validator)

#1 ──► #3 (validation logic)
#2 ──► #3

#3 ──► #4 (streaming)
```

## Usage

1. Start LM Studio with Qwen 3.5 0.8B or 4B loaded
2. Start proxy: `python sampling_proxy.py`
3. Configure Claude Code to use `http://localhost:8001` as API endpoint
