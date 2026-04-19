# Throttle/Pause Feature Design

**Date:** 2026-04-17
**Status:** Approved
**Author:** Claude (with user collaboration)

## Overview

Add configurable pause timers to throttle requests to the backend, preventing overload. Two independent timers per model:
- **Start-timer:** Cooldown before sending request upstream
- **End-timer:** Cooldown after response completes

Both timers are optional and can be configured globally or per-model.

## Architecture

### Component Diagram

```
Request → parallel_limits (existing) → throttle_manager (new) → upstream client
```

### Key Components

1. **ThrottleManager class** - Manages per-model and global throttle timers
2. **ModelThrottle class** - Holds state for a single model's timers (start/end semaphores, last timestamps)
3. **Config schema** - New `throttle` section in config.json

### Data Flow

1. Request arrives at `proxy_target_requests()`
2. Passes through existing `parallel_limits` check
3. **NEW:** Passes through `throttle_manager.wait_before_send(model)`
4. Request sent upstream
5. **NEW:** `throttle_manager.record_request_start(model)` called
6. Response completes
7. **NEW:** `throttle_manager.record_request_end(model)` called

The throttle manager uses `asyncio.Semaphore` with `asyncio.create_task()` to auto-release semaphores after the configured delay.

## Configuration Schema

```json
{
  "throttle": {
    "enabled": true,
    "global": {
      "start_pause_seconds": 0.5,
      "end_pause_seconds": 3.0
    },
    "per_model": {
      "GLM-5-turbo": {
        "start_pause_seconds": 1.0,
        "end_pause_seconds": 5.0
      },
      "GLM-4.7": {
        "start_pause_seconds": null,
        "end_pause_seconds": null
      }
    }
  }
}
```

### Config Rules

- `enabled`: Master switch (default: `false` for backward compatibility)
- `start_pause_seconds`: Cooldown before sending request upstream
- `end_pause_seconds`: Cooldown after response completes
- `null` values disable that specific timer
- Per-model settings override global for that model
- Both timers optional - if both `null`, model is unthrottled

### Default Values

When `enabled: true`:
- `start_pause_seconds`: `null` (disabled)
- `end_pause_seconds`: `3.0`

### Validation

- Invalid config (negative values, wrong types) throws exception on startup (fail-fast)
- Missing config section → feature disabled

## Implementation

### ThrottleManager Class

```python
class ThrottleManager:
    def __init__(self, config: dict):
        self.enabled = config.get("enabled", False)
        self.global_config = config.get("global", {})
        self.per_model_config = config.get("per_model", {})
        self.lock = asyncio.Lock()

        # State tracking
        self.model_state = {}  # {model_name: ModelThrottle}

    async def wait_before_send(self, model: str):
        """Call before sending request upstream."""
        if not self.enabled:
            return
        # Acquire start-timer semaphore (if configured)

    async def wait_after_send(self, model: str):
        """Call after response completes."""
        if not self.enabled:
            return
        # Acquire end-timer semaphore (if configured)
```

### ModelThrottle Class

```python
class ModelThrottle:
    def __init__(self, start_pause: Optional[float], end_pause: Optional[float]):
        self.start_semaphore = asyncio.Semaphore(1) if start_pause else None
        self.end_semaphore = asyncio.Semaphore(1) if end_pause else None
        self.start_pause = start_pause
        self.end_pause = end_pause
```

### Key Implementation Details

- `ModelThrottle` holds two `asyncio.Semaphore` objects (one per timer type)
- `asyncio.create_task()` schedules auto-release after configured delay
- Thread-safe using `asyncio.Lock()` for state updates
- Lazy initialization - creates `ModelThrottle` on first use per model
- Model name matching is case-insensitive (like `parallel_limits`)

### Semaphore Lifecycle

1. Request arrives → `acquire()` semaphore (blocks if timer active)
2. Timer task created → releases semaphore after N seconds
3. Request proceeds
4. On next request → repeat

## Integration Points

### Modifications to sampling_proxy.py

**1. Load throttle config (after line 200):**
```python
THROTTLE_CONFIG = {}
```

**2. Initialize ThrottleManager (in main block):**
```python
throttle_manager = ThrottleManager(THROTTLE_CONFIG) if THROTTLE_CONFIG.get("enabled") else None
```

**3. Add wait call before upstream send (around line 680, 1422, 1482):**
```python
# Before client.request() or client.send()
if throttle_manager:
    await throttle_manager.wait_before_send(model_name)
```

**4. Record request end (after response.aclose(), around line 1391, 1496):**
```python
# After closing response
if throttle_manager:
    await throttle_manager.wait_after_send(model_name)
```

**5. Model name extraction:**
- Parse from request body for `/chat/completions` and `/messages`
- Use "global" if model not found

### Integration Locations

- Anthropic passthrough streaming (line ~680)
- Anthropic streaming with validation retry (line ~1422, 1482)
- OpenAI streaming paths (similar locations)

## Error Handling

### Error Scenarios

1. **Invalid config values:** Throw exception on startup (fail-fast)
2. **Timer task failure:** Wrap timer tasks in try/except, log errors, ensure semaphore gets released
3. **Model name not found:** Fall back to global throttle settings, log debug message
4. **Disabled feature:** All calls become no-ops, zero performance overhead
5. **Request cancelled while waiting:** Semaphore released via `try/finally`

### Edge Cases

- Multiple requests to same model arrive simultaneously → queue on semaphore
- Very short pause values (<0.1s) → still honored, but may not be precise
- No timeout on semaphore acquisition → requests wait as long as needed

## Logging & Observability

### Log Messages

```
[INFO][R:123] Throttle: waiting 2.3s before sending to GLM-5-turbo (start-timer)
[INFO][R:123] Throttle: waiting 3.0s after response from GLM-5-turbo (end-timer)
[DEBUG] Throttle: GLM-5-turbo start-timer released after 2.3s
[DEBUG] Throttle: Created throttle state for new model: GLM-5-turbo
```

### Metrics to Track

- Number of requests throttled per model
- Average wait time per timer type
- Current active timers per model

### Config Validation Logging

```
[ERROR] Throttle config error: start_pause_seconds must be >= 0, got -1
```

Logs use existing `log_info()` and `ENABLE_DEBUG_LOGS` pattern.

## Testing Considerations

### Unit Tests for ThrottleManager

- Semaphore acquisition/release timing
- Config parsing and validation
- Concurrent request handling
- Model name fallback to global

### Integration Tests

- End-to-end request flow with throttle enabled
- Per-model override behavior
- Interaction with parallel_limits
- Disabled feature (no overhead)

### Manual Testing Scenarios

- Set short pause (0.5s), verify timing with logs
- Send concurrent requests, verify queuing
- Test with models that have no throttle config
- Test error responses still trigger end-timer

### Test Config Example

```json
{
  "throttle": {
    "enabled": true,
    "global": {
      "start_pause_seconds": 0.5,
      "end_pause_seconds": 1.0
    },
    "per_model": {
      "test-model": {
        "start_pause_seconds": 2.0,
        "end_pause_seconds": 3.0
      }
    }
  }
}
```

## Summary

This design adds flexible, per-model throttling to sampling-proxy using semaphore-based timing. The feature:
- Is fully backward compatible (disabled by default)
- Supports both global and per-model configuration
- Provides two independent timers (start and end)
- Integrates cleanly with existing parallel_limits
- Uses async-native patterns for minimal overhead
