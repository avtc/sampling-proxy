# Throttle/Pause Feature Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:executing-plans to implement this plan task-by-task.

**Goal:** Add configurable pause timers to throttle backend requests, preventing overload with global and per-model cooldowns.

**Architecture:** Semaphore-based throttling using `ThrottleManager` class with `asyncio.Semaphore` and auto-release timer tasks. Two independent timers (start/end) per model, both optional.

**Tech Stack:** Python 3.x, asyncio, FastAPI, httpx

---

### Task 0: Add Throttle Config Schema to load_config()

**Files:**
- Modify: `sampling_proxy.py:48-91` (load_config function)

**Step 1: Add throttle config to default_config dict**

Add after line 88 (after `"parallel_limits": {}`):

```python
        "throttle": {
            "enabled": False,
            "global": {
                "start_pause_seconds": None,
                "end_pause_seconds": 3.0
            },
            "per_model": {}
        }
```

**Step 2: Run to verify no errors**

Run: `python sampling_proxy.py --help`
Expected: No errors, help message displays

**Step 3: Commit**

```bash
git add sampling_proxy.py
git commit -m "feat: add throttle config schema to defaults"
```

---

### Task 1: Create ThrottleManager and ModelThrottle Classes

**Files:**
- Create: `throttle_manager.py` (new file)

**Step 1: Write the failing test**

Create test file `tests/test_throttle_manager.py`:

```python
import pytest
import asyncio
from throttle_manager import ThrottleManager, ModelThrottle

@pytest.mark.asyncio
async def test_model_throttle_initialization():
    """Test ModelThrottle initializes correctly with valid config."""
    throttle = ModelThrottle(start_pause=1.0, end_pause=2.0)
    assert throttle.start_semaphore is not None
    assert throttle.end_semaphore is not None
    assert throttle.start_pause == 1.0
    assert throttle.end_pause == 2.0

@pytest.mark.asyncio
async def test_model_throttle_none_values():
    """Test ModelThrottle with None values (disabled timers)."""
    throttle = ModelThrottle(start_pause=None, end_pause=None)
    assert throttle.start_semaphore is None
    assert throttle.end_semaphore is None

@pytest.mark.asyncio
async def test_throttle_manager_disabled():
    """Test ThrottleManager when disabled."""
    config = {"enabled": False}
    manager = ThrottleManager(config)
    assert not manager.enabled

@pytest.mark.asyncio
async def test_throttle_manager_enabled():
    """Test ThrottleManager when enabled."""
    config = {
        "enabled": True,
        "global": {
            "start_pause_seconds": 0.5,
            "end_pause_seconds": 1.0
        },
        "per_model": {}
    }
    manager = ThrottleManager(config)
    assert manager.enabled
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_throttle_manager.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'throttle_manager'"

**Step 3: Write minimal implementation**

Create `throttle_manager.py`:

```python
import asyncio
from typing import Optional, Dict

class ModelThrottle:
    """Holds throttle state for a single model."""

    def __init__(self, start_pause: Optional[float], end_pause: Optional[float]):
        self.start_pause = start_pause
        self.end_pause = end_pause
        self.start_semaphore = asyncio.Semaphore(1) if start_pause else None
        self.end_semaphore = asyncio.Semaphore(1) if end_pause else None

class ThrottleManager:
    """Manages throttle timers for backend requests."""

    def __init__(self, config: dict):
        self.enabled = config.get("enabled", False)
        self.global_config = config.get("global", {})
        self.per_model_config = config.get("per_model", {})
        self.lock = asyncio.Lock()
        self.model_state: Dict[str, ModelThrottle] = {}

    async def wait_before_send(self, model: str):
        """Wait for start-timer before sending request upstream."""
        if not self.enabled:
            return
        # Implementation in next task

    async def wait_after_send(self, model: str):
        """Wait for end-timer after response completes."""
        if not self.enabled:
            return
        # Implementation in next task
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_throttle_manager.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_throttle_manager.py throttle_manager.py
git commit -m "feat: add ThrottleManager and ModelThrottle classes with tests"
```

---

### Task 2: Implement wait_before_send() Method

**Files:**
- Modify: `throttle_manager.py`

**Step 1: Write the failing test**

Add to `tests/test_throttle_manager.py`:

```python
@pytest.mark.asyncio
async def test_wait_before_send_throttles():
    """Test that wait_before_send actually waits for configured time."""
    config = {
        "enabled": True,
        "global": {"start_pause_seconds": 0.1, "end_pause_seconds": None},
        "per_model": {}
    }
    manager = ThrottleManager(config)

    import time
    start = time.time()
    await manager.wait_before_send("test-model")
    elapsed = time.time() - start

    # First call should not wait (no previous request)
    assert elapsed < 0.05

@pytest.mark.asyncio
async def test_wait_before_send_second_call_waits():
    """Test that second call waits for start-timer."""
    config = {
        "enabled": True,
        "global": {"start_pause_seconds": 0.2, "end_pause_seconds": None},
        "per_model": {}
    }
    manager = ThrottleManager(config)

    import time
    await manager.wait_before_send("test-model")
    start = time.time()
    await manager.wait_before_send("test-model")
    elapsed = time.time() - start

    # Second call should wait approximately 0.2s
    assert 0.15 < elapsed < 0.3
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_throttle_manager.py::test_wait_before_send_second_call_waits -v`
Expected: FAIL (test passes but second call doesn't wait)

**Step 3: Write implementation**

Update `ThrottleManager` class in `throttle_manager.py`:

```python
    def _get_or_create_throttle(self, model: str) -> ModelThrottle:
        """Get or create ModelThrottle for a model."""
        model_lower = model.lower()
        if model_lower not in self.model_state:
            # Get config for this model
            per_model_config = self.per_model_config.get(model, {})
            start_pause = per_model_config.get("start_pause_seconds")
            if start_pause is None:
                start_pause = self.global_config.get("start_pause_seconds")

            end_pause = per_model_config.get("end_pause_seconds")
            if end_pause is None:
                end_pause = self.global_config.get("end_pause_seconds")

            self.model_state[model_lower] = ModelThrottle(start_pause, end_pause)

        return self.model_state[model_lower]

    async def wait_before_send(self, model: str):
        """Wait for start-timer before sending request upstream."""
        if not self.enabled:
            return

        throttle = self._get_or_create_throttle(model)
        if throttle.start_semaphore is None:
            return

        # Acquire semaphore (blocks if timer is active)
        await throttle.start_semaphore.acquire()

        # Schedule auto-release after configured delay
        if throttle.start_pause:
            async def release_after_delay():
                await asyncio.sleep(throttle.start_pause)
                throttle.start_semaphore.release()

            asyncio.create_task(release_after_delay())
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_throttle_manager.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add throttle_manager.py tests/test_throttle_manager.py
git commit -m "feat: implement wait_before_send with semaphore throttling"
```

---

### Task 3: Implement wait_after_send() Method

**Files:**
- Modify: `throttle_manager.py`

**Step 1: Write the failing test**

Add to `tests/test_throttle_manager.py`:

```python
@pytest.mark.asyncio
async def test_wait_after_send_throttles():
    """Test that wait_after_send actually waits for configured time."""
    config = {
        "enabled": True,
        "global": {"start_pause_seconds": None, "end_pause_seconds": 0.1},
        "per_model": {}
    }
    manager = ThrottleManager(config)

    import time
    await manager.wait_after_send("test-model")
    start = time.time()
    await manager.wait_after_send("test-model")
    elapsed = time.time() - start

    # Second call should wait approximately 0.1s
    assert 0.08 < elapsed < 0.15
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_throttle_manager.py::test_wait_after_send_throttles -v`
Expected: FAIL (test doesn't wait)

**Step 3: Write implementation**

Add to `ThrottleManager` class in `throttle_manager.py`:

```python
    async def wait_after_send(self, model: str):
        """Wait for end-timer after response completes."""
        if not self.enabled:
            return

        throttle = self._get_or_create_throttle(model)
        if throttle.end_semaphore is None:
            return

        # Acquire semaphore (blocks if timer is active)
        await throttle.end_semaphore.acquire()

        # Schedule auto-release after configured delay
        if throttle.end_pause:
            async def release_after_delay():
                await asyncio.sleep(throttle.end_pause)
                throttle.end_semaphore.release()

            asyncio.create_task(release_after_delay())
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_throttle_manager.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add throttle_manager.py tests/test_throttle_manager.py
git commit -m "feat: implement wait_after_send with semaphore throttling"
```

---

### Task 4: Add Config Validation

**Files:**
- Modify: `throttle_manager.py`

**Step 1: Write the failing test**

Add to `tests/test_throttle_manager.py`:

```python
def test_throttle_config_validation_negative_values():
    """Test that negative pause values raise exception."""
    config = {
        "enabled": True,
        "global": {"start_pause_seconds": -1.0, "end_pause_seconds": 3.0},
        "per_model": {}
    }
    with pytest.raises(ValueError, match="start_pause_seconds must be >= 0"):
        ThrottleManager(config)

def test_throttle_config_validation_invalid_types():
    """Test that non-numeric values raise exception."""
    config = {
        "enabled": True,
        "global": {"start_pause_seconds": "invalid", "end_pause_seconds": 3.0},
        "per_model": {}
    }
    with pytest.raises(ValueError, match="start_pause_seconds must be a number or null"):
        ThrottleManager(config)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_throttle_manager.py -k validation -v`
Expected: FAIL (no validation yet)

**Step 3: Write implementation**

Add validation method to `ThrottleManager` class:

```python
    def __init__(self, config: dict):
        if config.get("enabled", False):
            self._validate_config(config)

        self.enabled = config.get("enabled", False)
        self.global_config = config.get("global", {})
        self.per_model_config = config.get("per_model", {})
        self.lock = asyncio.Lock()
        self.model_state: Dict[str, ModelThrottle] = {}

    def _validate_config(self, config: dict):
        """Validate throttle configuration."""
        global_config = config.get("global", {})
        self._validate_pause_config(global_config, "global")

        per_model = config.get("per_model", {})
        for model_name, model_config in per_model.items():
            self._validate_pause_config(model_config, f"per_model.{model_name}")

    def _validate_pause_config(self, pause_config: dict, location: str):
        """Validate a single pause config (global or per-model)."""
        for key in ["start_pause_seconds", "end_pause_seconds"]:
            value = pause_config.get(key)
            if value is not None:
                if not isinstance(value, (int, float)):
                    raise ValueError(f"{location}.{key} must be a number or null, got {type(value).__name__}")
                if value < 0:
                    raise ValueError(f"{location}.{key} must be >= 0, got {value}")
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_throttle_manager.py -k validation -v`
Expected: PASS

**Step 5: Commit**

```bash
git add throttle_manager.py tests/test_throttle_manager.py
git commit -m "feat: add throttle config validation with fail-fast on startup"
```

---

### Task 5: Add Logging to ThrottleManager

**Files:**
- Modify: `throttle_manager.py`, `sampling_proxy.py`

**Step 1: Update ThrottleManager to accept logging parameters**

Modify `throttle_manager.py`:

```python
class ThrottleManager:
    """Manages throttle timers for backend requests."""

    def __init__(self, config: dict, enable_debug_logs: bool = False, request_id: int = 0):
        if config.get("enabled", False):
            self._validate_config(config)

        self.enabled = config.get("enabled", False)
        self.global_config = config.get("global", {})
        self.per_model_config = config.get("per_model", {})
        self.lock = asyncio.Lock()
        self.model_state: Dict[str, ModelThrottle] = {}
        self.enable_debug_logs = enable_debug_logs
        self.request_id = request_id

    def _log_debug(self, message: str):
        """Log debug message if enabled."""
        if self.enable_debug_logs:
            print(f"[DEBUG][R:{self.request_id}] Throttle: {message}")

    def _log_info(self, message: str):
        """Log info message."""
        print(f"[INFO][R:{self.request_id}] Throttle: {message}")
```

**Step 2: Add logging to wait methods**

Update `wait_before_send` and `wait_after_send`:

```python
    async def wait_before_send(self, model: str):
        """Wait for start-timer before sending request upstream."""
        if not self.enabled:
            return

        throttle = self._get_or_create_throttle(model)
        if throttle.start_semaphore is None:
            self._log_debug(f"start-timer disabled for {model}")
            return

        if throttle.start_pause:
            self._log_info(f"waiting {throttle.start_pause}s before sending to {model} (start-timer)")

        # Acquire semaphore (blocks if timer is active)
        await throttle.start_semaphore.acquire()

        # Schedule auto-release after configured delay
        if throttle.start_pause:
            async def release_after_delay():
                await asyncio.sleep(throttle.start_pause)
                throttle.start_semaphore.release()
                self._log_debug(f"{model} start-timer released after {throttle.start_pause}s")

            asyncio.create_task(release_after_delay())

    async def wait_after_send(self, model: str):
        """Wait for end-timer after response completes."""
        if not self.enabled:
            return

        throttle = self._get_or_create_throttle(model)
        if throttle.end_semaphore is None:
            self._log_debug(f"end-timer disabled for {model}")
            return

        if throttle.end_pause:
            self._log_info(f"waiting {throttle.end_pause}s after response from {model} (end-timer)")

        # Acquire semaphore (blocks if timer is active)
        await throttle.end_semaphore.acquire()

        # Schedule auto-release after configured delay
        if throttle.end_pause:
            async def release_after_delay():
                await asyncio.sleep(throttle.end_pause)
                throttle.end_semaphore.release()
                self._log_debug(f"{model} end-timer released after {throttle.end_pause}s")

            asyncio.create_task(release_after_delay())
```

**Step 3: Run tests to verify**

Run: `pytest tests/test_throttle_manager.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add throttle_manager.py
git commit -m "feat: add logging to ThrottleManager"
```

---

### Task 6: Integrate ThrottleManager into sampling_proxy.py - Config Loading

**Files:**
- Modify: `sampling_proxy.py`

**Step 1: Add THROTTLE_CONFIG global variable**

Add after line 206 (after `VALIDATION_CONFIG`):

```python
THROTTLE_CONFIG = {"enabled": False}
```

**Step 2: Load throttle config in main block**

Add after line 2402 (after `VALIDATION_CONFIG = CONFIG.get("validation", ...)`):

```python
    THROTTLE_CONFIG = CONFIG.get("throttle", {"enabled": False})
    print(f"Throttle config loaded: enabled={THROTTLE_CONFIG.get('enabled')}")
```

**Step 3: Run to verify**

Run: `python sampling_proxy.py --help`
Expected: No errors

**Step 4: Commit**

```bash
git add sampling_proxy.py
git commit -m "feat: load throttle config from config.json"
```

---

### Task 7: Initialize ThrottleManager in Main Block

**Files:**
- Modify: `sampling_proxy.py`

**Step 1: Import ThrottleManager**

Add after line 45 (after validator imports):

```python
# Import throttle manager for request throttling
from throttle_manager import ThrottleManager
```

**Step 2: Initialize global throttle_manager variable**

Add after line 214 (after `GLOBAL_SEMAPHORE = None`):

```python
# Throttle manager for request pacing
throttle_manager = None
```

**Step 3: Initialize ThrottleManager in main block**

Add after line 2466 (after parallel limits initialization):

```python
    # Initialize throttle manager
    throttle_manager = None
    if THROTTLE_CONFIG.get("enabled"):
        try:
            throttle_manager = ThrottleManager(THROTTLE_CONFIG, ENABLE_DEBUG_LOGS, 0)
            print(f"Throttle manager initialized: enabled={throttle_manager.enabled}")
        except ValueError as e:
            print(f"ERROR: Invalid throttle configuration: {e}")
            raise
```

**Step 4: Run to verify**

Run: `python sampling_proxy.py --help`
Expected: No errors

**Step 5: Commit**

```bash
git add sampling_proxy.py
git commit -m "feat: initialize ThrottleManager in main block"
```

---

### Task 8: Add Throttle Calls to Anthropic Passthrough Path

**Files:**
- Modify: `sampling_proxy.py:679-684`

**Step 1: Add throttle call before upstream send**

Find the line `upstream_response = await client.request(...)` around line 679.

Add before it:

```python
                # Apply throttle before sending upstream
                if throttle_manager:
                    model_for_throttle = passthrough_headers.get("anthropic-model", "global")
                    await throttle_manager.wait_before_send(model_for_throttle)
```

**Step 2: Add throttle call after response**

After the upstream request (after line 689), add:

```python
                # Apply throttle after response completes
                if throttle_manager:
                    model_for_throttle = passthrough_headers.get("anthropic-model", "global")
                    await throttle_manager.wait_after_send(model_for_throttle)
```

**Step 3: Run to verify**

Run: `python sampling_proxy.py`
Expected: Server starts without errors

**Step 4: Commit**

```bash
git add sampling_proxy.py
git commit -m "feat: integrate throttle into Anthropic passthrough path"
```

---

### Task 9: Add Throttle Calls to Streaming Validation Retry Path

**Files:**
- Modify: `sampling_proxy.py:1420-1437` (first retry location)

**Step 1: Add throttle call before retry**

Find the retry block around line 1422. Add before `retry_request_obj = client.build_request(...)`:

```python
                                # Apply throttle before retry
                                if throttle_manager:
                                    await throttle_manager.wait_before_send(model_name)
```

**Step 2: Add throttle call after retry response**

After `retry_response = await client.send(...)` around line 1429, add:

```python
                                if throttle_manager:
                                    await throttle_manager.wait_after_send(model_name)
```

**Step 3: Repeat for second retry location (around line 1482)**

Find the second retry block and add the same throttle calls.

**Step 4: Run to verify**

Run: `python sampling_proxy.py`
Expected: Server starts without errors

**Step 5: Commit**

```bash
git add sampling_proxy.py
git commit -m "feat: integrate throttle into streaming validation retry paths"
```

---

### Task 10: Extract Model Name Helper Function

**Files:**
- Modify: `sampling_proxy.py`

**Step 1: Create helper function to extract model name**

Add after line 246 (after `get_global_semaphore()` function):

```python
def extract_model_for_throttle(request_data: dict) -> str:
    """Extract model name from request data for throttle lookup."""
    model = request_data.get("model")
    if model:
        return model
    return "global"
```

**Step 2: Update throttle calls to use helper**

Replace existing throttle calls with:

```python
model_for_throttle = extract_model_for_throttle(incoming_json_body)
```

**Step 3: Run tests to verify**

Run: `pytest tests/ -v`
Expected: All tests pass

**Step 4: Commit**

```bash
git add sampling_proxy.py
git commit -m "refactor: add extract_model_for_throttle helper function"
```

---

### Task 11: Update Config Sample Files

**Files:**
- Modify: `config_sample.json`, `config_zai_sample.json`

**Step 1: Add throttle section to config_sample.json**

Add after `"parallel_limits"` section:

```json
  "throttle": {
    "enabled": false,
    "global": {
      "start_pause_seconds": null,
      "end_pause_seconds": 3.0
    },
    "per_model": {
      "example-model": {
        "start_pause_seconds": 1.0,
        "end_pause_seconds": 5.0
      }
    }
  }
```

**Step 2: Add same section to config_zai_sample.json**

**Step 3: Verify JSON is valid**

Run: `python -m json.tool config_sample.json > /dev/null && echo "Valid JSON"`
Expected: "Valid JSON"

**Step 4: Commit**

```bash
git add config_sample.json config_zai_sample.json
git commit -m "docs: add throttle config to sample files"
```

---

### Task 12: Update README.md Documentation

**Files:**
- Modify: `README.md`

**Step 1: Add throttle section to README**

Add after "Parallel Request Limits" section (around line 115):

```markdown
## Request Throttling

Add cooldown delays between requests to prevent backend overload:

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
      }
    }
  }
}
```

**Timers:**
- **start_pause_seconds:** Cooldown before sending request upstream
- **end_pause_seconds:** Cooldown after response completes
- `null` disables the timer

Set `null` for per-model to use global values, or both `null` to disable throttling for that model.
```

**Step 2: Update features list**

Add to features list (around line 13):

```markdown
- **Request Throttling:** Configurable cooldown delays between requests
```

**Step 3: Verify README renders correctly**

**Step 4: Commit**

```bash
git add README.md
git commit -m "docs: add throttle feature documentation to README"
```

---

### Task 13: Integration Test - End-to-End

**Files:**
- Create: `tests/test_throttle_integration.py`

**Step 1: Write integration test**

```python
import pytest
import asyncio
import json
from httpx import AsyncClient, ASGITransport
from sampling_proxy import app, load_config, CONFIG

@pytest.mark.asyncio
async def test_throttle_integration():
    """Test throttle feature end-to-end."""
    # Load test config with throttle enabled
    test_config = {
        "server": {
            "target_base_url": "http://127.0.0.1:8000",
            "sampling_proxy_host": "127.0.0.1",
            "sampling_proxy_port": 8002,
            "connect_timeout_seconds": 5.0,
            "timeout_seconds": 10.0,
            "supports_openai": True,
            "supports_anthropic": False
        },
        "throttle": {
            "enabled": True,
            "global": {
                "start_pause_seconds": 0.1,
                "end_pause_seconds": 0.1
            },
            "per_model": {}
        },
        "parallel_limits": {},
        "logging": {"enable_debug_logs": False}
    }

    # This test requires a mock backend server
    # For now, just verify config loads without error
    from throttle_manager import ThrottleManager
    manager = ThrottleManager(test_config["throttle"])
    assert manager.enabled

    import time
    start = time.time()
    await manager.wait_before_send("test-model")
    await manager.wait_after_send("test-model")
    elapsed = time.time() - start

    # Should have waited for end-timer
    assert elapsed >= 0.1
```

**Step 2: Run integration test**

Run: `pytest tests/test_throttle_integration.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/test_throttle_integration.py
git commit -m "test: add throttle integration test"
```

---

### Task 14: Manual Testing and Verification

**Files:**
- Manual testing with actual backend

**Step 1: Create test config**

Create `config_throttle_test.json`:

```json
{
  "server": {
    "target_base_url": "http://127.0.0.1:8000",
    "sampling_proxy_host": "127.0.0.1",
    "sampling_proxy_port": 8002,
    "connect_timeout_seconds": 5.0,
    "timeout_seconds": 1200.0,
    "supports_openai": true,
    "supports_anthropic": true
  },
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
  },
  "parallel_limits": {},
  "logging": {
    "enable_debug_logs": true,
    "enable_override_logs": false,
    "enable_validation_logs": false
  }
}
```

**Step 2: Start proxy with test config**

Run: `python sampling_proxy.py -c config_throttle_test.json`
Expected: Server starts, shows "Throttle manager initialized"

**Step 3: Send test requests**

```bash
curl -X POST http://127.0.0.1:8002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "test-model", "messages": [{"role": "user", "content": "Hello"}]}'
```

**Step 4: Verify logs show throttle messages**

Expected logs:
```
[INFO][R:1] Throttle: waiting 2.0s before sending to test-model (start-timer)
[INFO][R:1] Throttle: waiting 3.0s after response from test-model (end-timer)
```

**Step 5: Test with disabled throttle**

Modify config to `"enabled": false`, verify no throttle logs appear.

**Step 6: Commit test config**

```bash
git add config_throttle_test.json
git commit -m "test: add throttle test config for manual verification"
```

---

### Task 15: Final Cleanup and Documentation

**Files:**
- Various

**Step 1: Run all tests**

Run: `pytest tests/ -v`
Expected: All tests pass

**Step 2: Check for TODO comments**

Run: `grep -r "TODO" sampling_proxy.py throttle_manager.py`
Expected: No TODOs remaining

**Step 3: Verify config validation**

Test with invalid config (negative values):

```bash
cat > config_invalid.json << 'EOF'
{
  "throttle": {
    "enabled": true,
    "global": {"start_pause_seconds": -1.0}
  }
}
EOF
python sampling_proxy.py -c config_invalid.json
```

Expected: ERROR message and exit

**Step 4: Clean up test files**

```bash
rm config_invalid.json
```

**Step 5: Final commit**

```bash
git add -A
git commit -m "feat: complete throttle/pause feature implementation

- Add ThrottleManager and ModelThrottle classes
- Implement start and end pause timers
- Add per-model and global configuration
- Integrate with existing request paths
- Add comprehensive tests and documentation

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Summary

This plan implements the throttle/pause feature in 15 bite-sized tasks:
1. Config schema additions
2. Core ThrottleManager and ModelThrottle classes
3. Semaphore-based throttling implementation
4. Config validation with fail-fast
5. Logging integration
6. Integration into existing request paths
7. Documentation and testing

Each task follows TDD: write test → verify fail → implement → verify pass → commit.
