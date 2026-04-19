import pytest
import asyncio
import time
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

@pytest.mark.asyncio
async def test_wait_before_send_throttles():
    """Test that wait_before_send actually waits for configured time."""
    config = {
        "enabled": True,
        "global": {"start_pause_seconds": 0.1, "end_pause_seconds": None},
        "per_model": {}
    }
    manager = ThrottleManager(config)

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

    await manager.wait_before_send("test-model")
    start = time.time()
    await manager.wait_before_send("test-model")
    elapsed = time.time() - start

    # Second call should wait approximately 0.2s
    assert 0.15 < elapsed < 0.3

@pytest.mark.asyncio
async def test_wait_after_send_throttles():
    """Test that wait_after_send actually waits for configured time."""
    config = {
        "enabled": True,
        "global": {"start_pause_seconds": None, "end_pause_seconds": 0.1},
        "per_model": {}
    }
    manager = ThrottleManager(config)

    await manager.wait_after_send("test-model")
    start = time.time()
    await manager.wait_after_send("test-model")
    elapsed = time.time() - start

    # Second call should wait approximately 0.1s
    assert 0.08 < elapsed < 0.15


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


def test_throttle_config_validation_per_model():
    """Test that invalid per_model values raise exception."""
    config = {
        "enabled": True,
        "global": {"start_pause_seconds": None, "end_pause_seconds": 3.0},
        "per_model": {
            "test-model": {"start_pause_seconds": -1.0}
        }
    }
    with pytest.raises(ValueError, match="per_model.test-model.start_pause_seconds must be >= 0"):
        ThrottleManager(config)
