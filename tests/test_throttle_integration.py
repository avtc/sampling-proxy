import pytest
import asyncio
import time
from throttle_manager import ThrottleManager


@pytest.mark.asyncio
async def test_throttle_integration():
    """Test throttle feature end-to-end."""
    test_config = {
        "enabled": True,
        "global": {
            "start_pause_seconds": 0.1,
            "end_pause_seconds": 0.1
        },
        "per_model": {}
    }

    manager = ThrottleManager(test_config)
    assert manager.enabled

    # First request should go through immediately
    start = time.time()
    await manager.wait_before_send("test-model")
    await manager.wait_after_send("test-model")
    elapsed = time.time() - start

    # First call should not wait (no previous request)
    assert elapsed < 0.05

    # Second request should wait for both timers
    start = time.time()
    await manager.wait_before_send("test-model")
    elapsed = time.time() - start
    assert elapsed >= 0.08  # Should wait for start-timer

    await manager.wait_after_send("test-model")


@pytest.mark.asyncio
async def test_throttle_per_model_override():
    """Test that per-model config overrides global."""
    config = {
        "enabled": True,
        "global": {
            "start_pause_seconds": 0.1,
            "end_pause_seconds": 0.1
        },
        "per_model": {
            "fast-model": {
                "start_pause_seconds": None,
                "end_pause_seconds": None
            }
        }
    }

    manager = ThrottleManager(config)

    # fast-model should have no throttling (both timers disabled)
    start = time.time()
    await manager.wait_before_send("fast-model")
    await manager.wait_after_send("fast-model")
    start = time.time()
    await manager.wait_before_send("fast-model")
    await manager.wait_after_send("fast-model")
    elapsed = time.time() - start
    assert elapsed < 0.05  # No wait


@pytest.mark.asyncio
async def test_throttle_disabled():
    """Test that disabled throttle has zero overhead."""
    config = {"enabled": False}
    manager = ThrottleManager(config)

    start = time.time()
    for _ in range(100):
        await manager.wait_before_send("any-model")
        await manager.wait_after_send("any-model")
    elapsed = time.time() - start

    # 100 no-op calls should complete very quickly
    assert elapsed < 0.1


@pytest.mark.asyncio
async def test_throttle_concurrent_requests():
    """Test that throttle handles concurrent requests correctly."""
    config = {
        "enabled": True,
        "global": {
            "start_pause_seconds": 0.1,
            "end_pause_seconds": None
        },
        "per_model": {}
    }
    manager = ThrottleManager(config)

    # First request acquires the semaphore
    await manager.wait_before_send("concurrent-model")

    # Second request should wait
    start = time.time()

    async def second_request():
        await manager.wait_before_send("concurrent-model")

    task = asyncio.create_task(second_request())
    await asyncio.sleep(0.15)  # Wait for timer to release
    await task
    elapsed = time.time() - start

    # Should have waited approximately 0.1s
    assert 0.08 < elapsed < 0.2
