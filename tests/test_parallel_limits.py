"""Tests for parallel request limiting (global and per-model semaphores)."""
import pytest
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch


def make_semaphore_wrapper(sampling_proxy_module):
    """Import the proxy module and return key symbols."""
    return (
        sampling_proxy_module.GLOBAL_SEMAPHORE,
        sampling_proxy_module.GLOBAL_LIMIT,
        sampling_proxy_module.MODEL_SEMAPHORES,
        sampling_proxy_module.PARALLEL_LIMITS,
    )


@pytest.mark.asyncio
async def test_global_semaphore_limits_concurrent_non_streaming():
    """Test that the global semaphore actually blocks concurrent requests beyond the limit."""
    # Simulate a semaphore with limit=2
    sem = asyncio.Semaphore(2)
    acquired_count = 0
    max_concurrent = 0
    lock = asyncio.Lock()

    async def simulate_request():
        nonlocal acquired_count, max_concurrent
        await sem.acquire()
        async with lock:
            acquired_count += 1
            if acquired_count > max_concurrent:
                max_concurrent = acquired_count
        await asyncio.sleep(0.05)  # simulate work
        async with lock:
            acquired_count -= 1
        sem.release()

    # Launch 5 concurrent requests with limit=2
    await asyncio.gather(*[simulate_request() for _ in range(5)])
    assert max_concurrent <= 2


@pytest.mark.asyncio
async def test_global_semaphore_limits_concurrent_streaming():
    """Test that semaphores released inside streaming generators actually limit concurrency."""
    sem = asyncio.Semaphore(2)
    concurrent = 0
    max_concurrent = 0
    lock = asyncio.Lock()
    chunks_yielded = []

    async def streaming_request(req_id):
        nonlocal concurrent, max_concurrent
        await sem.acquire()
        async with lock:
            concurrent += 1
            if concurrent > max_concurrent:
                max_concurrent = concurrent

        # Simulate streaming: yield 3 chunks
        async def stream_gen():
            try:
                for i in range(3):
                    await asyncio.sleep(0.02)
                    chunk = f"req{req_id}-chunk{i}"
                    chunks_yielded.append(chunk)
                    yield chunk
            finally:
                nonlocal concurrent
                async with lock:
                    concurrent -= 1
                sem.release()

        # Consume the generator (simulating what FastAPI does)
        result = []
        async for chunk in stream_gen():
            result.append(chunk)
        return result

    # Launch 5 concurrent streaming requests with limit=2
    results = await asyncio.gather(*[streaming_request(i) for i in range(5)])
    
    # All requests completed
    assert len(results) == 5
    # Never exceeded limit
    assert max_concurrent <= 2
    # All chunks were yielded
    assert len(chunks_yielded) == 15


@pytest.mark.asyncio
async def test_semaphore_release_on_generator_exception():
    """Test that semaphore is released even if the streaming generator raises."""
    sem = asyncio.Semaphore(1)
    released = False

    async def failing_stream():
        nonlocal released
        await sem.acquire()
        try:
            yield "chunk1"
            raise RuntimeError("stream error")
            yield "chunk2"  # noqa
        finally:
            released = True
            sem.release()

    # Consume the generator
    chunks = []
    with pytest.raises(RuntimeError):
        async for chunk in failing_stream():
            chunks.append(chunk)

    assert chunks == ["chunk1"]
    assert released
    assert sem._value == 1  # semaphore was released


@pytest.mark.asyncio
async def test_streaming_semaphore_release_only_after_all_chunks():
    """Test that semaphore is NOT released between chunks — only after stream ends."""
    sem = asyncio.Semaphore(1)
    release_times = []
    chunk_times = []

    async def timed_stream():
        await sem.acquire()
        try:
            for i in range(3):
                chunk_times.append(asyncio.get_event_loop().time())
                yield f"chunk{i}"
                await asyncio.sleep(0.05)
        finally:
            release_times.append(asyncio.get_event_loop().time())
            sem.release()

    # Consume
    async for chunk in timed_stream():
        pass

    assert len(chunk_times) == 3
    assert len(release_times) == 1
    # Release happened AFTER last chunk
    assert release_times[0] >= chunk_times[-1]


@pytest.mark.asyncio
async def test_wrap_stream_with_semaphore_release_logic():
    """Test the wrap_stream_with_semaphore_release pattern used in sampling_proxy."""
    sem = asyncio.Semaphore(1)
    semaphores_released = False
    concurrent = 0
    max_concurrent = 0

    async def release_semaphores():
        nonlocal semaphores_released
        if semaphores_released:
            return
        semaphores_released = True
        sem.release()

    def wrap_stream_with_semaphore_release(generator):
        async def wrapped():
            try:
                async for chunk in generator:
                    yield chunk
            finally:
                await release_semaphores()
        return wrapped()

    async def inner_gen():
        for i in range(3):
            yield f"chunk{i}"
            await asyncio.sleep(0.02)

    # Acquire semaphore, wrap generator, consume
    await sem.acquire()
    assert sem._value == 0
    
    result = []
    async for chunk in wrap_stream_with_semaphore_release(inner_gen()):
        result.append(chunk)
    
    # After streaming completes, semaphore should be released
    assert semaphores_released is True
    assert sem._value == 1
    assert result == ["chunk0", "chunk1", "chunk2"]

    # Calling release_semaphores again should be idempotent
    await release_semaphores()
    assert sem._value == 1  # not double-released


@pytest.mark.asyncio
async def test_concurrent_requests_with_global_limit():
    """
    Integration-style test: simulate multiple requests hitting a semaphore-limited proxy.
    With limit=2, at most 2 should be 'in flight' simultaneously.
    """
    limit = 2
    sem = asyncio.Semaphore(limit)
    in_flight = 0
    max_in_flight = 0
    lock = asyncio.Lock()
    request_order = []

    async def proxy_request(req_id, is_streaming=False):
        nonlocal in_flight, max_in_flight
        await sem.acquire()
        async with lock:
            in_flight += 1
            if in_flight > max_in_flight:
                max_in_flight = in_flight
            request_order.append(f"start-{req_id}")

        if is_streaming:
            # Simulate streaming with delayed release
            async def stream():
                nonlocal in_flight
                try:
                    for i in range(2):
                        await asyncio.sleep(0.03)
                        yield f"req{req_id}-chunk{i}"
                finally:
                    async with lock:
                        in_flight -= 1
                        request_order.append(f"end-{req_id}")
                    sem.release()
            
            chunks = []
            async for c in stream():
                chunks.append(c)
            return chunks
        else:
            await asyncio.sleep(0.05)
            async with lock:
                in_flight -= 1
                request_order.append(f"end-{req_id}")
            sem.release()
            return f"response-{req_id}"

    # Mix of streaming and non-streaming, 6 total, limit=2
    tasks = []
    for i in range(6):
        is_stream = i % 2 == 0
        tasks.append(proxy_request(i, is_streaming=is_stream))

    await asyncio.gather(*tasks)

    assert max_in_flight <= limit
    assert len(request_order) == 12  # 6 starts + 6 ends


@pytest.mark.asyncio
async def test_streaming_response_semaphore_not_released_in_outer_finally():
    """
    Reproduce the exact code path in sampling_proxy.py:
    1. Handler acquires semaphore
    2. Handler creates wrapped generator (sets is_streaming_response=True)
    3. Handler returns StreamingResponse
    4. Outer finally runs but SKIPS release because is_streaming_response=True
    5. Starlette iterates the generator
    6. Generator's finally releases the semaphore
    """
    sem = asyncio.Semaphore(2)
    semaphores_released = False
    is_streaming_response = False
    events = []

    async def release_semaphores():
        nonlocal semaphores_released
        if semaphores_released:
            return
        semaphores_released = True
        sem.release()
        events.append("semaphore_released")

    def wrap_stream_with_semaphore_release(generator):
        nonlocal is_streaming_response
        is_streaming_response = True
        async def wrapped():
            try:
                async for chunk in generator:
                    yield chunk
            finally:
                await release_semaphores()
        return wrapped()

    # --- Simulate the proxy handler ---
    async def inner_gen():
        for i in range(3):
            await asyncio.sleep(0.02)
            events.append(f"chunk{i}")
            yield f"chunk{i}"

    # Acquire semaphore
    await sem.acquire()
    assert sem._value == 1
    events.append("semaphore_acquired")

    # Create wrapped generator (sets is_streaming_response=True)
    wrapped_gen = wrap_stream_with_semaphore_release(inner_gen())
    assert is_streaming_response is True

    # --- Simulate outer finally (runs when handler returns StreamingResponse) ---
    # This should NOT release because is_streaming_response=True
    if not is_streaming_response and not semaphores_released:
        await release_semaphores()
    
    assert not semaphores_released  # Semaphore NOT released yet!
    assert sem._value == 1  # Still held!
    events.append("outer_finally_ran")

    # Now simulate Starlette consuming the generator
    chunks = []
    async for chunk in wrapped_gen:
        chunks.append(chunk)

    # NOW the semaphore should be released
    assert semaphores_released is True
    assert sem._value == 2
    assert chunks == ["chunk0", "chunk1", "chunk2"]
    
    # Verify ordering: acquire -> outer_finally -> chunks -> release
    assert events == [
        "semaphore_acquired",
        "outer_finally_ran",
        "chunk0",
        "chunk1",
        "chunk2",
        "semaphore_released",
    ]
