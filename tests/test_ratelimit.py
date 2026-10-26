import time

from costrouter.llm.ratelimit import RateLimiter


async def test_limiter_blocks_when_request_budget_is_spent():
    limiter = RateLimiter(rpm=2, tpm=None, window_s=0.3)
    t0 = time.monotonic()
    for _ in range(3):
        await limiter.acquire(10)
    assert time.monotonic() - t0 >= 0.25  # the third call waited for the window to roll


async def test_limiter_counts_tokens():
    limiter = RateLimiter(rpm=None, tpm=100, window_s=0.3)
    t0 = time.monotonic()
    await limiter.acquire(80)
    await limiter.acquire(80)
    assert time.monotonic() - t0 >= 0.25


async def test_oversized_request_still_goes_through():
    limiter = RateLimiter(rpm=None, tpm=100, window_s=10)
    await limiter.acquire(500)  # must not deadlock
