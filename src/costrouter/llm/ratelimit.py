"""Client-side sliding-window limiter for requests and tokens per minute, so long
collection runs stay under free-tier limits instead of living on 429 retries."""

from __future__ import annotations

import asyncio
import time
from collections import deque


class RateLimiter:
    def __init__(self, rpm: int | None, tpm: int | None, window_s: float = 60.0):
        self.rpm = rpm
        self.tpm = tpm
        self.window_s = window_s
        self._events: deque[list[float]] = deque()  # [timestamp, tokens]
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int) -> list[float]:
        """Wait for capacity, then reserve it. Returns the reservation so the caller can
        replace the estimate with the actual token count once the response arrives."""
        async with self._lock:
            while True:
                now = time.monotonic()
                while self._events and now - self._events[0][0] >= self.window_s:
                    self._events.popleft()
                used = sum(tokens for _, tokens in self._events)
                rpm_ok = self.rpm is None or len(self._events) < self.rpm
                # A single request larger than the whole budget still goes through alone.
                tpm_ok = self.tpm is None or not self._events or used + tokens <= self.tpm
                if rpm_ok and tpm_ok:
                    event = [now, float(tokens)]
                    self._events.append(event)
                    return event
                await asyncio.sleep(self.window_s - (now - self._events[0][0]) + 0.05)
