"""Minimal in-process metrics rendered in Prometheus text format."""

from __future__ import annotations

import threading
from collections import defaultdict


class Metrics:
    """Thread-safe counters and latency totals for one process."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._requests: dict[tuple[str, str, int], int] = defaultdict(int)
        self._latency_sum = 0.0
        self._latency_count = 0

    def observe(self, method: str, path: str, status: int, duration: float) -> None:
        with self._lock:
            self._requests[(method, path, status)] += 1
            self._latency_sum += duration
            self._latency_count += 1

    def render(self) -> str:
        lines = [
            "# HELP http_requests_total Total HTTP requests",
            "# TYPE http_requests_total counter",
        ]
        with self._lock:
            for (method, path, status), count in sorted(self._requests.items()):
                lines.append(
                    f'http_requests_total{{method="{method}",path="{path}",status="{status}"}} {count}'
                )
            lines.extend(
                [
                    "# HELP http_request_duration_seconds_sum Total request time",
                    "# TYPE http_request_duration_seconds_sum counter",
                    f"http_request_duration_seconds_sum {self._latency_sum:.6f}",
                    "# HELP http_request_duration_seconds_count Request count",
                    "# TYPE http_request_duration_seconds_count counter",
                    f"http_request_duration_seconds_count {self._latency_count}",
                ]
            )
        return "\n".join(lines) + "\n"
