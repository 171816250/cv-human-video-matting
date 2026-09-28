"""Cancellation tokens scoped to the submitting browser session."""

from __future__ import annotations

from contextlib import contextmanager
from threading import Event, Lock


class BatchJobs:
    def __init__(self):
        self._lock = Lock()
        self._active: dict[str, Event] = {}

    @contextmanager
    def start(self, session_id: str):
        with self._lock:
            if session_id in self._active:
                raise ValueError("当前会话已有正在运行的任务，请等待完成或先停止。")
            token = Event()
            self._active[session_id] = token
        try:
            yield token
        finally:
            with self._lock:
                if self._active.get(session_id) is token:
                    del self._active[session_id]

    def cancel(self, session_id: str) -> bool:
        with self._lock:
            token = self._active.get(session_id)
            if token is None:
                return False
            token.set()
            return True
