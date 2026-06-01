"""Server-Sent Events helper: run a synchronous, callback-driven workload in
a worker thread and stream `progress`/`result`/`error` frames to the client.

Why threading + queue rather than an async generator: every service we wrap
makes blocking SDK calls (model serving, Vector Search, SQL). Those calls
must run off the event loop. A daemon thread + queue.Queue gives us a clean
producer/consumer split — the worker fires `progress_callback(pct, msg)`
freely; the generator pulls frames off the queue and times out every ~20s
to emit `: keepalive\\n\\n` comments so the Databricks Apps proxy doesn't
close the connection during slow phases (its upstream-silence timeout is
~60s)."""
from __future__ import annotations

import json
import logging
import queue
import threading
from typing import Any, Callable, Iterator

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, str], None]
KEEPALIVE_TIMEOUT_S = 20


def _sse_event(event: str, data: Any) -> str:
    payload = json.dumps(data, default=str)
    return f"event: {event}\ndata: {payload}\n\n"


def stream_with_progress(work_fn: Callable[[ProgressCallback], Any]) -> Iterator[str]:
    """Run `work_fn(progress_callback) -> result` in a daemon thread and
    yield SSE frames. `work_fn` must return a JSON-serializable result, or
    raise — the generator emits one of:

      event: progress    data: {"pct": int, "msg": str}
      event: result      data: <whatever work_fn returned>
      event: error       data: {"message": str}

    The generator returns after the terminal `result` or `error` frame."""
    q: queue.Queue = queue.Queue()

    def on_progress(pct: int, msg: str) -> None:
        try:
            q.put_nowait(("progress", {"pct": int(pct), "msg": str(msg)}))
        except Exception:
            logger.exception("Failed to enqueue progress event")

    def worker() -> None:
        try:
            result = work_fn(on_progress)
            q.put(("result", result))
        except Exception as e:
            logger.exception("SSE worker failed")
            q.put(("error", {"message": str(e)}))

    threading.Thread(target=worker, daemon=True).start()

    while True:
        try:
            kind, data = q.get(timeout=KEEPALIVE_TIMEOUT_S)
        except queue.Empty:
            yield ": keepalive\n\n"
            continue
        yield _sse_event(kind, data)
        if kind in ("result", "error"):
            return
