"""Agent-side controller that runs the data-trace recorder in a child process.

Architecture
------------
The main training process **never** touches disk I/O for trace recording.
Instead, a dedicated child process (``DataTraceProcess``) drains a
``multiprocessing.Queue`` and forwards every command to an in-process
:class:`~utils.data_trace_recorder.DataTraceRecorder`.

The :class:`DataTraceController` exposed here mirrors the public
``DataTraceRecorder`` API (``begin_episode``, ``record_step``,
``end_episode``, ``close``) so ``simulator_vec_env.py`` can swap one for
the other transparently.  IPC is kept simple — a single
``mp.Queue(maxsize=4096)`` carrying lightweight command tuples.
The ``MonitorPayload`` dataclass is pickle-serialised automatically by
the queue (all fields are numpy arrays or plain Python types).

Usage
-----
.. code-block:: python

    from utils.data_trace_schema import RecordingConfig
    from utils.data_trace_proxy  import DataTraceController

    cfg = RecordingConfig(trace_root="./traces")
    ctrl = DataTraceController(cfg)

    ctrl.begin_episode(env_id=0)
    for step in range(max_steps):
        payload = env.get_monitor_payload()
        ctrl.record_step(env_id=0, payload=payload, step_index=step)
    ctrl.end_episode(env_id=0)

    ctrl.close()
"""
from __future__ import annotations

import atexit
import logging
import multiprocessing as mp
import threading

from .data_trace_schema import RecordingConfig
from .telemetry import MonitorPayload

logger = logging.getLogger(__name__)

# ── Command tags used inside the queue ───────────────────────────────
_CMD_BEGIN = "begin_episode"
_CMD_RECORD = "record_step"
_CMD_END = "end_episode"
_CMD_CLOSE = "close"
_CMD_REGISTER_SPEC = "register_spec"


# =====================================================================
#  Child-process entry point
# =====================================================================

def _recorder_main(config: RecordingConfig, queue: mp.Queue) -> None:
    """Entry point executed inside the recorder child process.

    Instantiates a :class:`DataTraceRecorder`, then enters an infinite
    drain loop that processes commands arriving on *queue* until the
    ``close`` sentinel is received.

    Args:
        config: Recording configuration forwarded from the parent.
        queue: ``multiprocessing.Queue`` through which the parent sends
            ``(command_tag, *args)`` tuples.
    """
    # Import inside the child so the parent never pays the cost of
    # heavy optional deps (pyarrow, etc.) at import time.
    from .data_trace_recorder import DataTraceRecorder  # noqa: F811

    recorder = DataTraceRecorder(config)

    try:
        while True:
            msg = queue.get()  # blocks until a command arrives
            if msg is None:
                # Poison-pill — treat as close
                break

            tag = msg[0]

            if tag == _CMD_BEGIN:
                _, env_id = msg
                recorder.begin_episode(env_id)

            elif tag == _CMD_RECORD:
                _, env_id, payload, step_index = msg
                recorder.record_step(env_id, payload, step_index)

            elif tag == _CMD_END:
                _, env_id = msg
                recorder.end_episode(env_id)

            elif tag == _CMD_REGISTER_SPEC:
                _, env_id, spec_config = msg
                recorder.register_spec(env_id, spec_config)

            elif tag == _CMD_CLOSE:
                break

    except KeyboardInterrupt:
        pass
    except Exception as exc:
        logger.error("DataTrace recorder process crashed: %s", exc, exc_info=True)
    finally:
        recorder.close()


# =====================================================================
#  Parent-side controller
# =====================================================================

class DataTraceController:
    """Proxy that offloads all recording I/O to a dedicated child process.

    The public API is intentionally identical to
    :class:`~utils.data_trace_recorder.DataTraceRecorder` so the two can
    be swapped transparently in ``simulator_vec_env.py``.

    Args:
        config: Recording configuration (channels, output directory, …).
    """

    def __init__(self, config: RecordingConfig) -> None:
        self._cfg = config
        self._closed = False
        self._overflow_warned = False
        self._lock = threading.Lock()

        ctx = mp.get_context("spawn")
        self._queue: mp.Queue = ctx.Queue(maxsize=4096)
        self._process = ctx.Process(
            target=_recorder_main,
            args=(config, self._queue),
            name="DataTraceProcess",
            daemon=True,
        )
        self._process.start()

        atexit.register(self.close)
        logger.info(
            "DataTraceController started (pid=%d) → %s",
            self._process.pid,
            config.trace_root,
        )

    # ── public API (matches DataTraceRecorder) ───────────────────────

    def begin_episode(self, env_id: int) -> None:
        """Signal the start of a new recording episode for *env_id*.

        Args:
            env_id: Integer identifier of the environment instance.
        """
        self._put((_CMD_BEGIN, env_id))

    def register_spec(self, env_id: int, spec_config: dict) -> None:
        """Forward a monitor spec configuration to the recorder process.

        Args:
            env_id: Environment identifier.
            spec_config: The ``MonitorSpec.config`` dict.
        """
        self._put((_CMD_REGISTER_SPEC, env_id, spec_config))

    def record_step(
        self,
        env_id: int,
        payload: MonitorPayload,
        step_index: int,
    ) -> None:
        """Enqueue one simulation step for recording.

        The *payload* is pickled and sent over the queue for the child
        process to persist.

        Args:
            env_id: Environment instance identifier.
            payload: Telemetry bundle from the environment.
            step_index: Zero-based step counter within the current episode.
        """
        self._put((_CMD_RECORD, env_id, payload, step_index))

    def end_episode(self, env_id: int) -> None:
        """Signal the end of the current episode for *env_id*.

        Args:
            env_id: Environment instance identifier.
        """
        self._put((_CMD_END, env_id))

    def close(self) -> None:
        """Shut down the child process and release resources.

        All buffered data is flushed before the process exits.
        Safe to call multiple times.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True

        # Send the close sentinel
        try:
            self._queue.put_nowait((_CMD_CLOSE,))
        except Exception:
            pass

        # Prevent the queue's background feeder thread from blocking at exit
        try:
            self._queue.cancel_join_thread()
        except Exception:
            pass

        # Wait for a clean shutdown — NPZ flush can be I/O heavy
        try:
            self._process.join(timeout=30.0)
        except (KeyboardInterrupt, SystemExit):
            pass

        if self._process.is_alive():
            self._process.terminate()
            try:
                self._process.join(timeout=5.0)
            except (KeyboardInterrupt, SystemExit):
                pass

        if self._process.is_alive():
            self._process.kill()

        logger.info("DataTraceController closed.")

    # ── context-manager support ──────────────────────────────────────

    def __enter__(self) -> DataTraceController:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ── internal ─────────────────────────────────────────────────────

    def _put(self, item: tuple) -> None:
        """Thread-safe, non-blocking enqueue with back-pressure guard.

        If the queue is full the item is dropped silently so the main
        training loop is never stalled by slow disk I/O.

        Args:
            item: Command tuple to send to the child process.
        """
        if self._closed:
            return
        try:
            self._queue.put_nowait(item)
        except Exception:
            # Queue full or broken — drop the sample rather than block the
            # training loop.  A warning is logged once per overflow burst.
            if not self._overflow_warned:
                logger.warning(
                    "DataTrace queue full — dropping samples.  "
                    "Consider increasing flush_interval_steps or reducing channels."
                )
                self._overflow_warned = True
