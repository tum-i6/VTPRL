"""Agent-side controller that manages the web-monitor subprocess.

Drop-in alternative to :class:`TaskMonitorController` that launches
:func:`run_task_monitor_web_process` instead of the Qt monitor process.
The public API (``register_environment``, ``update_environment``,
``remove_environment``, ``close``) is identical so callers can swap
implementations transparently.
"""
from __future__ import annotations

import atexit
import multiprocessing as mp
import threading
import time
from dataclasses import asdict
from typing import TYPE_CHECKING, Dict, Optional

_PIPE_SEND_INTERVAL = 0.10  # seconds between pipe-based updates (10 Hz)

import numpy as np

if TYPE_CHECKING:
    from .task_monitor import MonitorSpec

from .task_monitor_ipc import (
    Command,
    MonitorMessage,
    SharedTelemetryDescriptor,
    SharedTelemetryWriter,
    create_command_pipe,
    encode_payload,
)
from .task_monitor_web_process import run_task_monitor_web_process

__all__ = ["TaskMonitorWebController"]


def _compact_inline_payload(data) -> Optional[Dict[str, object]]:
    """Build a small fallback payload safe for control-pipe transport.

    Shared memory remains the primary telemetry channel.  This lightweight
    backup ensures server can still update when SHM reads are delayed.

    Parameters
    ----------
    data:
        Full telemetry dictionary from the environment step.

    Returns
    -------
    dict or None
        Compact dictionary with scalar fields and truncated arrays, or
        ``None`` if the input is not a dict.
    """
    if not isinstance(data, dict):
        return None

    def _to_small_list(value, max_len: int = 16):
        """Convert *value* to a list of at most *max_len* floats.

        Parameters
        ----------
        value:
            Input value — ``None``, scalar, list, tuple, or NumPy array.
        max_len : int
            Maximum number of elements to keep.

        Returns
        -------
        list[float] or None
            Truncated float list, empty list for empty arrays, or
            ``None`` when *value* is ``None`` or unconvertible.
        """
        if value is None:
            return None
        try:
            arr = np.asarray(value).flatten()
            if arr.size == 0:
                return []
            return arr[:max_len].astype(float).tolist()
        except Exception:
            if isinstance(value, (list, tuple)):
                return list(value)[:max_len]
            return None

    inline: Dict[str, object] = {}
    for key in (
        "robot_position",
        "robot_yaw",
        "robot_velocity",
        "target_delta",
        "target_position",
        "reward",
        "success",
        "collision",
    ):
        if key in data:
            inline[key] = data[key]

    for key, limit in (("agent_state", 16), ("agent_action", 8), ("agent_reward", 8), ("laser_scan", 256)):
        if key in data:
            compact = _to_small_list(data.get(key), max_len=limit)
            if compact is not None:
                inline[key] = compact

    # Keep planner map lightweight to avoid multi-MB control messages.
    planner = data.get("planner_map")
    if isinstance(planner, dict):
        pm = {
            "robot_position": planner.get("robot_position"),
            "robot_yaw": planner.get("robot_yaw"),
            "target_position": planner.get("target_position"),
            "global_path": planner.get("global_path") or [],
            "dwa_traj": planner.get("dwa_traj") or [],
            "p_traj": planner.get("p_traj") or [],
        }
        inline["planner_map"] = pm

    robots = data.get("robots")
    if isinstance(robots, list) and robots:
        inline["robots"] = robots[:4]

    return inline if inline else None


class TaskMonitorWebController:
    """Coordinate the lifecycle and IPC with the web-monitor process.

    Parameters
    ----------
    host:
        Network interface the server will bind to.
    port:
        TCP port for the Flask server.
    startup_timeout:
        Seconds to wait for the subprocess ACK before giving up.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8050,
        startup_timeout: float = 10.0,
    ) -> None:
        """Initialise the controller, spawn the web-monitor subprocess, and wait for readiness.

        Creates a ``spawn``-context subprocess running
        :func:`run_task_monitor_web_process`, sets up the IPC pipe and
        shared-memory writer, and blocks until the child sends an ACK
        message or *startup_timeout* elapses.

        Parameters
        ----------
        host : str
            Network interface for the uvicorn server (default ``"0.0.0.0"``).
        port : int
            TCP port for the web dashboard (default ``8050``).
        startup_timeout : float
            Maximum seconds to wait for the subprocess ACK before
            raising an exception.
        """
        self._host = host
        self._port = port
        self._ctx = mp.get_context("spawn")
        parent_conn, child_conn = create_command_pipe(self._ctx)
        self._parent_conn = parent_conn
        self._process = self._ctx.Process(
            target=run_task_monitor_web_process,
            args=(child_conn, host, port),
            name="TaskMonitorWebProcess",
        )
        self._process.daemon = True
        self._process.start()
        child_conn.close()

        self._writer = SharedTelemetryWriter(ctx=self._ctx)
        self._specs: Dict[int, MonitorSpec] = {}
        self._descriptors: Dict[int, SharedTelemetryDescriptor] = {}
        self._last_pipe_send: Dict[int, float] = {}
        self._closed = False
        self._lock = threading.RLock()
        self._disconnect_logged = False

        atexit.register(self.close)

        try:
            self._wait_for_ready(startup_timeout)
        except Exception:
            self.close()
            raise

        display_host = "localhost" if host == "0.0.0.0" else host
        url = f"http://{display_host}:{port}"
        print(f"\n{'=' * 60}")
        print(f"  Task Monitor Web Dashboard is ready!")
        print(f"  Open in browser: {url}")
        print(f"{'=' * 60}\n")

    # ------------------------------------------------------------------
    # Public API  (mirrors TaskMonitorController)
    # ------------------------------------------------------------------

    def register_environment(self, spec: MonitorSpec) -> bool:
        """Create a shared-memory slot and announce the environment.

        Parameters
        ----------
        spec:
            Full :class:`MonitorSpec` dataclass for the new environment.

        Returns
        -------
        bool
            ``True`` if the message was sent successfully.
        """
        with self._lock:
            if self._closed:
                return False
            descriptor, _ = self._writer.ensure_slot(spec.env_id, 1024)
            self._descriptors[spec.env_id] = descriptor
            self._specs[spec.env_id] = spec
            try:
                self._send(MonitorMessage(
                    command=Command.INIT_ENV,
                    env_id=spec.env_id,
                    payload={
                        "spec": asdict(spec),
                        "descriptor": asdict(descriptor),
                    },
                    version=0,
                ))
            except (BrokenPipeError, EOFError):
                self._handle_transport_failure("register_environment")
                return False
            return True

    def update_environment(self, env_id: int, data) -> None:
        """Serialize *data* into shared memory and notify the dashboard.

        On every call the payload is written to the SHM slot.  An
        ``INIT_ENV`` is sent only when the descriptor changes (e.g. after
        a reallocation).  A throttled ``UPDATE_ENV`` with inline data is
        sent at ``_PIPE_SEND_INTERVAL`` as a reliable fallback.

        Parameters
        ----------
        env_id:
            Target environment identifier.
        data:
            Full telemetry dictionary to serialize.
        """
        with self._lock:
            if self._closed or env_id not in self._specs:
                return
            payload_bytes = encode_payload(data)
            descriptor, created = self._writer.ensure_slot(env_id, len(payload_bytes))
            version = self._writer.write(env_id, payload_bytes)

            descriptor_changed = created or self._descriptors.get(env_id) != descriptor
            if descriptor_changed:
                self._descriptors[env_id] = descriptor
                inline_data = _compact_inline_payload(data)
                message = MonitorMessage(
                    command=Command.INIT_ENV,
                    env_id=env_id,
                    payload={
                        "spec": asdict(self._specs[env_id]),
                        "descriptor": asdict(descriptor),
                        "inline_data": inline_data,
                    },
                    version=version,
                )
                try:
                    self._send(message)
                except (BrokenPipeError, EOFError):
                    self._handle_transport_failure("update_environment:init")
                self._last_pipe_send[env_id] = time.monotonic()
                return

            # Throttled pipe-based fallback — guarantees data delivery
            # even when SHM reads fail in the subprocess.
            now = time.monotonic()
            if now - self._last_pipe_send.get(env_id, 0.0) >= _PIPE_SEND_INTERVAL:
                self._last_pipe_send[env_id] = now
                inline_data = _compact_inline_payload(data)
                if inline_data:
                    message = MonitorMessage(
                        command=Command.UPDATE_ENV,
                        env_id=env_id,
                        payload={"inline_data": inline_data},
                        version=version,
                    )
                    try:
                        self._send(message)
                    except (BrokenPipeError, EOFError):
                        self._handle_transport_failure("update_environment:pipe")

    def remove_environment(self, env_id: int) -> None:
        """Release the shared-memory slot and remove the environment panel.

        Parameters
        ----------
        env_id:
            Environment to remove.
        """
        with self._lock:
            if env_id in self._specs:
                self._writer.release(env_id)
                self._descriptors.pop(env_id, None)
                self._specs.pop(env_id, None)
                self._last_pipe_send.pop(env_id, None)
                try:
                    self._send(MonitorMessage(command=Command.REMOVE_ENV, env_id=env_id))
                except (BrokenPipeError, EOFError):
                    self._handle_transport_failure("remove_environment")

    def close(self) -> None:
        """Tear down IPC and terminate the web-monitor subprocess.

        Sends a ``SHUTDOWN`` command and waits briefly for an ACK before
        terminating the child process.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
        try:
            self._send(MonitorMessage(command=Command.SHUTDOWN, payload={"reason": "close"}))
            self._wait_for_ack(timeout=2.0)
        except RuntimeError:
            pass
        except (BrokenPipeError, EOFError):
            pass
        finally:
            self._finalize_process()

    @property
    def url(self) -> str:
        """Return the dashboard URL."""
        return f"http://{self._host}:{self._port}"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _wait_for_ready(self, timeout: float) -> None:
        """Block until the subprocess sends an initial ``ACK`` message.

        Parameters
        ----------
        timeout : float
            Maximum seconds to wait.  If ``<= 0``, returns immediately.

        Raises
        ------
        BrokenPipeError / EOFError
            If the pipe is closed before an ACK arrives (handled by
            :meth:`_handle_transport_failure`).
        """
        if timeout <= 0:
            return
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            remaining = max(0.0, deadline - time.monotonic())
            if self._parent_conn.poll(remaining):
                try:
                    message = self._parent_conn.recv()
                except (BrokenPipeError, EOFError):
                    self._handle_transport_failure("wait_for_ready")
                    return
                if isinstance(message, MonitorMessage) and message.command is Command.ACK:
                    return

    def _wait_for_ack(self, timeout: float) -> None:
        """Block until the subprocess acknowledges a command.

        Parameters
        ----------
        timeout : float
            Maximum seconds to wait for the ACK.

        Raises
        ------
        RuntimeError
            If the timeout elapses without receiving an ACK.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            remaining = max(0.0, deadline - time.monotonic())
            if not self._parent_conn.poll(remaining):
                continue
            try:
                message = self._parent_conn.recv()
            except (BrokenPipeError, EOFError):
                self._handle_transport_failure("wait_for_ack")
                return
            if isinstance(message, MonitorMessage) and message.command is Command.ACK:
                return
        raise RuntimeError("Web monitor process did not acknowledge command")

    def _send(self, message: MonitorMessage) -> None:
        """Send a :class:`MonitorMessage` over the IPC pipe.

        Parameters
        ----------
        message : MonitorMessage
            The message to send.

        Raises
        ------
        BrokenPipeError
            If the connection is already closed.
        """
        if self._parent_conn.closed:
            raise BrokenPipeError("Web monitor connection already closed")
        self._parent_conn.send(message)

    def _finalize_process(self) -> None:
        """Close the IPC pipe, shut down the SHM writer, and terminate the subprocess.

        Waits up to 3 seconds for the child process to exit gracefully
        before sending ``SIGTERM``.
        """
        try:
            if not self._parent_conn.closed:
                self._parent_conn.close()
        except Exception:
            pass
        self._writer.shutdown()
        if self._process.is_alive():
            self._process.join(timeout=3.0)
            if self._process.is_alive():
                self._process.terminate()

    def _handle_transport_failure(self, context: str) -> None:
        """Handle a broken pipe or EOF error on the IPC connection.

        Marks the controller as closed, logs a one-time disconnect
        message, clears all internal state, and finalises the
        subprocess.

        Parameters
        ----------
        context : str
            Caller description for the log message (e.g.
            ``"register_environment"``).
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if not self._disconnect_logged:
                print(
                    "Web monitor disconnected (%s); telemetry updates will be skipped." % context
                )
                self._disconnect_logged = True
            self._specs.clear()
            self._descriptors.clear()
            self._inline_counter.clear()
            self._finalize_process()
