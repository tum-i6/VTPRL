"""Agent-side controller that manages the Task Monitor subprocess."""
from __future__ import annotations

import atexit
import multiprocessing as mp
import threading
import time
from dataclasses import asdict
from typing import Dict

from .task_monitor import MonitorSpec
from .task_monitor_ipc import (
    Command,
    MonitorMessage,
    SharedTelemetryDescriptor,
    SharedTelemetryWriter,
    create_command_pipe,
    encode_payload,
)
from .task_monitor_process import run_monitor_process


class TaskMonitorController:
    """Coordinate the lifecycle and IPC with the external Task Monitor.

    Attributes
    ----------
    _ctx:
        Multiprocessing context used to spawn the subprocess in a platform-safe
        manner.
    _parent_conn:
        Connection endpoint residing in the agent process for sending control
        messages to the monitor runtime.
    _process:
        ``multiprocessing.Process`` instance running :func:`run_monitor_process`.
    _writer:
        Instance of :class:`SharedTelemetryWriter` that owns shared-memory slots
        for each registered environment.
    _specs:
        Mapping of environment identifiers to their :class:`MonitorSpec`
        definitions.
    _descriptors:
        Mapping of environment identifiers to the shared-memory descriptors that
        were last announced to the monitor process.
    _closed:
        ``True`` once :meth:`close` has been invoked and resources released.
    _lock:
        Re-entrant lock guarding multi-threaded access to controller state.
    """

    def __init__(self, startup_timeout: float = 5.0):
        """Launch the monitor subprocess and prepare shared state.

        Parameters
        ----------
        startup_timeout:
            Seconds to wait for the monitor process to send its ready ACK
            before treating initialization as failed.

        Raises
        ------
        RuntimeError
            If the subprocess fails to acknowledge readiness within the allotted
            timeout or the command pipe closes unexpectedly.
        """
        self._ctx = mp.get_context("spawn")
        parent_conn, child_conn = create_command_pipe(self._ctx)
        self._parent_conn = parent_conn
        self._process = self._ctx.Process(
            target=run_monitor_process,
            args=(child_conn,),
            name="TaskMonitorProcess",
        )
        self._process.daemon = True
        self._process.start()
        child_conn.close()

        self._writer = SharedTelemetryWriter(ctx=self._ctx)
        self._specs: Dict[int, MonitorSpec] = {}
        self._descriptors: Dict[int, SharedTelemetryDescriptor] = {}
        self._closed = False
        self._lock = threading.RLock()
        self._disconnect_logged = False

        atexit.register(self.close)

        try:
            self._wait_for_ready(startup_timeout)
        except Exception:
            # Ensure the subprocess is torn down if initialization fails.
            self.close()
            raise

    def register_environment(self, spec: MonitorSpec) -> bool:
        """Create a telemetry slot and register a new environment panel.

        Parameters
        ----------
        spec:
            Metadata describing the environment and the panel configuration
            that should be instantiated inside the monitor process.

        Returns
        -------
        bool
            ``True`` when the environment was registered, ``False`` if the
            controller has already been closed.
        """
        with self._lock:
            if self._closed:
                return False
            descriptor, _ = self._writer.ensure_slot(spec.env_id, 1024)
            self._descriptors[spec.env_id] = descriptor
            self._specs[spec.env_id] = spec
            try:
                self._send(
                    MonitorMessage(
                        command=Command.INIT_ENV,
                        env_id=spec.env_id,
                        payload={
                            "spec": asdict(spec),
                            "descriptor": asdict(descriptor),
                        },
                        version=0,
                    )
                )
            except (BrokenPipeError, EOFError):
                self._handle_transport_failure("register_environment")
                return False
            return True

    def update_environment(self, env_id: int, data):
        """Write telemetry for ``env_id`` and notify the monitor of changes.

        Parameters
        ----------
        env_id:
            Identifier assigned to the environment when it was registered.
        data:
            Arbitrary, pickle-serializable payload containing telemetry to be
            plotted or displayed by the monitor window.
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
                message = MonitorMessage(
                    command=Command.INIT_ENV,
                    env_id=env_id,
                    payload={
                        "spec": asdict(self._specs[env_id]),
                        "descriptor": asdict(descriptor),
                    },
                    version=version,
                )
            else:
                message = MonitorMessage(
                    command=Command.UPDATE_ENV,
                    env_id=env_id,
                    version=version,
                )

            try:
                self._send(message)
            except (BrokenPipeError, EOFError):
                context = "update_environment:init" if descriptor_changed else "update_environment"
                self._handle_transport_failure(context)
                return

    def remove_environment(self, env_id: int):
        """Release the shared-memory slot and remove the panel for ``env_id``.

        Parameters
        ----------
        env_id:
            Identifier of the environment to remove from the monitor.

        Returns
        -------
        None
        """
        with self._lock:
            if env_id in self._specs:
                self._writer.release(env_id)
                self._descriptors.pop(env_id, None)
                self._specs.pop(env_id, None)
                try:
                    self._send(MonitorMessage(command=Command.REMOVE_ENV, env_id=env_id))
                except (BrokenPipeError, EOFError):
                    self._handle_transport_failure("remove_environment")

    def close(self):
        """Tear down IPC resources and terminate the monitor subprocess.

        Returns
        -------
        None
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

    # ------------------------------------------------------------------
    # Internal helpers

    def _wait_for_ready(self, timeout: float) -> None:
        """Block until the subprocess sends an ACK or the timeout elapses.

        Parameters
        ----------
        timeout:
            Maximum seconds to wait for the ready acknowledgement.

        Returns
        -------
        None
        """
        if timeout <= 0:
            return
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._parent_conn.poll(max(0.0, deadline - time.monotonic())):
                try:
                    message = self._parent_conn.recv()
                except (BrokenPipeError, EOFError):
                    self._handle_transport_failure("wait_for_ready")
                    return
                if isinstance(message, MonitorMessage) and message.command is Command.ACK:
                    return
                self._handle_async_message(message)

    def _wait_for_ack(self, timeout: float) -> None:
        """Wait for a command acknowledgement, raising if it never arrives.

        Parameters
        ----------
        timeout:
            Maximum seconds to wait for an acknowledgement.

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            If the acknowledgement is not received before timeout.
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
            self._handle_async_message(message)
        raise RuntimeError("Task monitor process did not acknowledge command")

    def _handle_async_message(self, message):
        """Process unsolicited messages from the monitor process.

        Parameters
        ----------
        message:
            Message received outside the standard request/ack flow.

        Returns
        -------
        None

        Notes
        -----
        Currently a no-op; kept for future expansion (e.g., log streaming).
        """
        # Future expansion: handle unsolicited messages/logging.
        pass

    def _send(self, message: MonitorMessage) -> None:
        """Transmit ``message`` over the control pipe, propagating pipe errors.

        Parameters
        ----------
        message:
            Control message to send to the monitor subprocess.

        Returns
        -------
        None

        Raises
        ------
        BrokenPipeError, EOFError
            If the control pipe is closed or broken.
        """
        if self._parent_conn.closed:
            raise BrokenPipeError("Task monitor connection already closed")
        try:
            self._parent_conn.send(message)
        except (BrokenPipeError, EOFError):
            raise

    def _finalize_process(self):
        """Clean up pipe, writer, and subprocess resources.

        Returns
        -------
        None
        """
        try:
            if not self._parent_conn.closed:
                self._parent_conn.close()
        except Exception:
            pass
        self._writer.shutdown()
        if self._process.is_alive():
            self._process.join(timeout=2.0)
            if self._process.is_alive():
                self._process.terminate()

    def _handle_transport_failure(self, context: str) -> None:
        """Handle pipe failures by shutting down cleanly and logging once.

        Parameters
        ----------
        context:
            String describing the operation during which the failure occurred.

        Returns
        -------
        None
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if not self._disconnect_logged:
                print(
                    "Task monitor disconnected (%s); telemetry updates will be skipped." % context
                )
                self._disconnect_logged = True
            self._specs.clear()
            self._descriptors.clear()
            self._finalize_process()
