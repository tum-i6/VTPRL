"""Qt monitor subprocess entrypoint.

This module hosts the runtime that is launched in a dedicated process so the
main agent can remain free of Qt threading constraints.  Communication happens
via the primitives defined in ``task_monitor_ipc``.
"""
from __future__ import annotations

from multiprocessing.connection import Connection
from typing import Dict, Optional, Set, Tuple, Any

import numpy as np

from PySide2.QtCore import QTimer
from PySide2.QtWidgets import QApplication

from .task_monitor import MonitorSpec, TaskMonitorWindow, create_monitor_panel
from .task_monitor_ipc import (
    Command,
    MonitorMessage,
    SharedTelemetryDescriptor,
    SharedTelemetryReader,
    decode_payload,
)

__all__ = ["run_monitor_process"]


class _MonitorRuntime:
    """Manage IPC and UI updates inside the monitor subprocess.

    Attributes
    ----------
    _conn:
        Connection endpoint receiving control :class:`MonitorMessage` instances
        from the agent process.
    _window:
        The :class:`TaskMonitorWindow` that hosts environment panels.
    _readers:
        Mapping from environment identifiers to active
        :class:`SharedTelemetryReader` instances.
    _metadata:
        Cache of :class:`MonitorSpec` objects for environments currently shown
        in the monitor UI.
    _dirty:
        Set of environment identifiers whose telemetry data needs to be read
        from shared memory on the next processing pass.
    _shutdown_requested:
        Flag indicating that the subprocess should exit once pending updates
        are flushed.
    _ack_sent:
        Tracks whether the ready acknowledgement has already been issued.
    _pending_attach:
        Mapping of env_id to a tuple of (spec, descriptor, last version,
        reuse_panel flag) awaiting reader attachment when shared memory becomes
        available.
    _poll_timer:
        ``QTimer`` responsible for multiplexing command processing with UI
        updates.
    """

    def __init__(self, connection: Connection):
        """Initialise the runtime with the control pipe.

        Parameters
        ----------
        connection:
            Duplex pipe endpoint receiving :class:`MonitorMessage` instances
            from the agent process.

        Returns
        -------
        None
        """
        self._conn = connection
        self._window = TaskMonitorWindow()
        self._window.closed.connect(self._request_shutdown)
        self._window.show()
        self._readers: Dict[int, SharedTelemetryReader] = {}
        self._metadata: Dict[int, MonitorSpec] = {}
        self._dirty: Set[int] = set()
        self._shutdown_requested = False
        self._ack_sent = False
        self._pending_attach: Dict[int, Tuple[MonitorSpec, SharedTelemetryDescriptor, Optional[int], bool]] = {}

        self._poll_timer = QTimer(self._window)
        self._poll_timer.setInterval(15)
        self._poll_timer.timeout.connect(self._on_poll_timer)
        self._poll_timer.start()

    # ------------------------------------------------------------------
    # Message handling

    def _on_poll_timer(self) -> None:
        """Handle timer ticks by draining the pipe and applying queued work.

        Returns
        -------
        None
        """
        try:
            while self._conn.poll():
                message = self._conn.recv()
                self._handle_message(message)
        except EOFError:
            self._request_shutdown()

        self._process_dirty_updates()

        if self._shutdown_requested:
            self._finalize_and_quit()

        if not self._ack_sent:
            self._send_ack({"ready": True})
            self._ack_sent = True

    def _handle_message(self, message: MonitorMessage) -> None:
        """Dispatch a :class:`MonitorMessage` to the appropriate handler.

        Parameters
        ----------
        message:
            Incoming monitor message from the agent process.

        Returns
        -------
        None
        """
        if not isinstance(message, MonitorMessage):
            return

        command = message.command
        env_id = message.env_id

        if command is Command.INIT_ENV and env_id is not None:
            self._handle_init_env(env_id, message.payload or {}, message.version)
        elif command is Command.UPDATE_ENV and env_id is not None:
            self._handle_update_env(env_id)
        elif command is Command.REMOVE_ENV and env_id is not None:
            self._handle_remove_env(env_id)
        elif command is Command.SHUTDOWN:
            self._handle_shutdown(message.payload or {})
        elif command is Command.PING:
            self._send_ack(message.payload or {})

    def _handle_init_env(self, env_id: int, payload: Dict[str, object], version: Optional[int]) -> None:
        """Create or replace telemetry bindings for ``env_id``.

        Parameters
        ----------
        env_id:
            Identifier for the environment panel to initialise.
        payload:
            Mapping containing a serialized :class:`MonitorSpec` and the shared
            memory descriptor advertised by the agent.
        version:
            Optional telemetry version that triggered the initialisation. When
            provided the environment is marked dirty immediately so the first
            update is rendered without delay.

        Returns
        -------
        None
        """
        spec_dict = payload.get("spec")
        descriptor_dict = payload.get("descriptor")
        if spec_dict is None or descriptor_dict is None:
            return

        try:
            spec = MonitorSpec(**spec_dict)
            descriptor = SharedTelemetryDescriptor(**descriptor_dict)
        except Exception:
            return

        existing_spec = self._metadata.get(env_id)
        reuse_panel = existing_spec is not None and self._specs_equivalent(existing_spec, spec)

        existing_reader = self._readers.pop(env_id, None)
        if existing_reader is not None:
            existing_reader.close()

        if not reuse_panel and existing_spec is not None:
            self._window.remove_environment(env_id)
            self._metadata.pop(env_id, None)

        self._pending_attach[env_id] = (spec, descriptor, version, reuse_panel)
        self._attempt_attach(env_id)

    @staticmethod
    def _specs_equivalent(a: MonitorSpec, b: MonitorSpec) -> bool:
        def _normalize(value: Any) -> Any:
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, (list, tuple)):
                return [_normalize(v) for v in value]
            if isinstance(value, dict):
                return {k: _normalize(v) for k, v in value.items()}
            return value

        if a.env_id != b.env_id or a.name != b.name or a.type != b.type:
            return False
        return _normalize(a.config) == _normalize(b.config)

    def _handle_update_env(self, env_id: int) -> None:
        """Mark ``env_id`` for processing on the next poll tick.

        Parameters
        ----------
        env_id:
            Identifier of the environment whose telemetry was updated.

        Returns
        -------
        None
        """
        if env_id not in self._readers:
            if env_id in self._pending_attach:
                self._attempt_attach(env_id)
            return
        self._dirty.add(env_id)

    def _attempt_attach(self, env_id: int, retry_delay_ms: int = 50) -> None:
        """Try to attach a reader and panel for a pending environment.

        Parameters
        ----------
        env_id:
            Identifier of the environment being attached.
        retry_delay_ms:
            Delay in milliseconds before retrying if the shared memory is not
            yet available.

        Returns
        -------
        None
        """
        pending = self._pending_attach.get(env_id)
        if pending is None:
            return

        spec, descriptor, version, reuse_panel = pending
        try:
            reader = SharedTelemetryReader(descriptor)
        except FileNotFoundError:
            if retry_delay_ms <= 0:
                return
            QTimer.singleShot(retry_delay_ms, lambda env_id=env_id: self._attempt_attach(env_id, min(retry_delay_ms * 2, 1000)))
            return
        except Exception:
            self._pending_attach.pop(env_id, None)
            return

        panel = None
        if reuse_panel:
            panel = self._window.get_environment_panel(env_id)
            if panel is None:
                reuse_panel = False

        if not reuse_panel:
            panel = create_monitor_panel(spec)
            if panel is None:
                reader.close()
                self._pending_attach.pop(env_id, None)
                return
            self._window.add_environment(spec, panel)

        self._pending_attach.pop(env_id, None)
        self._readers[env_id] = reader
        self._metadata[env_id] = spec
        if version is not None:
            self._dirty.add(env_id)

    def _handle_remove_env(self, env_id: int) -> None:
        """Remove UI and shared-memory bindings for ``env_id``.

        Parameters
        ----------
        env_id:
            Identifier of the environment to remove.

        Returns
        -------
        None
        """
        reader = self._readers.pop(env_id, None)
        if reader is not None:
            reader.close()
        self._metadata.pop(env_id, None)
        self._dirty.discard(env_id)
        self._pending_attach.pop(env_id, None)
        self._window.remove_environment(env_id)

    def _handle_shutdown(self, payload: Dict[str, object]) -> None:
        """Acknowledge the shutdown request and prepare to exit.

        Parameters
        ----------
        payload:
            Optional payload to echo back in the ACK message.

        Returns
        -------
        None
        """
        self._send_ack(payload)
        self._request_shutdown()

    def _request_shutdown(self) -> None:
        """Signal that the process should terminate after draining work.

        Returns
        -------
        None
        """
        self._shutdown_requested = True

    def _process_dirty_updates(self) -> None:
        """Apply telemetry updates for environments marked as dirty.

        Returns
        -------
        None
        """
        if not self._dirty:
            return

        dirty_envs = list(self._dirty)
        self._dirty.clear()
        for env_id in dirty_envs:
            reader = self._readers.get(env_id)
            if reader is None:
                continue
            updated = reader.read_if_updated()
            if not updated:
                continue
            _, payload = updated
            try:
                data = decode_payload(payload)
            except Exception:
                continue
            self._window.update_environment(env_id, data)

    def _send_ack(self, payload: Dict[str, object]) -> None:
        """Send an acknowledgement message with optional ``payload``.

        Parameters
        ----------
        payload:
            Optional dictionary to include in the acknowledgement.

        Returns
        -------
        None
        """
        if self._conn.closed:
            return
        try:
            self._conn.send(MonitorMessage(command=Command.ACK, payload=payload))
        except (BrokenPipeError, EOFError):
            pass

    def _finalize_and_quit(self) -> None:
        """Release resources and request application exit.

        Returns
        -------
        None
        """
        self._poll_timer.stop()
        for reader in self._readers.values():
            try:
                reader.close()
            except Exception:
                pass
        self._readers.clear()
        self._metadata.clear()
        self._pending_attach.clear()
        if not self._conn.closed:
            try:
                self._conn.close()
            except Exception:
                pass
        QApplication.instance().quit()


def run_monitor_process(connection: Connection) -> None:
    """Start the Qt application that hosts the Task Monitor window.

    Parameters
    ----------
    connection:
        The child endpoint of the control pipe used to receive commands from
        the agent process.

    Returns
    -------
    None
    """

    app = QApplication.instance() or QApplication([])
    runtime = _MonitorRuntime(connection)
    try:
        app.exec_()
    finally:
        runtime._finalize_and_quit()
