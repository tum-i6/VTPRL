"""FastAPI + WebSocket web-monitor subprocess entrypoint.

Mirrors ``task_monitor_process.py`` but runs a FastAPI/uvicorn server instead
of the PySide2/Qt event loop.  Communication with the agent process happens
via the same IPC primitives (shared memory + control pipe) used by the Qt
monitor.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from multiprocessing.connection import Connection
from typing import Any, Dict, Optional, Set, Tuple

from .task_monitor_ipc import (
    Command,
    MonitorMessage,
    SharedTelemetryDescriptor,
    SharedTelemetryReader,
    decode_payload,
)
from .task_monitor_web import TelemetryStore, create_app

__all__ = ["run_task_monitor_web_process"]

_log = logging.getLogger(__name__)


@dataclass
class _MonitorSpecLite:
    """Lightweight mirror of ``task_monitor.MonitorSpec`` without PySide2.

    Attributes
    ----------
    env_id : int
        Unique environment identifier.
    name : str
        Human-readable environment name.
    type : str
        ``"warehouse"`` or ``"manipulator"``.
    config : dict
        Parameter specification dictionary forwarded to figure builders.
    """
    env_id: int
    name: str
    type: str
    config: Dict[str, Any]


class _TaskMonitorWebRuntime:
    """Poll the IPC pipe, read shared-memory telemetry, and feed the store.

    The uvicorn ASGI server runs on the main thread.
    IPC polling happens on a background daemon thread that reads the
    control pipe and shared-memory slots, then pushes decoded payloads
    into the :class:`TelemetryStore` consumed by WebSocket handlers.
    """

    def __init__(self, connection: Connection, host: str, port: int) -> None:
        """Initialise the runtime with IPC connection and server bind address.

        Parameters
        ----------
        connection : Connection
            Duplex pipe endpoint for receiving :class:`MonitorMessage`
            commands from the parent (agent) process.
        host : str
            Network interface the uvicorn ASGI server will bind to
            (e.g. ``"0.0.0.0"``).
        port : int
            TCP port for the uvicorn ASGI server.
        """
        self._conn = connection
        self._host = host
        self._port = port
        self._store = TelemetryStore()
        self._readers: Dict[int, SharedTelemetryReader] = {}
        self._metadata: Dict[int, _MonitorSpecLite] = {}
        self._dirty: Set[int] = set()
        self._shutdown = threading.Event()
        self._ack_sent = False
        self._pending: Dict[int, Tuple[_MonitorSpecLite, SharedTelemetryDescriptor, Optional[int], Optional[Dict[str, Any]]]] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # public
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Start the IPC poller thread and then the uvicorn server (blocking).

        The poller thread is a daemon so it terminates automatically when
        the main thread (uvicorn) exits.
        """
        import uvicorn

        poller = threading.Thread(target=self._poll_loop, daemon=True, name="ipc-poller")
        poller.start()

        app = create_app(self._store)

        uvicorn.run(
            app,
            host=self._host,
            port=self._port,
            log_level="warning",
        )

    # ------------------------------------------------------------------
    # IPC polling (runs on background thread)
    # ------------------------------------------------------------------

    def _poll_loop(self) -> None:
        """Continuously drain the pipe and refresh dirty environments.

        Runs on a background daemon thread.  Exits when the pipe closes
        or the ``_shutdown`` event is set.
        """
        # Send initial ACK before first poll
        self._send_ack({"ready": True})

        while not self._shutdown.is_set():
            try:
                while self._conn.poll(timeout=0.01):
                    message = self._conn.recv()
                    self._handle_message(message)
            except EOFError:
                _log.info("IPC pipe closed — shutting down web monitor.")
                self._shutdown.set()
                break
            except Exception:
                _log.exception("Error polling IPC pipe")

            self._process_dirty()
            time.sleep(0.005)
        # Clean up all readers before the process exits
        self._cleanup_readers()
        import os
        os._exit(0)

    def _cleanup_readers(self) -> None:
        """Close all shared-memory readers to keep the resource tracker clean.

        Called during shutdown to release OS-level shared-memory handles
        and clear internal reader/pending/metadata dictionaries.
        """
        with self._lock:
            for reader in self._readers.values():
                try:
                    reader.close()
                except Exception:
                    pass
            self._readers.clear()
            self._pending.clear()
            self._metadata.clear()

    def _handle_message(self, message: MonitorMessage) -> None:
        """Dispatch a single :class:`MonitorMessage` by command type.

        Routes the message to the appropriate handler based on
        :attr:`MonitorMessage.command`:

        - ``INIT_ENV``   → :meth:`_handle_init`
        - ``UPDATE_ENV`` → :meth:`_handle_update`
        - ``REMOVE_ENV`` → :meth:`_handle_remove`
        - ``SHUTDOWN``   → sets the shutdown event
        - ``PING``       → replies with an ACK

        Parameters
        ----------
        message : MonitorMessage
            Incoming IPC message.  Non-:class:`MonitorMessage` values
            are silently ignored.
        """
        if not isinstance(message, MonitorMessage):
            return
        cmd = message.command
        eid = message.env_id

        if cmd is Command.INIT_ENV and eid is not None:
            self._handle_init(eid, message.payload or {}, message.version)
        elif cmd is Command.UPDATE_ENV and eid is not None:
            self._handle_update(eid, message.payload or {})
        elif cmd is Command.REMOVE_ENV and eid is not None:
            self._handle_remove(eid)
        elif cmd is Command.SHUTDOWN:
            self._shutdown.set()
        elif cmd is Command.PING:
            self._send_ack(message.payload or {})

    def _handle_init(self, env_id: int, payload: Dict[str, Any], version: Optional[int]) -> None:
        """Unpack an ``INIT_ENV`` message: create spec, descriptor, and schedule SHM attach.

        Parses the ``spec`` and ``descriptor`` sub-dicts from *payload*,
        closes any pre-existing reader for *env_id*, and attempts to
        attach to the shared-memory segment.  If the segment is not yet
        visible, the entry is queued in ``_pending`` for later retry.

        Parameters
        ----------
        env_id : int
            Unique environment identifier.
        payload : dict
            Must contain ``"spec"`` and ``"descriptor"`` sub-dicts.
            May also contain ``"inline_data"`` as a fallback telemetry
            snapshot.
        version : int or None
            Writer version counter from the parent process. If not
            ``None``, the environment is immediately marked dirty.
        """
        spec_dict = payload.get("spec")
        desc_dict = payload.get("descriptor")
        if spec_dict is None or desc_dict is None:
            return
        try:
            spec = _MonitorSpecLite(**spec_dict)
            descriptor = SharedTelemetryDescriptor(**desc_dict)
        except Exception:
            _log.exception("Failed to unpack INIT_ENV for env %s", env_id)
            return

        with self._lock:
            old_reader = self._readers.pop(env_id, None)
            if old_reader is not None:
                old_reader.close()
            inline_data = payload.get("inline_data") if isinstance(payload.get("inline_data"), dict) else None
            self._pending[env_id] = (spec, descriptor, version, inline_data)

        self._attempt_attach(env_id)

    def _attempt_attach(self, env_id: int) -> None:
        """Try to open the SHM segment for *env_id*.

        If the segment is not yet visible (race with the writer process),
        the pending entry is kept so ``_process_dirty`` can retry later.

        On success, registers the environment in the
        :class:`TelemetryStore`, optionally pushes inline data, and
        marks the environment dirty for an immediate SHM read.

        Parameters
        ----------
        env_id : int
            Environment whose shared-memory segment to attach.
        """
        with self._lock:
            pending = self._pending.get(env_id)
            if pending is None:
                return
            spec, descriptor, version, inline_data = pending
            try:
                reader = SharedTelemetryReader(descriptor)
            except FileNotFoundError:
                return
            except Exception:
                _log.exception("Failed to attach reader for env %s", env_id)
                self._pending.pop(env_id, None)
                return

            self._pending.pop(env_id, None)
            self._readers[env_id] = reader
            self._metadata[env_id] = spec

            self._store.register_env(env_id, {
                "env_id": spec.env_id,
                "name": spec.name,
                "type": spec.type,
                "config": spec.config,
            })

            if isinstance(inline_data, dict):
                self._store.update_data(env_id, inline_data)

            if version is not None:
                self._dirty.add(env_id)

    def _handle_update(self, env_id: int, payload: Optional[Dict[str, Any]] = None) -> None:
        """Process an ``UPDATE_ENV`` message: ingest inline data and mark dirty.

        If inline telemetry is present in *payload*, it is pushed to the
        :class:`TelemetryStore` immediately.  The environment is then
        marked dirty so that the next :meth:`_process_dirty` cycle reads
        the full shared-memory snapshot.

        Parameters
        ----------
        env_id : int
            Target environment identifier.
        payload : dict or None
            May contain ``"inline_data"`` with a compact telemetry dict.
        """
        inline_data = (payload or {}).get("inline_data") if payload is not None else None
        if isinstance(inline_data, dict):
            self._store.update_data(env_id, inline_data)
        with self._lock:
            if env_id in self._readers:
                self._dirty.add(env_id)
            elif env_id in self._pending:
                self._attempt_attach(env_id)

    def _handle_remove(self, env_id: int) -> None:
        """Process a ``REMOVE_ENV`` message: close reader and clean up state.

        Closes the shared-memory reader, removes all metadata, and
        notifies the :class:`TelemetryStore` to drop the environment.

        Parameters
        ----------
        env_id : int
            Environment to remove.
        """
        with self._lock:
            reader = self._readers.pop(env_id, None)
            if reader is not None:
                reader.close()
            self._metadata.pop(env_id, None)
            self._dirty.discard(env_id)
            self._pending.pop(env_id, None)
        self._store.remove_env(env_id)

    def _process_dirty(self) -> None:
        """Read shared-memory for all attached environments.

        We still keep the `_dirty` set for backwards compatibility, but data
        ingestion no longer depends on control-pipe UPDATE notifications.
        This avoids a stale UI when notifications are dropped or delayed.
        """
        with self._lock:
            pending_ids = list(self._pending.keys())
            # Drain legacy dirty markers and also poll every attached reader.
            self._dirty.clear()
            readers = list(self._readers.items())

        # Retry delayed SHM attachments (INIT can arrive before SHM becomes visible).
        for eid in pending_ids:
            self._attempt_attach(eid)

        for eid, reader in readers:
            try:
                result = reader.read_if_updated()
            except Exception:
                continue
            if result is None:
                continue
            _version, raw = result
            try:
                data = decode_payload(raw)
            except Exception:
                continue
            if isinstance(data, dict):
                self._store.update_data(eid, data)

    def _send_ack(self, payload: Dict[str, Any]) -> None:
        """Send an ``ACK`` message back to the parent process.

        Parameters
        ----------
        payload : dict
            Arbitrary payload to include in the acknowledgement (e.g.
            ``{"ready": True}`` during startup).
        """
        try:
            self._conn.send(MonitorMessage(command=Command.ACK, payload=payload))
        except Exception:
            pass


def run_task_monitor_web_process(
    connection: Connection,
    host: str = "0.0.0.0",
    port: int = 8050,
) -> None:
    """Subprocess entry-point that starts the FastAPI/uvicorn web monitor.

    Parameters
    ----------
    connection:
        Duplex pipe endpoint for receiving :class:`MonitorMessage` from the
        agent process.
    host:
        Network interface for the uvicorn ASGI server.
    port:
        TCP port for the uvicorn ASGI server.
    """
    # The parent (writer) process owns the lifecycle of all shared-memory
    # segments.  Prevent this subprocess's resource_tracker from registering
    # them — otherwise the tracker tries to unlink segments it does not own
    # on exit and prints noisy KeyError tracebacks.
    from multiprocessing import resource_tracker as _rt
    _orig_register = _rt.register
    _orig_unregister = _rt.unregister

    def _skip_shm_register(name, rtype):
        if rtype == "shared_memory":
            return
        return _orig_register(name, rtype)

    def _skip_shm_unregister(name, rtype):
        if rtype == "shared_memory":
            return
        return _orig_unregister(name, rtype)

    _rt.register = _skip_shm_register
    _rt.unregister = _skip_shm_unregister

    runtime = _TaskMonitorWebRuntime(connection, host, port)
    try:
        runtime.run()
    except KeyboardInterrupt:
        pass
    finally:
        connection.close()
