"""IPC primitives for the external Task Monitor process.

This module provides the low-level shared-memory buffers and message
structures that the agent process (writer) and the Qt monitor process
(reader) will use to communicate.  It intentionally stays free of any Qt
imports so it can be safely imported in subprocess bootstrap code.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
from multiprocessing import resource_tracker
import pickle
import struct
import threading
from typing import Any, Dict, Optional, Tuple

__all__ = [
    "Command",
    "MonitorMessage",
    "SharedTelemetryDescriptor",
    "SharedTelemetryWriter",
    "SharedTelemetryReader",
    "encode_payload",
    "decode_payload",
    "create_command_pipe",
]


class Command(str, Enum):
    """High-level command identifiers passed over the control pipe.

    The values are string-based so they are easy to serialise across the pipe
    and remain human-readable when inspecting debug logs.
    """

    INIT_ENV = "init_env"
    UPDATE_ENV = "update_env"
    REMOVE_ENV = "remove_env"
    SHUTDOWN = "shutdown"
    PING = "ping"
    ACK = "ack"


@dataclass
class MonitorMessage:
    """Control message exchanged between the agent and monitor process.

    Attributes
    ----------
    command:
        The :class:`Command` describing the action to perform.
    env_id:
        Optional environment identifier that the command is scoped to.
    payload:
        Optional structured payload accompanying the command. The format is
        specific to each command, e.g. ``INIT_ENV`` includes specs and shared
        memory descriptors.
    version:
        Optional telemetry version used to order updates for ``UPDATE_ENV``.
    """

    command: Command
    env_id: Optional[int] = None
    payload: Optional[Dict[str, Any]] = None
    version: Optional[int] = None


@dataclass(frozen=True)
class SharedTelemetryDescriptor:
    """Metadata describing a shared-memory telemetry slot.

    Attributes
    ----------
    env_id:
        Environment identifier owning this slot.
    name:
        OS-level shared memory name of the slot.
    size:
        Total size of the shared memory region in bytes (including header).
    """

    env_id: int
    name: str
    size: int


def encode_payload(data: Any) -> bytes:
    """Serialize telemetry payloads for transport.

    Pickle keeps the implementation simple for stage one; the buffer lives in
    shared memory so we avoid the additional copies that a Queue would incur.

    Parameters
    ----------
    data:
        Arbitrary Python object to serialize.

    Returns
    -------
    bytes
        Pickled representation of the input object.
    """

    return pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)


def decode_payload(buffer: bytes) -> Any:
    """Deserialize payloads produced by :func:`encode_payload`.

    Parameters
    ----------
    buffer:
        Bytes produced by :func:`encode_payload`.

    Returns
    -------
    Any
        Original Python object reconstructed from the pickle.
    """

    return pickle.loads(buffer)


_HEADER_STRUCT = struct.Struct("<QQ")  # (version, size)
_HEADER_SIZE = _HEADER_STRUCT.size


class _SharedMemorySlot:
    """Internal helper managing a single shared-memory buffer."""

    def __init__(self, shm: SharedMemory):
        """Create a slot wrapper around a shared-memory block.

        Parameters
        ----------
        shm:
            Existing shared-memory handle allocated by the writer.

        Returns
        -------
        None
        """
        self._shm = shm
        self.capacity = shm.size - _HEADER_SIZE
        self._lock = threading.Lock()  # single-writer within the agent process
        # zero the header to mark the slot as unused
        _HEADER_STRUCT.pack_into(self._shm.buf, 0, 0, 0)

    @property
    def name(self) -> str:
        """Return the shared-memory block name.

        Returns
        -------
        str
            OS-level shared memory name for this slot.
        """
        return self._shm.name

    @property
    def size(self) -> int:
        """Return total buffer size including header.

        Returns
        -------
        int
            Size of the shared memory region in bytes.
        """
        return self._shm.size

    def write(self, payload: bytes, version: int) -> None:
        """Write a payload into the slot with version stamping.

        Parameters
        ----------
        payload:
            Raw bytes to store.
        version:
            Monotonic version tag for readers.

        Returns
        -------
        None
        """
        if len(payload) > self.capacity:
            raise ValueError(
                f"Payload of {len(payload)} bytes exceeds slot capacity {self.capacity}"
            )
        with self._lock:
            buf = self._shm.buf
            start = _HEADER_SIZE
            stop = start + len(payload)
            buf[start:stop] = payload
            _HEADER_STRUCT.pack_into(buf, 0, version, len(payload))

    def close(self, unlink: bool = False) -> None:
        """Close the shared-memory slot and optionally unlink it.

        Parameters
        ----------
        unlink:
            If True, remove the shared-memory object from the OS.

        Returns
        -------
        None
        """
        self._shm.close()
        if unlink:
            try:
                self._shm.unlink()
            except FileNotFoundError:
                pass


class SharedTelemetryWriter:
    """Manage shared-memory telemetry slots for the agent process."""

    def __init__(self, ctx: Optional[mp.context.BaseContext] = None):
        """Create a writer for telemetry shared-memory slots.

        Parameters
        ----------
        ctx:
            Optional multiprocessing context; defaults to the ``spawn`` context
            to stay consistent with the Task Monitor subprocess.

        Returns
        -------
        None
        """
        self._ctx = ctx or mp.get_context("spawn")
        self._slots: Dict[int, _SharedMemorySlot] = {}
        self._versions: Dict[int, int] = {}

    def ensure_slot(
        self, env_id: int, minimum_capacity: int
    ) -> Tuple[SharedTelemetryDescriptor, bool]:
        """Create (or resize) the telemetry slot for ``env_id``.

        Parameters
        ----------
        env_id:
            Environment identifier that owns the slot.
        minimum_capacity:
            Payload capacity in bytes that must be supported by the slot. The
            method automatically allocates extra header space.

        Returns
        -------
        Tuple[SharedTelemetryDescriptor, bool]
            The descriptor describing the slot and a flag indicating whether a
            new shared-memory region was created.
        """

        slot = self._slots.get(env_id)
        if slot is not None and slot.capacity >= minimum_capacity:
            descriptor = SharedTelemetryDescriptor(env_id, slot.name, slot.size)
            return descriptor, False

        if slot is not None:
            slot.close(unlink=True)

        total_size = _HEADER_SIZE + max(minimum_capacity, 1024)
        shm = SharedMemory(create=True, size=total_size)
        slot = _SharedMemorySlot(shm)
        self._slots[env_id] = slot
        self._versions[env_id] = 0
        descriptor = SharedTelemetryDescriptor(env_id, slot.name, slot.size)
        return descriptor, True

    def write(self, env_id: int, payload: bytes) -> int:
        """Write ``payload`` to ``env_id``'s slot and return the new version.

        Parameters
        ----------
        env_id:
            Environment identifier whose slot is targeted.
        payload:
            Raw bytes to write.

        Returns
        -------
        int
            The incremented version number stored with the payload.
        """

        slot = self._slots.get(env_id)
        if slot is None:
            raise KeyError(f"No telemetry slot registered for env {env_id}")
        version = self._versions.get(env_id, 0) + 1
        self._versions[env_id] = version
        slot.write(payload, version)
        return version

    def release(self, env_id: int) -> None:
        """Free the shared-memory slot associated with ``env_id``.

        Parameters
        ----------
        env_id:
            Environment identifier whose slot should be released.

        Returns
        -------
        None
        """
        slot = self._slots.pop(env_id, None)
        self._versions.pop(env_id, None)
        if slot is not None:
            slot.close(unlink=True)

    def shutdown(self) -> None:
        """Release all managed shared-memory slots.

        Returns
        -------
        None
        """
        for env_id in list(self._slots.keys()):
            self.release(env_id)


class SharedTelemetryReader:
    """Read telemetry data from shared-memory slots in the monitor process."""

    def __init__(self, descriptor: SharedTelemetryDescriptor):
        """Attach to an existing shared-memory region described by ``descriptor``.

        Parameters
        ----------
        descriptor:
            Shared telemetry descriptor produced by the writer.

        Returns
        -------
        None
        """
        self._descriptor = descriptor
        self._shm: Optional[SharedMemory] = None
        self._last_version = 0
        self._is_tracked = False
        if not self._reattach():
            raise FileNotFoundError(descriptor.name)

    @property
    def descriptor(self) -> SharedTelemetryDescriptor:
        """Return the :class:`SharedTelemetryDescriptor` for the reader.

        Returns
        -------
        SharedTelemetryDescriptor
            Descriptor that was used to attach this reader.
        """
        return self._descriptor

    def read_if_updated(self) -> Optional[Tuple[int, bytes]]:
        """Return the latest payload if a new version is available.

        Returns
        -------
        Optional[Tuple[int, bytes]]
            ``None`` when the shared-memory slot has no new data, otherwise a
            tuple of ``(version, payload)``.
        """

        buffer = self._ensure_buffer()
        if buffer is None:
            return None

        try:
            version, size = _HEADER_STRUCT.unpack_from(buffer, 0)
        except (BufferError, ValueError, TypeError):
            if not self._reattach():
                return None
            buffer = self._ensure_buffer()
            if buffer is None:
                return None
            try:
                version, size = _HEADER_STRUCT.unpack_from(buffer, 0)
            except Exception:
                return None

        if version == 0 or version == self._last_version or size == 0:
            return None

        start = _HEADER_SIZE
        stop = start + size
        try:
            payload = bytes(buffer[start:stop])
        except (BufferError, ValueError, TypeError):
            if not self._reattach():
                return None
            buffer = self._ensure_buffer()
            if buffer is None:
                return None
            try:
                payload = bytes(buffer[start:stop])
            except Exception:
                return None

        self._last_version = version
        return version, payload

    def close(self) -> None:
        """Detach from the shared-memory buffer without unlinking it.

        Returns
        -------
        None
        """
        self._close_internal()

    # ------------------------------------------------------------------
    # Internal helpers

    def _ensure_buffer(self) -> Optional[memoryview]:
        """Return a valid memoryview for the shared buffer, reattaching if needed.

        Returns
        -------
        Optional[memoryview]
            Active buffer view or ``None`` when unavailable.
        """
        shm = self._shm
        if shm is None:
            if not self._reattach():
                return None
            shm = self._shm
            if shm is None:
                return None
        try:
            buffer = shm.buf
        except (BufferError, ValueError):
            if not self._reattach():
                return None
            shm = self._shm
            if shm is None:
                return None
            try:
                buffer = shm.buf
            except Exception:
                return None
        if buffer is None:
            if not self._reattach():
                return None
            shm = self._shm
            if shm is None:
                return None
            try:
                buffer = shm.buf
            except Exception:
                return None
        return buffer

    def _reattach(self) -> bool:
        """Re-open the shared memory based on the stored descriptor.

        Returns
        -------
        bool
            True if reattachment succeeded, False otherwise.
        """
        self._close_internal()
        try:
            self._shm = SharedMemory(name=self._descriptor.name)
        except FileNotFoundError:
            self._shm = None
            return False
        except Exception:
            self._shm = None
            return False
        name = getattr(self._descriptor, "name", None)
        if name:
            try:
                resource_tracker.register(name, "shared_memory")
                self._is_tracked = True
            except ValueError:
                # Already tracked; keep flag so we attempt to unregister later.
                self._is_tracked = True
            except Exception:
                self._is_tracked = False
        self._last_version = 0
        return True

    def _close_internal(self) -> None:
        """Internal helper to close shared memory and unregister tracking.

        Returns
        -------
        None
        """
        if self._shm is not None:
            try:
                self._shm.close()
            except Exception:
                pass
            finally:
                self._shm = None
        if self._is_tracked:
            name = getattr(self._descriptor, "name", None)
            if name:
                try:
                    resource_tracker.unregister(name, "shared_memory")
                except Exception:
                    pass
        self._is_tracked = False


def create_command_pipe(
    ctx: Optional[mp.context.BaseContext] = None,
) -> Tuple[mp.connection.Connection, mp.connection.Connection]:
    """Create a duplex pipe for control messages using the spawn context.

    Parameters
    ----------
    ctx:
        Optional multiprocessing context. When omitted the function uses a
        ``spawn`` context to stay consistent with the controller.

    Returns
    -------
    Tuple[Connection, Connection]
        The parent and child endpoints of the duplex pipe.
    """

    ctx = ctx or mp.get_context("spawn")
    return ctx.Pipe(duplex=True)
