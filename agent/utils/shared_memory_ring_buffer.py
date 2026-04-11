"""
Lock-free single-producer / single-consumer (SPSC) ring buffer backed by a
memory-mapped file.

This module mirrors the C# ``SharedMemoryRingBuffer`` class binary-for-binary
so that a Unity server (Windows) and a Python agent (WSL2 / Docker / Linux)
can share the same backing file via a bind-mounted path.

Typical usage
-------------
**Producer (client writing actions)**::

    ring = SharedMemoryRingBuffer("/mnt/shm/shm_request.bin", create=False)
    ring.write(action_bytes)

**Consumer (client reading responses)**::

    ring = SharedMemoryRingBuffer("/mnt/shm/shm_response.bin", create=False)
    data = ring.blocking_read(timeout=5.0)

Binary layout
-------------
See the C# ``SharedMemoryRingBuffer`` class docstring for the canonical file
layout specification.  Both implementations share the same header, slot
header, and slot data format.
"""

from __future__ import annotations

import mmap
import os
import struct
import time
from pathlib import Path
from typing import Optional, Tuple

# ──────────────────── constants ────────────────────

MAGIC: int = 0x564D4950  # "VMIP"
VERSION: int = 1
HEADER_SIZE: int = 128
SLOT_HEADER_SIZE: int = 32

# Header field offsets
_OFF_MAGIC = 0
_OFF_VERSION = 4
_OFF_CAPACITY = 8
_OFF_SLOT_SIZE = 12
_OFF_WRITE_SEQ = 16
_OFF_READ_SEQ = 24
_OFF_FLAGS = 32

# Slot header offsets (relative to slot start)
_SLOT_OFF_STATE = 0
_SLOT_OFF_PAYLOAD_LEN = 4
_SLOT_OFF_MSG_TYPE = 8
_SLOT_OFF_SEQUENCE = 16

# Slot states
_SLOT_EMPTY = 0
_SLOT_WRITING = 1
_SLOT_COMMITTED = 2

# Flags
FLAG_SHUTDOWN: int = 1

# Struct formats (little-endian, matching C# MemoryMappedViewAccessor default)
_U32 = struct.Struct("<I")
_I64 = struct.Struct("<q")
_U64 = struct.Struct("<Q")


class SharedMemoryRingBuffer:
    """SPSC ring buffer on a memory-mapped file.

    Parameters
    ----------
    path : str | Path
        Absolute path to the backing file.
    capacity : int
        Number of slots (must be power-of-2).  Ignored when ``create=False``.
    slot_size : int
        Max payload bytes per slot.  Ignored when ``create=False``.
    create : bool
        If *True*, create (or overwrite) the backing file and initialize the
        header.  If *False*, open an existing file and validate its header.
    cross_os : bool
        If *True* (default), flush dirty mmap pages after every write/read
        and invalidate cached pages before blocking reads.  Required for
        cross-OS coherency (Windows ↔ WSL2/Docker via 9P bind-mount).
        Set to *False* when both processes share the same OS kernel for
        maximum performance (true zero-copy IPC, no flush overhead).
    """

    __slots__ = (
        "_path",
        "_fd",
        "_mm",
        "_capacity",
        "_slot_size",
        "_file_size",
        "_is_creator",
        "_cross_os",
        "_closed",
    )

    def __init__(
        self,
        path: str | Path,
        capacity: int = 4,
        slot_size: int = 64 * 1024 * 1024,
        create: bool = False,
        cross_os: bool = True,
    ) -> None:
        self._path = str(path)
        self._is_creator = create
        self._cross_os = cross_os
        self._closed = False

        if create:
            if capacity <= 0 or (capacity & (capacity - 1)) != 0:
                raise ValueError(f"Capacity must be a positive power of two, got {capacity}.")
            if slot_size <= 0:
                raise ValueError(f"Slot size must be positive, got {slot_size}.")

            self._capacity = capacity
            self._slot_size = slot_size
            self._file_size = HEADER_SIZE + capacity * (SLOT_HEADER_SIZE + slot_size)

            # Ensure parent directory exists
            os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)

            # Create / truncate and pre-allocate
            with open(self._path, "wb") as f:
                f.truncate(self._file_size)

            self._fd = os.open(self._path, os.O_RDWR)
            self._mm = mmap.mmap(self._fd, self._file_size)

            # Write header
            self._put_u32(_OFF_MAGIC, MAGIC)
            self._put_u32(_OFF_VERSION, VERSION)
            self._put_u32(_OFF_CAPACITY, capacity)
            self._put_u32(_OFF_SLOT_SIZE, slot_size)
            self._put_i64(_OFF_WRITE_SEQ, 0)
            self._put_i64(_OFF_READ_SEQ, 0)
            self._put_u64(_OFF_FLAGS, 0)

            # Zero-init slot headers
            for i in range(capacity):
                off = self._slot_offset(i)
                self._put_u32(off + _SLOT_OFF_STATE, _SLOT_EMPTY)
                self._put_u32(off + _SLOT_OFF_PAYLOAD_LEN, 0)
                self._put_u32(off + _SLOT_OFF_MSG_TYPE, 0)
                self._put_i64(off + _SLOT_OFF_SEQUENCE, 0)

            self._mm.flush()
        else:
            if not os.path.exists(self._path):
                raise FileNotFoundError(f"SHM backing file not found: {self._path}")

            self._fd = os.open(self._path, os.O_RDWR)
            file_stat = os.fstat(self._fd)
            self._mm = mmap.mmap(self._fd, file_stat.st_size)

            magic = self._get_u32(_OFF_MAGIC)
            if magic != MAGIC:
                raise ValueError(f"Bad magic 0x{magic:08X} (expected 0x{MAGIC:08X}).")

            version = self._get_u32(_OFF_VERSION)
            if version != VERSION:
                raise ValueError(f"Unsupported version {version} (expected {VERSION}).")

            self._capacity = self._get_u32(_OFF_CAPACITY)
            self._slot_size = self._get_u32(_OFF_SLOT_SIZE)
            self._file_size = HEADER_SIZE + self._capacity * (SLOT_HEADER_SIZE + self._slot_size)

    # ──────────────────── properties ────────────────────

    @property
    def capacity(self) -> int:
        """Number of slots in the ring buffer (always a power of two)."""
        return self._capacity

    @property
    def slot_size(self) -> int:
        """Maximum payload size per slot in bytes."""
        return self._slot_size

    @property
    def path(self) -> str:
        """Filesystem path of the backing memory-mapped file."""
        return self._path

    @property
    def file_size(self) -> int:
        """Total size of the memory-mapped file in bytes."""
        return self._file_size

    @property
    def write_sequence(self) -> int:
        """Current monotonic write sequence number."""
        return self._get_i64(_OFF_WRITE_SEQ)

    @property
    def read_sequence(self) -> int:
        """Current monotonic read sequence number."""
        return self._get_i64(_OFF_READ_SEQ)

    @property
    def available(self) -> int:
        """Number of committed, unread messages currently in the ring."""
        return max(0, self.write_sequence - self.read_sequence)

    @property
    def flags(self) -> int:
        """Raw 64-bit flags word from the shared header."""
        return self._get_u64(_OFF_FLAGS)

    @property
    def is_shutdown(self) -> bool:
        """``True`` if the shutdown flag has been set by either endpoint."""
        return bool(self.flags & FLAG_SHUTDOWN)

    # ──────────────────── producer API ────────────────────

    def try_write(
        self,
        payload: bytes | bytearray | memoryview,
        msg_type: int = 0,
    ) -> bool:
        """Try to write a message into the next available slot.

        Parameters
        ----------
        payload : bytes | bytearray | memoryview
            The raw bytes to write.  Must not exceed ``slot_size``.
        msg_type : int, optional
            Application-defined message type tag (default ``0``).

        Returns
        -------
        bool
            ``True`` if the write succeeded, ``False`` if the ring is full.

        Raises
        ------
        ValueError
            If *payload* length exceeds the configured slot size.
        """
        length = len(payload)
        if length > self._slot_size:
            raise ValueError(
                f"Payload length {length} exceeds slot size {self._slot_size}."
            )

        wseq = self._get_i64(_OFF_WRITE_SEQ)
        rseq = self._get_i64(_OFF_READ_SEQ)

        if wseq - rseq >= self._capacity:
            return False  # full

        idx = wseq & (self._capacity - 1)
        slot_off = self._slot_offset(idx)

        # Mark writing
        self._put_u32(slot_off + _SLOT_OFF_STATE, _SLOT_WRITING)

        # Write payload
        data_off = slot_off + SLOT_HEADER_SIZE
        self._mm[data_off : data_off + length] = bytes(payload)

        # Write slot metadata
        self._put_u32(slot_off + _SLOT_OFF_PAYLOAD_LEN, length)
        self._put_u32(slot_off + _SLOT_OFF_MSG_TYPE, msg_type)
        self._put_i64(slot_off + _SLOT_OFF_SEQUENCE, wseq)

        # Commit
        self._put_u32(slot_off + _SLOT_OFF_STATE, _SLOT_COMMITTED)

        # Advance write sequence
        self._put_i64(_OFF_WRITE_SEQ, wseq + 1)

        # Flush dirty pages to the backing filesystem so that a reader on
        # another OS (Windows ↔ WSL2/Docker via 9P) can see the update.
        # Skipped when both processes share the same OS kernel (same page cache).
        if self._cross_os:
            self._mm.flush()
        return True

    def write(
        self,
        payload: bytes | bytearray | memoryview,
        msg_type: int = 0,
        timeout: float = 5.0,
    ) -> None:
        """Write a message, spinning until a slot is free or *timeout* expires.

        Parameters
        ----------
        payload : bytes | bytearray | memoryview
            The raw bytes to write.
        msg_type : int, optional
            Application-defined message type tag (default ``0``).
        timeout : float, optional
            Maximum seconds to wait for a free slot (default ``5.0``).

        Raises
        ------
        TimeoutError
            If the ring stays full for the entire *timeout* period.
        """
        deadline = time.monotonic() + timeout
        while not self.try_write(payload, msg_type):
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Ring buffer full for {timeout}s (write_seq={self.write_sequence}, "
                    f"read_seq={self.read_sequence})."
                )
            # Invalidate stale page-cache pages so we can see the remote
            # reader's updated read_seq (cross-OS 9P coherency).
            if self._cross_os:
                self.invalidate()
            # Brief sleep to avoid busy-waiting too aggressively
            time.sleep(0.0001)

    def write_string(self, text: str, msg_type: int = 0, timeout: float = 5.0) -> None:
        """Encode *text* as UTF-8 and write it to the ring buffer.

        Parameters
        ----------
        text : str
            The string to encode and write.
        msg_type : int, optional
            Application-defined message type tag (default ``0``).
        timeout : float, optional
            Maximum seconds to wait for a free slot (default ``5.0``).
        """
        self.write(text.encode("utf-8"), msg_type, timeout)

    # ──────────────────── consumer API ────────────────────

    def try_read(self) -> Optional[Tuple[bytes, int]]:
        """Read the next committed message without blocking.

        Returns
        -------
        tuple[bytes, int] | None
            ``(payload_bytes, msg_type)`` if a message was available,
            or ``None`` if the ring is empty.
        """
        rseq = self._get_i64(_OFF_READ_SEQ)
        wseq = self._get_i64(_OFF_WRITE_SEQ)

        if rseq >= wseq:
            return None  # empty

        idx = rseq & (self._capacity - 1)
        slot_off = self._slot_offset(idx)

        state = self._get_u32(slot_off + _SLOT_OFF_STATE)
        if state != _SLOT_COMMITTED:
            return None

        payload_len = self._get_u32(slot_off + _SLOT_OFF_PAYLOAD_LEN)
        msg_type = self._get_u32(slot_off + _SLOT_OFF_MSG_TYPE)

        data_off = slot_off + SLOT_HEADER_SIZE
        payload = bytes(self._mm[data_off : data_off + payload_len])

        # Release
        self._put_u32(slot_off + _SLOT_OFF_STATE, _SLOT_EMPTY)
        self._put_i64(_OFF_READ_SEQ, rseq + 1)

        # Flush updated read_seq + slot state so the remote writer can see
        # slots have been freed.
        if self._cross_os:
            self._mm.flush()
        return payload, msg_type

    def blocking_read(self, timeout: float = 30.0) -> Tuple[bytes, int]:
        """Spin-read until data is available or *timeout* expires.

        Parameters
        ----------
        timeout : float, optional
            Maximum seconds to wait (default ``30.0``).

        Returns
        -------
        tuple[bytes, int]
            ``(payload_bytes, msg_type)``.

        Raises
        ------
        TimeoutError
            If no data becomes available within *timeout* seconds.
        """
        deadline = time.monotonic() + timeout
        while True:
            result = self.try_read()
            if result is not None:
                return result
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"No data available for {timeout}s (write_seq={self.write_sequence}, "
                    f"read_seq={self.read_sequence})."
                )
            # Invalidate stale page-cache pages so we can see the remote
            # writer's updated write_seq / slot state (cross-OS 9P coherency).
            if self._cross_os:
                self.invalidate()
            time.sleep(0.0001)

    def read_string(self, timeout: float = 30.0) -> Tuple[str, int]:
        """Read a message and decode as UTF-8.

        Parameters
        ----------
        timeout : float, optional
            Maximum seconds to wait (default ``30.0``).

        Returns
        -------
        tuple[str, int]
            ``(decoded_text, msg_type)``.
        """
        data, msg_type = self.blocking_read(timeout)
        return data.decode("utf-8"), msg_type

    # ──────────────────── cross-OS cache coherency ────────────────────

    def flush(self) -> None:
        """Flush dirty mmap pages to the backing filesystem (9P/NTFS).

        Must be called after writes when the reader is on a different OS
        (e.g., Windows ↔ WSL2/Docker via 9P bind-mount).  This triggers
        ``msync(MS_SYNC)`` which pushes pages through the 9P client to
        the Windows 9P server.
        """
        self._mm.flush()

    def invalidate(self) -> None:
        """Invalidate cached mmap pages so the next access fetches fresh
        data from the backing filesystem.

        Two-step invalidation is required for cross-OS coherency via
        9P / Docker bind mounts:

        1. ``madvise(MADV_DONTNEED)`` unmaps pages from the process page
           table, decrementing their mapcount to 0.
        2. ``posix_fadvise(FADV_DONTNEED)`` evicts the now-unmapped pages
           from the Linux page cache.

        Without step 2, the pages remain in the page cache and re-fault
        returns stale data instead of re-reading from the 9P server.
        """
        # Step 1: Unmap from process page table (sets mapcount → 0)
        try:
            self._mm.madvise(mmap.MADV_DONTNEED)
        except (AttributeError, OSError):
            pass
        # Step 2: Evict from page cache (only works when mapcount is 0)
        try:
            os.posix_fadvise(self._fd, 0, self._file_size, os.POSIX_FADV_DONTNEED)
        except (AttributeError, OSError):
            # posix_fadvise not available (non-Linux) — fall back to flush
            try:
                self._mm.flush()
            except OSError:
                pass

    # ──────────────────── control ────────────────────

    def signal_shutdown(self) -> None:
        """Set the shutdown flag and flush to notify the peer process."""
        self._put_u64(_OFF_FLAGS, self.flags | FLAG_SHUTDOWN)
        self._mm.flush()

    # ──────────────────── internal helpers ────────────────────

    def _slot_offset(self, index: int) -> int:
        """Return the byte offset of the slot at *index* from the file start."""
        return HEADER_SIZE + index * (SLOT_HEADER_SIZE + self._slot_size)

    def _get_u32(self, offset: int) -> int:
        """Read a little-endian unsigned 32-bit integer from the mmap at *offset*."""
        return _U32.unpack_from(self._mm, offset)[0]

    def _put_u32(self, offset: int, value: int) -> None:
        """Write a little-endian unsigned 32-bit integer to the mmap at *offset*."""
        _U32.pack_into(self._mm, offset, value)

    def _get_i64(self, offset: int) -> int:
        """Read a little-endian signed 64-bit integer from the mmap at *offset*."""
        return _I64.unpack_from(self._mm, offset)[0]

    def _put_i64(self, offset: int, value: int) -> None:
        """Write a little-endian signed 64-bit integer to the mmap at *offset*."""
        _I64.pack_into(self._mm, offset, value)

    def _get_u64(self, offset: int) -> int:
        """Read a little-endian unsigned 64-bit integer from the mmap at *offset*."""
        return _U64.unpack_from(self._mm, offset)[0]

    def _put_u64(self, offset: int, value: int) -> None:
        """Write a little-endian unsigned 64-bit integer to the mmap at *offset*."""
        _U64.pack_into(self._mm, offset, value)

    # ──────────────────── lifecycle ────────────────────

    def close(self) -> None:
        """Unmap the file and close the file descriptor.

        If this instance created the backing file (``is_creator=True``),
        the file is also deleted from disk.  Safe to call multiple times.
        """
        if self._closed:
            return
        self._closed = True
        try:
            self._mm.close()
        except Exception:
            pass
        try:
            os.close(self._fd)
        except Exception:
            pass
        if self._is_creator:
            try:
                os.unlink(self._path)
            except OSError:
                pass

    def __enter__(self):
        """Context-manager entry; returns *self*."""
        return self

    def __exit__(self, *exc):
        """Context-manager exit; delegates to :meth:`close`."""
        self.close()

    def __del__(self):
        """Ensure the mmap is released when the object is garbage-collected."""
        self.close()
