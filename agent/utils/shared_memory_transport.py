"""
High-level shared-memory transport for the Python agent.

This module wraps :class:`SharedMemoryRingBuffer` in a bidirectional
channel and handles the gRPC handshake with the Unity server so that
the heavy observation / action payloads travel through memory-mapped
files instead of the gRPC data field.

Usage in ``SimulatorVecEnv``
----------------------------
::

    from utils.shared_memory_transport import SharedMemoryTransport

    # During __init__ (after gRPC channel is ready):
    self.shm = SharedMemoryTransport(
        stub=self.stub,
        shm_dir="/mnt/shm",          # path visible to both Unity & container
        capacity=4,
        slot_size_mb=64,
    )

    # In _send_request (instead of stub.step(StepRequest(data=content))):
    response_json = self.shm.step(content)   # content is the JSON string
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

from utils.shared_memory_ring_buffer import SharedMemoryRingBuffer

logger = logging.getLogger(__name__)

# Default path for the shared-memory directory.  On WSL2/Docker this should
# be a bind-mount from the Windows host (e.g., C:\\shm  →  /mnt/shm).
DEFAULT_SHM_DIR = os.environ.get("VTPRL_SHM_DIR", "/mnt/shm")

# Default parameters matching the C# SharedMemoryChannel defaults.
DEFAULT_CAPACITY = 4
DEFAULT_SLOT_SIZE_MB = 64


class SharedMemoryTransport:
    """Bidirectional shared-memory transport + gRPC control plane.

    Parameters
    ----------
    stub : CommunicationServiceStub
        The existing gRPC stub used for control messages.
    shm_dir : str
        Directory on the shared filesystem where the backing files will be
        created.  Must be accessible from both Unity (Windows/Linux) and the
        Python agent (WSL2/Docker/Linux).
    capacity : int
        Number of ring-buffer slots (power-of-2).
    slot_size_mb : int
        Payload size per slot in mebibytes.
    same_kernel : bool
        When *True*, both processes share the same OS kernel.  Disables
        flush/invalidate operations for maximum shared-memory performance.
        When *False* (default), enables cross-OS coherency via 9P.
    grpc_timeout : float | None
        Timeout for gRPC control calls in seconds.
    read_timeout : float
        Timeout for blocking reads on the response ring.
    """

    def __init__(
        self,
        stub,
        shm_dir: str = DEFAULT_SHM_DIR,
        capacity: int = DEFAULT_CAPACITY,
        slot_size_mb: int = DEFAULT_SLOT_SIZE_MB,
        same_kernel: bool = False,
        grpc_timeout: Optional[float] = None,
        read_timeout: float = 60.0,
    ) -> None:
        self._stub = stub
        self._shm_dir = shm_dir
        self._capacity = capacity
        self._slot_size_mb = slot_size_mb
        self._same_kernel = same_kernel
        self._grpc_timeout = grpc_timeout
        self._read_timeout = read_timeout

        self._request_ring: Optional[SharedMemoryRingBuffer] = None
        self._response_ring: Optional[SharedMemoryRingBuffer] = None
        self._ready = False

        # Perform handshake
        self._handshake()

    # ──────────────────── handshake ────────────────────

    def _handshake(self) -> None:
        """Send an ``SHM_INIT`` gRPC request to Unity and open the backing files.

        The method serializes the SHM configuration (directory, capacity,
        slot size) as a JSON envelope and sends it via the standard
        ``step`` RPC.  Once Unity replies with ``{"status": "ok"}``, the
        client polls for the backing files to appear (they may be delayed
        by cross-OS filesystem propagation) and opens them as
        :class:`SharedMemoryRingBuffer` instances.

        Raises
        ------
        RuntimeError
            If the server rejects the handshake.
        FileNotFoundError
            If the backing files do not appear within the poll timeout.
        """
        from utils.service_pb2 import StepRequest

        envelope = json.dumps({
            "shm_init": True,
            "shm_dir": self._shm_dir,
            "capacity": self._capacity,
            "slot_size_mb": self._slot_size_mb,
        })

        logger.info("[SHM] Sending handshake: %s", envelope)

        call_kwargs = {}
        if self._grpc_timeout:
            call_kwargs["timeout"] = self._grpc_timeout

        reply = self._stub.step(StepRequest(data=envelope), **call_kwargs)
        resp = json.loads(reply.data)

        if resp.get("status") != "ok":
            msg = resp.get("msg", "unknown error")
            raise RuntimeError(f"[SHM] Handshake failed: {msg}")

        # Log what the server reported for debugging path mismatches.
        server_dir = resp.get("server_shm_dir", "(not reported)")
        # Update same_kernel from server response (server-side config is authoritative)
        if "same_kernel" in resp:
            self._same_kernel = bool(resp["same_kernel"])
        logger.info(
            "[SHM] Server created files at: %s  \u2014  Python will read from: %s  (same_kernel=%s)",
            server_dir, self._shm_dir, self._same_kernel,
        )

        # Server has created the backing files — open them on the client side.
        request_path = os.path.join(self._shm_dir, "shm_request.bin")
        response_path = os.path.join(self._shm_dir, "shm_response.bin")

        # Poll for the files to appear through the filesystem mount (especially
        # across WSL2 / 9P / Docker bind-mount boundaries).
        import time
        _POLL_INTERVAL = 0.2
        _POLL_TIMEOUT = 15.0  # generous timeout for cross-OS FS propagation
        _elapsed = 0.0
        while _elapsed < _POLL_TIMEOUT:
            if os.path.exists(request_path) and os.path.exists(response_path):
                break
            time.sleep(_POLL_INTERVAL)
            _elapsed += _POLL_INTERVAL
        else:
            # Files never appeared — give a detailed diagnostic message.
            dir_exists = os.path.isdir(self._shm_dir)
            dir_contents = []
            if dir_exists:
                try:
                    dir_contents = os.listdir(self._shm_dir)
                except OSError:
                    pass
            raise FileNotFoundError(
                f"[SHM] Backing files not found at {self._shm_dir} after "
                f"{_POLL_TIMEOUT}s.\n"
                f"  Unity server created files at: {server_dir}\n"
                f"  Python expected them at:       {self._shm_dir}\n"
                f"  Directory '{self._shm_dir}' exists: {dir_exists}\n"
                f"  Directory contents: {dir_contents}\n\n"
                f"This usually means the Docker container does not have the "
                f"host directory bind-mounted.  Add a volume mount, e.g.:\n"
                f"  docker run ... -v {server_dir}:{self._shm_dir}:rw ...\n"
                f"Or on WSL2:\n"
                f"  docker run ... -v /mnt/c/shm:{self._shm_dir}:rw ..."
            )

        cross_os = not self._same_kernel
        self._request_ring = SharedMemoryRingBuffer(request_path, create=False, cross_os=cross_os)
        self._response_ring = SharedMemoryRingBuffer(response_path, create=False, cross_os=cross_os)
        self._ready = True

        logger.info(
            "[SHM] Handshake complete — capacity=%d, slot_size=%dMiB, "
            "same_kernel=%s, request=%s, response=%s",
            self._request_ring.capacity,
            self._request_ring.slot_size // (1024 * 1024),
            self._same_kernel,
            request_path,
            response_path,
        )

    # ──────────────────── step ────────────────────

    def step(self, request_json: str) -> str:
        """Execute one simulation step via the SHM data plane.

        1. Write *request_json* (command payload) into the SHM request ring
           and flush dirty pages through 9P.
        2. Send a lightweight gRPC notification ``{"shm": true}`` to wake
           the Unity server.
        3. Invalidate the response ring's cached pages and block-read the
           response.
        4. Return the response JSON string.

        Parameters
        ----------
        request_json : str
            JSON-encoded command payload to send to the server.

        Returns
        -------
        str
            JSON-encoded response from the Unity server.

        Raises
        ------
        RuntimeError
            If the transport has not completed its handshake.
        TimeoutError
            If the response is not received within the configured
            ``read_timeout``.
        """
        if not self._ready:
            raise RuntimeError("[SHM] Transport not ready (handshake not complete).")

        from utils.service_pb2 import StepRequest

        # 1. Write payload into SHM (try_write already calls mm.flush)
        payload_bytes = request_json.encode("utf-8")
        self._request_ring.write(payload_bytes, timeout=self._read_timeout)

        # 2. Notify Unity via gRPC (tiny envelope)
        call_kwargs = {}
        if self._grpc_timeout:
            call_kwargs["timeout"] = self._grpc_timeout
        reply = self._stub.step(StepRequest(data='{"shm":true}'), **call_kwargs)

        # 3. Invalidate stale response-ring pages so the next read fetches
        #    fresh data written by Unity (crosses the 9P boundary).
        #    Skipped when same_kernel=True (same page cache, no stale data).
        if not self._same_kernel:
            self._response_ring.invalidate()

        # 4. Read response from SHM
        resp_bytes, _msg_type = self._response_ring.blocking_read(
            timeout=self._read_timeout
        )
        return resp_bytes.decode("utf-8")

    # ──────────────────── raw binary step ────────────────────

    def step_raw(self, request_bytes: bytes, msg_type: int = 0) -> bytes:
        """Like :meth:`step` but operates on raw bytes instead of strings.

        Useful for binary serialization formats (e.g., MessagePack).

        Parameters
        ----------
        request_bytes : bytes
            Raw payload to send to the server.
        msg_type : int, optional
            Application-defined message type tag (default ``0``).

        Returns
        -------
        bytes
            Raw response payload from the server.

        Raises
        ------
        RuntimeError
            If the transport has not completed its handshake.
        TimeoutError
            If the response is not received within the configured
            ``read_timeout``.
        """
        if not self._ready:
            raise RuntimeError("[SHM] Transport not ready.")

        from utils.service_pb2 import StepRequest

        self._request_ring.write(request_bytes, msg_type=msg_type, timeout=self._read_timeout)

        call_kwargs = {}
        if self._grpc_timeout:
            call_kwargs["timeout"] = self._grpc_timeout
        self._stub.step(StepRequest(data='{"shm":true}'), **call_kwargs)

        if not self._same_kernel:
            self._response_ring.invalidate()
        resp_bytes, resp_msg_type = self._response_ring.blocking_read(
            timeout=self._read_timeout
        )
        return resp_bytes

    # ──────────────────── lifecycle ────────────────────

    @property
    def ready(self) -> bool:
        """``True`` if the handshake has completed and the rings are open."""
        return self._ready

    def close(self) -> None:
        """Release shared memory resources and mark the transport as not ready.

        Both ring buffers are closed (and their backing files are deleted
        if this side was the creator).  Safe to call multiple times.
        """
        self._ready = False
        if self._request_ring:
            self._request_ring.close()
            self._request_ring = None
        if self._response_ring:
            self._response_ring.close()
            self._response_ring = None

    def __del__(self):
        """Ensure resources are released when the object is garbage-collected."""
        self.close()
