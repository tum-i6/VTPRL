"""
Cross-language step profiler for the simulator control loop.

Collects wall-clock timings on the Python side and merges them with
profiling data sent by the C# Unity simulator (when ``EnableProfiling``
is enabled in the simulator configuration).

Measured sections (Python side):
    - ``python_action_conversion``  — IK / action formatting
    - ``python_create_request``     — building command dicts + JSON serialization
    - ``python_send_request``       — overall transport round-trip
        - ``python_request_encode`` — MsgPack pack + Base64 encode (GRPC_BIN only)
        - ``python_grpc_call``      — blocking RPC call (GRPC / GRPC_BIN / GRPC_SHM)
        - ``python_response_decode``— JSON / MsgPack decode (GRPC / GRPC_BIN / GRPC_SHM)
    - ``python_update_envs``        — parsing observations, updating DART chain, rewards
    - ``python_total``              — total ``step()`` wall-clock time (synthetic)

Measured sections (C# / Unity side, received via ``ProfilingData``):
    - ``unity_shm_read``                — shared-memory read (GRPC_SHM only)
    - ``unity_request_decode``          — Base64 + MsgPack deserialization
    - ``unity_command_parsing``         — iterating commands and applying actions
    - ``unity_physics``                 — ``Physics.Simulate()`` loop
    - ``unity_observation_collection``  — ``GetObservationPayload()`` per environment
    - ``unity_response_serialize``      — MsgPack pack + Base64 encode
    - ``unity_total``                   — total server-side wall-clock time (synthetic)

Derived metrics:
    - ``communication_overhead``   — ``python_grpc_call - unity_total``
    - ``step_gap``                 — wall-clock time *between* successive ``step()`` calls

Usage::

    profiler = StepProfiler(enabled=True)

    profiler.begin_step()
    profiler.begin("python_create_request")
    request = _create_request(...)
    profiler.end("python_create_request")
    ...
    profiler.merge_unity_timings(unity_profiling_dict)
    profiler.end_step()
    profiler.print_report()
"""

from __future__ import annotations

import time
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple


class StepProfiler:
    """Lightweight wall-clock profiler for a single action→observation loop.

    Records ``begin`` / ``end`` timing pairs for labelled code sections,
    merges C# server-side timings received through the observation payload,
    and prints a formatted table to stdout.

    Attributes:
        enabled (bool): Whether profiling is active.  All public methods
            are no-ops when ``False``, so call-sites do not need guards.
    """

    # Known execution order of Unity sections (used for report ordering).
    _UNITY_SECTION_ORDER: List[str] = [
        "unity_shm_read",
        "unity_request_decode",
        "unity_command_parsing",
        "unity_physics",
        "unity_observation_collection",
        "unity_response_serialize",
    ]

    # Python sections that are sub-sections of python_send_request.
    _PYTHON_SUBSECTIONS = frozenset({
        "python_request_encode",
        "python_grpc_call",
        "python_response_decode",
    })

    def __init__(self, enabled: bool = False):
        """Initialise the profiler.

        Args:
            enabled (bool): Activate profiling.  When ``False`` every
                public method returns immediately with zero overhead.
        """
        self.enabled: bool = enabled
        self._starts: Dict[str, float] = {}
        self._sections: OrderedDict[str, float] = OrderedDict()
        self._unity_sections: OrderedDict[str, float] = OrderedDict()
        self._step_start: float = 0.0
        self._step_count: int = 0
        self._last_step_end: float = 0.0
        self._step_gap_ms: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def begin_step(self) -> None:
        """Mark the beginning of a new simulator step.

        Clears previously accumulated section timings, computes the
        *step_gap* (time since the previous ``end_step``), and records
        the new step start timestamp.

        Returns:
            None
        """
        if not self.enabled:
            return
        now = time.perf_counter()
        if self._last_step_end > 0.0:
            self._step_gap_ms = (now - self._last_step_end) * 1000.0
        else:
            self._step_gap_ms = 0.0
        self._starts.clear()
        self._sections.clear()
        self._unity_sections.clear()
        self._step_start = now

    def begin(self, label: str) -> None:
        """Start timing a named section.

        Args:
            label (str): Human-readable section name (e.g.
                ``"python_create_request"``).  Must be unique within
                a single step unless intentional accumulation is desired.

        Returns:
            None
        """
        if not self.enabled:
            return
        self._starts[label] = time.perf_counter()

    def end(self, label: str) -> None:
        """Stop timing *label* and accumulate elapsed milliseconds.

        If ``begin(label)`` was never called (or was already ended),
        the call is silently ignored.

        Args:
            label (str): Must match a prior ``begin()`` call.

        Returns:
            None
        """
        if not self.enabled:
            return
        start = self._starts.pop(label, None)
        if start is None:
            return
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self._sections[label] = self._sections.get(label, 0.0) + elapsed_ms

    def end_step(self) -> None:
        """Finalise the step and record total Python time.

        Adds a synthetic ``python_total`` entry, increments the internal
        step counter, and stores the timestamp for the next step-gap
        calculation.

        Returns:
            None
        """
        if not self.enabled:
            return
        now = time.perf_counter()
        total_ms = (now - self._step_start) * 1000.0
        self._sections["python_total"] = total_ms
        self._step_count += 1
        self._last_step_end = now

    def merge_unity_timings(self, profiling_data: Optional[Dict[str, float]]) -> None:
        """Incorporate C# profiling data received from the simulator.

        The Unity simulator attaches a ``ProfilingData`` dictionary to
        the first environment's observation payload.  Pass that
        dictionary here so the report includes server-side sections.

        Args:
            profiling_data (dict or None): Mapping of C# section labels
                to elapsed milliseconds.  ``None`` or empty dicts are
                silently ignored.

        Returns:
            None
        """
        if not self.enabled or not profiling_data:
            return
        for key, value in profiling_data.items():
            self._unity_sections[str(key)] = float(value)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def print_report(self, every_n: int = 1) -> None:
        """Print a formatted profiling table to stdout.

        The table groups Python and Unity sections with sub-sections
        indented under their parent.  Columns show absolute milliseconds
        and the percentage relative to ``python_total``.

        Args:
            every_n (int): Only print every *every_n*-th step.
                Use ``1`` (default) to print on every step.

        Returns:
            None
        """
        if not self.enabled:
            return
        if every_n > 1 and (self._step_count % every_n) != 0:
            return

        total_ms = self._sections.get("python_total", 0.0)
        unity_total_ms = self._unity_sections.get("unity_total", 0.0)

        def pct(val: float) -> str:
            if total_ms > 0:
                return f"{val / total_ms * 100:6.1f}%"
            return "   N/A"

        W = 72
        SEP = "=" * W
        THIN = "-" * W

        lines = [
            "",
            SEP,
            f"  STEP PROFILER   (step #{self._step_count})"
            f"        total: {total_ms:.3f} ms",
            SEP,
            f"  {'Section':<40s} {'Time (ms)':>10s} {'% Total':>8s}",
            THIN,
        ]

        # ── Python sections ──────────────────────────────────────────
        lines.append(f"  {'[Python]':<40s}")
        for label, ms in self._sections.items():
            if label == "python_total":
                continue
            if label in self._PYTHON_SUBSECTIONS:
                lines.append(f"      {label:<36s} {ms:10.3f} {pct(ms):>8s}")
            else:
                lines.append(f"    {label:<38s} {ms:10.3f} {pct(ms):>8s}")

        lines.append(THIN)

        # ── Unity / C# sections ──────────────────────────────────────
        if self._unity_sections:
            lines.append(f"  {'[Unity / C#]':<40s}")
            for label, dur in self._ordered_unity_sections():
                if label == "unity_total":
                    continue
                lines.append(f"    {label:<38s} {dur:10.3f} {pct(dur):>8s}")
            if unity_total_ms > 0:
                lines.append(THIN)
                lines.append(f"    {'unity_total':<38s} {unity_total_ms:10.3f} {pct(unity_total_ms):>8s}")
            lines.append(THIN)

        # ── Totals & derived metrics ─────────────────────────────────
        lines.append(f"  {'python_total':<40s} {total_ms:10.3f} {'  100.0%':>8s}")

        # Communication overhead = grpc_call - unity_total
        grpc_dur = self._sections.get("python_grpc_call")
        if grpc_dur is None:
            grpc_dur = self._sections.get("python_send_request")

        if grpc_dur is not None and unity_total_ms > 0:
            overhead = grpc_dur - unity_total_ms
            if overhead >= 0:
                src = "grpc_call" if "python_grpc_call" in self._sections else "send_request"
                lines.append(
                    f"  {'communication_overhead':<40s} {overhead:10.3f} {pct(overhead):>8s}"
                )

        # Step gap (time between consecutive step() calls)
        if self._step_gap_ms > 0 and self._step_count > 1:
            lines.append(
                f"  {'step_gap (between steps)':<40s} {self._step_gap_ms:10.3f} {pct(self._step_gap_ms):>8s}"
            )

        if not self._unity_sections:
            lines.append("")
            lines.append(
                "  (Unity profiling data not received"
                " -- set EnableProfiling=true in simulator config)"
            )

        lines.append(SEP)
        lines.append("")

        print("\n".join(lines))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ordered_unity_sections(self) -> List[Tuple[str, float]]:
        """Return Unity sections in their known execution order.

        Sections not in ``_UNITY_SECTION_ORDER`` are appended at the
        end in the order they were received.

        Returns:
            List[Tuple[str, float]]: ``(label, duration_ms)`` pairs.
        """
        ordered: List[Tuple[str, float]] = []
        seen: set = set()
        for label in self._UNITY_SECTION_ORDER:
            if label in self._unity_sections:
                ordered.append((label, self._unity_sections[label]))
                seen.add(label)
        for label, dur in self._unity_sections.items():
            if label not in seen and label != "unity_total":
                ordered.append((label, dur))
        return ordered
