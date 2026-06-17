"""
Lightweight, crash-surviving memory monitor.

A background daemon thread samples this process's RSS (plus all child
processes, e.g. joblib workers) and system memory at a fixed interval and
appends each sample to a CSV log.  The file is line-buffered and explicitly
flushed after every row, so the trajectory survives even if the process is
OOM-killed (flush pushes the data to the OS page cache, which outlives the
dying process).

Usage (standalone runner)::

    from _memlog import MemoryMonitor, set_phase
    mon = MemoryMonitor("Data_.../_mem_log/run.csv", interval=1.0).start()
    set_phase("loading")
    ...                       # do work; call set_phase(...) at each stage
    mon.stop()                # writes a peak summary row

Library code (e.g. feature_derivation) can call the module-level
``set_phase()`` / ``note()`` helpers unconditionally — they are no-ops when no
monitor is running, so importing this module never forces monitoring on.
"""

from __future__ import annotations

import csv
import datetime
import os
import threading
import time

import psutil

_ACTIVE: "MemoryMonitor | None" = None

_HEADER = [
    "iso_time",
    "elapsed_s",
    "phase",
    "rss_main_mb",
    "rss_children_mb",
    "rss_total_mb",
    "sys_used_mb",
    "sys_available_mb",
    "sys_percent",
    "event",
]


class MemoryMonitor:
    """Background memory sampler that writes a flush-on-every-line CSV log."""

    def __init__(self, log_path: str, interval: float = 1.0):
        self.log_path = log_path
        self.interval = float(interval)
        self._phase = "init"
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._proc = psutil.Process(os.getpid())
        self._lock = threading.Lock()
        self.peak_total_mb = 0.0
        self.peak_phase = "init"
        self.peak_time_s = 0.0

        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        # buffering=1 -> line buffered; we also flush() explicitly each write.
        self._f = open(log_path, "w", newline="", buffering=1)
        self._writer = csv.writer(self._f)
        self._writer.writerow(_HEADER)
        self._f.flush()
        self._t0 = time.time()

    # -- sampling -----------------------------------------------------------
    def _sample_rss(self) -> tuple[float, float]:
        main_rss = 0.0
        child_rss = 0.0
        try:
            main_rss = float(self._proc.memory_info().rss)
            for child in self._proc.children(recursive=True):
                try:
                    child_rss += float(child.memory_info().rss)
                except psutil.Error:
                    pass
        except psutil.Error:
            pass
        return main_rss, child_rss

    def _write_row(self, event: str = "") -> None:
        main_rss, child_rss = self._sample_rss()
        total_mb = (main_rss + child_rss) / 1e6
        vm = psutil.virtual_memory()
        elapsed = round(time.time() - self._t0, 1)

        if total_mb > self.peak_total_mb:
            self.peak_total_mb = total_mb
            self.peak_phase = self._phase
            self.peak_time_s = elapsed

        with self._lock:
            self._writer.writerow(
                [
                    datetime.datetime.now().isoformat(timespec="seconds"),
                    elapsed,
                    self._phase,
                    round(main_rss / 1e6, 1),
                    round(child_rss / 1e6, 1),
                    round(total_mb, 1),
                    round(vm.used / 1e6, 1),
                    round(vm.available / 1e6, 1),
                    vm.percent,
                    event,
                ]
            )
            self._f.flush()

    def _run(self) -> None:
        while not self._stop.is_set():
            self._write_row()
            self._stop.wait(self.interval)

    # -- public API ---------------------------------------------------------
    def set_phase(self, name: str) -> None:
        self._phase = name
        # Emit an immediate marker so the phase boundary timestamp is captured.
        self._write_row(event=f"PHASE -> {name}")

    def note(self, msg: str) -> None:
        self._write_row(event=msg)

    def start(self) -> "MemoryMonitor":
        global _ACTIVE
        _ACTIVE = self
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        global _ACTIVE
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        with self._lock:
            self._writer.writerow([])
            self._writer.writerow(
                [
                    "# PEAK_TOTAL_MB",
                    self.peak_total_mb,
                    "phase",
                    self.peak_phase,
                    "at_s",
                    self.peak_time_s,
                ]
            )
            self._f.flush()
            self._f.close()
        if _ACTIVE is self:
            _ACTIVE = None


def set_phase(name: str) -> None:
    """Set the current phase label on the active monitor (no-op if none)."""
    if _ACTIVE is not None:
        _ACTIVE.set_phase(name)


def note(msg: str) -> None:
    """Record a one-off event on the active monitor (no-op if none)."""
    if _ACTIVE is not None:
        _ACTIVE.note(msg)
