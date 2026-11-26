"""Worker-count limits based on free memory, so a big machine's core count
doesn't translate into dozens of numpy-importing processes that exhaust RAM
or the Windows paging file."""

from __future__ import annotations

import os

# Rough resident cost of one spawned worker (interpreter + numpy + skyfield +
# ephemeris + its slice of work), with headroom.
WORKER_MB = 800
_MAX_AUTO_WORKERS = 8


def available_memory_mb() -> "float | None":
    """Free physical RAM + page-file headroom (MB) on Windows, available RAM
    on Linux; None if unknown."""
    try:
        if os.name == "nt":
            import ctypes

            class _MS(ctypes.Structure):
                _fields_ = [(n, ctypes.c_ulonglong) for n in (
                    "dwLength_pad", "total_phys", "avail_phys", "total_page", "avail_page", "total_virt", "avail_virt", "ext")]

            class _MEM(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("total_phys", ctypes.c_ulonglong), ("avail_phys", ctypes.c_ulonglong),
                            ("total_page", ctypes.c_ulonglong), ("avail_page", ctypes.c_ulonglong),
                            ("total_virt", ctypes.c_ulonglong), ("avail_virt", ctypes.c_ulonglong),
                            ("avail_ext", ctypes.c_ulonglong)]

            m = _MEM()
            m.dwLength = ctypes.sizeof(_MEM)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            # avail_page is the commit limit left -- the quantity that actually
            # runs out ("paging file is too small").
            return min(m.avail_phys, m.avail_page) / 2**20
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024
    except Exception:  # noqa: BLE001
        pass
    return None


def safe_workers(requested: "int | None") -> int:
    """`requested` workers (None = auto from CPU count), clamped by CPU count,
    a cap of 8 when auto, and available memory."""
    cpus = max(1, (os.cpu_count() or 1) - 1)
    n = min(requested, cpus) if requested else min(cpus, _MAX_AUTO_WORKERS)
    mem = available_memory_mb()
    if mem is not None:
        n = min(n, max(1, int(mem * 0.7 // WORKER_MB)))
    return max(1, n)
