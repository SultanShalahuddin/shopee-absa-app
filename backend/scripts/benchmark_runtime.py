"""Measure local single-thread ONNX model load, inference, and process memory."""

from __future__ import annotations

import ctypes
import os
import platform
import statistics
import time
from ctypes import wintypes
from pathlib import Path

from app.config import Settings
from app.services.model_service import ABSAModelService


BACKEND_DIR = Path(__file__).resolve().parents[1]
REVIEW = (
    "Aplikasinya cepat dan mudah digunakan, tetapi kurirnya lambat dan voucher "
    "tidak bisa dipakai."
)


def windows_memory_mib() -> dict[str, float]:
    class ProcessMemoryCountersEx(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    counters = ProcessMemoryCountersEx()
    counters.cb = ctypes.sizeof(counters)
    get_current_process = ctypes.windll.kernel32.GetCurrentProcess
    get_current_process.restype = wintypes.HANDLE
    get_process_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
    get_process_memory_info.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ProcessMemoryCountersEx),
        wintypes.DWORD,
    ]
    get_process_memory_info.restype = wintypes.BOOL
    process = get_current_process()
    success = get_process_memory_info(
        process, ctypes.byref(counters), counters.cb
    )
    if not success:
        raise OSError("GetProcessMemoryInfo gagal.")
    divisor = 1024 * 1024
    return {
        "working_set_mib": counters.WorkingSetSize / divisor,
        "peak_working_set_mib": counters.PeakWorkingSetSize / divisor,
        "private_usage_mib": counters.PrivateUsage / divisor,
    }


def main() -> None:
    settings = Settings(
        _env_file=None,
        model_dir=BACKEND_DIR / "model",
        allow_demo_mode=False,
        supabase_url="",
        supabase_service_role_key="",
    )
    started = time.perf_counter()
    service = ABSAModelService(settings)
    load_ms = (time.perf_counter() - started) * 1000

    durations = []
    for _ in range(10):
        started = time.perf_counter()
        predictions = service.predict(REVIEW)
        durations.append((time.perf_counter() - started) * 1000)
    assert len(predictions) == 8

    print(f"platform={platform.platform()}")
    print("onnx_threads=1")
    print(f"model_load_ms={load_ms:.3f}")
    print(f"first_inference_ms={durations[0]:.3f}")
    print(f"warm_inference_median_ms={statistics.median(durations[1:]):.3f}")
    print(f"warm_inference_max_ms={max(durations[1:]):.3f}")
    if os.name == "nt":
        for name, value in windows_memory_mib().items():
            print(f"{name}={value:.3f}")


if __name__ == "__main__":
    main()
