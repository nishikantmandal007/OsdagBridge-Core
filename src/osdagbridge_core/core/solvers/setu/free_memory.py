import ctypes
import os
import re
import subprocess
import sys

MACOS_PAGE_LINE = r"{}:\s+(\d+)"


# Memory the system can give out now, in GB, or None where it cannot be asked: sizes setu's workers.
def available_memory_gb():
    try:
        if sys.platform == "darwin":  # free, inactive and speculative pages
            output = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
            pages = sum(int(re.search(MACOS_PAGE_LINE.format(name), output).group(1)) for name in ("Pages free", "Pages inactive", "Pages speculative"))
            return pages * os.sysconf("SC_PAGE_SIZE") / 1024 ** 3
        if sys.platform.startswith("linux"):
            with open("/proc/meminfo") as meminfo:
                return next(int(line.split()[1]) for line in meminfo if line.startswith("MemAvailable:")) / 1024 ** 2
        if sys.platform == "win32":
            return windows_available_gb()
    except (OSError, ValueError, AttributeError, StopIteration):
        pass
    return None


# Physical memory Windows can give out now (GlobalMemoryStatusEx), in GB.
def windows_available_gb():
    class MemoryStatus(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total_physical", ctypes.c_ulonglong),
                    ("available_physical", ctypes.c_ulonglong), ("total_page_file", ctypes.c_ulonglong),
                    ("available_page_file", ctypes.c_ulonglong), ("total_virtual", ctypes.c_ulonglong),
                    ("available_virtual", ctypes.c_ulonglong), ("available_extended_virtual", ctypes.c_ulonglong)]

    status = MemoryStatus()
    status.length = ctypes.sizeof(MemoryStatus)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return status.available_physical / 1024 ** 3
