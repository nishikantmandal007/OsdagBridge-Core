# Memory handling for macOS only, kept apart from the Linux and Windows paths in memory_guard.py.
# glibc's malloc_trim and /proc do not exist here.
import ctypes
import os
import re
import subprocess

PAGE_LINE = r"{}:\s+(\d+)"


# Ask the system allocator to hand freed memory back: macOS's counterpart of malloc_trim.
def release():
    allocator = ctypes.CDLL("libSystem.B.dylib")
    allocator.malloc_zone_pressure_relief.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    allocator.malloc_zone_pressure_relief.restype = ctypes.c_size_t
    return allocator.malloc_zone_pressure_relief(None, 0)


# This process's resident and virtual memory in MB, as memory_guard's proc_mem_mb gives them on Linux.
def process_mb():
    output = subprocess.run(["ps", "-o", "rss=,vsz=", "-p", str(os.getpid())], capture_output=True, text=True).stdout
    resident_kb, virtual_kb = output.split()
    return int(resident_kb) / 1024.0, int(virtual_kb) / 1024.0


# Memory the system can give out now, in GB: free, inactive and speculative pages.
def available_gb():
    output = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    page_size = os.sysconf("SC_PAGE_SIZE")
    pages = sum(int(re.search(PAGE_LINE.format(name), output).group(1)) for name in ("Pages free", "Pages inactive", "Pages speculative"))
    return pages * page_size / 1024 ** 3


if __name__ == "__main__":
    resident, virtual = process_mb()
    assert 0 < resident <= virtual and available_gb() > 0
    assert isinstance(release(), int)
