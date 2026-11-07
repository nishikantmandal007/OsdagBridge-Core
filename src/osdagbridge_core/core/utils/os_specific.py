# Everything in OsdagBridge-Core that differs between Windows, macOS and Linux, in one place: debugging a platform starts here.
import ctypes
import os
import re
import subprocess
import sys
from pathlib import Path

MACOS_PAGE_LINE = r"{}:\s+(\d+)"


# Hand memory the process has freed back to the system: macOS's malloc_zone_pressure_relief, glibc's malloc_trim on Linux.
# gc.collect() frees objects into the allocator, which does not return it by itself. Best effort: nothing elsewhere.
def release_freed_memory():
    try:
        if sys.platform == "darwin":
            allocator = ctypes.CDLL("libSystem.B.dylib")
            allocator.malloc_zone_pressure_relief.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            allocator.malloc_zone_pressure_relief.restype = ctypes.c_size_t
            allocator.malloc_zone_pressure_relief(None, 0)
            return
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


# This process's resident and virtual memory in MB, or (None, None) where it cannot be read.
def process_mb():
    rss = virt = None
    try:
        if sys.platform == "darwin":
            output = subprocess.run(["ps", "-o", "rss=,vsz=", "-p", str(os.getpid())], capture_output=True, text=True).stdout
            resident_kb, virtual_kb = output.split()
            return int(resident_kb) / 1024.0, int(virtual_kb) / 1024.0
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss = int(line.split()[1]) / 1024.0
                elif line.startswith("VmSize:"):
                    virt = int(line.split()[1]) / 1024.0
    except Exception:
        pass
    return rss, virt


# Memory the system can give out now, in GB, or None where it cannot be asked.
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


# The Windows console is not UTF-8 by default: print there without crashing on non-ASCII text.
def utf8_console():
    if sys.platform.startswith("win"):
        sys.stdout.reconfigure(encoding="utf-8", errors="ignore")
        sys.stderr.reconfigure(encoding="utf-8", errors="ignore")


_forkserver_preloaded = False


# A start method for design workers that is safe beside the GUI's threads: forkserver (its server starts by fork+exec, so workers
# fork from a clean single-threaded state, with these modules preloaded once), or spawn where there is no forkserver (Windows).
def design_process_context(preload):
    import multiprocessing
    global _forkserver_preloaded
    try:
        context = multiprocessing.get_context("forkserver")
        if not _forkserver_preloaded:
            context.set_forkserver_preload(preload)
            _forkserver_preloaded = True
        return context
    except ValueError:
        return multiprocessing.get_context("spawn")


# The user's Documents folder: OneDrive's on Windows when there is no local one, XDG_DOCUMENTS_DIR on Linux, home otherwise.
def documents_folder():
    if sys.platform.startswith("win"):
        docs_path = Path.home() / "Documents"
        if not docs_path.exists():
            docs_path = Path.home() / "OneDrive" / "Documents"
    elif sys.platform == "darwin":
        docs_path = Path.home() / "Documents"
    elif sys.platform.startswith("linux"):
        xdg_docs = os.environ.get("XDG_DOCUMENTS_DIR")
        docs_path = Path(xdg_docs) if xdg_docs else Path.home() / "Documents"
    else:
        docs_path = Path.home()
    if not docs_path.exists():
        docs_path = Path.home()
    return str(docs_path)


# subprocess options that keep a console window from flashing up on Windows; none elsewhere.
def no_console_window():
    return {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
