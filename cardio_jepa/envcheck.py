"""Pre-flight check for a local (non-Kaggle) run.

``check_environment`` answers one question before hours of training are committed:
*does this machine match what the chosen run mode needs?* It verifies

  * Python and package versions against ``pyproject.toml``;
  * that PyTorch can really execute a CUDA kernel on the installed GPU. A wheel built
    for older architectures imports fine and reports ``cuda.is_available() == True`` on
    a new card, then fails on the first op -- so we run one rather than trust the flag;
  * RAM, free disk on the output drive, and VRAM;
  * that every configured dataset directory exists and holds ECG files.

Hard failures are returned in ``report["errors"]`` and make ``report["ok"]`` False;
anything merely suboptimal goes in ``report["warnings"]``.
"""

from __future__ import annotations

import importlib
import os
import platform
import shutil
import sys
from typing import Dict, List, Optional, Tuple

# name -> (minimum version, required?)
REQUIREMENTS: Dict[str, Tuple[str, bool]] = {
    "torch": ("2.0", True), "numpy": ("1.22", True), "scipy": ("1.9", True),
    "sklearn": ("1.1", True), "pandas": ("1.5", True),
    "matplotlib": ("3.5", False),   # Stage 3 plots
    "h5py": ("3.7", False),         # CODE-15 only
    "wfdb": ("4.0", False),         # optional: the bundled reader covers these corpora
}

#: What each run mode needs from the machine (free RAM / output-disk GiB, min VRAM GiB).
#: These are floors for "will it plausibly work", not predictions of speed -- the
#: runtime-estimate cell measures real step time.
MODE_NEEDS = {
    "smoke":  {"ram": 4,  "disk": 2,   "vram": 0},
    "kaggle": {"ram": 12, "disk": 20,  "vram": 8},
    "full":   {"ram": 24, "disk": 100, "vram": 24},
}


def _vtuple(v: str) -> Tuple[int, ...]:
    out = []
    for p in v.replace("+", ".").split("."):
        if p.isdigit():
            out.append(int(p))
        else:
            break
    return tuple(out)


def _ram_gib() -> Tuple[Optional[float], Optional[float]]:
    """(total, available) physical memory in GiB, or (None, None)."""
    try:
        if os.name == "nt":
            import ctypes

            class _MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
            ms = _MS()
            ms.dwLength = ctypes.sizeof(_MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
            return ms.ullTotalPhys / 2**30, ms.ullAvailPhys / 2**30
        pages, avail = os.sysconf("SC_PHYS_PAGES"), os.sysconf("SC_AVPHYS_PAGES")
        size = os.sysconf("SC_PAGE_SIZE")
        return pages * size / 2**30, avail * size / 2**30
    except Exception:
        return None, None


def check_environment(datasets: Dict[str, str], run_mode: str = "full",
                      out_dir: str = ".", verbose: bool = True) -> Dict[str, object]:
    errors: List[str] = []
    warnings: List[str] = []
    info: Dict[str, object] = {}
    need = MODE_NEEDS.get(run_mode, MODE_NEEDS["full"])

    def say(msg: str = ""):
        if verbose:
            print(msg)

    # ---- python / packages -------------------------------------------------------------
    say(f"python   {platform.python_version()}  ({platform.system()} {platform.release()})")
    if sys.version_info < (3, 9):
        errors.append(f"Python >= 3.9 required, found {platform.python_version()}")

    mods = {}
    for name, (minv, required) in REQUIREMENTS.items():
        try:
            m = importlib.import_module(name)
            ver = str(getattr(m, "__version__", "?"))
            mods[name] = m
            ok = _vtuple(ver) >= _vtuple(minv) if _vtuple(ver) else True
            say(f"  {name:11s} {ver:14s} {'ok' if ok else f'< {minv} REQUIRED'}")
            if not ok:
                (errors if required else warnings).append(
                    f"{name} {ver} is older than the required {minv}")
        except ImportError:
            say(f"  {name:11s} {'missing':14s} {'REQUIRED' if required else '(optional)'}")
            if required:
                errors.append(f"required package {name!r} is not installed "
                              f"(pip install -e .[data,dev])")
            elif name == "wfdb":
                say("              -> fine: the bundled dependency-free reader is used")
            elif name == "matplotlib":
                warnings.append("matplotlib missing: Stage 3 plots will be skipped")

    # ---- GPU ---------------------------------------------------------------------------
    torch = mods.get("torch")
    info["cuda"] = False
    if torch is not None:
        say(f"\ntorch    {torch.__version__}   built for CUDA {torch.version.cuda}")
        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(idx)
            vram = props.total_memory / 2**30
            cap = f"sm_{props.major}{props.minor}"
            say(f"gpu      {props.name}  {vram:.1f} GiB  compute {props.major}.{props.minor}")
            archs = torch.cuda.get_arch_list()
            try:
                # run a real kernel: arch mismatch only shows up at execution time
                a = torch.randn(256, 256, device="cuda")
                float((a @ a).sum())
                with torch.autocast("cuda", dtype=torch.float16):
                    float((a @ a).sum())
                torch.cuda.synchronize()
                say(f"  kernel test  ok (matmul fp32 + fp16 autocast on {cap})")
                info["cuda"] = True
            except Exception as exc:
                errors.append(f"CUDA is visible but a test kernel failed on {cap}: "
                              f"{type(exc).__name__}: {exc}. This torch build supports "
                              f"{archs}; install a wheel that includes {cap}.")
            say(f"  bf16 supported: {torch.cuda.is_bf16_supported()}")
            info.update(gpu=props.name, vram_gib=round(vram, 1), arch=cap)
            if vram < need["vram"]:
                warnings.append(f"{vram:.0f} GiB VRAM is below the ~{need['vram']} GiB "
                                f"suggested for '{run_mode}'; lower optim.batch_size if "
                                f"you hit out-of-memory")
        else:
            say("gpu      none visible to torch")
            if run_mode != "smoke":
                errors.append(f"no CUDA device available but run mode is '{run_mode}'. "
                              f"Training the {run_mode} model on CPU is impractical; "
                              f"use RUN_MODE='smoke' or install a CUDA build of torch.")

    # ---- RAM / disk --------------------------------------------------------------------
    total, avail = _ram_gib()
    if total is not None:
        say(f"\nram      {total:.1f} GiB total, {avail:.1f} GiB free")
        info.update(ram_gib=round(total, 1))
        if avail < need["ram"]:
            warnings.append(f"only {avail:.1f} GiB RAM free; '{run_mode}' suggests "
                            f"{need['ram']}+ GiB (close other programs)")
    os.makedirs(out_dir, exist_ok=True)
    free = shutil.disk_usage(out_dir).free / 2**30
    say(f"disk     {free:.0f} GiB free on the drive holding {out_dir}")
    if free < need["disk"]:
        warnings.append(f"{free:.0f} GiB free for checkpoints; '{run_mode}' suggests "
                        f"{need['disk']}+ GiB")
    cpu = os.cpu_count() or 1
    say(f"cpu      {cpu} logical cores")
    info["cpu"] = cpu

    # ---- datasets ----------------------------------------------------------------------
    say("\ndatasets")
    for key, path in datasets.items():
        if not os.path.isdir(path):
            say(f"  {key:8s} MISSING  {path}")
            warnings.append(f"dataset '{key}' not found at {path}")
            continue
        n = sum(1 for e in os.scandir(path) if e.is_file())
        sub = sum(1 for e in os.scandir(path) if e.is_dir())
        say(f"  {key:8s} ok       {n} files, {sub} subdirs   {path}")
        if n == 0 and sub == 0:
            warnings.append(f"dataset '{key}' directory is empty: {path}")
    if not any(os.path.isdir(p) for p in datasets.values()):
        errors.append("none of the configured dataset directories exist")

    ok = not errors
    say("\n" + ("ENVIRONMENT OK for run mode '%s'" % run_mode if ok
                else "ENVIRONMENT NOT READY:"))
    for e in errors:
        say(f"  ERROR   {e}")
    for w in warnings:
        say(f"  warning {w}")
    return {"ok": ok, "errors": errors, "warnings": warnings, "info": info}


if __name__ == "__main__":                                   # pragma: no cover
    root = sys.argv[1] if len(sys.argv) > 1 else r"D:\Ishtiaque\Dataset"
    mode = sys.argv[2] if len(sys.argv) > 2 else "full"
    sub = {d: os.path.join(root, d) for d in os.listdir(root)
           if os.path.isdir(os.path.join(root, d))} if os.path.isdir(root) else {}
    check_environment(sub, mode, out_dir=os.getcwd())
