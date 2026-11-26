"""Execute a notebook headless in a chosen Python env, cell by cell.

* the executed notebook is saved after **every** cell (so outputs survive a crash),
* everything the kernel prints is also appended live to a log file (tail it to watch),
* ``--set 'OLD=>NEW'`` patches cell sources in the executed copy only (the source
  notebook is never modified), e.g. ``--set 'RUN_MODE = "full"=>RUN_MODE = "smoke"'``.

Usage (needs nbclient/nbformat in the interpreter running *this* script; the kernel runs
in ``--python``):

    python tools/run_notebook.py notebooks/cardio_jepa_local_v2.ipynb \
        --out notebooks/cardio_jepa_local_v2.executed.ipynb \
        --log runs/notebook_run.log \
        --python C:/ProgramData/anaconda3/envs/CardioJepa/python.exe
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time


def _make_kernelspec(python: str) -> str:
    root = tempfile.mkdtemp(prefix="nbrun_")
    kdir = os.path.join(root, "kernels", "nbrun")
    os.makedirs(kdir)
    with open(os.path.join(kdir, "kernel.json"), "w") as fh:
        json.dump({"argv": [python, "-Xfrozen_modules=off", "-m", "ipykernel_launcher",
                            "-f", "{connection_file}"],
                   "display_name": "nbrun", "language": "python"}, fh)
    return root


TEE = r'''
import sys as _s, io as _io
class _Tee:
    def __init__(self, a, path):
        self._a, self._f = a, open(path, "a", encoding="utf-8", buffering=1)
    def write(self, x):
        try: self._f.write(x)
        except Exception: pass
        return self._a.write(x)
    def flush(self):
        try: self._f.flush()
        except Exception: pass
        return self._a.flush()
    def __getattr__(self, k): return getattr(self._a, k)
if not isinstance(_s.stdout, _Tee):
    _s.stdout = _Tee(_s.stdout, LOGPATH)
    _s.stderr = _Tee(_s.stderr, LOGPATH)
del _io
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook")
    ap.add_argument("--out", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--python", required=True, help="interpreter the kernel runs in")
    ap.add_argument("--set", action="append", default=[], help="'OLD=>NEW' source patch")
    ap.add_argument("--start", type=int, default=0, help="first cell index to execute")
    args = ap.parse_args()

    os.environ["JUPYTER_PATH"] = _make_kernelspec(args.python) + os.pathsep \
        + os.environ.get("JUPYTER_PATH", "")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    import nbformat
    from nbclient import NotebookClient

    nb = nbformat.read(args.notebook, as_version=4)
    for patch in args.set:
        old, new = patch.split("=>", 1)
        hits = 0
        for c in nb.cells:
            if c.cell_type == "code" and old in c.source:
                c.source = c.source.replace(old, new)
                hits += 1
        if not hits:
            sys.exit(f"--set pattern not found: {old!r}")
        print(f"patched {hits} cell(s): {old!r} -> {new!r}")

    log = os.path.abspath(args.log)
    os.makedirs(os.path.dirname(log), exist_ok=True)

    def note(msg):
        line = f"\n##### [{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n"
        with open(log, "a", encoding="utf-8") as fh:
            fh.write(line)
        print(line.strip(), flush=True)

    client = NotebookClient(nb, timeout=None, kernel_name="nbrun", allow_errors=False,
                            resources={"metadata": {"path": os.path.dirname(
                                os.path.abspath(args.notebook))}})
    t0 = time.time()
    with client.setup_kernel():
        client.kc.execute_interactive(TEE.replace("LOGPATH", repr(log)), timeout=60)
        for i, cell in enumerate(nb.cells):
            if cell.cell_type != "code":
                continue
            if i < args.start:
                continue
            first = (cell.source.strip().splitlines() or [""])[0][:90]
            note(f"cell {i} start: {first}")
            tc = time.time()
            try:
                client.execute_cell(cell, i)
            except Exception as exc:
                nbformat.write(nb, args.out)
                note(f"cell {i} FAILED after {time.time() - tc:.0f}s: "
                     f"{type(exc).__name__}: {str(exc)[-2000:]}")
                sys.exit(1)
            nbformat.write(nb, args.out)
            note(f"cell {i} done in {time.time() - tc:.0f}s "
                 f"(total {(time.time() - t0) / 3600:.2f} h)")
    note(f"ALL CELLS DONE in {(time.time() - t0) / 3600:.2f} h")


if __name__ == "__main__":
    main()
