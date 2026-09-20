#!/usr/bin/env python3
"""One-file entry point; installation is explicit, never a side effect of analysis."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def python_in(environment: Path) -> Path:
    # Do not resolve this symlink: doing so escapes the virtual environment.
    return environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def command_line(argv: list[str], root: Path = ROOT) -> list[str]:
    args = list(argv)
    explicit_config = any(x == "--config" or x.startswith("--config=") for x in args)
    if not explicit_config:
        config = root / ".tcr/runtime.json"
        for i, arg in enumerate(args):
            device = (args[i + 1] if arg == "--device" and i + 1 < len(args)
                      else arg.partition("=")[2] if arg.startswith("--device=") else "")
            if device:
                device = device.strip().lower()
                name = ("gpu" if device.startswith("cuda") else
                        "apple" if device == "mlx" else device)
                candidate = root / ".tcr" / f"runtime-{name}.json"
                if candidate.is_file():
                    config = candidate
        if config.is_file():
            args = ["--config", str(config), *args]
    return [str(python_in(root / ".tcr/envs/core")), "-m", "tcr_workbench", *args]


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in {"setup", "doctor"}:
        import bootstrap
        return bootstrap.main(args, root=ROOT)
    interpreter = python_in(ROOT / ".tcr/envs/core")
    if not args or args in (["--help"], ["-h"]):
        print("TCR-Workbench — local antigen hypothesis workflows\n\n"
              "First use:\n"
              "  python3 tcr.py setup --device apple   # Apple Silicon MLX, ESM-C 300M\n"
              "  python3 tcr.py setup --device cpu     # CPU, ESM-C 300M\n"
              "  python3 tcr.py setup --core-only      # reference matching; no model\n"
              "  python3 tcr.py doctor                 # check local setup\n\n"
              "New checkpoints: python3 tcr.py prepare-model --help\n\n"
              "Workflows: pmhc-score, pmhc-profile, tcr-score, tcr-profile,\n"
              "           repertoire-score, screen, validate\n"
              "Use COMMAND --help for flags. See README.md and docs/usage.md.\n",
              flush=True)
        if not interpreter.is_file():
            return 0
        args = ["--help"]
    if not interpreter.is_file():
        print("Setup is needed: python3 tcr.py setup --device cpu (or apple).\n"
              "For reference matching only: python3 tcr.py setup --core-only", file=sys.stderr)
        return 2
    env = dict(os.environ, PYTHONNOUSERSITE="1", TCR_WORKBENCH_PREPARE_DIR=str(ROOT / ".tcr/prepared"))
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    # Preserve the user's working directory so relative input/output paths keep their meaning.
    return subprocess.call(command_line(args), env=env)


if __name__ == "__main__":
    if sys.version_info < (3, 9):
        sys.exit("The launcher needs Python 3.9 or newer; setup provisions Python 3.12 locally.")
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
