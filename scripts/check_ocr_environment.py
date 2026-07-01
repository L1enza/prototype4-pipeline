#!/usr/bin/env python3
"""Inspect lightweight OCR availability without installing packages or models."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import platform
import shutil
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = "outputs/nll_test4/jersey_number_ocr_baseline/ocr_environment_check.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check whether lightweight Tesseract OCR can run.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def run_command(command: list[str], timeout: int = 15) -> dict:
    executable = shutil.which(command[0]) if "/" not in command[0] else command[0]
    if not executable or not Path(executable).exists():
        return {
            "command": command,
            "available": False,
            "path": None,
            "returncode": None,
            "stdout": "",
            "stderr": "command not found",
        }
    try:
        result = subprocess.run(
            [executable, *command[1:]],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        return {
            "command": command,
            "available": True,
            "path": executable,
            "returncode": result.returncode,
            "stdout": stdout[:2000],
            "stderr": stderr[:2000],
            "output_truncated": len(stdout) > 2000 or len(stderr) > 2000,
        }
    except Exception as exc:
        return {
            "command": command,
            "available": True,
            "path": executable,
            "returncode": None,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
        }


def package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def read_os_release() -> dict:
    path = Path("/etc/os-release")
    values = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key] = value.strip().strip('"')
    return values


def check_module_tesseract() -> dict:
    modulecmd = shutil.which("modulecmd")
    if not modulecmd:
        return {
            "module_system_available": False,
            "modulecmd": None,
            "query": "module avail tesseract",
            "query_returncode": None,
            "query_output": "",
            "tesseract_module_available": False,
        }
    result = subprocess.run(
        [modulecmd, "sh", "avail", "tesseract"],
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )
    output = "\n".join(part for part in (result.stdout.strip(), result.stderr.strip()) if part)
    lowered = output.casefold()
    module_available = "tesseract" in lowered and "no module" not in lowered
    return {
        "module_system_available": True,
        "modulecmd": modulecmd,
        "query": "module avail tesseract",
        "query_returncode": result.returncode,
        "query_output": output,
        "tesseract_module_available": module_available,
    }


def main() -> int:
    args = parse_args()
    output_path = project_path(args.output)
    tesseract_path = shutil.which("tesseract")
    tesseract_which = {
        "command": ["which", "tesseract"],
        "available": bool(tesseract_path),
        "path": "/usr/bin/which" if shutil.which("which") else None,
        "returncode": 0 if tesseract_path else 1,
        "stdout": tesseract_path or "",
        "stderr": "" if tesseract_path else "tesseract not found on PATH",
    }
    tesseract_version = run_command(["tesseract", "--version"])
    pytesseract_available = importlib.util.find_spec("pytesseract") is not None
    module_check = check_module_tesseract()
    conda = run_command(["conda", "--version"])
    mamba = run_command(["mamba", "--version"])
    micromamba = run_command(["micromamba", "--version"])
    dnf_path = shutil.which("dnf")
    dnf = {"available": bool(dnf_path), "path": dnf_path, "command_executed": False}
    os_release = read_os_release()

    tesseract_binary_available = bool(
        tesseract_version["available"] and tesseract_version["returncode"] == 0
    )
    ocr_can_run_now = tesseract_binary_available
    baseline_command = (
        ".venv/bin/python scripts/run_jersey_number_ocr_baseline.py --engine tesseract"
        if ocr_can_run_now
        else None
    )

    if tesseract_binary_available and not pytesseract_available:
        minimal_install_command = ".venv/bin/python -m pip install pytesseract"
        install_note = (
            "Optional Python wrapper only. The current baseline can already run through the "
            "Tesseract CLI without pytesseract."
        )
    elif pytesseract_available and not tesseract_binary_available:
        minimal_install_command = "sudo dnf install -y tesseract"
        install_note = (
            "pytesseract is only a wrapper; OCR cannot run until the system Tesseract binary is installed."
        )
    elif not tesseract_binary_available:
        minimal_install_command = "sudo dnf install -y tesseract"
        install_note = (
            "ECE is RHEL 8 and no Tesseract environment module or conda/mamba executable was found. "
            "This command requires administrator privileges; otherwise request that ECE support install "
            "or expose a Tesseract module. pytesseract is not required by the current CLI-based baseline."
        )
    else:
        minimal_install_command = None
        install_note = "No installation is needed for the current CLI-based baseline."

    payload = {
        "status": "ready" if ocr_can_run_now else "unavailable",
        "stage": "ocr_environment_check",
        "checks_are_read_only": True,
        "packages_installed": False,
        "models_downloaded": False,
        "system": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "os_release": os_release,
        },
        "tesseract": {
            "binary_available": tesseract_binary_available,
            "which": tesseract_which,
            "version": tesseract_version,
        },
        "pytesseract": {
            "import_available": pytesseract_available,
            "version": package_version("pytesseract"),
            "required_by_current_baseline": False,
        },
        "environment_options": {
            "module": module_check,
            "conda": conda,
            "mamba": mamba,
            "micromamba": micromamba,
            "dnf": dnf,
        },
        "ocr_can_run_now": ocr_can_run_now,
        "next_baseline_command": baseline_command,
        "minimal_install_command": minimal_install_command,
        "optional_pytesseract_install_command": ".venv/bin/python -m pip install pytesseract",
        "install_note": install_note,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**payload, "output": str(output_path)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
