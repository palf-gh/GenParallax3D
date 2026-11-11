"""Build the standalone app and an archive ready for a GitHub Release.

Usage (from the repository root, in an environment with the app's dependencies,
PyInstaller, and *without* the AI extra):

    python packaging/build.py              # build, check, self-test, archive
    python packaging/build.py --no-gl      # self-test without OpenGL (CI runners)
    python packaging/build.py --skip-self-test

Results:
    Windows: dist/GenParallax3D/GenParallax3D.exe  and  dist/GenParallax3D-<version>-windows-<arch>.zip
    macOS:   dist/GenParallax3D.app                and  dist/GenParallax3D-<version>-macos-<arch>.zip
"""

from __future__ import annotations

import argparse
import platform
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"
APP_NAME = "GenParallax3D"
FORBIDDEN = ("torch", "transformers", "torchvision")  # the AI extra must never be bundled


def version() -> str:
    text = (ROOT / "genparallax" / "__init__.py").read_text(encoding="utf-8")
    return re.search(r'__version__\s*=\s*"([^"]+)"', text).group(1)


def arch() -> str:
    machine = platform.machine().lower()
    return {"amd64": "x64", "x86_64": "x64", "arm64": "arm64", "aarch64": "arm64"}.get(machine, machine)


def run_pyinstaller() -> None:
    subprocess.run([sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean",
                    "--distpath", str(DIST), "--workpath", str(ROOT / "build" / "pyinstaller"),
                    str(ROOT / "packaging" / f"{APP_NAME}.spec")], check=True, cwd=ROOT)


def app_paths() -> tuple[Path, Path]:
    """(thing to archive, executable)."""
    if sys.platform == "darwin":
        app = DIST / f"{APP_NAME}.app"
        return app, app / "Contents" / "MacOS" / APP_NAME
    folder = DIST / APP_NAME
    return folder, folder / (f"{APP_NAME}.exe" if sys.platform == "win32" else APP_NAME)


def check_contents(bundle: Path) -> None:
    bad = sorted({p.name for p in bundle.rglob("*") if p.is_dir() and p.name.lower() in FORBIDDEN})
    if bad:
        raise SystemExit(f"error: the AI extra leaked into the build: {', '.join(bad)}")
    size = sum(p.stat().st_size for p in bundle.rglob("*") if p.is_file() and not p.is_symlink())
    print(f"bundle: {bundle} ({size / 2**20:.0f} MiB)")


def self_test(exe: Path, no_gl: bool) -> None:
    report = ROOT / "build" / "self-test.txt"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.unlink(missing_ok=True)
    args = [str(exe), "--self-test", str(report)] + (["--no-gl"] if no_gl else [])
    # The windowed executable has no console; the exit code and the report carry the result.
    result = subprocess.run(args, timeout=300)
    if report.exists():
        print(report.read_text(encoding="utf-8"), end="")
    if result.returncode != 0:
        raise SystemExit(f"error: self-test failed with exit code {result.returncode}")


def archive(bundle: Path) -> Path:
    plat = {"win32": "windows", "darwin": "macos"}.get(sys.platform, sys.platform)
    out = DIST / f"{APP_NAME}-{version()}-{plat}-{arch()}.zip"
    out.unlink(missing_ok=True)
    if sys.platform == "darwin":
        # ditto keeps the framework symlinks and the code signature intact; zipfile would not.
        subprocess.run(["ditto", "-c", "-k", "--sequesterRsrc", "--keepParent", str(bundle), str(out)], check=True)
    else:
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
            for path in sorted(bundle.rglob("*")):
                zf.write(path, path.relative_to(bundle.parent))
    print(f"archive: {out} ({out.stat().st_size / 2**20:.0f} MiB)")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-gl", action="store_true", help="skip the OpenGL part of the self-test")
    parser.add_argument("--skip-self-test", action="store_true")
    parser.add_argument("--no-archive", action="store_true")
    args = parser.parse_args()

    for name in (f"{APP_NAME}", f"{APP_NAME}.app"):
        shutil.rmtree(DIST / name, ignore_errors=True)
    run_pyinstaller()
    bundle, exe = app_paths()
    check_contents(bundle)
    if not args.skip_self_test:
        self_test(exe, args.no_gl)
    if not args.no_archive:
        archive(bundle)
    return 0


if __name__ == "__main__":
    sys.exit(main())
