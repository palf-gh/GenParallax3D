# PyInstaller spec for the standalone GenParallax3D app (Windows folder / macOS .app).
#
# Build with:  python packaging/build.py
# (or directly: pyinstaller --noconfirm --clean packaging/GenParallax3D.spec)
#
# One-folder mode on purpose: it starts faster than one-file, upsets virus scanners
# less, and lets spawned worker processes start without unpacking everything again.

import re
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules

ROOT = Path(SPECPATH).parent
APP_NAME = "GenParallax3D"
VERSION = re.search(r'__version__\s*=\s*"([^"]+)"', (ROOT / "genparallax" / "__init__.py").read_text()).group(1)

# The optional AI extra is never bundled: it would add gigabytes. The app checks for it
# at run time and explains how to get depth maps without it. Listing it here also keeps
# a developer's environment (where torch may be installed) from leaking into the build.
AI_EXCLUDES = [
    "torch", "torchvision", "torchaudio", "transformers", "sentence_transformers", "tokenizers",
    "safetensors", "huggingface_hub", "accelerate", "timm", "tensorflow", "jax", "onnxruntime",
]
# Qt modules the app does not use (it needs QtCore, QtGui, QtWidgets, QtOpenGL(Widgets)).
QT_EXCLUDES = [
    f"PySide6.{m}" for m in (
        "Qt3DAnimation", "Qt3DCore", "Qt3DExtras", "Qt3DInput", "Qt3DLogic", "Qt3DRender",
        "QtBluetooth", "QtCharts", "QtDataVisualization", "QtDesigner", "QtGraphs", "QtHelp",
        "QtLocation", "QtMultimedia", "QtMultimediaWidgets", "QtNetworkAuth", "QtNfc", "QtPdf",
        "QtPdfWidgets", "QtPositioning", "QtQml", "QtQuick", "QtQuick3D", "QtQuickControls2",
        "QtQuickWidgets", "QtRemoteObjects", "QtScxml", "QtSensors", "QtSerialBus", "QtSerialPort",
        "QtSpatialAudio", "QtSql", "QtStateMachine", "QtSvgWidgets", "QtTest", "QtTextToSpeech",
        "QtUiTools", "QtWebChannel", "QtWebEngineCore", "QtWebEngineQuick", "QtWebEngineWidgets",
        "QtWebSockets", "QtWebView", "QtXml",
    )
]
OTHER_EXCLUDES = ["tkinter", "_tkinter", "matplotlib", "IPython", "pytest", "PyQt5", "PyQt6", "PySide2"]

a = Analysis(
    [str(ROOT / "packaging" / "launcher.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=[(str(ROOT / "LICENSE"), "."), (str(ROOT / "THIRD_PARTY_NOTICES.md"), ".")],
    # moderngl picks its OpenGL loader (glcontext backend) by name at run time.
    hiddenimports=collect_submodules("glcontext") + ["genparallax.selftest"],
    hookspath=[],
    runtime_hooks=[],
    excludes=AI_EXCLUDES + QT_EXCLUDES + OTHER_EXCLUDES,
    noarchive=False,
)


# Qt plug-ins pull in libraries the app never uses (the PDF image-format plug-in brings
# Qt Pdf, Qml and Quick; the virtual keyboard brings Quick too). Drop them, and keep only
# the Japanese Qt translations (English needs none).
_UNUSED_QT = re.compile(
    r"(Qt6?(Quick|Qml|Pdf|VirtualKeyboard)\w*(\.dll|\.framework|\.so)"
    r"|(lib)?qpdf\.(dll|dylib)|(lib)?qtvirtualkeyboardplugin\.(dll|dylib))",
    re.IGNORECASE,
)


def _keep(entry) -> bool:
    dest = entry[0].replace("\\", "/")
    if _UNUSED_QT.search(dest):
        return False
    if "/translations/" in dest and dest.endswith(".qm"):
        name = dest.rsplit("/", 1)[-1]
        return name in ("qt_ja.qm", "qtbase_ja.qm")
    return True


a.binaries = [b for b in a.binaries if _keep(b)]
a.datas = [d for d in a.datas if _keep(d)]

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    strip=False,
    upx=False,
    console=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name=APP_NAME)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name=f"{APP_NAME}.app",
        bundle_identifier="io.github.palf-gh.genparallax3d",
        version=VERSION,
        info_plist={
            "CFBundleName": APP_NAME,
            "CFBundleDisplayName": APP_NAME,
            "CFBundleShortVersionString": VERSION,
            "CFBundleVersion": VERSION,
            "NSHighResolutionCapable": True,
            "LSMinimumSystemVersion": "12.0",
        },
    )
