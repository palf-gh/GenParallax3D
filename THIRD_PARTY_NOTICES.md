# Third-party notices

GenParallax3D itself is released under the MIT License (see `LICENSE`). It uses the
following third-party software and models, which keep their own licences.

## Bundled in the standalone app

| Component | Licence | Notes |
|---|---|---|
| [Qt for Python (PySide6, shiboken6)](https://www.qt.io/qt-for-python) and Qt 6 | LGPL-3.0 | Shipped as separate shared libraries in the app folder, so they can be replaced. Source: <https://code.qt.io/> |
| [NumPy](https://numpy.org/) | BSD-3-Clause | |
| [SciPy](https://scipy.org/) | BSD-3-Clause | Includes further permissively licensed libraries (see SciPy's `LICENSE.txt`) |
| [Pillow](https://python-pillow.org/) | MIT-CMU (HPND) | |
| [psd-tools](https://github.com/psd-tools/psd-tools) | MIT | |
| [ModernGL](https://github.com/moderngl/moderngl) and glcontext | MIT | |
| [pygltflib](https://gitlab.com/dodgyville/pygltflib) | MIT | |
| [CPython](https://www.python.org/) | PSF License | Python runtime embedded by PyInstaller |
| [PyInstaller](https://pyinstaller.org/) bootloader | GPL-2.0 with a bootloader exception | The exception allows distributing apps built with it under any licence |

## Optional, not bundled (AI depth estimation from source)

| Component | Licence | Notes |
|---|---|---|
| [Depth Anything V2 Small](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf) model weights | Apache-2.0 | Downloaded from Hugging Face on first use. Only the Small model is used: the Base and Large models are under CC-BY-NC-4.0 (non-commercial) and are not used by this app. |
| [PyTorch](https://pytorch.org/) | BSD-3-Clause | |
| [Hugging Face Transformers](https://github.com/huggingface/transformers) | Apache-2.0 | |

Depth Anything V2: Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu,
Jiashi Feng, Hengshuang Zhao. *Depth Anything V2*. arXiv:2406.09414, 2024.
