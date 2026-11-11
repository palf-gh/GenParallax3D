from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw
from psd_tools import PSDImage
from psd_tools.api.layers import Group, PixelLayer

W, H = 160, 120


def _disc(size, colour, alpha=255):
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    ImageDraw.Draw(img).ellipse([2, 2, size[0] - 3, size[1] - 3], fill=(*colour, alpha))
    return img


@pytest.fixture
def psd_path(tmp_path):
    """Background, a group with two layers (one antialiased disc), and a hidden layer."""
    psd = PSDImage.new("RGBA", (W, H))  # RGB documents drop layer alpha in psd-tools
    bg = Image.new("RGBA", (W, H), (40, 90, 160, 255))
    psd.append(PixelLayer.frompil(bg, psd, "background", top=0, left=0))
    grp = Group.new(name="building", parent=psd)
    grp.append(PixelLayer.frompil(Image.new("RGBA", (50, 60), (200, 180, 120, 255)), psd, "wall", top=40, left=20))
    grp.append(PixelLayer.frompil(_disc((40, 40), (220, 40, 40)), psd, "roof", top=20, left=25))
    psd.append(PixelLayer.frompil(_disc((30, 30), (30, 200, 60)), psd, "character", top=70, left=100))
    hidden = PixelLayer.frompil(Image.new("RGBA", (10, 10), (255, 255, 0, 255)), psd, "hidden", top=0, left=0)
    hidden.visible = False
    psd.append(hidden)
    path = tmp_path / "sample.psd"
    psd.save(path)
    return path


@pytest.fixture
def depth_path(tmp_path):
    """Radial nearness map: centre near (white), edges far (black)."""
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.hypot(xx - W / 2, yy - H / 2) / np.hypot(W / 2, H / 2)
    img = Image.fromarray(((1 - r) * 255).astype(np.uint8), "L")
    path = tmp_path / "depth.png"
    img.save(path)
    return path


@pytest.fixture
def haze_psd(tmp_path):
    """Background, a translucent haze band (overlay) and an opaque disc above it."""
    from PIL import ImageDraw
    from psd_tools import PSDImage
    from psd_tools.api.layers import PixelLayer

    psd = PSDImage.new("RGBA", (W, H))
    yy, xx = np.mgrid[0:H, 0:W]
    bg = np.zeros((H, W, 4), np.uint8)
    bg[..., 0] = 60 + xx
    bg[..., 1] = 80 + yy
    bg[..., 2] = 120
    bg[..., 3] = 255
    psd.append(PixelLayer.frompil(Image.fromarray(bg, "RGBA"), psd, "ground", top=0, left=0))
    haze = np.zeros((60, W, 4), np.uint8)
    haze[..., :3] = 245
    haze[..., 3] = (90 + 60 * np.sin(xx[:60] / 9.0)).astype(np.uint8)
    psd.append(PixelLayer.frompil(Image.fromarray(haze, "RGBA"), psd, "haze", top=30, left=0))
    disc = Image.new("RGBA", (40, 40), (0, 0, 0, 0))
    ImageDraw.Draw(disc).ellipse([2, 2, 37, 37], fill=(200, 40, 40, 255))
    psd.append(PixelLayer.frompil(disc, psd, "ball", top=40, left=60))
    path = tmp_path / "haze.psd"
    psd.save(path)
    return path
