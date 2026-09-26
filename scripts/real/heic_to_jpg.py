"""Convert iPhone HEIC photos to JPEG for COLMAP.

EXIF orientation is baked into the pixels (COLMAP ignores the orientation tag) and the tag reset to 1;
the rest of EXIF (focal length etc.) is kept. Reports the resulting sizes: one shared camera model
needs all frames in the same orientation.

Usage: python scripts/real/heic_to_jpg.py --data data/real/chair   (moves images/*.HEIC to heic/)
"""

import argparse
import collections
import shutil
from pathlib import Path

from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

register_heif_opener()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--quality", type=int, default=95)
    a = p.parse_args()
    src_dir, dst_dir = a.data / "heic", a.data / "images"
    src_dir.mkdir(exist_ok=True)
    for f in dst_dir.glob("*"):
        if f.suffix.lower() == ".heic":
            shutil.move(str(f), src_dir / f.name)

    sizes = collections.Counter()
    for f in sorted(src_dir.glob("*")):
        if f.suffix.lower() != ".heic":
            continue
        im = Image.open(f)
        exif = im.getexif()
        im = ImageOps.exif_transpose(im)
        exif[0x0112] = 1  # orientation: already applied to the pixels
        im.convert("RGB").save(dst_dir / (f.stem + ".jpg"), quality=a.quality, exif=exif.tobytes())
        sizes[im.size] += 1
    print("converted:", sum(sizes.values()), "sizes:", dict(sizes))


if __name__ == "__main__":
    main()
