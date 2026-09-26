"""COLMAP (sequential matching) on a Tanks-and-Temples image set. Usage: python scripts/tnt/run_colmap_tnt.py --scene_dir /root/data/tnt/Meetingroom"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "real"))
from run_real import run_colmap  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--scene_dir", type=Path, required=True)
p.add_argument("--max_size", type=int, default=1920)
a = p.parse_args()
print(run_colmap(a.scene_dir / "images", a.scene_dir / "work", a.max_size, matcher="sequential"))
