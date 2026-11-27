import os
import urllib.request
from pathlib import Path
from typing import Tuple

CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt"
CHECKPOINT_NAME = "sam2.1_hiera_small.pt"

SAM2_CHECKPOINTS = {
    "small": (
        "sam2.1_hiera_small.pt",
        "configs/sam2.1/sam2.1_hiera_s.yaml",
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
        100_000_000,
    ),
    "base_plus": (
        "sam2.1_hiera_base_plus.pt",
        "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
        250_000_000,
    ),
}


def ensure_checkpoint(checkpoints_dir: str | Path = "checkpoints", variant: str = "small") -> Path:
    ckpt_dir = Path(checkpoints_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    name, _cfg, url, min_size = SAM2_CHECKPOINTS.get(variant, SAM2_CHECKPOINTS["small"])
    ckpt_path = ckpt_dir / name
    if ckpt_path.exists() and ckpt_path.stat().st_size > min_size:
        print(f"[Checkpoint] Found existing {ckpt_path} ({ckpt_path.stat().st_size / 1e6:.1f} MB)")
        return ckpt_path

    print(f"[Checkpoint] Downloading {name} from {url}...")
    tmp_path = ckpt_path.with_suffix(".pt.tmp")
    urllib.request.urlretrieve(url, tmp_path)
    os.replace(tmp_path, ckpt_path)
    print(f"[Checkpoint] Saved to {ckpt_path} ({ckpt_path.stat().st_size / 1e6:.1f} MB)")
    return ckpt_path


def ensure_best_sam2_checkpoint(checkpoints_dir: str | Path = "checkpoints") -> Tuple[Path, str]:
    """Return (ckpt_path, config_yaml), preferring sam2.1_hiera_base_plus.pt when available on disk."""
    ckpt_dir = Path(checkpoints_dir)
    base_name, base_cfg, _url, base_min = SAM2_CHECKPOINTS["base_plus"]
    base_path = ckpt_dir / base_name
    if base_path.exists() and base_path.stat().st_size > base_min:
        return base_path, base_cfg
    small_path = ensure_checkpoint(ckpt_dir, variant="small")
    return small_path, SAM2_CHECKPOINTS["small"][1]


if __name__ == "__main__":
    ckpt_root = Path(__file__).resolve().parent / "checkpoints"
    ensure_checkpoint(ckpt_root, variant="small")
    ensure_checkpoint(ckpt_root, variant="base_plus")
