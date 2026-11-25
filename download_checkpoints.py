import os
import urllib.request
from pathlib import Path

CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt"
CHECKPOINT_NAME = "sam2.1_hiera_small.pt"


def ensure_checkpoint(checkpoints_dir: str | Path = "checkpoints") -> Path:
    ckpt_dir = Path(checkpoints_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / CHECKPOINT_NAME
    if ckpt_path.exists() and ckpt_path.stat().st_size > 100_000_000:
        print(f"[Checkpoint] Found existing {ckpt_path} ({ckpt_path.stat().st_size / 1e6:.1f} MB)")
        return ckpt_path

    print(f"[Checkpoint] Downloading {CHECKPOINT_NAME} from {CHECKPOINT_URL}...")
    tmp_path = ckpt_path.with_suffix(".pt.tmp")
    urllib.request.urlretrieve(CHECKPOINT_URL, tmp_path)
    os.replace(tmp_path, ckpt_path)
    print(f"[Checkpoint] Saved to {ckpt_path} ({ckpt_path.stat().st_size / 1e6:.1f} MB)")
    return ckpt_path


if __name__ == "__main__":
    ensure_checkpoint(Path(__file__).resolve().parent / "checkpoints")
