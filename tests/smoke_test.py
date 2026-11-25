from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cli_test import run_pipeline
from src.compositor import CompositeConfig
from src.video_io import get_video_metadata


def generate_synthetic_videos(out_dir: Path, num_frames: int = 24, w: int = 1280, h: int = 720):
    out_dir.mkdir(parents=True, exist_ok=True)
    src_path = out_dir / "smoke_src.mp4"
    ai_path = out_dir / "smoke_ai.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw_src = cv2.VideoWriter(str(src_path), fourcc, 24.0, (w, h))
    vw_ai = cv2.VideoWriter(str(ai_path), fourcc, 24.0, (w, h))

    # Phone/card rectangle centered around (640, 360), size 320x440
    cx, cy = 640, 360
    bw, bh = 320, 440

    for t in range(num_frames):
        # --- Source frame ---
        src = np.full((h, w, 3), (45, 50, 55), dtype=np.uint8)
        # Background texture
        cv2.circle(src, (200, 180), 80, (70, 80, 90), -1)
        cv2.rectangle(src, (cx - bw // 2, cy - bh // 2), (cx + bw // 2, cy + bh // 2), (245, 245, 245), -1)
        cv2.rectangle(src, (cx - bw // 2, cy - bh // 2), (cx + bw // 2, cy + bh // 2), (20, 20, 20), 6)
        cv2.putText(src, "SOURCE ANCHOR", (cx - 125, cy - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (15, 15, 15), 2, cv2.LINE_AA)
        cv2.putText(src, "100% SHARP TEXT", (cx - 130, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 90, 200), 2, cv2.LINE_AA)
        cv2.putText(src, f"ITEM #{t:02d}", (cx - 75, cy + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (30, 30, 30), 2, cv2.LINE_AA)

        # Original hand swiping across frames 10..14
        if 10 <= t <= 14:
            hx = cx - 40 + (t - 10) * 25
            hy = cy + 60
            cv2.ellipse(src, (hx, hy), (55, 25), -20, 0, 360, (140, 175, 215), -1)

        vw_src.write(src)

        # --- AI frame (shifted 30px, rotated 2 deg, 2-frame timing offset, blurred text, different bg & hand) ---
        t_ai_phase = min(num_frames - 1, t + 2)
        ai_base = np.full((h, w, 3), (90, 60, 110), dtype=np.uint8)
        cv2.circle(ai_base, (1050, 560), 110, (130, 90, 150), -1)
        cv2.rectangle(ai_base, (cx - bw // 2, cy - bh // 2), (cx + bw // 2, cy + bh // 2), (235, 230, 240), -1)
        cv2.rectangle(ai_base, (cx - bw // 2, cy - bh // 2), (cx + bw // 2, cy + bh // 2), (35, 25, 45), 6)
        # Blurry/garbled AI text
        cv2.putText(ai_base, "S0URC3 ANCH0R", (cx - 125, cy - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (80, 80, 80), 2)
        cv2.putText(ai_base, "BLURRY AI TXT", (cx - 125, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (100, 100, 150), 2)
        # Blur the card interior
        y0, y1 = cy - bh // 2 + 12, cy + bh // 2 - 12
        x0, x1 = cx - bw // 2 + 12, cx + bw // 2 - 12
        ai_base[y0:y1, x0:x1] = cv2.GaussianBlur(ai_base[y0:y1, x0:x1], (15, 15), 5.0)

        # AI hand (different color) swiping across frames 10..14
        if 10 <= t_ai_phase <= 14:
            hx = cx - 40 + (t_ai_phase - 10) * 25
            hy = cy + 60
            cv2.ellipse(ai_base, (hx, hy), (55, 25), -20, 0, 360, (110, 145, 235), -1)

        # Shift 30px and rotate 2 degrees
        M = cv2.getRotationMatrix2D((cx, cy), 2.0, 1.0)
        M[0, 2] += 30.0
        M[1, 2] += 18.0
        ai_warped = cv2.warpAffine(ai_base, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        vw_ai.write(ai_warped)

    vw_src.release()
    vw_ai.release()
    return src_path, ai_path


def main():
    smoke_dir = ROOT / "workspace" / "smoke_test"
    src_path, ai_path = generate_synthetic_videos(smoke_dir)
    out_path = smoke_dir / "smoke_output.mp4"

    # Points for SAM 2:
    # Source Obj (id=1) at frame 0, plus negative point on original hand at frame 12
    # AI Obj (id=1) at frame 0, AI Hand (id=2) at frame 10
    points_dict = {
        ("src", 1, 0): [(640, 360, 1), (640, 220, 1)],
        ("src", 1, 12): [(650, 420, 0)],  # Negative point on original hand
        ("ai", 1, 0): [(670, 378, 1), (670, 240, 1)],
        ("ai", 2, 10): [(670, 438, 1)],
    }

    cfg = CompositeConfig(
        mode="Emissive Screen",
        lighting_strength=0.25,
        micro_soften=0.0,
        grain_match=0.1,
        mask_erode_dilate=-1,
        edge_feather=2,
        hand_feather=2,
        auto_crop=True,
    )

    report = run_pipeline(
        src_video=src_path,
        ai_video=ai_path,
        output_video=out_path,
        points_dict=points_dict,
        work_dir=smoke_dir / "temp_run",
        sync_mode="Auto DTW",
        comp_cfg=cfg,
        audio_from="none",
    )

    assert out_path.exists() and out_path.stat().st_size > 10_000, "Output MP4 missing or too small"
    meta = get_video_metadata(out_path)
    assert meta.frame_count == 24, f"Expected 24 frames, got {meta.frame_count}"
    assert meta.width == 1280 and meta.height == 720, f"Unexpected resolution {meta.size}"
    assert report["peak_vram_gb"] < 7.0, f"Peak VRAM {report['peak_vram_gb']} GB exceeded 7.0 GB!"

    print("=== SMOKE TEST PASSED ===")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
