from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from src.compositor import CompositeConfig, composite_frame
from src.segmentation import PointsDict, clear_vram, temporal_smooth_mask, track_video_masks
from src.spatial_align import (
    apply_border_hug_flow,
    compute_safe_crop_box,
    estimate_blur_sigma,
    estimate_pair_similarity,
    interpolate_ai_pair,
    smooth_transforms,
    warp_ai_frame,
)
from src.temporal_sync import SyncMode, compute_temporal_mapping, smooth_fractional_mapping
from src.video_io import encode_video, extract_frames, get_video_metadata, load_mask


class AIResampler:
    """Returns the AI frame (+ object/occluder masks) at a FRACTIONAL position, using
    motion-compensated interpolation so AI frames are never held or skipped."""

    def __init__(self, frames: List[Path], obj_masks: List[Path], occ_masks: Optional[List[Path]]):
        self.frames, self.obj_masks, self.occ_masks = frames, obj_masks, occ_masks
        self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        self._cache: Dict[tuple, tuple] = {}

    def _load(self, i: int, scale: float):
        key = (i, scale)
        if key not in self._cache:
            if len(self._cache) > 12:
                self._cache.clear()
            img = cv2.imread(str(self.frames[i]))
            obj = load_mask(self.obj_masks[i]).astype(np.float32)
            occ = load_mask(self.occ_masks[i]).astype(np.float32) if self.occ_masks else None
            if scale != 1.0:
                sz = (int(img.shape[1] * scale), int(img.shape[0] * scale))
                img = cv2.resize(img, sz, interpolation=cv2.INTER_AREA)
                obj = cv2.resize(obj, sz, interpolation=cv2.INTER_LINEAR)
                occ = cv2.resize(occ, sz, interpolation=cv2.INTER_LINEAR) if occ is not None else None
            self._cache[key] = (img, obj, occ)
        return self._cache[key]

    def get(self, pos: float, scale: float = 1.0):
        i0 = int(np.floor(pos))
        i1 = min(i0 + 1, len(self.frames) - 1)
        a = float(pos - i0) if i1 != i0 else 0.0
        im0, ob0, oc0 = self._load(i0, scale)
        if a < 0.03 or i1 == i0:
            return im0, (ob0 > 0.5).astype(np.uint8), (None if oc0 is None else (oc0 > 0.5).astype(np.uint8))
        im1, ob1, oc1 = self._load(i1, scale)
        img = interpolate_ai_pair(im0, im1, a, self.dis)
        obj = ((1 - a) * ob0 + a * ob1 > 0.5).astype(np.uint8)
        occ = None if oc0 is None else ((1 - a) * oc0 + a * oc1 > 0.5).astype(np.uint8)
        return img, obj, occ


def run_pipeline(
    src_video: str | Path,
    ai_video: str | Path,
    output_video: str | Path,
    points_dict: PointsDict,
    work_dir: str | Path = "workspace/temp_run",
    sync_mode: SyncMode = "Auto DTW",
    sakoe_chiba_window: int = 15,
    manual_offset: int = 0,
    border_hug_flow: bool = False,
    comp_cfg: Optional[CompositeConfig] = None,
    audio_from: str = "ai",
) -> dict:
    if comp_cfg is None:
        comp_cfg = CompositeConfig()

    work = Path(work_dir)
    src_frames_dir = work / "src_frames"
    ai_frames_dir = work / "ai_frames"
    src_masks_dir = work / "src_masks"
    ai_masks_dir = work / "ai_masks"
    out_frames_dir = work / "out_frames"
    out_frames_dir.mkdir(parents=True, exist_ok=True)
    for old in out_frames_dir.glob("*.png"):
        old.unlink()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    meta_src = get_video_metadata(src_video)
    src_frames = extract_frames(src_video, src_frames_dir)
    ai_frames = extract_frames(ai_video, ai_frames_dir, target_size=meta_src.size)

    # Phase 1: SAM 2 Tracking (sequential, offload to CPU, clear VRAM after)
    src_tracked = track_video_masks(
        src_frames_dir, src_masks_dir, "src", points_dict, obj_ids=[1]
    )
    ai_obj_ids = [1]
    if any(vt == "ai" and oid == 2 and pts for (vt, oid, _), pts in points_dict.items()):
        ai_obj_ids.append(2)
    ai_tracked = track_video_masks(
        ai_frames_dir, ai_masks_dir, "ai", points_dict, obj_ids=ai_obj_ids
    )
    clear_vram()

    peak_vram_gb = (
        torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0
    )

    src_masks = src_tracked[1]
    ai_masks = ai_tracked[1]
    ai_occ_masks = ai_tracked.get(2)

    # Phase 2: Temporal Sync -> smooth fractional AI position per source frame
    mapping = compute_temporal_mapping(
        src_frames,
        ai_frames,
        src_masks,
        ai_masks,
        mode=sync_mode,
        sakoe_chiba_window=sakoe_chiba_window,
        manual_offset=manual_offset,
    )
    pos = smooth_fractional_mapping(mapping, len(ai_frames))
    resampler = AIResampler(ai_frames, ai_masks, ai_occ_masks)

    # Phase 3: Spatial alignment at 0.5x on the INTERPOLATED AI frame (so the estimate matches
    # exactly what is rendered), SIFT + sub-pixel ECC, outlier rejection + smoothing.
    S = 0.5
    raw_transforms = []
    for t in range(len(src_frames)):
        src_s = cv2.resize(cv2.imread(str(src_frames[t])), None, fx=S, fy=S, interpolation=cv2.INTER_AREA)
        m_src_s = cv2.resize(temporal_smooth_mask(src_masks, t), (src_s.shape[1], src_s.shape[0]), interpolation=cv2.INTER_NEAREST)
        ai_s, m_ai_s, m_occ_s = resampler.get(pos[t], S)
        p = estimate_pair_similarity(src_s, ai_s, m_src_s, m_ai_s, m_occ_s)
        p.tx /= S
        p.ty /= S
        raw_transforms.append(p)

    smooth_tf = smooth_transforms(raw_transforms)
    crop_box = compute_safe_crop_box(smooth_tf, meta_src.width, meta_src.height) if comp_cfg.auto_crop else None

    # Resolution/sharpness compensation: AI clips are a mix of native 720p (upscaled) and 1080p.
    # Measure, per clip, how much softer the AI is than the Source at the object boundary
    # (aligned pairs, saturated/failed measurements dropped) and soften the Source layer to match.
    measured_sigma = None
    if comp_cfg.auto_match_sharpness:
        vals = []
        for t in np.linspace(0, len(src_frames) - 1, 16).astype(int):
            src_bgr = cv2.imread(str(src_frames[t]))
            m_src = temporal_smooth_mask(src_masks, int(t))
            ai_bgr, _, m_occ = resampler.get(pos[t], 1.0)
            wa, wo = warp_ai_frame(ai_bgr, smooth_tf[t].to_matrix(), meta_src.size, mask_occ=m_occ)
            vals.append(estimate_blur_sigma(src_bgr, wa, m_src, wo))
        ok = [v for v in vals if v < 2.9]
        if len(ok) >= 4:
            measured_sigma = float(np.median(ok))
            comp_cfg = replace(comp_cfg, micro_soften=measured_sigma)

    # Phase 4: Warp + Composite + Save PNG sequentially
    for t in range(len(src_frames)):
        src_bgr = cv2.imread(str(src_frames[t]))
        m_src = temporal_smooth_mask(src_masks, t)
        ai_bgr, _, m_occ = resampler.get(pos[t], 1.0)

        warped_ai, warped_occ = warp_ai_frame(
            ai_bgr, smooth_tf[t].to_matrix(), meta_src.size, mask_occ=m_occ
        )
        if border_hug_flow:
            warped_ai, warped_occ = apply_border_hug_flow(warped_ai, src_bgr, m_src, warped_occ)

        out_bgr = composite_frame(src_bgr, warped_ai, m_src, warped_occ, comp_cfg, crop_box=crop_box)
        cv2.imwrite(str(out_frames_dir / f"{t:05d}.png"), out_bgr)

    # Phase 5: Encode MP4
    audio_src = ai_video if audio_from == "ai" else (src_video if audio_from == "src" else None)
    out_mp4 = encode_video(out_frames_dir, output_video, fps=meta_src.fps, audio_source_path=audio_src)

    steps = np.diff(pos)
    return {
        "output_video": str(out_mp4),
        "num_frames": len(src_frames),
        "peak_vram_gb": round(peak_vram_gb, 3),
        "crop_box": crop_box,
        "ai_pos_step_mean": round(float(steps.mean()), 3),
        "ai_pos_step_std": round(float(steps.std()), 3),
        "invalid_align_frames": sum(1 for p in raw_transforms if not p.valid),
        "measured_ai_softness_sigma_px": None if measured_sigma is None else round(measured_sigma, 2),
        "micro_soften_used": round(float(comp_cfg.micro_soften), 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Source-Anchor Video Compositor CLI")
    parser.add_argument("--src", required=True)
    parser.add_argument("--ai", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--points_json", required=True, help="JSON file with points list")
    parser.add_argument("--mode", default="Emissive Screen", choices=["Emissive Screen", "Surface / Product"])
    parser.add_argument("--sync", default="Auto DTW", choices=["Auto DTW", "Manual Frame Offset", "Lock 1:1 (No Sync)"])
    args = parser.parse_args()

    raw_pts = json.loads(Path(args.points_json).read_text(encoding="utf-8"))
    points_dict: PointsDict = {}
    for item in raw_pts:
        key = (str(item["video_type"]), int(item["obj_id"]), int(item["frame_idx"]))
        points_dict.setdefault(key, []).append((int(item["x"]), int(item["y"]), int(item["label"])))

    cfg = CompositeConfig(mode=args.mode)
    report = run_pipeline(args.src, args.ai, args.out, points_dict, sync_mode=args.sync, comp_cfg=cfg)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
