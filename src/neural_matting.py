from __future__ import annotations

import gc
import sys
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
_MATANYONE2_ROOT = _REPO_ROOT / "third_party" / "MatAnyone2"

_SLOT_LOCK = threading.Lock()
_MODEL_SLOT: Dict[str, object] = {"kind": None, "payload": {}}


def _ensure_matanyone2_path() -> None:
    if _MATANYONE2_ROOT.is_dir():
        root_str = str(_MATANYONE2_ROOT)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)


def release_neural_models(keep: Optional[str] = None) -> None:
    """Unload cached GPU neural matting models except `keep` so peak VRAM stays bounded."""
    with _SLOT_LOCK:
        current = _MODEL_SLOT.get("kind")
        if current is None or current == keep:
            return
        _MODEL_SLOT["kind"] = None
        _MODEL_SLOT["payload"] = {}
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _release_sam2_if_loaded() -> None:
    try:
        from src.segmentation import release_sam2_cache

        release_sam2_cache()
    except Exception:
        pass


def estimate_foreground_pymatting(
    bgr: np.ndarray,
    alpha: np.ndarray,
    strength: float = 1.0,
    despill_green: bool = False,
) -> np.ndarray:
    """
    Solve the physical compositing equation I = alpha * F + (1 - alpha) * B using
    PyMatting's multi-level foreground estimator (`estimate_foreground_ml`) on a tight ROI,
    while keeping strictly opaque interior pixels (alpha >= 0.995) 100% lossless.
    """
    alpha_f = np.clip(alpha.astype(np.float32), 0.0, 1.0)
    fg_f = bgr.astype(np.float32) / 255.0
    s = float(np.clip(strength, 0.0, 1.0))

    soft_band = (alpha_f > 0.004) & (alpha_f < 0.996)
    if s > 1e-3 and np.any(soft_band):
        from pymatting.foreground import estimate_foreground_ml

        h, w = bgr.shape[:2]
        ys, xs = np.where(alpha_f > 0.002)
        if len(xs) > 0:
            pad = 24
            y0 = max(0, int(ys.min()) - pad)
            y1 = min(h, int(ys.max()) + pad + 1)
            x0 = max(0, int(xs.min()) - pad)
            x1 = min(w, int(xs.max()) + pad + 1)

            roi_bgr = bgr[y0:y1, x0:x1]
            rh, rw = roi_bgr.shape[:2]
            roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
            roi_a = alpha_f[y0:y1, x0:x1].astype(np.float64)

            max_dim = 960
            if max(rh, rw) > max_dim:
                scale = max_dim / float(max(rh, rw))
                sw = max(32, int(round(rw * scale)))
                sh = max(32, int(round(rh * scale)))
                in_rgb = cv2.resize(roi_rgb, (sw, sh), interpolation=cv2.INTER_AREA)
                in_a = cv2.resize(roi_a, (sw, sh), interpolation=cv2.INTER_AREA)
                est_small = estimate_foreground_ml(
                    in_rgb,
                    in_a,
                    n_small_iterations=8,
                    n_big_iterations=1,
                )
                est_rgb = cv2.resize(est_small, (rw, rh), interpolation=cv2.INTER_LINEAR)
            else:
                est_rgb = estimate_foreground_ml(
                    roi_rgb,
                    roi_a,
                    n_small_iterations=10,
                    n_big_iterations=2,
                )
            est_bgr = np.clip(est_rgb[..., ::-1], 0.0, 1.0).astype(np.float32)

            # Full ML foreground on fractional edges; smooth taper + 1px near-opaque rim (alpha >= 0.996 stays lossless)
            roi_a_f = alpha_f[y0:y1, x0:x1]
            edge_w = np.clip((0.996 - roi_a_f) / (0.996 - 0.985), 0.0, 1.0) * s
            near_rim = (
                (cv2.dilate((roi_a_f < 0.94).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))) > 0)
                & (roi_a_f >= 0.985)
                & (roi_a_f < 0.996)
            )
            edge_w[near_rim] = np.maximum(edge_w[near_rim], 0.35 * s)
            edge_w[roi_a_f <= 0.002] = 0.0
            edge_w[roi_a_f >= 0.996] = 0.0
            orig_roi = fg_f[y0:y1, x0:x1]
            fg_f[y0:y1, x0:x1] = orig_roi * (1.0 - edge_w[..., None]) + est_bgr * edge_w[..., None]

    if despill_green:
        b, g, r = fg_f[..., 0], fg_f[..., 1], fg_f[..., 2]
        max_rb = np.maximum(r, b)
        excess_g = np.maximum(0.0, g - max_rb)
        edge_weight = np.clip(1.15 - 0.90 * alpha_f, 0.15, 1.0) * np.clip(
            (0.995 - alpha_f) / 0.15, 0.0, 1.0
        )
        g_clean = g - excess_g * edge_weight
        lum_loss = (g - g_clean) * 0.42
        fg_f = np.stack(
            [
                np.clip(b + lum_loss * 0.4, 0.0, 1.0),
                np.clip(g_clean, 0.0, 1.0),
                np.clip(r + lum_loss * 0.6, 0.0, 1.0),
            ],
            axis=-1,
        )

    return np.clip(fg_f * 255.0, 0.0, 255.0).astype(np.uint8)


def _load_birefnet(device: str):
    _release_sam2_if_loaded()
    release_neural_models(keep="birefnet")
    with _SLOT_LOCK:
        if _MODEL_SLOT.get("kind") == "birefnet" and _MODEL_SLOT["payload"].get("device") == device:
            return _MODEL_SLOT["payload"]["model"]

    from transformers import AutoModelForImageSegmentation

    model = None
    last_exc: Optional[Exception] = None
    for repo_id in ("ZhengPeng7/BiRefNet_HR-matting", "ZhengPeng7/BiRefNet-matting"):
        for local_only in (True, False):
            try:
                model = AutoModelForImageSegmentation.from_pretrained(
                    repo_id,
                    trust_remote_code=True,
                    low_cpu_mem_usage=False,
                    local_files_only=local_only,
                )
                break
            except Exception as exc:
                last_exc = exc
        if model is not None:
            break
    if model is None:
        raise RuntimeError(f"Failed to load BiRefNet matting model: {last_exc}")

    model = model.to(device).eval()
    with _SLOT_LOCK:
        _MODEL_SLOT["kind"] = "birefnet"
        _MODEL_SLOT["payload"] = {"model": model, "device": device}
    return model


def _round_mult32(val: int) -> int:
    return max(256, int(round(val / 32.0)) * 32)


def run_birefnet_matting(bgr: np.ndarray, max_size: int = 1024) -> np.ndarray:
    """
    Run ZhengPeng7/BiRefNet_HR-matting on CUDA (with CPU/size fallback) to produce
    trimap-free sub-pixel alpha matte in [0.0, 1.0] at native aspect resolution.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _load_birefnet(device)
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    target_size = _round_mult32(int(max_size))
    if device == "cuda" and min(h, w) >= 1280 and max_size > 1024:
        free_mb = torch.cuda.mem_get_info()[0] // (1024 ** 2)
        if free_mb < 3500:
            target_size = 1024

    base_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
    base_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)

    candidate_sizes = [target_size]
    if device == "cuda":
        for fallback_sz in (1024, 768, 512):
            if fallback_sz < target_size and fallback_sz not in candidate_sizes:
                candidate_sizes.append(fallback_sz)

    attempts: list[tuple[str, int]] = [(device, sz) for sz in candidate_sizes]
    if device == "cuda":
        attempts.append(("cpu", min(768, target_size)))

    last_exc: Optional[Exception] = None
    active_dev = device
    for attempt_dev, sz in attempts:
        try:
            if attempt_dev != active_dev:
                model = model.to(attempt_dev).eval()
                with _SLOT_LOCK:
                    if _MODEL_SLOT.get("kind") == "birefnet" and _MODEL_SLOT.get("payload"):
                        _MODEL_SLOT["payload"]["device"] = attempt_dev
                active_dev = attempt_dev
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            mean = base_mean.to(attempt_dev)
            std = base_std.to(attempt_dev)
            resized = cv2.resize(rgb, (sz, sz), interpolation=cv2.INTER_AREA if min(h, w) > sz else cv2.INTER_LINEAR)
            tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).to(device=attempt_dev, dtype=torch.float32) / 255.0
            tensor = (tensor - mean) / std
            with torch.inference_mode():
                if attempt_dev == "cuda":
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        preds = model(tensor)[-1].sigmoid()
                else:
                    preds = model(tensor)[-1].sigmoid()
            alpha_small = preds[0].squeeze().float().cpu().numpy()
            alpha = cv2.resize(alpha_small, (w, h), interpolation=cv2.INTER_LINEAR)
            return np.clip(alpha, 0.0, 1.0).astype(np.float32)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            last_exc = exc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    raise RuntimeError(f"BiRefNet OOM across attempts {attempts}: {last_exc}")


def run_birefnet_guided_on_mask(
    bgr: np.ndarray,
    coarse_mask: np.ndarray,
    max_size: int = 1024,
) -> np.ndarray:
    """
    Run BiRefNet_HR-matting and gate by a generous soft support zone around `coarse_mask`
    so user-selected objects keep full hair/edge strands while distant unselected objects stay excluded.
    """
    alpha = run_birefnet_matting(bgr, max_size=max_size)
    m = (coarse_mask > 0).astype(np.uint8)
    if not np.any(m > 0) or np.all(m > 0):
        return alpha

    h, w = bgr.shape[:2]
    pad = max(24, int(min(h, w) * 0.04))
    k = pad * 2 + 1
    support = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).astype(np.float32)
    support = cv2.GaussianBlur(support, (7, 7), 0)
    return np.clip(alpha * support, 0.0, 1.0).astype(np.float32)


def _load_vitmatte(model_id: str, device: str):
    _release_sam2_if_loaded()
    release_neural_models(keep="vitmatte")
    with _SLOT_LOCK:
        payload = _MODEL_SLOT.get("payload", {})
        if (
            _MODEL_SLOT.get("kind") == "vitmatte"
            and payload.get("model_id") == model_id
            and payload.get("device") == device
        ):
            return payload["proc"], payload["model"]

    from transformers import VitMatteForImageMatting, VitMatteImageProcessor

    candidates = [model_id]
    if model_id != "hustvl/vitmatte-small-composition-1k":
        candidates.append("hustvl/vitmatte-small-composition-1k")

    for cid in candidates:
        for local_only in (True, False):
            try:
                proc = VitMatteImageProcessor.from_pretrained(cid, local_files_only=local_only)
                model = (
                    VitMatteForImageMatting.from_pretrained(cid, local_files_only=local_only)
                    .to(device)
                    .eval()
                )
                with _SLOT_LOCK:
                    _MODEL_SLOT["kind"] = "vitmatte"
                    _MODEL_SLOT["payload"] = {
                        "proc": proc,
                        "model": model,
                        "model_id": cid,
                        "device": device,
                    }
                return proc, model
            except Exception:
                continue
    return None, None


def run_vitmatte_neural(
    bgr: np.ndarray,
    trimap: np.ndarray,
    model_id: str = "hustvl/vitmatte-base-composition-1k",
    max_side: int = 1536,
) -> Optional[np.ndarray]:
    """
    Run ViTMatte (Base by default, Small fallback) on CUDA inside the active trimap bounding box,
    with automatic OOM-safe downscaling that preserves discrete trimap regions.
    """
    from PIL import Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc, model = _load_vitmatte(model_id, device)
    if proc is None or model is None:
        return None

    h, w = bgr.shape[:2]
    ys, xs = np.where(trimap > 0)
    if len(xs) == 0:
        return np.zeros((h, w), dtype=np.float32)

    y0 = max(0, int(ys.min()) - 32)
    y1 = min(h, int(ys.max()) + 32)
    x0 = max(0, int(xs.min()) - 32)
    x1 = min(w, int(xs.max()) + 32)
    ch, cw = y1 - y0, x1 - x0

    crop_rgb = cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
    crop_tri = trimap[y0:y1, x0:x1]

    for limit in (int(max_side), 1024, 768):
        if limit < int(max_side) and max(ch, cw) <= limit:
            continue
        try:
            if max(ch, cw) > limit:
                scale = limit / float(max(ch, cw))
                rw = max(32, int(round(cw * scale)))
                rh = max(32, int(round(ch * scale)))
                in_rgb = cv2.resize(crop_rgb, (rw, rh), interpolation=cv2.INTER_AREA)
                in_tri = cv2.resize(crop_tri, (rw, rh), interpolation=cv2.INTER_NEAREST)
            else:
                rw, rh = cw, ch
                in_rgb, in_tri = crop_rgb, crop_tri

            inputs = proc(
                images=Image.fromarray(in_rgb),
                trimaps=Image.fromarray(in_tri),
                return_tensors="pt",
            ).to(device)

            with torch.inference_mode():
                if device == "cuda":
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        pred = model(**inputs).alphas[0, 0].float().cpu().numpy()
                else:
                    pred = model(**inputs).alphas[0, 0].float().cpu().numpy()

            pred_crop = pred[:rh, :rw] if (pred.shape[0] >= rh and pred.shape[1] >= rw) else pred
            pred_resized = cv2.resize(pred_crop, (cw, ch), interpolation=cv2.INTER_LINEAR)
            alpha = np.zeros((h, w), dtype=np.float32)
            alpha[y0:y1, x0:x1] = np.clip(pred_resized, 0.0, 1.0)
            alpha[trimap == 255] = 1.0
            alpha[trimap == 0] = 0.0
            return alpha
        except torch.cuda.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            return None
    return None


def _load_matanyone2(device: str):
    _ensure_matanyone2_path()
    _release_sam2_if_loaded()
    release_neural_models(keep="matanyone2")
    with _SLOT_LOCK:
        payload = _MODEL_SLOT.get("payload", {})
        if _MODEL_SLOT.get("kind") == "matanyone2" and payload.get("device") == device:
            return payload["model"]

    from matanyone2.model.matanyone2 import MatAnyone2

    model = None
    for local_only in (True, False):
        try:
            model = MatAnyone2.from_pretrained("PeiqingYang/MatAnyone2", local_files_only=local_only)
            break
        except Exception:
            continue
    if model is None:
        model = MatAnyone2.from_pretrained("PeiqingYang/MatAnyone2")
    model = model.to(device).eval()
    with _SLOT_LOCK:
        _MODEL_SLOT["kind"] = "matanyone2"
        _MODEL_SLOT["payload"] = {"model": model, "device": device}
    return model


def _load_rgb_cpu_tensor(
    frame_path: Path, resize_hw: Optional[tuple[int, int]] = None
) -> torch.Tensor:
    bgr = cv2.imread(str(frame_path))
    if bgr is None:
        raise FileNotFoundError(f"Cannot read frame: {frame_path}")
    if resize_hw is not None:
        rh, rw = resize_hw
        if bgr.shape[:2] != (rh, rw):
            interp = cv2.INTER_AREA if bgr.shape[0] > rh else cv2.INTER_LINEAR
            bgr = cv2.resize(bgr, (rw, rh), interpolation=interp)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0)


def _write_alpha_np_png(alpha_np: np.ndarray, path: Path) -> None:
    alpha_u8 = np.round(alpha_np * 255.0).astype(np.uint8)
    cv2.imwrite(str(path), alpha_u8, [cv2.IMWRITE_PNG_COMPRESSION, 1])


def _merge_alpha_np_png(alpha_np: np.ndarray, path: Path, ramp: float = 1.0) -> None:
    bwd = alpha_np * float(np.clip(ramp, 0.0, 1.0))
    fwd_u8 = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if fwd_u8 is None or fwd_u8.shape != bwd.shape:
        merged = bwd
    else:
        fwd = fwd_u8.astype(np.float32) / 255.0
        merged = np.maximum(fwd, bwd)
    cv2.imwrite(str(path), np.round(np.clip(merged, 0.0, 1.0) * 255.0).astype(np.uint8), [cv2.IMWRITE_PNG_COMPRESSION, 1])


def run_matanyone2_video(
    frame_paths: List[Path],
    confirmed_masks: Dict[int, np.ndarray],
    output_dir: Path,
    max_internal_size: int = 720,
    n_warmup: int = 6,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> List[Path]:
    """
    Propagate high-precision temporal video alpha mattes using MatAnyone 2 from one or more
    confirmed keyframe masks, saving 8-bit full-resolution alpha PNGs into `output_dir`.
    Uses async CPU frame prefetching & async PNG writing so the GPU stays saturated.
    """
    if not frame_paths or not confirmed_masks:
        return []

    _ensure_matanyone2_path()
    from matanyone2.inference.inference_core import InferenceCore

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _load_matanyone2(device)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample = cv2.imread(str(frame_paths[0]))
    if sample is None:
        raise FileNotFoundError(f"Cannot read first frame: {frame_paths[0]}")
    h, w = sample.shape[:2]
    n_frames = len(frame_paths)

    resize_hw: Optional[tuple[int, int]] = None
    max_sz = int(max_internal_size)
    if max_sz > 0 and min(h, w) > max_sz:
        scale = max_sz / float(min(h, w))
        new_h = max(32, int(round(h * scale)))
        new_w = max(32, int(round(w * scale)))
        resize_hw = (new_h, new_w)

    anchors = {int(k): v for k, v in confirmed_masks.items() if 0 <= int(k) < n_frames}
    if not anchors:
        raise ValueError("MatAnyone 2 requires at least one valid confirmed keyframe mask.")

    bounds = sorted(anchors)
    # Each entry: (indices, seed_mask, merge_mode, ramp_frames)
    segments: List[tuple[List[int], np.ndarray, str, int]] = []
    if bounds[0] > 0:
        segments.append((list(range(bounds[0], -1, -1)), anchors[bounds[0]], "overwrite", 0))
    for idx, start in enumerate(bounds):
        has_next = (idx + 1 < len(bounds))
        stop = bounds[idx + 1] if has_next else n_frames
        segments.append((list(range(start, stop)), anchors[start], "overwrite", 0))
        if has_next and stop - start > 1:
            back_min = max(start + 1, stop - 60)
            lookback = list(range(stop, back_min - 1, -1))
            if len(lookback) > 1:
                ramp_len = 8 if (stop - back_min) >= 16 else 1
                segments.append((lookback, anchors[stop], "merge_max", ramp_len))

    out_paths = [output_dir / f"{t:05d}.png" for t in range(n_frames)]
    total_steps = max(1, sum(len(s[0]) for s in segments))
    processed_count = 0

    try:
        with ThreadPoolExecutor(max_workers=2) as reader_pool, ThreadPoolExecutor(max_workers=2) as writer_pool:
            for seg_indices, seed_mask, merge_mode, ramp_frames in segments:
                core = InferenceCore(model, device=device)
                # Pre-resize frames/masks before core.step() because InferenceCore's internal
                # max_internal_size path assumes a 3D mask tensor while matting=True expects 2D (H, W).
                core.max_internal_size = -1

                target_h, target_w = resize_hw if resize_hw is not None else (h, w)
                seed_bin = (seed_mask > 0).astype(np.float32)
                if seed_bin.shape[:2] != (target_h, target_w):
                    seed_bin = cv2.resize(seed_bin, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
                seed_tensor = torch.from_numpy(seed_bin * 255.0).to(device=device, dtype=torch.float32)

                # Prefetch queue (depth 4) for upcoming segment frames
                prefetch_q: deque = deque()
                for pre_idx in seg_indices[:4]:
                    prefetch_q.append(reader_pool.submit(_load_rgb_cpu_tensor, frame_paths[pre_idx], resize_hw))
                next_submit = 4
                pending_writes: deque = deque()

                def _next_gpu_frame():
                    nonlocal next_submit
                    cpu_t = prefetch_q.popleft().result()
                    if next_submit < len(seg_indices):
                        idx_sub = seg_indices[next_submit]
                        prefetch_q.append(reader_pool.submit(_load_rgb_cpu_tensor, frame_paths[idx_sub], resize_hw))
                        next_submit += 1
                    return cpu_t.to(device=device, non_blocking=True)

                def _enqueue_write(fn, *args):
                    while len(pending_writes) >= 8:
                        pending_writes.popleft().result()
                    pending_writes.append(writer_pool.submit(fn, *args))

                seed_img = _next_gpu_frame()
                with torch.inference_mode():
                    ctx = (
                        torch.amp.autocast("cuda", dtype=torch.float16)
                        if device == "cuda"
                        else _null_ctx()
                    )
                    with ctx:
                        output_prob = core.step(seed_img, seed_tensor, objects=[1])
                        output_prob = core.step(seed_img, first_frame_pred=True)
                        for _ in range(1, max(1, int(n_warmup))):
                            output_prob = core.step(seed_img, first_frame_pred=True)
                        seed_alpha = core.output_prob_to_mask(output_prob)
                        if merge_mode == "overwrite":
                            seed_np = _tensor_to_alpha_np(seed_alpha, (w, h))
                            _enqueue_write(_write_alpha_np_png, seed_np, out_paths[seg_indices[0]])
                        processed_count += 1
                        if progress_cb:
                            progress_cb(min(processed_count, total_steps), total_steps)

                        tail_min = min(seg_indices)
                        for frame_idx in seg_indices[1:]:
                            img_t = _next_gpu_frame()
                            output_prob = core.step(img_t)
                            alpha_t = core.output_prob_to_mask(output_prob)
                            alpha_np = _tensor_to_alpha_np(alpha_t, (w, h))
                            if merge_mode == "merge_max" and out_paths[frame_idx].is_file():
                                ramp = min(1.0, (frame_idx - tail_min + 1) / float(max(1, ramp_frames)))
                                _enqueue_write(_merge_alpha_np_png, alpha_np, out_paths[frame_idx], ramp)
                            else:
                                _enqueue_write(_write_alpha_np_png, alpha_np, out_paths[frame_idx])
                            processed_count += 1
                            if progress_cb:
                                progress_cb(min(processed_count, total_steps), total_steps)

                # Flush all segment alpha PNGs before next segment reads them (required for merge_max!)
                while pending_writes:
                    pending_writes.popleft().result()
                core.clear_memory()
                del core
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    finally:
        release_neural_models()

    return out_paths


def _tensor_to_alpha_np(alpha_tensor: torch.Tensor, size_wh: tuple[int, int]) -> np.ndarray:
    alpha_np = alpha_tensor.detach().float().cpu().numpy()
    if alpha_np.ndim == 3:
        alpha_np = alpha_np.squeeze(0)
    w, h = size_wh
    if alpha_np.shape[:2] != (h, w):
        alpha_np = cv2.resize(alpha_np, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(alpha_np, 0.0, 1.0).astype(np.float32)


def _save_alpha_png(alpha_tensor: torch.Tensor, path: Path, size_wh: tuple[int, int]) -> None:
    alpha_np = _tensor_to_alpha_np(alpha_tensor, size_wh)
    alpha_u8 = np.round(alpha_np * 255.0).astype(np.uint8)
    cv2.imwrite(str(path), alpha_u8)


def _merge_alpha_png(
    alpha_tensor: torch.Tensor, path: Path, size_wh: tuple[int, int], ramp: float = 1.0
) -> None:
    bwd = _tensor_to_alpha_np(alpha_tensor, size_wh) * float(np.clip(ramp, 0.0, 1.0))
    fwd_u8 = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if fwd_u8 is None or fwd_u8.shape != bwd.shape:
        merged = bwd
    else:
        fwd = fwd_u8.astype(np.float32) / 255.0
        merged = np.maximum(fwd, bwd)
    cv2.imwrite(str(path), np.round(np.clip(merged, 0.0, 1.0) * 255.0).astype(np.uint8))


class _null_ctx:
    def __enter__(self):
        return None

    def __exit__(self, *args):
        return False
