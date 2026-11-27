from __future__ import annotations

import argparse
import base64
import hashlib
import mimetypes
import os
import shutil
import threading
import time
import traceback
import uuid
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import APIRouter, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

import app as legacy_app
from src.alpha_matting import (
    auto_chroma_key_mask,
    auto_seed_roto_strokes,
    extract_pixel_accurate_matte,
    refine_alpha_edge_controls,
    stabilize_alpha_temporal,
)
from src.flow_prompt_engine import (
    analyze_video_optical_profile,
    generate_matched_background_prompts,
)
from src.matte_qa import audit_and_heal_video_mattes, heal_universal_structural_defects
from src.neural_matting import (
    estimate_foreground_pymatting,
    release_neural_models,
    run_birefnet_matting,
    run_matanyone2_video,
)
from src.relight_harmonizer import (
    RelightConfig,
    composite_with_relighting,
    detect_emissive_screen_mask,
)
from src.segmentation import (
    auto_cut_universal_hybrid,
    clear_vram,
    preview_single_frame_mask,
    refine_mask_with_sam2_negative,
    smart_roto_brush_refine,
    temporal_smooth_mask,
    track_roto_video_masks,
)
from src.spatial_align import (
    estimate_pair_similarity,
    estimate_source_camera_trajectory,
    smooth_transforms,
    warp_ai_frame,
    warp_bg_with_camera_motion,
)
from src.video_io import VideoPipeEncoder, encode_video, extract_frames, get_video_metadata, load_alpha, load_mask, save_mask

ROOT_DIR = Path(__file__).resolve().parent
WEB_DIR = ROOT_DIR / "web"
REMOTE_SESSIONS_DIR = ROOT_DIR / "workspace" / "remote_sessions"
REMOTE_SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MAX_CHUNK_BYTES = 16 * 1024 * 1024
SESSION_TTL_SECONDS = max(300, int(os.environ.get("MATTING_SESSION_TTL_SECONDS", "3600")))

ENGINE_MAP = {
    "matanyone2": legacy_app.MATTING_ENGINE_MATANYONE2,
    "birefnet": legacy_app.MATTING_ENGINE_BIREFNET,
    "vitmatte": legacy_app.MATTING_ENGINE_VITMATTE,
    "classic": legacy_app.MATTING_ENGINE_CLASSIC,
}

DIAG_MAP = {
    "split": legacy_app.PREVIEW_DIAG_SPLIT,
    "halo_black": legacy_app.PREVIEW_DIAG_BLACK,
    "halo_white": legacy_app.PREVIEW_DIAG_WHITE,
    "screen_mask": legacy_app.PREVIEW_DIAG_SCREEN,
}


def _resolve_engine_label(value: str) -> str:
    key = str(value or "").strip().lower()
    if key in ENGINE_MAP:
        return ENGINE_MAP[key]
    if value in ENGINE_MAP.values():
        return value
    return legacy_app.MATTING_ENGINE_MATANYONE2


def _dir_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return round(total / (1024 * 1024), 2)


def _decode_b64_mask(b64_data: Optional[str], h: int, w: int) -> np.ndarray:
    if not b64_data:
        return np.zeros((h, w), dtype=np.uint8)
    raw_txt = str(b64_data)
    if "," in raw_txt:
        raw_txt = raw_txt.split(",", 1)[1]
    buf = np.frombuffer(base64.b64decode(raw_txt), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if img is None:
        return np.zeros((h, w), dtype=np.uint8)
    if img.ndim == 3 and img.shape[2] == 4:
        mask = (img[..., 3] > 20) | (np.max(img[..., :3], axis=2) > 25)
    elif img.ndim == 3:
        mask = np.max(img, axis=2) > 25
    else:
        mask = img > 25
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    return mask.astype(np.uint8)


def _encode_jpeg_b64(rgb: np.ndarray, quality: int = 88) -> str:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("Không thể nén ảnh JPEG.")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


@dataclass
class SessionState:
    session_id: str
    work_dir: Path
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    lock: threading.RLock = field(default_factory=threading.RLock)

    src_media: Optional[str] = None
    src_is_image: bool = False
    src_frames: List[Path] = field(default_factory=list)
    fps: float = 30.0
    width: int = 0
    height: int = 0

    bg_media: Optional[str] = None
    bg_is_image: bool = False
    bg_frames: List[Path] = field(default_factory=list)
    bg_fps: float = 30.0

    confirmed_masks: Dict[int, Path] = field(default_factory=dict)
    undo_stacks: Dict[int, List[np.ndarray]] = field(default_factory=dict)
    birefnet_alpha_by_frame: Dict[int, np.ndarray] = field(default_factory=dict)
    tracked_masks: List[Path] = field(default_factory=list)

    uploads: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    jobs: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    matte_cache: OrderedDict = field(default_factory=OrderedDict)
    bg_cache: OrderedDict = field(default_factory=OrderedDict)
    comp_cache: OrderedDict = field(default_factory=OrderedDict)

    def touch(self) -> None:
        self.last_active = time.time()


class SessionStore:
    def __init__(self, base_dir: Path = REMOTE_SESSIONS_DIR, ttl_seconds: int = SESSION_TTL_SECONDS):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
        self._sessions: Dict[str, SessionState] = {}

    def create(self) -> SessionState:
        sid = uuid.uuid4().hex[:16]
        wdir = self.base_dir / sid
        wdir.mkdir(parents=True, exist_ok=True)
        sess = SessionState(session_id=sid, work_dir=wdir)
        with self._lock:
            self._sessions[sid] = sess
        return sess

    def get(self, sid: str) -> SessionState:
        with self._lock:
            sess = self._sessions.get(sid)
        if sess is None or not sess.work_dir.exists():
            raise HTTPException(status_code=404, detail="Phiên làm việc không tồn tại hoặc đã được dọn dẹp.")
        sess.touch()
        return sess

    def cleanup(self, sid: str, release_gpu: bool = True) -> float:
        with self._lock:
            sess = self._sessions.pop(sid, None)
        freed_mb = 0.0
        target_dir = sess.work_dir if sess is not None else (self.base_dir / sid)
        if sess is not None:
            with sess.lock:
                sess.matte_cache.clear()
                sess.bg_cache.clear()
                sess.comp_cache.clear()
                sess.birefnet_alpha_by_frame.clear()
                sess.undo_stacks.clear()
        if target_dir.exists():
            freed_mb = _dir_size_mb(target_dir)
            shutil.rmtree(target_dir, ignore_errors=True)
        if release_gpu:
            try:
                release_neural_models()
                clear_vram()
            except Exception:
                pass
        return freed_mb

    def sweep_expired(self) -> int:
        now = time.time()
        with self._lock:
            expired = [
                sid
                for sid, s in self._sessions.items()
                if (now - s.last_active) > self.ttl_seconds
                and not any(j.get("status") == "running" for j in s.jobs.values())
            ]
        for sid in expired:
            self.cleanup(sid, release_gpu=True)
        # Also remove orphan directories left from earlier runs past TTL
        for child in self.base_dir.iterdir():
            if not child.is_dir():
                continue
            with self._lock:
                tracked = child.name in self._sessions
            if not tracked:
                try:
                    age = now - child.stat().st_mtime
                    if age > self.ttl_seconds:
                        shutil.rmtree(child, ignore_errors=True)
                except OSError:
                    pass
        return len(expired)

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            count = len(self._sessions)
        return {
            "active_sessions": count,
            "total_disk_mb": _dir_size_mb(self.base_dir),
        }


STORE = SessionStore()


def _ttl_reaper_loop() -> None:
    while True:
        time.sleep(600)
        try:
            STORE.sweep_expired()
        except Exception:
            pass


_reaper_started = False


def _ensure_reaper() -> None:
    global _reaper_started
    if not _reaper_started:
        _reaper_started = True
        t = threading.Thread(target=_ttl_reaper_loop, daemon=True)
        t.start()


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------
class UploadInitReq(BaseModel):
    role: str = Field(..., pattern="^(source|background)$")
    filename: str
    total_size: int = 0


class AutoCutReq(BaseModel):
    frame_idx: int = 0
    mode: str = "birefnet"  # birefnet | hands_product | chroma
    engine: str = "matanyone2"
    use_pymatting_fg: bool = True


class StrokesReq(BaseModel):
    frame_idx: int = 0
    fg_mask_b64: Optional[str] = None
    bg_mask_b64: Optional[str] = None
    box_xyxy: Optional[List[int]] = None
    smart_red_subtract: bool = True
    engine: str = "matanyone2"
    use_pymatting_fg: bool = True


class FrameActionReq(BaseModel):
    frame_idx: int = 0


class AnalyzePromptsReq(BaseModel):
    target_background_vi: str = "Quán cà phê hiện đại hoặc phòng đọc sách ánh sáng tự nhiên"
    subject_description_vi: str = ""
    use_local_llm: bool = True
    use_vision_ai: bool = True


class PreviewReq(BaseModel):
    frame_idx: int = 0
    diag_mode: str = "split"  # split | halo_black | halo_white | screen_mask
    max_preview_dim: int = 1080
    matting_engine: str = "matanyone2"
    use_pymatting_fg: bool = True
    trimap_radius: int = 8
    guided_radius: int = 2
    decont_strength: float = 0.95
    despill_green: bool = False
    erode_dilate: float = 0.0
    edge_preserve: float = 0.45
    edge_blur_match: float = 0.0
    ambient_color_strength: float = 0.68
    directional_light_strength: float = 0.45
    black_point_match: float = 0.35
    light_wrap_radius: int = 4
    light_wrap_strength: float = 0.65
    protect_screen: bool = True
    screen_glare: float = 0.06
    grain_match: float = 0.12
    bg_defocus: float = 0.6
    clean_ai_ghost: bool = False
    linear_compositing: bool = True
    pass2_legacy: bool = False


class RenderReq(PreviewReq):
    tracking_mode: str = "Camera Motion Lock"  # Camera Motion Lock | Flow V2V Spatial Sync | Lock 1:1
    audio_choice: str = "source"  # source | background | none
    matanyone_max_size: int = 720
    auto_qa_heal: bool = True
    qa_sample_pct: float = 0.10


# ---------------------------------------------------------------------------
# Core session processing helpers
# ---------------------------------------------------------------------------
def _load_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"Không thể đọc ảnh: {path}")
    return img


def _load_session_frame_mask(sess: SessionState, f_idx: int) -> Optional[np.ndarray]:
    if f_idx in sess.confirmed_masks and sess.confirmed_masks[f_idx].is_file():
        return load_mask(sess.confirmed_masks[f_idx])
    if sess.tracked_masks and f_idx < len(sess.tracked_masks) and sess.tracked_masks[f_idx].is_file():
        return load_mask(sess.tracked_masks[f_idx])
    return None


def _save_confirmed_mask(sess: SessionState, f_idx: int, mask: np.ndarray, push_undo: bool = True) -> Path:
    conf_dir = sess.work_dir / "confirmed_masks"
    conf_dir.mkdir(parents=True, exist_ok=True)
    if push_undo:
        prev = _load_session_frame_mask(sess, f_idx)
        prev_alpha = sess.birefnet_alpha_by_frame.get(f_idx)
        stack = sess.undo_stacks.setdefault(f_idx, [])
        if prev is not None:
            stack.append((prev.copy(), prev_alpha.copy() if prev_alpha is not None else None))
            if len(stack) > 12:
                stack.pop(0)
    out_p = conf_dir / f"conf_{f_idx:05d}.png"
    save_mask(mask, out_p)
    sess.confirmed_masks[f_idx] = out_p
    sess.matte_cache.clear()
    sess.comp_cache.clear()
    return out_p


def _get_bg_frame_bgr(sess: SessionState, f_idx: int, src_bgr: np.ndarray) -> Tuple[np.ndarray, Tuple[object, ...]]:
    h, w = src_bgr.shape[:2]
    if not sess.bg_frames:
        bg_synth = cv2.GaussianBlur(src_bgr, (0, 0), 22.0)
        return bg_synth, ("synth", h, w)

    if len(sess.bg_frames) == 1:
        t_bg = 0
    else:
        mapped = int(round(f_idx * (float(sess.bg_fps or sess.fps or 30.0) / max(float(sess.fps or 30.0), 1.0))))
        t_bg = mapped % len(sess.bg_frames)

    bg_path = sess.bg_frames[t_bg]
    st = bg_path.stat()
    key = (str(bg_path), st.st_size, st.st_mtime_ns, h, w)
    cached = sess.bg_cache.get(key)
    if cached is not None:
        sess.bg_cache.move_to_end(key)
        return cached, key

    bg_raw = _load_bgr(bg_path)
    bg_resized = cv2.resize(bg_raw, (w, h), interpolation=cv2.INTER_CUBIC)
    sess.bg_cache[key] = bg_resized
    sess.bg_cache.move_to_end(key)
    while len(sess.bg_cache) > 8:
        sess.bg_cache.popitem(last=False)
    return bg_resized, key


def _get_matte_and_fg(
    sess: SessionState,
    f_idx: int,
    src_bgr: np.ndarray,
    req: PreviewReq,
) -> Tuple[np.ndarray, np.ndarray, Tuple[object, ...]]:
    h, w = src_bgr.shape[:2]
    engine_label = _resolve_engine_label(req.matting_engine)
    mask = _load_session_frame_mask(sess, f_idx)
    if mask is None:
        raise HTTPException(
            status_code=400,
            detail=f"Khung {f_idx} chưa có vùng chọn. Hãy bấm 'Tự tách AI 1-Click' hoặc quét cọ trên Bước 1 trước.",
        )

    mask_path = sess.confirmed_masks.get(f_idx) or (
        sess.tracked_masks[f_idx] if f_idx < len(sess.tracked_masks) else None
    )
    m_sig = (str(mask_path), mask_path.stat().st_mtime_ns) if mask_path and mask_path.is_file() else (f_idx,)
    biref_cached = sess.birefnet_alpha_by_frame.get(f_idx)
    biref_sig = hashlib.sha1(biref_cached[:16, :16].tobytes()).hexdigest() if biref_cached is not None else "none"

    m_key = (
        f_idx,
        m_sig,
        biref_sig,
        engine_label,
        bool(req.use_pymatting_fg),
        bool(req.pass2_legacy),
        int(req.trimap_radius),
        int(req.guided_radius),
        round(float(req.decont_strength), 3),
        bool(req.despill_green),
        round(float(req.erode_dilate), 2),
        round(float(req.edge_preserve), 3),
        round(float(req.edge_blur_match), 2),
    )
    cached = sess.matte_cache.get(m_key)
    if cached is not None:
        sess.matte_cache.move_to_end(m_key)
        return cached[0], cached[1], m_key

    if (
        biref_cached is not None
        and biref_cached.shape == (h, w)
        and f_idx in sess.confirmed_masks
        and req.matting_engine in ("birefnet", "matanyone2")
    ):
        base_alpha = biref_cached
        if float(req.edge_preserve) > 0.01:
            base_alpha, _, _ = heal_universal_structural_defects(src_bgr, base_alpha)
        alpha = refine_alpha_edge_controls(
            src_bgr,
            base_alpha,
            erode_dilate=float(req.erode_dilate),
            guided_radius=int(req.guided_radius),
            trimap_radius=int(req.trimap_radius),
            edge_preserve=float(req.edge_preserve),
            edge_feather=float(req.edge_blur_match),
        )
        fg_clean = (
            estimate_foreground_pymatting(
                src_bgr,
                alpha,
                strength=float(req.decont_strength),
                despill_green=bool(req.despill_green),
            )
            if req.use_pymatting_fg
            else src_bgr
        )
    else:
        mcfg = legacy_app._resolve_matting_cfg(
            engine_label,
            int(req.trimap_radius),
            int(req.guided_radius),
            float(req.decont_strength),
            bool(req.despill_green),
            0,
            bool(req.use_pymatting_fg),
            bool(req.pass2_legacy),
        )
        mres = extract_pixel_accurate_matte(src_bgr, mask, mcfg)
        base_alpha = mres.alpha
        if float(req.edge_preserve) > 0.01:
            base_alpha, _, _ = heal_universal_structural_defects(src_bgr, base_alpha)
        alpha = refine_alpha_edge_controls(
            src_bgr,
            base_alpha,
            erode_dilate=float(req.erode_dilate),
            guided_radius=int(req.guided_radius),
            trimap_radius=int(req.trimap_radius),
            edge_preserve=float(req.edge_preserve),
            edge_feather=float(req.edge_blur_match),
        )
        alpha_shifted = (
            abs(float(req.erode_dilate)) >= 0.05
            or float(req.edge_blur_match) > 0.05
            or abs(float(req.edge_preserve) - 0.45) > 0.02
        )
        if req.use_pymatting_fg and alpha_shifted:
            fg_clean = estimate_foreground_pymatting(
                src_bgr,
                alpha,
                strength=float(req.decont_strength),
                despill_green=bool(req.despill_green),
            )
        else:
            fg_clean = mres.fg_clean_bgr

    sess.matte_cache[m_key] = (fg_clean, alpha)
    sess.matte_cache.move_to_end(m_key)
    while len(sess.matte_cache) > 4:
        sess.matte_cache.popitem(last=False)
    return fg_clean, alpha, m_key


def _compute_preview_bundle(sess: SessionState, req: PreviewReq) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not sess.src_frames:
        raise HTTPException(status_code=400, detail="Chưa tải video/ảnh gốc lên phiên.")
    f_idx = int(np.clip(req.frame_idx, 0, len(sess.src_frames) - 1))
    src_bgr = _load_bgr(sess.src_frames[f_idx])
    bg_bgr, bg_sig = _get_bg_frame_bgr(sess, f_idx, src_bgr)
    fg_clean, alpha, m_key = _get_matte_and_fg(sess, f_idx, src_bgr, req)

    comp_key = (
        m_key,
        bg_sig,
        round(float(req.ambient_color_strength), 3),
        round(float(req.directional_light_strength), 3),
        round(float(req.black_point_match), 3),
        int(req.light_wrap_radius),
        round(float(req.light_wrap_strength), 3),
        bool(req.protect_screen),
        round(float(req.screen_glare), 3),
        round(float(req.grain_match), 3),
        round(float(req.bg_defocus), 2),
        bool(req.clean_ai_ghost),
        bool(req.linear_compositing),
        round(float(req.edge_blur_match), 2),
        bool(req.pass2_legacy),
    )
    cached = sess.comp_cache.get(comp_key)
    if cached is not None:
        sess.comp_cache.move_to_end(comp_key)
        return cached

    engine_label = _resolve_engine_label(req.matting_engine)
    ai_on = legacy_app._is_ai_engine(engine_label)
    rcfg = RelightConfig(
        ambient_color_strength=float(req.ambient_color_strength),
        directional_light_strength=float(req.directional_light_strength),
        light_wrap_Radius=int(req.light_wrap_radius),
        light_wrap_strength=float(req.light_wrap_strength),
        protect_emissive_screen=bool(req.protect_screen),
        screen_glare_opacity=float(req.screen_glare),
        grain_match=float(req.grain_match),
        bg_defocus_sigma=float(req.bg_defocus),
        clean_ai_ghost=bool(req.clean_ai_ghost),
        pass3_aa_and_restore=bool(req.pass2_legacy) if ai_on else True,
        linear_compositing=bool(req.linear_compositing),
        edge_blur_match=float(req.edge_blur_match),
        black_point_match=float(req.black_point_match),
    )
    scr_mask = (
        detect_emissive_screen_mask(fg_clean, (alpha > 0.15).astype(np.uint8))
        if req.protect_screen
        else np.zeros(src_bgr.shape[:2], dtype=np.uint8)
    )
    comp_bgr = composite_with_relighting(
        fg_clean_bgr=fg_clean,
        alpha=alpha,
        new_bg_bgr=bg_bgr,
        src_full_bgr=src_bgr,
        screen_mask=scr_mask,
        cfg=rcfg,
        frame_idx=f_idx,
    )
    bundle = (
        cv2.cvtColor(src_bgr, cv2.COLOR_BGR2RGB),
        cv2.cvtColor(comp_bgr, cv2.COLOR_BGR2RGB),
        cv2.cvtColor(fg_clean, cv2.COLOR_BGR2RGB),
        alpha,
        scr_mask,
    )
    sess.comp_cache[comp_key] = bundle
    sess.comp_cache.move_to_end(comp_key)
    while len(sess.comp_cache) > 6:
        sess.comp_cache.popitem(last=False)
    return bundle


def _run_render_job(sess: SessionState, job_id: str, req: RenderReq) -> None:
    job = sess.jobs[job_id]
    try:
        with sess.lock:
            src_frames = list(sess.src_frames)
            bg_frames = list(sess.bg_frames)
            confirmed = {int(k): load_mask(v) for k, v in sess.confirmed_masks.items() if v.is_file()}
            src_fps = float(sess.fps or 30.0)
            bg_fps = float(sess.bg_fps or src_fps)
            w, h = int(sess.width), int(sess.height)
            src_media = sess.src_media
            bg_media = sess.bg_media
            tracked_paths = list(sess.tracked_masks)
            work = sess.work_dir

        n_src = len(src_frames)
        if n_src == 0:
            raise RuntimeError("Không có khung hình nguồn để xuất.")

        engine_label = _resolve_engine_label(req.matting_engine)
        ai_on = legacy_app._is_ai_engine(engine_label)

        job.update(progress=0.05, message="Đang khởi tạo keyframe tách nền...")
        if not confirmed:
            bgr0 = _load_bgr(src_frames[0])
            alpha0 = run_birefnet_matting(bgr0, max_size=1024)
            alpha0 = refine_alpha_edge_controls(
                bgr0,
                alpha0,
                guided_radius=2,
                trimap_radius=6,
                edge_preserve=float(req.edge_preserve),
            )
            alpha0, _, _ = heal_universal_structural_defects(bgr0, alpha0)
            m0 = (alpha0 >= 0.42).astype(np.uint8)
            sess.birefnet_alpha_by_frame[0] = alpha0
            p0 = _save_confirmed_mask(sess, 0, m0, push_undo=False)
            confirmed[0] = load_mask(p0)

        matanyone2_paths: Optional[List[Path]] = None
        if n_src > 1 and req.matting_engine == "matanyone2":
            job.update(progress=0.10, message="Đang chạy MatAnyone 2 GPU Video Matting...")
            matanyone2_paths = run_matanyone2_video(
                src_frames,
                confirmed,
                work / "matanyone2_alpha",
                max_internal_size=int(req.matanyone_max_size),
                progress_cb=lambda done, tot: job.update(
                    progress=0.10 + 0.35 * (done / max(tot, 1)),
                    message=f"MatAnyone 2 GPU: {done}/{tot} frames...",
                ),
            )

        if matanyone2_paths is not None:
            alpha_or_mask_paths = [Path(p) for p in matanyone2_paths]
            is_alpha_stream = True
        elif n_src == 1:
            p_single = sess.confirmed_masks[next(iter(sess.confirmed_masks))]
            alpha_or_mask_paths = [p_single]
            is_alpha_stream = False
        else:
            if len(tracked_paths) != n_src:
                job.update(progress=0.12, message="Đang chạy SAM 2.1 Roto-Brush Tracking toàn bộ video...")
                tracked = track_roto_video_masks(
                    work / "src_frames",
                    work / "tracked_masks",
                    confirmed_masks=confirmed,
                    prefer_best_checkpoint=ai_on,
                    progress_cb=lambda f, tot: job.update(
                        progress=0.12 + 0.30 * ((f + 1) / max(tot, 1)),
                        message=f"SAM 2.1 Tracking: {f + 1}/{tot} frames...",
                    ),
                )
                with sess.lock:
                    sess.tracked_masks = [Path(p) for p in tracked]
                tracked_paths = [Path(p) for p in tracked]
            alpha_or_mask_paths = tracked_paths
            is_alpha_stream = False

        # Stratified 10% Multi-Model QA Audit & Closed-Loop Auto-Healing
        qa_report: Optional[Dict[str, Any]] = None
        if n_src > 2 and bool(req.auto_qa_heal):
            def _repropagate_healed(healed_kfs: Dict[int, np.ndarray]) -> None:
                if not healed_kfs:
                    return
                if is_alpha_stream and req.matting_engine == "matanyone2":
                    merged_kfs = {**confirmed, **healed_kfs}
                    h_sorted = sorted(healed_kfs.keys())
                    win_lo = max(0, h_sorted[0] - 24)
                    win_hi = min(n_src, h_sorted[-1] + 25)
                    slice_frames = src_frames[win_lo:win_hi]
                    slice_kfs = {k - win_lo: v for k, v in merged_kfs.items() if win_lo <= k < win_hi}
                    if slice_frames and slice_kfs:
                        heal_dir = work / "matanyone2_qa_heal"
                        reprop_paths = run_matanyone2_video(
                            slice_frames,
                            slice_kfs,
                            heal_dir,
                            max_internal_size=int(req.matanyone_max_size),
                        )
                        for local_i, rp in enumerate(reprop_paths):
                            g_i = win_lo + local_i
                            if g_i in healed_kfs:
                                continue
                            if rp.is_file():
                                shutil.copy2(rp, alpha_or_mask_paths[g_i])
                        shutil.rmtree(heal_dir, ignore_errors=True)

            qa_report = audit_and_heal_video_mattes(
                src_frames,
                alpha_or_mask_paths,
                is_alpha_stream=is_alpha_stream,
                sample_ratio=float(np.clip(req.qa_sample_pct, 0.05, 0.25)),
                min_samples=min(4, n_src),
                max_samples=20,
                use_birefnet_crosscheck=ai_on,
                heal_repropagate_cb=_repropagate_healed if is_alpha_stream else None,
                progress_cb=lambda frac, msg: job.update(
                    progress=0.45 + 0.05 * float(frac),
                    message=msg,
                ),
                confirmed_frame_indices=set(confirmed.keys()),
            )
            job["qa_report"] = qa_report

        # Camera tracking if requested and multiple frames exist
        cam_transforms = None
        v2v_transforms = None
        if bg_frames and n_src > 1:
            if "Camera Motion Lock" in req.tracking_mode:
                job.update(progress=0.46, message="Đang trích xuất chuyển động Camera gốc...")
                cam_transforms = smooth_transforms(estimate_source_camera_trajectory(src_frames, alpha_or_mask_paths))
            elif "Flow V2V" in req.tracking_mode and len(bg_frames) >= n_src:
                job.update(progress=0.46, message="Đang đồng bộ không gian Flow V2V...")
                v2v_transforms = smooth_transforms(
                    [estimate_pair_similarity(_load_bgr(src_frames[t]), _load_bgr(bg_frames[t % len(bg_frames)]), load_mask(alpha_or_mask_paths[t])) for t in range(n_src)]
                )

        # Sequential temporal alpha stabilization
        job.update(progress=0.50, message="Đang ổn định Alpha Matte theo thời gian...")
        mcfg = legacy_app._resolve_matting_cfg(
            engine_label,
            int(req.trimap_radius),
            int(req.guided_radius),
            float(req.decont_strength),
            bool(req.despill_green),
            0,
            bool(req.use_pymatting_fg),
            bool(req.pass2_legacy),
        )
        rcfg = RelightConfig(
            ambient_color_strength=float(req.ambient_color_strength),
            directional_light_strength=float(req.directional_light_strength),
            light_wrap_Radius=int(req.light_wrap_radius),
            light_wrap_strength=float(req.light_wrap_strength),
            protect_emissive_screen=bool(req.protect_screen),
            screen_glare_opacity=float(req.screen_glare),
            grain_match=float(req.grain_match),
            bg_defocus_sigma=float(req.bg_defocus),
            clean_ai_ghost=bool(req.clean_ai_ghost),
            pass3_aa_and_restore=bool(req.pass2_legacy) if ai_on else True,
            linear_compositing=bool(req.linear_compositing),
            edge_blur_match=float(req.edge_blur_match),
            black_point_match=float(req.black_point_match),
        )

        job.update(progress=0.52, message="Đang bóc màu viền, ghép ánh sáng & nén GPU H.264...")
        audio_src = (
            src_media
            if (req.audio_choice == "source" and not sess.src_is_image)
            else (bg_media if (req.audio_choice == "background" and not sess.bg_is_image) else None)
        )
        out_mp4 = work / "AGY_TachNen_Final.mp4"

        def _render_frame(
            t: int,
            src_bgr: np.ndarray,
            alpha_t: np.ndarray,
            fg_clean_opt: Optional[np.ndarray],
        ) -> np.ndarray:
            fg_clean = fg_clean_opt
            if fg_clean is None:
                fg_clean = (
                    estimate_foreground_pymatting(
                        src_bgr,
                        alpha_t,
                        strength=float(req.decont_strength),
                        despill_green=bool(req.despill_green),
                    )
                    if req.use_pymatting_fg
                    else src_bgr
                )

            if not bg_frames:
                bg_bgr = cv2.GaussianBlur(src_bgr, (0, 0), 22.0)
            else:
                if len(bg_frames) == 1:
                    t_bg = 0
                else:
                    mapped = int(round(t * (bg_fps / max(src_fps, 1.0))))
                    t_bg = mapped % len(bg_frames)
                bg_bgr = _load_bgr(bg_frames[t_bg])
                if (bg_bgr.shape[1], bg_bgr.shape[0]) != (w, h):
                    bg_bgr = cv2.resize(bg_bgr, (w, h), interpolation=cv2.INTER_CUBIC)
                if cam_transforms is not None and t < len(cam_transforms):
                    bg_bgr = warp_bg_with_camera_motion(bg_bgr, cam_transforms[t], (w, h))
                elif v2v_transforms is not None and t < len(v2v_transforms):
                    bg_bgr, _ = warp_ai_frame(bg_bgr, v2v_transforms[t].to_matrix(), (w, h))

            return composite_with_relighting(
                fg_clean_bgr=fg_clean,
                alpha=alpha_t,
                new_bg_bgr=bg_bgr,
                src_full_bgr=src_bgr,
                cfg=rcfg,
                frame_idx=t,
            )

        workers = min(8, max(4, os.cpu_count() or 4))
        max_in_flight = workers * 2
        in_flight: deque = deque()
        written_count = 0
        prev_alpha = None
        prev_src = None

        with (
            VideoPipeEncoder(out_mp4, w, h, fps=src_fps, audio_source_path=audio_src, prefer_nvenc=True) as enc,
            ThreadPoolExecutor(max_workers=workers) as ex,
        ):
            for t in range(n_src):
                src_bgr = _load_bgr(src_frames[t])
                raw_fg_opt: Optional[np.ndarray] = None
                if is_alpha_stream:
                    alpha_t = load_alpha(alpha_or_mask_paths[t])
                else:
                    if t in sess.birefnet_alpha_by_frame and req.matting_engine in ("birefnet", "matanyone2"):
                        alpha_t = sess.birefnet_alpha_by_frame[t]
                    else:
                        mask_t = load_mask(alpha_or_mask_paths[t])
                        mres = extract_pixel_accurate_matte(src_bgr, mask_t, mcfg)
                        alpha_t = mres.alpha
                        raw_fg_opt = mres.fg_clean_bgr

                alpha_t, bezel_repaired, _ = heal_universal_structural_defects(src_bgr, alpha_t)
                alpha_t = refine_alpha_edge_controls(
                    src_bgr,
                    alpha_t,
                    erode_dilate=float(req.erode_dilate),
                    guided_radius=int(req.guided_radius),
                    trimap_radius=int(req.trimap_radius),
                    edge_preserve=float(req.edge_preserve),
                    edge_feather=float(req.edge_blur_match),
                )
                if raw_fg_opt is not None and (
                    bezel_repaired > 0
                    or abs(float(req.erode_dilate)) >= 0.05
                    or float(req.edge_blur_match) > 0.05
                    or float(req.edge_preserve) > 0.01
                ):
                    raw_fg_opt = None

                alpha_stable = stabilize_alpha_temporal(
                    alpha_t,
                    prev_alpha,
                    src_bgr,
                    prev_src,
                    temporal_weight=0.25 if is_alpha_stream else 0.35,
                )
                prev_alpha = alpha_stable
                prev_src = src_bgr

                in_flight.append(ex.submit(_render_frame, t, src_bgr, alpha_stable, raw_fg_opt))
                if len(in_flight) >= max_in_flight:
                    enc.write(in_flight.popleft().result())
                    written_count += 1
                    if written_count % 8 == 0:
                        job.update(
                            progress=0.52 + 0.43 * (written_count / max(n_src, 1)),
                            message=f"Đang xử lý & xuất video GPU: {written_count}/{n_src} frames...",
                        )

            while in_flight:
                enc.write(in_flight.popleft().result())
                written_count += 1
                if written_count % 8 == 0 or written_count == n_src:
                    job.update(
                        progress=0.52 + 0.43 * (written_count / max(n_src, 1)),
                        message=f"Đang xử lý & xuất video GPU: {written_count}/{n_src} frames...",
                    )

        # Immediately purge heavy intermediate mask/alpha directories once MP4 is finalized
        shutil.rmtree(work / "matanyone2_alpha", ignore_errors=True)
        shutil.rmtree(work / "tracked_masks", ignore_errors=True)
        sess.tracked_masks = []
        release_neural_models()
        clear_vram()

        out_mb = round(out_mp4.stat().st_size / (1024 * 1024), 2)
        job.update(
            status="done",
            progress=1.0,
            message=f"Hoàn tất {n_src} frames ({out_mb} MB). Đã tự động dọn file trung gian.",
            output_path=str(out_mp4),
            output_mb=out_mb,
            disk_mb=_dir_size_mb(work),
        )
    except Exception as exc:
        traceback.print_exc()
        job.update(status="error", message=f"Lỗi render: {exc}")


# ---------------------------------------------------------------------------
# FastAPI Router (mounted at both "" and "/matting")
# ---------------------------------------------------------------------------
router = APIRouter()


@router.get("/", response_class=HTMLResponse)
def serve_index():
    idx_file = WEB_DIR / "index.html"
    if not idx_file.is_file():
        raise HTTPException(status_code=500, detail="Missing web/index.html")
    return HTMLResponse(
        content=idx_file.read_text(encoding="utf-8"),
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )


@router.get("/static/{filepath:path}")
def serve_static(filepath: str):
    target = (WEB_DIR / filepath).resolve()
    web_root = WEB_DIR.resolve()
    if not str(target).startswith(str(web_root)) or not target.is_file():
        raise HTTPException(status_code=404, detail="Not found")
    ctype, _ = mimetypes.guess_type(str(target))
    return FileResponse(str(target), media_type=ctype or "application/octet-stream", headers={"Cache-Control": "no-cache"})


@router.get("/api/health")
def health():
    stats = STORE.stats()
    return {
        "ok": True,
        "service": "matting-studio",
        "cuda_available": bool(torch.cuda.is_available()),
        **stats,
    }


@router.post("/api/session")
def create_session():
    _ensure_reaper()
    sess = STORE.create()
    return {"session_id": sess.session_id, "ttl_seconds": STORE.ttl_seconds}


@router.post("/api/session/{sid}/upload/init")
def upload_init(sid: str, req: UploadInitReq):
    sess = STORE.get(sid)
    uid = uuid.uuid4().hex[:12]
    safe_name = Path(req.filename or "upload.bin").name
    ext = Path(safe_name).suffix.lower() or ".bin"
    up_dir = sess.work_dir / "uploads"
    up_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = up_dir / f"{req.role}_{uid}{ext}"
    tmp_path.write_bytes(b"")
    with sess.lock:
        sess.uploads[uid] = {
            "role": req.role,
            "filename": safe_name,
            "path": tmp_path,
            "chunks": {},
            "total_size": int(req.total_size),
        }
    return {"upload_id": uid}


@router.put("/api/session/{sid}/upload/{uid}/chunk")
async def upload_chunk(sid: str, uid: str, request: Request, index: int = Query(0, ge=0)):
    sess = STORE.get(sid)
    with sess.lock:
        up = sess.uploads.get(uid)
        if up is None:
            raise HTTPException(status_code=404, detail="Upload session not found.")
        chunk_dir = up["path"].parent / f"{uid}_chunks"
        chunk_dir.mkdir(parents=True, exist_ok=True)

    body = await request.body()
    if len(body) > MAX_CHUNK_BYTES:
        raise HTTPException(status_code=413, detail="Chunk quá 16 MB.")
    chunk_file = chunk_dir / f"{int(index):06d}.part"
    chunk_file.write_bytes(body)
    with sess.lock:
        up["chunks"][int(index)] = chunk_file
    return {"ok": True, "index": int(index), "bytes": len(body)}


def _ingest_uploaded_media(dst_path: Path, frames_dir: Path, is_image: bool, filename: str) -> Tuple[List[Path], int, int, float, float]:
    shutil.rmtree(frames_dir, ignore_errors=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    try:
        if is_image:
            bgr = _load_bgr(dst_path)
            fp = frames_dir / "00000.jpg"
            cv2.imwrite(str(fp), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            h, w = bgr.shape[:2]
            return [fp], w, h, 30.0, 0.0

        meta = get_video_metadata(dst_path)
        frames = extract_frames(dst_path, frames_dir)
        fps = float(meta.fps or 30.0)
        w, h = int(meta.width), int(meta.height)
        if w <= 0 or h <= 0:
            sample_bgr = cv2.imread(str(frames[0]))
            if sample_bgr is None:
                raise RuntimeError("Không thể giải mã khung hình đầu tiên.")
            h, w = sample_bgr.shape[:2]
        dur = float(len(frames)) / max(fps, 1.0)
        return [Path(p) for p in frames], w, h, fps, dur
    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        dst_path.unlink(missing_ok=True)
        shutil.rmtree(frames_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=f"Không thể đọc file media '{filename}': {exc}") from exc


@router.post("/api/session/{sid}/upload/{uid}/complete")
def upload_complete(sid: str, uid: str):
    sess = STORE.get(sid)
    with sess.lock:
        up = sess.uploads.pop(uid, None)
    if up is None:
        raise HTTPException(status_code=404, detail="Upload not found.")

    dst_path: Path = up["path"]
    chunk_dir = dst_path.parent / f"{uid}_chunks"
    if up["chunks"]:
        with open(dst_path, "wb") as out_f:
            for idx in sorted(up["chunks"].keys()):
                part = up["chunks"][idx]
                out_f.write(part.read_bytes())
        shutil.rmtree(chunk_dir, ignore_errors=True)

    if not dst_path.exists() or dst_path.stat().st_size <= 0:
        dst_path.unlink(missing_ok=True)
        shutil.rmtree(chunk_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail="File tải lên bị rỗng (0 byte). Hãy chọn lại file.")

    ext = dst_path.suffix.lower()
    role = up["role"]
    is_image = ext in IMAGE_SUFFIXES
    frames_dir = sess.work_dir / ("src_frames" if role == "source" else "bg_frames")
    frames, w, h, fps, dur = _ingest_uploaded_media(dst_path, frames_dir, is_image, up["filename"])

    if role == "source":
        with sess.lock:
            if sess.src_media and Path(sess.src_media).exists():
                Path(sess.src_media).unlink(missing_ok=True)
            sess.src_media = str(dst_path)
            sess.src_is_image = is_image
            sess.src_frames = frames
            sess.fps = fps
            sess.width = w
            sess.height = h
            sess.confirmed_masks.clear()
            sess.undo_stacks.clear()
            sess.birefnet_alpha_by_frame.clear()
            sess.tracked_masks = []
            sess.matte_cache.clear()
            sess.comp_cache.clear()
    else:
        with sess.lock:
            if sess.bg_media and Path(sess.bg_media).exists():
                Path(sess.bg_media).unlink(missing_ok=True)
            sess.bg_media = str(dst_path)
            sess.bg_is_image = is_image
            sess.bg_frames = frames
            sess.bg_fps = fps
            sess.bg_cache.clear()
            sess.comp_cache.clear()

    return {
        "role": role,
        "filename": up["filename"],
        "frame_count": len(frames),
        "fps": round(fps, 2),
        "width": w,
        "height": h,
        "duration": round(dur, 2),
        "is_image": is_image,
        "disk_mb": _dir_size_mb(sess.work_dir),
    }


@router.get("/api/session/{sid}/frame/{idx}")
def get_frame(
    sid: str,
    idx: int,
    view: str = Query("raw"),
    max_dim: int = Query(1440, ge=64, le=4096),
):
    sess = STORE.get(sid)
    with sess.lock:
        if not sess.src_frames:
            raise HTTPException(status_code=404, detail="Chưa có video/ảnh gốc.")
        f_idx = int(np.clip(idx, 0, len(sess.src_frames) - 1))
        src_bgr = _load_bgr(sess.src_frames[f_idx])
        mask = _load_session_frame_mask(sess, f_idx)
        alpha = sess.birefnet_alpha_by_frame.get(f_idx)

    src_rgb = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2RGB)
    out_rgb = src_rgb
    if view == "overlay" and mask is not None and np.any(mask > 0):
        out_rgb = legacy_app._render_photoshop_green_selection(src_rgb, mask, alpha)
    elif view == "cutout" and (alpha is not None or mask is not None):
        a_use = alpha if alpha is not None else mask.astype(np.float32)
        out_rgb = legacy_app._make_checkerboard_cutout(src_bgr, a_use)
    elif view == "alpha" and (alpha is not None or mask is not None):
        a_use = alpha if alpha is not None else mask.astype(np.float32)
        u8 = np.clip(a_use * 255.0, 0, 255).astype(np.uint8)
        out_rgb = np.stack([u8, u8, u8], axis=-1)

    h, w = out_rgb.shape[:2]
    if max(h, w) > max_dim:
        scale = float(max_dim) / float(max(h, w))
        out_rgb = cv2.resize(out_rgb, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)

    out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", out_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise HTTPException(status_code=500, detail="Encode error")
    return Response(content=buf.tobytes(), media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@router.post("/api/session/{sid}/auto-cut")
def auto_cut(sid: str, req: AutoCutReq):
    sess = STORE.get(sid)
    with sess.lock:
        if not sess.src_frames:
            raise HTTPException(status_code=400, detail="Chưa tải video/ảnh gốc.")
        f_idx = int(np.clip(req.frame_idx, 0, len(sess.src_frames) - 1))
        src_bgr = _load_bgr(sess.src_frames[f_idx])

        use_ai = legacy_app._is_ai_engine(_resolve_engine_label(req.engine))
        if req.mode == "chroma":
            mask = auto_chroma_key_mask(src_bgr)
            sess.birefnet_alpha_by_frame.pop(f_idx, None)
        elif req.mode in ("hands_product", "universal_hybrid"):
            mask, alpha = auto_cut_universal_hybrid(
                src_bgr,
                max_size=1024,
                prefer_best_checkpoint=use_ai,
            )
            sess.birefnet_alpha_by_frame[f_idx] = alpha
        else:
            alpha_raw = run_birefnet_matting(src_bgr, max_size=1024)
            alpha = refine_alpha_edge_controls(
                src_bgr,
                alpha_raw,
                guided_radius=2,
                trimap_radius=6,
                edge_preserve=0.45,
            )
            alpha, _, _ = heal_universal_structural_defects(src_bgr, alpha)
            mask = (alpha >= 0.42).astype(np.uint8)
            sess.birefnet_alpha_by_frame[f_idx] = alpha

        _save_confirmed_mask(sess, f_idx, mask, push_undo=True)
        fg_pct = round(float(np.mean(mask > 0)) * 100.0, 1)
        confirmed_list = sorted(sess.confirmed_masks.keys())
        disk_mb = _dir_size_mb(sess.work_dir)

    return {"ok": True, "frame_idx": f_idx, "fg_pct": fg_pct, "confirmed_frames": confirmed_list, "disk_mb": disk_mb}


@router.post("/api/session/{sid}/strokes")
def apply_strokes(sid: str, req: StrokesReq):
    sess = STORE.get(sid)
    with sess.lock:
        if not sess.src_frames:
            raise HTTPException(status_code=400, detail="Chưa tải video/ảnh gốc.")
        f_idx = int(np.clip(req.frame_idx, 0, len(sess.src_frames) - 1))
        src_bgr = _load_bgr(sess.src_frames[f_idx])
        h, w = src_bgr.shape[:2]
        prev_mask = _load_session_frame_mask(sess, f_idx)
        prev_alpha = sess.birefnet_alpha_by_frame.get(f_idx)
        fg_strokes = _decode_b64_mask(req.fg_mask_b64, h, w)
        bg_strokes = _decode_b64_mask(req.bg_mask_b64, h, w)
        box_xyxy: Optional[Tuple[int, int, int, int]] = None
        if req.box_xyxy is not None:
            if len(req.box_xyxy) != 4:
                raise HTTPException(status_code=400, detail="Khung khoanh vùng cần đúng 4 tọa độ [x0, y0, x1, y1].")
            bx0, by0, bx1, by1 = (int(v) for v in req.box_xyxy)
            if bx1 - bx0 < 8 or by1 - by0 < 8:
                raise HTTPException(status_code=400, detail="Khung khoanh vùng quá nhỏ, hãy kéo rộng hơn quanh chủ thể.")
            box_xyxy = (bx0, by0, bx1, by1)

        has_fg = bool(np.any(fg_strokes > 0))
        has_bg = bool(np.any(bg_strokes > 0))
        if not has_fg and not has_bg and box_xyxy is None:
            raise HTTPException(status_code=400, detail="Hãy kéo khung chọn hoặc tô nét Xanh/Đỏ trên ảnh trước khi áp dụng.")
        if not has_fg and box_xyxy is None and (prev_mask is None or not np.any(prev_mask > 0)):
            raise HTTPException(status_code=400, detail="Chưa có vùng chọn để xóa. Hãy khoanh vùng, bấm 'AI Toàn diện' hoặc tô nét Xanh trước.")

        rgb = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2RGB)
        use_ai = legacy_app._is_ai_engine(_resolve_engine_label(req.engine))
        new_mask, new_alpha = smart_roto_brush_refine(
            rgb,
            prev_mask,
            prev_alpha,
            fg_strokes,
            bg_strokes,
            box_xyxy=box_xyxy,
            smart_mode=bool(req.smart_red_subtract and use_ai),
            prefer_best_checkpoint=use_ai,
        )

        _save_confirmed_mask(sess, f_idx, new_mask, push_undo=True)
        sess.birefnet_alpha_by_frame[f_idx] = new_alpha
        fg_pct = round(float(np.mean(new_mask > 0)) * 100.0, 1)
        confirmed_list = sorted(sess.confirmed_masks.keys())
        disk_mb = _dir_size_mb(sess.work_dir)

    return {"ok": True, "frame_idx": f_idx, "fg_pct": fg_pct, "confirmed_frames": confirmed_list, "disk_mb": disk_mb}


@router.post("/api/session/{sid}/undo")
def undo_mask(sid: str, req: FrameActionReq):
    sess = STORE.get(sid)
    with sess.lock:
        f_idx = int(max(0, req.frame_idx))
        stack = sess.undo_stacks.get(f_idx) or []
        if not stack:
            return {"ok": False, "message": "Không còn bước hoàn tác.", "confirmed_frames": sorted(sess.confirmed_masks.keys())}
        item = stack.pop()
        if isinstance(item, tuple):
            prev_mask, prev_alpha = item
        else:
            prev_mask, prev_alpha = item, None
        if prev_alpha is not None:
            sess.birefnet_alpha_by_frame[f_idx] = prev_alpha
        else:
            sess.birefnet_alpha_by_frame.pop(f_idx, None)
        _save_confirmed_mask(sess, f_idx, prev_mask, push_undo=False)
        return {"ok": True, "confirmed_frames": sorted(sess.confirmed_masks.keys()), "disk_mb": _dir_size_mb(sess.work_dir)}


@router.post("/api/session/{sid}/reset-mask")
def reset_mask(sid: str, req: FrameActionReq):
    sess = STORE.get(sid)
    with sess.lock:
        f_idx = int(max(0, req.frame_idx))
        old_p = sess.confirmed_masks.pop(f_idx, None)
        if old_p is not None and old_p.is_file():
            old_p.unlink(missing_ok=True)
        sess.birefnet_alpha_by_frame.pop(f_idx, None)
        sess.matte_cache.clear()
        sess.comp_cache.clear()
        return {"ok": True, "confirmed_frames": sorted(sess.confirmed_masks.keys()), "disk_mb": _dir_size_mb(sess.work_dir)}


@router.post("/api/session/{sid}/analyze-prompts")
def analyze_prompts(sid: str, req: AnalyzePromptsReq):
    sess = STORE.get(sid)
    with sess.lock:
        if not sess.src_frames:
            raise HTTPException(status_code=400, detail="Chưa tải video/ảnh gốc lên để phân tích.")
        frames = list(sess.src_frames)
        masks = [_load_session_frame_mask(sess, i) for i in range(len(frames))]

    profile = analyze_video_optical_profile(frames, fg_masks=masks, sample_count=4, use_vision_ai=bool(req.use_vision_ai))
    prompts = generate_matched_background_prompts(
        profile,
        target_background_vi=req.target_background_vi,
        subject_description_vi=req.subject_description_vi,
        use_local_llm=bool(req.use_local_llm),
    )
    sampled_urls = [
        f"api/session/{sid}/frame/{i}?view=raw&max_dim=480"
        for i in profile.get("sampled_indices", [0])
    ]
    return {
        "ok": True,
        "sampled_indices": profile.get("sampled_indices", [0]),
        "sampled_urls": sampled_urls,
        "telemetry_cards": prompts["telemetry_cards"],
        "clean_plate_prompt": prompts["clean_plate_prompt"],
        "v2v_prompt": prompts["v2v_prompt"],
        "still_bg_prompt": prompts["still_bg_prompt"],
        "full_markdown": prompts["full_markdown"],
    }


@router.post("/api/session/{sid}/preview")
def preview_composite(sid: str, req: PreviewReq):
    t0 = time.perf_counter()
    sess = STORE.get(sid)
    with sess.lock:
        src_rgb, comp_rgb, fg_rgb, alpha, scr_mask = _compute_preview_bundle(sess, req)

    diag_label = DIAG_MAP.get(req.diag_mode, legacy_app.PREVIEW_DIAG_SPLIT)
    left_rgb, right_rgb = legacy_app._render_preview_view(
        src_rgb,
        comp_rgb,
        fg_rgb,
        alpha,
        scr_mask,
        diag_mode=diag_label,
        zoom=1.0,
        pan_x=0.5,
        pan_y=0.5,
    )
    h, w = left_rgb.shape[:2]
    max_dim = max(256, min(2160, int(req.max_preview_dim or 1080)))
    if max(h, w) > max_dim:
        scale = float(max_dim) / float(max(h, w))
        nw, nh = int(round(w * scale)), int(round(h * scale))
        left_rgb = cv2.resize(left_rgb, (nw, nh), interpolation=cv2.INTER_AREA)
        right_rgb = cv2.resize(right_rgb, (nw, nh), interpolation=cv2.INTER_AREA)
        h, w = nh, nw

    return {
        "left_b64": _encode_jpeg_b64(left_rgb, quality=88),
        "right_b64": _encode_jpeg_b64(right_rgb, quality=88),
        "width": w,
        "height": h,
        "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 1),
    }


@router.post("/api/session/{sid}/render")
def start_render(sid: str, req: RenderReq):
    sess = STORE.get(sid)
    with sess.lock:
        if not sess.src_frames:
            raise HTTPException(status_code=400, detail="Chưa có video/ảnh gốc để xuất.")
        for active in sess.jobs.values():
            if active.get("status") == "running":
                return {"job_id": active["job_id"], "status": "running"}
        jid = uuid.uuid4().hex[:10]
        sess.jobs[jid] = {
            "job_id": jid,
            "status": "running",
            "progress": 0.01,
            "message": "Đang chuẩn bị render...",
            "output_path": "",
            "output_mb": 0.0,
            "disk_mb": _dir_size_mb(sess.work_dir),
        }

    t = threading.Thread(target=_run_render_job, args=(sess, jid, req), daemon=True)
    t.start()
    return {"job_id": jid, "status": "running"}


@router.get("/api/session/{sid}/render/{jid}")
def render_status(sid: str, jid: str):
    sess = STORE.get(sid)
    with sess.lock:
        job = sess.jobs.get(jid)
    if job is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy tác vụ render.")
    dl_url = f"api/session/{sid}/download?kind=video&job_id={jid}&cleanup=1" if job.get("status") == "done" else ""
    return {
        "job_id": jid,
        "status": job["status"],
        "progress": round(float(job.get("progress", 0.0)), 3),
        "message": job.get("message", ""),
        "output_mb": job.get("output_mb", 0.0),
        "disk_mb": job.get("disk_mb", 0.0),
        "download_url": dl_url,
        "qa_report": job.get("qa_report"),
    }


@router.get("/api/session/{sid}/download")
def download_result(
    sid: str,
    kind: str = Query("video"),
    job_id: str = Query(""),
    frame_idx: int = Query(0),
    cleanup: int = Query(1),
    engine: str = Query("matanyone2"),
    use_pymatting_fg: int = Query(1),
):
    sess = STORE.get(sid)
    if kind == "cutout":
        with sess.lock:
            if not sess.src_frames:
                raise HTTPException(status_code=400, detail="Chưa có khung hình gốc.")
            f_idx = int(np.clip(frame_idx, 0, len(sess.src_frames) - 1))
            src_bgr = _load_bgr(sess.src_frames[f_idx])
            req = PreviewReq(frame_idx=f_idx, matting_engine=engine, use_pymatting_fg=bool(use_pymatting_fg))
            fg_clean, alpha, _ = _get_matte_and_fg(sess, f_idx, src_bgr, req)
            a_u8 = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
            rgba = np.dstack([fg_clean, a_u8])
            file_path = sess.work_dir / f"cutout_frame_{f_idx:05d}.png"
            cv2.imwrite(str(file_path), rgba)
        media_type = "image/png"
        dl_name = f"tach_nen_f{f_idx:05d}.png"
    else:
        with sess.lock:
            job = sess.jobs.get(job_id) if job_id else None
            if job is None:
                done_jobs = [j for j in sess.jobs.values() if j.get("status") == "done"]
                job = done_jobs[-1] if done_jobs else None
        if job is None or not job.get("output_path") or not Path(job["output_path"]).is_file():
            raise HTTPException(status_code=404, detail="Chưa có video thành phẩm để tải.")
        file_path = Path(job["output_path"])
        media_type = "video/mp4"
        dl_name = "AGY_TachNen_GhepCanh.mp4"

    file_size = file_path.stat().st_size

    def _stream_and_purge():
        sent = 0
        try:
            with open(file_path, "rb") as f:
                while True:
                    chunk = f.read(256 * 1024)
                    if not chunk:
                        break
                    sent += len(chunk)
                    yield chunk
        finally:
            if int(cleanup) == 1 and sent == file_size:
                STORE.cleanup(sid, release_gpu=True)

    return StreamingResponse(
        _stream_and_purge(),
        media_type=media_type,
        headers={
            "Content-Disposition": f'attachment; filename="{dl_name}"',
            "Content-Length": str(file_size),
            "Cache-Control": "no-store",
        },
    )


@router.post("/api/session/{sid}/cleanup")
@router.delete("/api/session/{sid}")
def cleanup_session(sid: str):
    freed = STORE.cleanup(sid, release_gpu=True)
    return {"cleaned": True, "freed_mb": freed}


def _redirect_matting_slash(request: Request):
    loc = "/matting/" + (f"?{request.url.query}" if request.url.query else "")
    return RedirectResponse(url=loc, status_code=302)


def create_app() -> FastAPI:
    app = FastAPI(title="AGY SOTA Roto-Brush & Matting Studio")
    app.include_router(router, prefix="")
    app.include_router(router, prefix="/matting")
    app.add_api_route("/matting", _redirect_matting_slash, methods=["GET"], include_in_schema=False)
    return app


app = create_app()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AGY Roto-Brush & SOTA Matting FastAPI Studio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7883)
    args, _ = parser.parse_known_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
