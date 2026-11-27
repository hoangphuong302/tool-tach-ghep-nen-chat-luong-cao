from __future__ import annotations

import base64
import json
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from src.alpha_matting import detect_active_frame_box

LOCAL_ROUTER_URL = "http://127.0.0.1:20128/v1/chat/completions"
LOCAL_ROUTER_KEY = "sk-00b59870f0cdb29e-od5qn5-26b35cc0"
LOCAL_ROUTER_MODEL = "editvideo"


def analyze_scene_telemetry(
    bgr: np.ndarray,
    fg_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Extract quantitative optical & cinematographic telemetry from a source frame:
      - Active aspect ratio (auto-detecting 9:16 vertical crop inside 16:9 black bars)
      - Camera framing, perspective, horizon line, and foreground subject placement
      - Background Depth-of-Field (bokeh blur sigma & equivalent lens focal length/aperture)
      - Key light direction (4-quadrant luminance gradient) & Color Temperature (Kelvin)
    """
    h, w = bgr.shape[:2]
    x0, y0, x1, y1 = detect_active_frame_box(bgr)
    active_bgr = bgr[y0:y1, x0:x1]
    ah, aw = active_bgr.shape[:2]
    ar = aw / float(max(ah, 1))

    is_pillarboxed = (x0 > int(w * 0.08)) or (x1 < int(w * 0.92))
    if ar < 0.65:
        ar_label = "9:16 (Vertical Crop in 16:9)" if is_pillarboxed else "9:16"
        ar_en = "9:16 vertical portrait aspect ratio"
    elif ar > 1.45:
        ar_label = "16:9"
        ar_en = "16:9 horizontal widescreen aspect ratio"
    elif 0.9 <= ar <= 1.1:
        ar_label = "1:1"
        ar_en = "1:1 square aspect ratio"
    else:
        ar_label = f"{aw}:{ah}"
        ar_en = f"{aw}:{ah} aspect ratio"

    if fg_mask is not None and fg_mask.shape[:2] == (h, w):
        active_fg = (fg_mask[y0:y1, x0:x1] > 0).astype(np.uint8)
    else:
        # Estimate central foreground region if user hasn't clicked mask yet
        active_fg = np.zeros((ah, aw), dtype=np.uint8)
        active_fg[int(ah * 0.30): int(ah * 0.98), int(aw * 0.18): int(aw * 0.82)] = 1

    bg_zone = active_fg == 0
    if np.sum(bg_zone) < 200:
        bg_zone = np.ones((ah, aw), dtype=bool)

    # 1. Measure Background Depth-of-Field & Estimate Equivalent Smartphone Focal Length (mm)
    gray = cv2.cvtColor(active_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    fg_sharp = float(np.percentile(lap[active_fg > 0], 90)) if np.any(active_fg > 0) else 25.0
    bg_sharp = float(np.percentile(lap[bg_zone], 90))
    sharp_ratio = fg_sharp / max(bg_sharp, 1.0)

    # Photogrammetric foreground scale, focal length (mm), camera pitch & distance estimation
    ys, xs = np.where(active_fg > 0)
    if len(xs) > 0:
        fg_ratio = round(float(len(xs)) / float(max(ah * aw, 1)) * 100.0, 1)
        cy_norm = float(np.mean(ys)) / float(max(ah, 1))
        x_span_ratio = float(np.max(xs) - np.min(xs) + 1) / float(max(aw, 1))
    else:
        fg_ratio = 35.0
        cy_norm = 0.62
        x_span_ratio = 0.58

    if x_span_ratio >= 0.62:
        focal_mm = 24
        subj_dist_cm = 30
    elif x_span_ratio >= 0.45:
        focal_mm = 26
        subj_dist_cm = 36
    elif x_span_ratio >= 0.32:
        focal_mm = 28
        subj_dist_cm = 45
    else:
        focal_mm = 35
        subj_dist_cm = 60

    pitch_deg = int(round(-8.0 - max(0.0, cy_norm - 0.50) * 28.0))
    roll_deg = 0
    horizon_pct = int(round(np.clip(52.0 - (pitch_deg + 10) * 0.6, 38.0, 58.0)))

    scale_calibration_vi = (
        f"Ống kính điện thoại {focal_mm}mm (cảm biến phổ thông), cách bàn tay ~{subj_dist_cm}cm, "
        f"góc nghiêng {pitch_deg}° hướng nhẹ xuống, đường chân trời ở ~{horizon_pct}% khung hình -> "
        f"người và vật thể ở xa có tỷ lệ nhỏ tự nhiên đúng phối cảnh góc rộng {focal_mm}mm."
    )
    scale_calibration_en = (
        f"shot on a casual mid-range consumer smartphone {focal_mm}mm wide lens from a {subj_dist_cm}cm foreground distance "
        f"at a {abs(pitch_deg)}-degree downward POV pitch angle (0-degree roll), horizon at {horizon_pct}% frame height, "
        f"keeping background people and architectural elements proportionally scaled to a natural {focal_mm}mm perspective"
    )

    if sharp_ratio > 2.2 or bg_sharp < 18.0:
        bokeh_vi = f"Chất lượng điện thoại bình dân ({focal_mm}mm): nét tại tay/máy, hậu cảnh xa chỉ mờ nhòe rất nhẹ tự nhiên + nhiễu cảm biến nhẹ (KHÔNG xóa phông gắt DSLR)"
        bokeh_en = (
            f"casual mid-range smartphone optics ({focal_mm}mm f/2.2 small sensor) focused on the foreground plane at {subj_dist_cm}cm, "
            "producing only very gentle natural far-background softness and subtle organic mobile sensor noise/grain — NO heavy DSLR f/1.8 bokeh and NO studio CGI sharpness"
        )
    else:
        bokeh_vi = f"Chất lượng điện thoại bình dân ({focal_mm}mm): độ sâu trường ảnh rộng tự nhiên, hậu cảnh hơi mềm nhẹ và có nhiễu hạt cảm biến điện thoại thật"
        bokeh_en = (
            f"everyday consumer smartphone {focal_mm}mm lens aesthetic with natural deep mobile depth of field, "
            "very slight natural softness in the far distance, and authentic subtle handheld phone camera grain (no artificial DSLR portrait blur)"
        )

    # 2. Key Light Direction & Ambient Luminance
    mid_y, mid_x = ah // 2, aw // 2
    lum_top = float(np.mean(gray[:mid_y, :]))
    lum_bot = float(np.mean(gray[mid_y:, :]))
    lum_left = float(np.mean(gray[:, :mid_x]))
    lum_right = float(np.mean(gray[:, mid_x:]))

    vert_dir_en = "overhead / upper" if lum_top >= lum_bot - 5 else "eye-level"
    vert_dir_vi = "từ phía trên chếch xuống" if lum_top >= lum_bot - 5 else "ngang tầm mắt"
    if lum_left > lum_right + 8:
        horiz_en = "upper-left key light with soft right-side fill"
        horiz_vi = f"Nguồn sáng chính từ bên TRÁI ({vert_dir_vi}), ánh sáng phụ dịu bên phải"
    elif lum_right > lum_left + 8:
        horiz_en = "upper-right key light with soft left-side fill"
        horiz_vi = f"Nguồn sáng chính từ bên PHẢI ({vert_dir_vi}), ánh sáng phụ dịu bên trái"
    else:
        horiz_en = f"balanced diffused {vert_dir_en} ambient lighting"
        horiz_vi = f"Ánh sáng môi trường khuếch tán cân bằng ({vert_dir_vi})"

    # 3. Color Temperature (Kelvin approximation from B/R ratio in background)
    b_mean = float(np.mean(active_bgr[..., 0][bg_zone]))
    g_mean = float(np.mean(active_bgr[..., 1][bg_zone]))
    r_mean = float(np.mean(active_bgr[..., 2][bg_zone]))

    # Check if background is a green screen
    is_green_screen = (g_mean > max(r_mean, b_mean) + 32.0)
    if is_green_screen:
        temp_vi = "Quay trên Phông Xanh Studio (Ánh sáng trắng trung tính ~5200K trên tay & sản phẩm)"
        temp_en = "neutral 5200K diffused daylight matching the natural skin tone illumination on the hands"
    elif r_mean > b_mean + 18:
        temp_vi = "Ánh sáng Ấm (Warm Daylight / Golden Indoor ~4200K - 4800K)"
        temp_en = "warm 4500K natural ambient lighting with soft golden-neutral highlights"
    elif b_mean > r_mean + 18:
        temp_vi = "Ánh sáng Lạnh / Hiện đại (Cool Daylight / Terminal LED ~6000K)"
        temp_en = "clean 6000K cool daylight / modern architectural ambient illumination"
    else:
        temp_vi = "Ánh sáng Ban ngày Trung tính (Neutral Daylight ~5400K)"
        temp_en = "natural 5400K neutral daylight with accurate skin-tone rendering"

    angle_vi = (
        f"Góc quay POV góc nhìn thứ nhất (nghiêng {pitch_deg}° xuống tay/máy, độ nghiêng ngang {roll_deg}°), "
        f"đường chân trời ở khoảng {horizon_pct}% chiều cao khung hình ({focal_mm}mm)."
    )
    angle_en = (
        f"first-person POV eye-level shot angled ~{abs(pitch_deg)} degrees downward at the handheld device in the lower-center foreground, "
        f"horizon line positioned at ~{horizon_pct}% height matching a {focal_mm}mm mobile camera perspective"
    )

    return {
        "raw_resolution": f"{w}x{h}",
        "active_box": (x0, y0, x1, y1),
        "active_resolution": f"{aw}x{ah}",
        "active_aspect_ratio": ar_label,
        "aspect_ratio_en": ar_en,
        "is_pillarboxed": is_pillarboxed,
        "is_green_screen": is_green_screen,
        "foreground_occupancy_pct": fg_ratio,
        "subject_center_y_norm": round(cy_norm, 2),
        "focal_length_mm": focal_mm,
        "camera_pitch_deg": pitch_deg,
        "camera_roll_deg": roll_deg,
        "subject_distance_cm": subj_dist_cm,
        "scale_calibration_vi": scale_calibration_vi,
        "scale_calibration_en": scale_calibration_en,
        "camera_angle_vi": angle_vi,
        "camera_angle_en": angle_en,
        "bokeh_blur_level": bokeh_vi,
        "bokeh_en": bokeh_en,
        "lighting_direction": horiz_vi,
        "lighting_en": horiz_en,
        "color_temperature": temp_vi,
        "color_temp_en": temp_en,
    }


def _translate_bg_concept_Via_agy_router(
    target_background_vi: str,
    subject_description_vi: str,
    telemetry: Dict[str, Any],
) -> Optional[str]:
    """
    Call local AGY / 9router LLM (ds/deepseek-v4-flash on 127.0.0.1:20128) to craft a vivid,
    optically matched English environment description for Google Flow. Falls back cleanly if offline.
    """
    sys_msg = (
        "You are an expert Cinematographer and Google Flow (Omni 1.2 / Veo) Prompt Architect. "
        "Translate and expand the user's Vietnamese background concept into a concise, photorealistic "
        "2-sentence English environment description that strictly matches the provided camera angle, "
        "lighting direction, and depth-of-field bokeh. Output ONLY the English environment sentences, no markdown."
    )
    user_msg = (
        f"Vietnamese Background Idea: {target_background_vi}\n"
        f"Foreground Subject: {subject_description_vi}\n"
        f"Photogrammetric Lens & Realism Telemetry: {telemetry['aspect_ratio_en']}, "
        f"{telemetry['scale_calibration_en']}, {telemetry['bokeh_en']}, "
        f"{telemetry['lighting_en']}, {telemetry['color_temp_en']}."
    )
    payload = {
        "model": LOCAL_ROUTER_MODEL,
        "messages": [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.25,
        "max_tokens": 160,
    }
    try:
        req = urllib.request.Request(
            LOCAL_ROUTER_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {LOCAL_ROUTER_KEY}",
            },
        )
        with urllib.request.urlopen(req, timeout=4.5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            txt = data["choices"][0]["message"]["content"].strip().strip('"')
            if len(txt) > 25:
                return txt
    except Exception:
        pass
    return None


def _offline_bg_translation(target_background_vi: str) -> str:
    """Deterministic fallback dictionary + builder when local LLM router is unreachable."""
    t = target_background_vi.lower()
    presets = [
        (["thư viện", "library", "đọc sách", "sách", "reading"], "a warm cozy library and sunlit reading room with wooden bookshelves, books softly out of focus in the distance, and soft natural window light"),
        (["cà phê", "cafe", "coffee"], "a sophisticated modern glass-walled specialty coffee shop in Tokyo with warm wood textures, soft morning sunlight filtering through floor-to-ceiling windows, and gently blurred patrons in the distance"),
        (["sân bay", "airport", "check-in", "nhà ga"], "a bright, spacious international airport terminal departure hall with sleek modern architectural columns, glowing flight information boards, and softly blurred travelers with luggage walking in the background"),
        (["khách sạn", "hotel", "sảnh", "lobby"], "a luxury 5-star hotel lobby with polished marble floors, warm architectural cove lighting, minimalist indoor greenery, and soft ambient depth in the background"),
        (["phố", "street", "tokyo", "châu âu", "đi bộ"], "a vibrant upscale pedestrian boulevard with modern storefronts, trees, and gently out-of-focus pedestrians strolling under natural daylight"),
        (["văn phòng", "office", "họp", "công sở"], "a sleek contemporary tech executive office and glass meeting lounge with clean architectural lines and soft natural window light"),
        (["taxi", "xe hơi", "ô tô", "ghế sau"], "the comfortable rear cabin of a modern executive taxi moving smoothly through a city avenue, with natural daylight entering through the side windows"),
    ]
    for keys, desc in presets:
        if any(k in t for k in keys):
            return desc
    return (
        f"a photorealistic real-world environment ({target_background_vi}) with authentic architectural depth, "
        "natural ambient activity in the distance, and physically accurate perspective"
    )


def generate_flow_prompts(
    telemetry: Dict[str, Any],
    target_background_vi: str,
    subject_description_vi: str = "Hai bàn tay đang cầm thiết bị máy phiên dịch cầm tay màu đen có màn hình hiển thị giao diện Tiếng Việt",
    use_local_llm: bool = True,
) -> Dict[str, str]:
    """
    Generate production-ready Google Flow prompts (following flow-omni-v2v KEEP/CHANGE/CONTROL standard):
      1. v2v_prompt: For Google Flow Omni 1.2 Video-to-Video Background Replacement
      2. clean_plate_prompt: For generating a 100% Clean Empty Background Plate Video (no hands/device)
         with identical camera angle, focal length, bokeh, lighting, and handheld micro-motion
         so our tool can composite the extracted real hands + product onto it seamlessly.
    """
    bg_idea = (target_background_vi or "").strip() or "Sảnh sân bay quốc tế hiện đại, ánh sáng tự nhiên"
    subj_vi = (subject_description_vi or "").strip() or "Hai bàn tay cầm thiết bị sản phẩm ở tiền cảnh"

    ai_bg_en = _translate_bg_concept_Via_agy_router(bg_idea, subj_vi, telemetry) if use_local_llm else None
    if not ai_bg_en:
        ai_bg_en = _offline_bg_translation(bg_idea)

    # PROMPT 1: Google Flow Omni 1.2 Video-to-Video (KEEP / CHANGE / CONTROL)
    v2v_prompt = (
        f"Video-to-video background transformation in {telemetry['aspect_ratio_en']}, "
        f"{telemetry['scale_calibration_en']}. "
        f"KEEP the exact foreground hands, finger positions, hand gestures, and the handheld electronic device completely unchanged, "
        f"preserving the device's exact rectangular geometry, black bezel, physical buttons, and original on-screen Vietnamese UI text without any alteration. "
        f"KEEP the exact original {telemetry['camera_angle_en']} and subtle handheld camera motion. "
        f"CHANGE only the background environment behind the hands and device into {ai_bg_en}. "
        f"Render the scene with {telemetry['bokeh_en']}, aligning the background horizon and perspective lines seamlessly with the camera's micro-movements. "
        f"Illuminate the scene with {telemetry['lighting_en']} and {telemetry['color_temp_en']}, casting realistic soft ambient light wrap onto the hands and device edges without altering the screen content. "
        f"Maintain strict frame-to-frame temporal consistency with zero background flickering, zero device morphing into a smartphone, and natural everyday mobile video realism."
    )

    # PROMPT 2: Clean Empty Background Plate (for 100% Real Hand + Product Cutout Compositing)
    clean_plate_prompt = (
        f"Photorealistic clean background plate video in {telemetry['aspect_ratio_en']}, "
        f"{telemetry['scale_calibration_en']}. "
        f"The scene depicts {ai_bg_en}, with a completely clear, unobstructed empty foreground (no hands, no person, no handheld device in the foreground). "
        f"Captured with {telemetry['bokeh_en']}, focused around the empty {telemetry['subject_distance_cm']}cm foreground plane so far-away objects stay naturally recognizable with only slight mobile softness and authentic consumer phone sensor texture. "
        f"Lighting features {telemetry['lighting_en']} at {telemetry['color_temp_en']} with natural spatial light gradients and realistic consumer mobile dynamic range. "
        f"Include subtle, organic handheld phone camera breathing and micro-sway with strict frame-to-frame temporal stability, ready as a clean V2V compositing plate."
    )

    full_markdown = f"""### A. Phân Tích Quang Học Từ Ảnh/Video Gốc & Chiến Lược (AGY Photogrammetry Telemetry)
- **Độ phân giải & Tỷ lệ khung hình thực:** `{telemetry['active_resolution']}` — **{telemetry['active_aspect_ratio']}** {"*(Đã phát hiện viền đen hai bên, tự động đo theo khung dọc thật ở giữa)*" if telemetry['is_pillarboxed'] else ""}
- **Tiêu cự & Tỷ lệ phối cảnh (Photogrammetry):** `{telemetry['focal_length_mm']}mm` | Khoảng cách tay-máy ~`{telemetry['subject_distance_cm']}cm` | {telemetry['scale_calibration_vi']}
- **Góc máy & Bố cục (Camera Framing):** {telemetry['camera_angle_vi']} (Chủ thể chiếm ~`{telemetry['foreground_occupancy_pct']}%` khung hình).
- **Quang học Điện thoại Bình dân (Smart DoF):** {telemetry['bokeh_blur_level']}
- **Hướng sáng & Nhiệt độ màu gốc:** {telemetry['lighting_direction']} | {telemetry['color_temperature']}
- **Giữ nguyên (KEEP):** {subj_vi} (Tách chuẩn từng sub-pixel từ video gốc: giữ 100% bàn tay thật, thiết bị thật và chữ tiếng Việt sắc nét trên màn hình).
- **Thay đổi (CHANGE):** Đổi phông nền sang **{bg_idea}** với đúng tiêu cự `{telemetry['focal_length_mm']}mm`, kích thước vật thể ở xa thực tế, góc nghiêng camera và ánh sáng gốc.

---

### B1. PROMPT TẠO VIDEO PHÔNG NỀN SẠCH (KHUYÊN DÙNG ĐỂ GHÉP TAY + SẢN PHẨM THẬT 100%)
*(Copy prompt này sang **Google Flow / Veo** để tạo 1 video nền trống đúng tiêu cự `{telemetry['focal_length_mm']}mm`, chất lượng quay điện thoại bình dân tự nhiên, không xóa phông giả, không dính tay/máy ảo)*:
```text
{clean_plate_prompt}
```

---

### B2. PROMPT GOOGLE FLOW OMNI 1.2 (CHẾ ĐỘ VIDEO-TO-VIDEO ĐỔI NỀN TRỰC TIẾP)
*(Dùng khi Sếp tải thẳng Video gốc lên **Google Flow Omni 1.2 V2V** để AI đổi nền bám theo rung lắc máy quay gốc, sau đó đưa video AI về Tool để đè tay thật + màn hình thật lên)*:
```text
{v2v_prompt}
```

---

### C. THÔNG TIN TÙY CHỈNH NHANH
- **[Bối cảnh nền mới]:** `{bg_idea}` -> `{ai_bg_en}`
- **[Tiêu cự & Tỷ lệ phối cảnh]:** `{telemetry['scale_calibration_en']}`
- **[Độ sâu trường ảnh Điện thoại]:** `{telemetry['bokeh_en']}`
- **[Hướng sáng & Màu nắng]:** `{telemetry['lighting_en']}`, `{telemetry['color_temp_en']}`
"""

    return {
        "v2v_prompt": v2v_prompt,
        "clean_plate_prompt": clean_plate_prompt,
        "ai_bg_en": ai_bg_en,
        "full_markdown": full_markdown,
    }


def _load_frame_bgr(item: Union[str, Path, np.ndarray]) -> Optional[np.ndarray]:
    if isinstance(item, np.ndarray):
        return item
    img = cv2.imread(str(item))
    return img


def _extract_skin_roi_mask(bgr: np.ndarray, fg_mask: np.ndarray) -> np.ndarray:
    """Segment true foreground skin/hand pixels while excluding glowing screens."""
    fg = (fg_mask > 0)
    if not np.any(fg):
        return fg
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    b = bgr[..., 0].astype(np.float32)
    g = bgr[..., 1].astype(np.float32)
    r = bgr[..., 2].astype(np.float32)
    skin = (
        fg
        & (r > g + 6.0)
        & (g >= b - 4.0)
        & (r > 70.0)
        & (hsv[..., 1] > 25)
        & ((hsv[..., 0] <= 24) | (hsv[..., 0] >= 160))
    )
    if int(np.sum(skin)) < max(80, int(bgr.shape[0] * bgr.shape[1] * 0.002)):
        return fg
    return skin


def _estimate_vision_notes(frames: List[np.ndarray]) -> str:
    """Best-effort compact 2x2 contact sheet Vision AI optical read via local 9router."""
    if not frames:
        return ""
    tiles: List[np.ndarray] = []
    for f in frames[:4]:
        h, w = f.shape[:2]
        scale = min(1.0, 448.0 / float(max(h, w, 1)))
        tiles.append(cv2.resize(f, (max(64, int(round(w * scale))), max(64, int(round(h * scale)))), interpolation=cv2.INTER_AREA))
    while len(tiles) < 4:
        tiles.append(tiles[-1].copy())
    th, tw = tiles[0].shape[:2]
    norm_tiles = [cv2.resize(t, (tw, th), interpolation=cv2.INTER_AREA) for t in tiles[:4]]
    sheet = np.vstack([np.hstack([norm_tiles[0], norm_tiles[1]]), np.hstack([norm_tiles[2], norm_tiles[3]])])
    ok, buf = cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not ok:
        return ""
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")

    candidates = ["gemini/gemini-3-flash-preview", "gemini/gemini-3.6-flash"]
    settings_path = Path.home() / ".ai-video-material-kit" / "settings.json"
    if settings_path.is_file():
        try:
            cfg = json.loads(settings_path.read_text(encoding="utf-8"))
            vm = str(cfg.get("vision_model") or "").strip()
            if vm and vm.upper() != "FREE" and vm not in candidates:
                candidates.insert(0, vm)
        except Exception:
            pass

    prompt = (
        "Inspect this 2x2 contact sheet of a video clip. In 2 concise sentences describe ONLY: "
        "(1) what the foreground subject is, camera POV/angle, shot distance, and real-world scale cues, and "
        "(2) key-light direction, shadow softness on the subject/clothes/hands, and background lens blur."
    )
    for model_name in candidates:
        payload = {
            "model": model_name,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            }],
            "max_tokens": 160,
            "temperature": 0.2,
            "stream": False,
        }
        try:
            req = urllib.request.Request(
                LOCAL_ROUTER_URL,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {LOCAL_ROUTER_KEY}"},
            )
            with urllib.request.urlopen(req, timeout=6.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                txt = str(((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()
                if len(txt) > 20:
                    return txt
        except Exception:
            continue
    return ""


def analyze_video_optical_profile(
    frames_or_paths: Sequence[Union[str, Path, np.ndarray]],
    fg_masks: Optional[Sequence[Optional[np.ndarray]]] = None,
    sample_count: int = 4,
    use_vision_ai: bool = True,
) -> Dict[str, Any]:
    """
    Analyze 1..N sampled frames from the source video/image to extract a full
    photogrammetric & lighting profile for photorealistic background generation:
      - Hand size/occupancy, wrist entry orientation & background scale calibration
      - True key-light vector & opposite cast-shadow direction measured on skin
      - Highlight-to-shadow hardness, exposure (EV) & Kelvin temperature
      - Background DoF blur (sigma/aperture) & sensor grain level
      - Multi-frame camera motion (handheld breathing / static / sway)
    """
    items = list(frames_or_paths or [])
    if not items:
        raise ValueError("Cần ít nhất 1 khung hình để phân tích quang học.")

    n = len(items)
    if n <= 1:
        sampled_indices = [0]
    else:
        ratios = np.linspace(0.08, 0.92, max(2, min(int(sample_count), n)))
        sampled_indices = sorted({int(np.clip(round(r * (n - 1)), 0, n - 1)) for r in ratios})

    loaded_frames: List[np.ndarray] = []
    loaded_masks: List[Optional[np.ndarray]] = []
    for idx in sampled_indices:
        bgr = _load_frame_bgr(items[idx])
        if bgr is not None:
            loaded_frames.append(bgr)
            m = fg_masks[idx] if (fg_masks is not None and idx < len(fg_masks)) else None
            loaded_masks.append(m)

    if not loaded_frames:
        raise ValueError("Không thể đọc khung hình hợp lệ để phân tích.")

    base = analyze_scene_telemetry(loaded_frames[0], fg_mask=loaded_masks[0])
    hand_w_list: List[float] = []
    hand_h_list: List[float] = []
    fg_occ_list: List[float] = []
    gx_list: List[float] = []
    gy_list: List[float] = []
    contrast_list: List[float] = []
    luma_list: List[float] = []
    kelvin_list: List[float] = []
    bg_blur_list: List[float] = []
    grain_list: List[float] = []
    entry_scores = {"left": 0.0, "right": 0.0, "center": 0.0}
    has_meaningful_skin = False
    fg_spans_portrait = False

    for bgr, fmask in zip(loaded_frames, loaded_masks):
        x0, y0, x1, y1 = detect_active_frame_box(bgr)
        act = bgr[y0:y1, x0:x1]
        ah, aw = act.shape[:2]
        if fmask is not None and fmask.shape[:2] == bgr.shape[:2]:
            fg_act = (fmask[y0:y1, x0:x1] > 0).astype(np.uint8)
        else:
            fg_act = np.zeros((ah, aw), dtype=np.uint8)
            fg_act[int(ah * 0.20):int(ah * 0.96), int(aw * 0.14):int(aw * 0.86)] = 1

        fg_bool = fg_act > 0
        skin = _extract_skin_roi_mask(act, fg_act)
        skin_frac_frame = float(np.mean(skin)) if np.any(skin) else 0.0
        fg_frac_frame = max(1e-4, float(np.mean(fg_bool)))
        use_skin_roi = (skin_frac_frame >= 0.02) and (skin_frac_frame / fg_frac_frame >= 0.15)
        if use_skin_roi:
            has_meaningful_skin = True
        measure_roi = skin if use_skin_roi else fg_bool

        fg_occ_list.append(float(np.mean(fg_bool)) * 100.0)
        fy, fx = np.where(fg_bool)
        if len(fx) > 0:
            hand_w_list.append(float(np.max(fx) - np.min(fx) + 1) / float(max(aw, 1)) * 100.0)
            f_h_ratio = float(np.max(fy) - np.min(fy) + 1) / float(max(ah, 1))
            hand_h_list.append(f_h_ratio * 100.0)
            if f_h_ratio >= 0.62 and float(np.min(fy)) <= float(ah) * 0.30:
                fg_spans_portrait = True
            bot = measure_roi[int(ah * 0.70):ah, :]
            if np.any(bot):
                bw = bot.shape[1]
                entry_scores["left"] += float(np.mean(bot[:, :max(1, bw // 3)]))
                entry_scores["center"] += float(np.mean(bot[:, bw // 3:max(bw // 3 + 1, (2 * bw) // 3)]))
                entry_scores["right"] += float(np.mean(bot[:, (2 * bw) // 3:]))

        gray = cv2.cvtColor(act, cv2.COLOR_BGR2GRAY).astype(np.float32)
        y_m, x_m = np.where(measure_roi)
        if len(x_m) > 40:
            vals = gray[y_m, x_m]
            xm = float(np.mean(x_m))
            ym = float(np.mean(y_m))
            dx = (x_m.astype(np.float32) - xm) / max(float(aw), 1.0)
            dy = (y_m.astype(np.float32) - ym) / max(float(ah), 1.0)
            dv = vals - float(np.mean(vals))
            gx_list.append(float(np.mean(dx * dv)))
            gy_list.append(float(np.mean(dy * dv)))
            p15, p50, p90 = np.percentile(vals, [15, 50, 90])
            contrast_list.append(float(p90 - p15) / max(float(p50), 12.0))
            luma_list.append(float(np.mean(vals)))
        else:
            contrast_list.append(0.45)
            luma_list.append(float(np.mean(gray)))

        bg_bool = ~fg_bool
        if int(np.sum(bg_bool)) < 150:
            bg_bool = np.ones((ah, aw), dtype=bool)
        b_m = float(np.mean(act[..., 0][bg_bool]))
        r_m = float(np.mean(act[..., 2][bg_bool]))
        rb_diff = r_m - b_m
        kelvin_list.append(float(np.clip(5400.0 - rb_diff * 24.0, 3200.0, 6800.0)))

        lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
        fg_s = float(np.percentile(lap[fg_bool], 90)) if np.any(fg_bool) else 25.0
        bg_s = float(np.percentile(lap[bg_bool], 90))
        ratio = fg_s / max(bg_s, 1.0)
        bg_blur_list.append(float(np.clip(0.4 + max(0.0, ratio - 1.1) * 0.75, 0.4, 3.8)))

        blur_ref = cv2.GaussianBlur(gray, (5, 5), 1.0)
        hp = np.abs(gray[bg_bool] - blur_ref[bg_bool])
        mad_sigma = float(np.median(hp)) * 1.4826 / 255.0
        grain_list.append(mad_sigma)

    # Multi-frame background camera motion
    motion_vals: List[float] = []
    if len(loaded_frames) >= 2:
        prev_small = cv2.resize(cv2.cvtColor(loaded_frames[0], cv2.COLOR_BGR2GRAY), (240, 240))
        for nxt in loaded_frames[1:]:
            cur_small = cv2.resize(cv2.cvtColor(nxt, cv2.COLOR_BGR2GRAY), (240, 240))
            flow = cv2.calcOpticalFlowFarneback(prev_small, cur_small, None, 0.5, 2, 13, 2, 5, 1.1, 0)
            mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            motion_vals.append(float(np.median(mag)))
            prev_small = cur_small
    avg_motion = round(float(np.mean(motion_vals)), 2) if motion_vals else 0.0

    hand_w_pct = round(float(np.mean(hand_w_list)) if hand_w_list else base["foreground_occupancy_pct"], 1)
    hand_h_pct = round(float(np.mean(hand_h_list)) if hand_h_list else 48.0, 1)
    fg_occ_pct = round(float(np.mean(fg_occ_list)), 1)

    if fg_spans_portrait:
        shot_kind = "portrait_medium"
        est_dist_label = "~1.2–2.0m"
    elif has_meaningful_skin and not fg_spans_portrait:
        shot_kind = "close_handheld"
        est_dist_label = f"~{base['subject_distance_cm']}cm"
    else:
        shot_kind = "subject_object"
        est_dist_label = "~60–120cm"

    if entry_scores["left"] > entry_scores["right"] * 1.35 and entry_scores["left"] > entry_scores["center"] * 1.15:
        wrist_vi = "Chủ thể / tay đưa vào từ góc DƯỚI-TRÁI khung hình"
        wrist_en = "foreground subject enters naturally from the bottom-left frame edge"
    elif entry_scores["right"] > entry_scores["left"] * 1.35 and entry_scores["right"] > entry_scores["center"] * 1.15:
        wrist_vi = "Chủ thể / tay đưa vào từ góc DƯỚI-PHẢI khung hình"
        wrist_en = "foreground subject enters naturally from the bottom-right frame edge"
    elif fg_spans_portrait:
        wrist_vi = "Chủ thể chính đứng/ngồi cân đối giữa khung hình"
        wrist_en = "foreground subject is framed in a balanced medium-shot composition"
    else:
        wrist_vi = "Chủ thể bố trí tập trung ở phần giữa-dưới khung hình"
        wrist_en = "foreground subject is anchored naturally near the center/lower-center frame"

    gx = float(np.mean(gx_list)) if gx_list else 0.0
    gy = float(np.mean(gy_list)) if gy_list else -1.0
    h_light = "left" if gx < -0.55 else ("right" if gx > 0.55 else "balanced")
    v_light = "top" if gy < -0.45 else ("bottom" if gy > 0.45 else "eye-level")

    if h_light == "left" and v_light == "top":
        light_dir_vi = "Trái-Trên chiếu chếch xuống"
        light_dir_en = "upper-left key light angled downward (~40° elevation)"
        shadow_vi = "Đổ bóng mềm về phía DƯỚI-PHẢI chủ thể"
        shadow_en = "soft cast shadows falling consistently toward the lower-right side of props and surfaces"
    elif h_light == "right" and v_light == "top":
        light_dir_vi = "Phải-Trên chiếu chếch xuống"
        light_dir_en = "upper-right key light angled downward (~40° elevation)"
        shadow_vi = "Đổ bóng mềm về phía DƯỚI-TRÁI chủ thể"
        shadow_en = "soft cast shadows falling consistently toward the lower-left side of props and surfaces"
    elif h_light == "left":
        light_dir_vi = "Chếch từ bên TRÁI"
        light_dir_en = "directional key light coming from the left side"
        shadow_vi = "Đổ bóng nhẹ sang bên PHẢI"
        shadow_en = "gentle natural shadows falling toward the right side"
    elif h_light == "right":
        light_dir_vi = "Chếch từ bên PHẢI"
        light_dir_en = "directional key light coming from the right side"
        shadow_vi = "Đổ bóng nhẹ sang bên TRÁI"
        shadow_en = "gentle natural shadows falling toward the left side"
    else:
        light_dir_vi = "Ánh sáng trần / khuếch tán đều phía trên"
        light_dir_en = "balanced overhead diffused ambient illumination"
        shadow_vi = "Bóng đổ mềm trực tiếp ngay dưới vật thể (không gắt cạnh)"
        shadow_en = "soft, diffused contact shadows directly beneath background objects with no harsh edges"

    contrast_score = float(np.mean(contrast_list))
    if contrast_score >= 0.92:
        hard_vi = "Ánh sáng có độ tương phản rõ (bóng viền khá sắc)"
        hard_en = "medium-crisp directional lighting with moderate shadow definition"
    elif contrast_score >= 0.50:
        hard_vi = "Ánh sáng tự nhiên bán khuếch tán (bóng mềm vừa)"
        hard_en = "natural semi-diffused lighting with soft-edged shadows"
    else:
        hard_vi = "Ánh sáng khuếch tán rất mịn (gần như không gắt bóng)"
        hard_en = "heavily diffused soft ambient illumination with ultra-soft shadow transitions"

    mean_luma = round(float(np.mean(luma_list)), 1)
    if mean_luma < 85.0:
        exp_vi = f"Phơi sáng dịu / thiếu sáng nhẹ (Luma {int(mean_luma)}/255)"
        exp_en = "slightly moody low-to-mid exposure typical of indoor handheld video (avoid blown highlights)"
    elif mean_luma > 165.0:
        exp_vi = f"Phơi sáng sáng rõ ban ngày (Luma {int(mean_luma)}/255)"
        exp_en = "bright well-lit daylight exposure matching natural daytime skin highlights"
    else:
        exp_vi = f"Phơi sáng cân bằng tự nhiên (Luma {int(mean_luma)}/255)"
        exp_en = "balanced natural exposure matching real-world consumer phone auto-exposure"

    kelvin_est = int(round(float(np.mean(kelvin_list)) / 100.0) * 100)
    bg_blur_sigma = round(float(np.mean(bg_blur_list)), 2)
    aperture_est = "f/1.8" if bg_blur_sigma >= 2.2 else ("f/2.2" if bg_blur_sigma >= 1.1 else "f/2.8")
    grain_sigma = round(float(np.mean(grain_list)), 4)

    if len(loaded_frames) == 1 or avg_motion < 0.35:
        motion_vi = "Máy quay rất tĩnh / gần như cố định"
        motion_en = "very steady near-locked handheld framing with minimal imperceptible breathing"
    elif avg_motion < 2.2:
        motion_vi = f"Rung nhẹ tự nhiên kiểu cầm tay (~{avg_motion}px/step)"
        motion_en = "subtle organic handheld smartphone micro-breathing and gentle natural sway"
    else:
        motion_vi = f"Di chuyển / lia tay vừa (~{avg_motion}px/step)"
        motion_en = "gentle continuous handheld smartphone walk/sway motion tracking smoothly"

    if shot_kind == "portrait_medium":
        bg_scale_rule_vi = (
            f"Chủ thể chiếm ~{hand_w_pct}%×{hand_h_pct}% khung ở khoảng trung cảnh {est_dist_label} -> "
            f"nội thất/ghế/kệ ngay phía sau và cảnh ở xa 3–6m phải đúng tỷ lệ góc nhìn {base['focal_length_mm']}mm."
        )
        bg_scale_rule_en = (
            f"Because the medium-framed foreground subject occupies ~{hand_w_pct}%x{hand_h_pct}% of the frame at {est_dist_label} "
            f"({wrist_en}), align mid-ground furniture/shelves (1.8–3m away) and distant architectural elements (4–8m away) "
            f"to a natural {base['focal_length_mm']}mm perspective with consistent floor/horizon geometry."
        )
    else:
        bg_scale_rule_vi = (
            f"Chủ thể tiền cảnh rộng ~{hand_w_pct}% khung ở khoảng cách {est_dist_label} -> "
            f"bàn/vật dụng cách ~1.2–2.5m và hậu cảnh xa 3–5m phải có kích thước nhỏ đúng phối cảnh {base['focal_length_mm']}mm."
        )
        bg_scale_rule_en = (
            f"Because the foreground subject occupies ~{hand_w_pct}% frame width at {est_dist_label} distance "
            f"({wrist_en}), keep mid-ground tables/surfaces (1.5–2.5m away) and distant people/architecture (3–6m away) "
            f"accurately scaled to a real {base['focal_length_mm']}mm lens perspective — never oversized close-up props right behind the subject."
        )

    vision_notes = _estimate_vision_notes(loaded_frames) if use_vision_ai else ""
    analysis_mode = f"Multi-Frame CV ({len(loaded_frames)} khung)" + (" + Vision AI" if vision_notes else "")

    horizon_pct = int(round(np.clip(52.0 - (base["camera_pitch_deg"] + 10) * 0.6, 38.0, 58.0)))
    telemetry_cards = [
        {
            "id": "lens_scale",
            "icon": "📐",
            "title": "Tiêu cự & Tỷ lệ chủ thể",
            "value": f"{base['focal_length_mm']}mm · {est_dist_label} · Chủ thể ~{hand_w_pct}%",
            "detail": bg_scale_rule_vi,
        },
        {
            "id": "camera_angle",
            "icon": "🎥",
            "title": "Góc máy & Chuyển động",
            "value": f"Nghiêng {base['camera_pitch_deg']}° · Chân trời ~{horizon_pct}% · {base['active_aspect_ratio']}",
            "detail": f"{wrist_vi}. {motion_vi}.",
        },
        {
            "id": "key_light",
            "icon": "☀️",
            "title": "Hướng sáng chủ thể & Bóng đổ",
            "value": light_dir_vi,
            "detail": f"{shadow_vi}. {hard_vi}.",
        },
        {
            "id": "exposure_kelvin",
            "icon": "🌡️",
            "title": "Nhiệt độ màu & Phơi sáng",
            "value": f"~{kelvin_est}K · Luma {int(mean_luma)}",
            "detail": exp_vi,
        },
        {
            "id": "dof_grain",
            "icon": "🔍",
            "title": "Độ nét, Độ mờ nền & Hạt nhiễu",
            "value": f"{aperture_est} · Blur ~{bg_blur_sigma}px · Grain {grain_sigma:.3f}",
            "detail": base["bokeh_blur_level"],
        },
        {
            "id": "mode",
            "icon": "🤖",
            "title": "Phân tích đa khung hình",
            "value": analysis_mode,
            "detail": vision_notes or f"Đã quét đồng bộ các frame {sampled_indices} để chống lệch góc sáng và tỷ lệ vật thể.",
        },
    ]

    profile = dict(base)
    profile.update({
        "sampled_indices": sampled_indices,
        "foreground_occupancy_pct": fg_occ_pct,
        "hand_width_pct": hand_w_pct,
        "hand_height_pct": hand_h_pct,
        "wrist_entry_vi": wrist_vi,
        "wrist_entry_en": wrist_en,
        "lighting_direction": light_dir_vi,
        "lighting_en": light_dir_en,
        "shadow_direction_vi": shadow_vi,
        "shadow_direction_en": shadow_en,
        "light_hardness_vi": hard_vi,
        "light_hardness_en": hard_en,
        "exposure_luma": mean_luma,
        "exposure_vi": exp_vi,
        "exposure_en": exp_en,
        "kelvin_est": kelvin_est,
        "bg_blur_sigma_px": bg_blur_sigma,
        "aperture_est": aperture_est,
        "grain_sigma": grain_sigma,
        "camera_motion_px": avg_motion,
        "camera_motion_vi": motion_vi,
        "camera_motion_en": motion_en,
        "shot_kind": shot_kind,
        "subject_distance_label": est_dist_label,
        "bg_scale_rule_vi": bg_scale_rule_vi,
        "bg_scale_rule_en": bg_scale_rule_en,
        "vision_notes": vision_notes,
        "analysis_mode": analysis_mode,
        "telemetry_cards": telemetry_cards,
    })
    return profile


def _craft_matched_environment_via_llm(
    target_background_vi: str,
    subject_description_vi: str,
    profile: Dict[str, Any],
) -> Optional[str]:
    sys_msg = (
        "You are an expert Cinematographer & VFX Compositing Supervisor. "
        "Translate the user's Vietnamese background description into 2-3 crisp, photorealistic English sentences "
        "describing ONLY the empty real-world environment and background props scaled accurately to match the camera telemetry. "
        "Never include hands, phones, or foreground objects in the description. Output ONLY English prose."
    )
    user_msg = (
        f"Vietnamese Background Idea: {target_background_vi}\n"
        f"Foreground Subject to Composite Later: {subject_description_vi}\n"
        f"Optical Profile: {profile['aspect_ratio_en']}, {profile['focal_length_mm']}mm smartphone lens, "
        f"{profile['camera_angle_en']}, {profile['bg_scale_rule_en']}, "
        f"{profile['lighting_en']}, {profile['shadow_direction_en']}, {profile['light_hardness_en']}, "
        f"~{profile['kelvin_est']}K, {profile['exposure_en']}, {profile['aperture_est']} subtle DoF (~{profile['bg_blur_sigma_px']}px), "
        f"{profile['camera_motion_en']}. Vision note: {profile.get('vision_notes') or 'none'}."
    )
    payload = {
        "model": LOCAL_ROUTER_MODEL,
        "messages": [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.25,
        "max_tokens": 180,
        "stream": False,
    }
    try:
        req = urllib.request.Request(
            LOCAL_ROUTER_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {LOCAL_ROUTER_KEY}"},
        )
        with urllib.request.urlopen(req, timeout=10.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            txt = str(((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip().strip('"')
            if len(txt) > 25:
                return txt
    except Exception:
        pass
    return None


def generate_matched_background_prompts(
    profile: Dict[str, Any],
    target_background_vi: str,
    subject_description_vi: str = "",
    use_local_llm: bool = True,
) -> Dict[str, Any]:
    """
    Generate 3 optics-matched background prompts (Clean Video Plate, Google Flow V2V,
    and Static Photo Plate) for ANY foreground subject (person, book, hands, product, gadget).
    """
    bg_idea = (target_background_vi or "").strip() or "Phòng khách hoặc không gian tự nhiên có chiều sâu chân thực"
    subj_vi = (subject_description_vi or "").strip() or "Chủ thể chính đã khoanh vùng trong video gốc"
    subj_en = (
        f"the exact foreground subject described as ({subj_vi})"
        if (subject_description_vi or "").strip()
        else "the exact foreground subject(s), pose, clothing/skin texture, and held items from the source video"
    )

    ai_bg_en = _craft_matched_environment_via_llm(bg_idea, subj_vi, profile) if use_local_llm else None
    if not ai_bg_en:
        ai_bg_en = _offline_bg_translation(bg_idea)

    scale_en = profile.get("bg_scale_rule_en") or profile["scale_calibration_en"]
    shadow_en = profile.get("shadow_direction_en") or "consistent soft contact shadows"
    hard_en = profile.get("light_hardness_en") or "natural semi-diffused light"
    exp_en = profile.get("exposure_en") or "balanced natural auto-exposure"
    kelvin_est = profile.get("kelvin_est", 5400)
    aperture_est = profile.get("aperture_est", "f/2.2")
    bg_blur = profile.get("bg_blur_sigma_px", 1.2)
    motion_en = profile.get("camera_motion_en") or "subtle organic natural camera motion"
    dist_focus = profile.get("subject_distance_label") or f"~{profile['subject_distance_cm']}cm"

    clean_plate_prompt = (
        f"Photorealistic clean empty background plate video in {profile['aspect_ratio_en']}, "
        f"{profile['scale_calibration_en']}. "
        f"Depict {ai_bg_en}, with a completely open, unobstructed empty foreground (strictly NO people, NO hands, NO held objects in the center foreground plane). "
        f"Scale rule: {scale_en} "
        f"Lighting & Shadows: match {profile['lighting_en']} at ~{kelvin_est}K with {hard_en}, {shadow_en}, and {exp_en}. "
        f"Optics & Motion: authentic natural camera capture ({profile['focal_length_mm']}mm {aperture_est}), focus plane around {dist_focus} so the background sits at subtle natural ~{bg_blur}px optical softness with authentic fine sensor grain (NO artificial CGI gloss), and {motion_en}."
    )

    v2v_prompt = (
        f"Video-to-video background replacement in {profile['aspect_ratio_en']}, {profile['scale_calibration_en']}. "
        f"KEEP {subj_en} 100% intact with identical silhouette, details, surface texture, and original motion. "
        f"CHANGE only the environment behind the foreground into {ai_bg_en}, strictly obeying the perspective rule: {scale_en} "
        f"Match illumination with {profile['lighting_en']} (~{kelvin_est}K), {hard_en}, {shadow_en}, {exp_en}, "
        f"and {aperture_est} natural background softness (~{bg_blur}px) with fine sensor grain and temporal frame-to-frame stability."
    )

    still_bg_prompt = (
        f"Authentic candid photograph of an empty background scene in {profile['aspect_ratio_en']}, "
        f"{profile['scale_calibration_en']}. Scene: {ai_bg_en}, completely unobstructed empty foreground (no subject blocking center foreground). "
        f"{scale_en} Illuminated by {profile['lighting_en']} at ~{kelvin_est}K with {hard_en}, {shadow_en}, and {exp_en}. "
        f"Captured at {profile['focal_length_mm']}mm {aperture_est} focused at {dist_focus} so background architecture and objects have ~{bg_blur}px gentle optical softness and realistic fine sensor grain."
    )

    full_markdown = (
        f"### Phân Tích Quang Học ({profile.get('analysis_mode', 'Multi-Frame')})\n"
        f"- **Tiêu cự & Tỷ lệ chủ thể:** `{profile['focal_length_mm']}mm`, rộng ~`{profile.get('hand_width_pct', profile['foreground_occupancy_pct'])}%` ở `{dist_focus}` — {profile.get('bg_scale_rule_vi', profile['scale_calibration_vi'])}\n"
        f"- **Hướng sáng & Bóng đổ:** `{profile['lighting_direction']}` — {profile.get('shadow_direction_vi', '')} ({profile.get('light_hardness_vi', '')})\n"
        f"- **Nhiệt độ màu & Phơi sáng:** `~{kelvin_est}K` | {profile.get('exposure_vi', '')}\n"
        f"- **Độ mờ hậu cảnh & Chuyển động:** `{aperture_est}` (~`{bg_blur}px` blur), Grain `{profile.get('grain_sigma', 0.01):.3f}`, {profile.get('camera_motion_vi', '')}\n"
        + (f"- **Nhận xét Vision AI:** {profile['vision_notes']}\n" if profile.get("vision_notes") else "")
    )

    return {
        "clean_plate_prompt": clean_plate_prompt,
        "v2v_prompt": v2v_prompt,
        "still_bg_prompt": still_bg_prompt,
        "ai_bg_en": ai_bg_en,
        "telemetry_cards": profile.get("telemetry_cards", []),
        "profile": profile,
        "full_markdown": full_markdown,
    }
