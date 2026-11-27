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

STATIONARY_CAMERA_LOCK_VI = (
    "KHÓA CỨNG CAMERA ĐỨNG IM 1 CHỖ + ĐUNG ĐƯA TAY CẦM RẤT NHẸ: người quay đứng yên 1 vị trí cố định "
    "(KHÔNG đi bộ, KHÔNG tiến/lùi, KHÔNG lia pan/tilt/zoom), nhưng có độ đung đưa & rung vi mô rất nhẹ nhàng "
    "tự nhiên như đang cầm điện thoại bằng tay thật (không cứng đơ như đặt trên tripod)."
)
STATIONARY_CAMERA_LOCK_EN = (
    "STATIONARY STANDING CAMERA WITH GENTLE HANDHELD SWAY: the person holding the smartphone stands in one fixed spot "
    "(STRICTLY NO walking, NO stepping forward or backward, NO dolly/tracking movement, NO panning, NO tilting, NO orbiting, NO zooming), "
    "with a very gentle, slow natural handheld sway and subtle real-life hand micro-tremor as if holding a phone casually in hand while standing still — never frozen stiff like a tripod"
)

RAW_SMARTPHONE_REALISM_EN = (
    "RAW UNEDITED EVERYDAY SMARTPHONE FOOTAGE LOOK: shot casually on an ordinary mid-range consumer phone camera with a small mobile sensor "
    "(deep mobile depth of field where background objects remain mostly recognizable with only mild natural unsharpness and slight everyday phone video compression — "
    "STRICTLY NO cinematic film look, NO DSLR/mirrorless camera look, NO creamy portrait bokeh, NO overly smooth denoised AI plastic surfaces, "
    "NO studio commercial color grading, NO glowing HDR gloss)"
)

_BG_VARIATION_FLAVORS = (
    "",
    "framed from a plain slightly angled everyday viewpoint with natural mid-ground depth and unpolished real-life clutter",
    "captured with ordinary unedited indoor/daylight ambient falloff across everyday surfaces and realistic distance scale",
    "showing an unremarkable candid everyday space with open center foreground space and natural unstyled background details",
)


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
        "lighting direction, and depth-of-field bokeh. CRITICAL: The camera MUST remain strictly stationary "
        "in one fixed standing spot (NEVER walking, NEVER dolly/moving forward, NEVER panning/tilting). "
        "Output ONLY the English environment sentences, no markdown."
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


def _offline_bg_translation(target_background_vi: str, variation_seed: int = 0) -> str:
    """Deterministic everyday raw-smartphone scene builder (anti-cinematic, no DSLR bokeh buzzwords)."""
    t = target_background_vi.lower()
    presets = [
        (["thư viện", "library", "đọc sách", "sách", "reading"], [
            "an ordinary real-world public library and reading room seen from one fixed standing spot, with plain wooden bookshelves filled with everyday books in the background and flat natural window daylight",
            "a candid everyday bookstore and study room viewed from a stationary standing spot, showing rows of regular bookshelves, simple wooden tables, and ordinary ceiling/window room lighting",
            "a quiet local library interior seen from a fixed standing position, with unstyled shelves, regular reading desks in the mid-ground, and plain natural daylight",
        ]),
        (["cà phê", "cafe", "coffee"], [
            "an ordinary everyday neighborhood coffee shop interior seen from one fixed standing spot, with regular wooden tables, glass storefront windows, plain indoor lighting, and casual customers sitting further back",
            "a candid casual cafe interior viewed from a stationary standing point, showing everyday counter equipment, plain wall tiles, regular chairs, and natural ungraded daylight from the window",
            "a real-life street-level coffee shop observed from a fixed standing position, with simple wood tables, ordinary hanging bulbs, and people sitting naturally in the background",
        ]),
        (["sân bay", "airport", "check-in", "nhà ga"], [
            "an ordinary real-world airport terminal check-in hall seen from one fixed standing spot, with plain tiled floors, everyday overhead terminal lights, departure screens, and casual travelers further back",
            "a candid airport departure concourse viewed from a stationary standing position, showing standard check-in counters in the mid-ground, regular signage, and flat everyday indoor lighting",
            "a regular international airport waiting hall observed from one fixed standing spot, with ordinary glass windows, plain metal seating rows, and everyday travelers in the distance",
        ]),
        (["khách sạn", "hotel", "sảnh", "lobby"], [
            "a real-world hotel lobby seen from one fixed standing spot, with plain stone floor tiles, regular reception desk in the distance, potted plants, and ordinary indoor ceiling lighting",
            "an everyday hotel reception lounge viewed from a stationary standing point, with standard lobby chairs, neutral walls, and natural ungraded ambient light",
        ]),
        (["phố", "street", "tokyo", "châu âu", "đi bộ"], [
            "an ordinary real-world city sidewalk and storefront street observed from one fixed standing spot, with everyday shop signs, regular street trees, concrete pavement, and natural ungraded daylight",
            "a candid everyday city street corner viewed from a stationary standing position on the pavement, with plain building facades, parked bicycles/signs, and ordinary outdoor daylight",
        ]),
        (["văn phòng", "office", "họp", "công sở"], [
            "an ordinary everyday office workspace seen from one fixed standing spot, with plain desks, office chairs, glass partition walls, and standard indoor ceiling/window light",
            "a candid real-world meeting room and office interior viewed from a stationary standing point, with regular desks, computer monitors further back, and flat everyday room lighting",
        ]),
        (["taxi", "xe hơi", "ô tô", "ghế sau"], [
            "the ordinary interior of a parked modern sedan rear seat observed from a fixed seated viewpoint, with plain car door trim, regular fabric/leather seats, and natural ungraded daylight through the side window",
            "the candid everyday back-seat cabin of a stationary car seen from a fixed passenger viewpoint, with ordinary window glass and realistic unpolished street view outside",
        ]),
        (["linh kiện", "điện tử", "thâm quyến", "sửa chữa"], [
            "an ordinary real-world Shenzhen electronics components shop counter viewed from one fixed standing spot, with cluttered parts bins, plain fluorescent shop lighting, and regular shelves behind",
            "a candid everyday electronics repair shop interior seen from a stationary standing position, with regular workbench tools, unstyled overhead lights, and component drawers in the background",
        ]),
    ]
    v_idx = max(0, int(variation_seed))
    base_desc = ""
    for keys, variants in presets:
        if any(k in t for k in keys):
            base_desc = variants[v_idx % len(variants)]
            break
    if not base_desc:
        base_desc = (
            f"an ordinary candid real-world environment ({target_background_vi}) viewed from a single fixed standing spot, "
            "with plain everyday textures, natural unpolished lighting, and realistic perspective"
        )
    flavor = _BG_VARIATION_FLAVORS[v_idx % len(_BG_VARIATION_FLAVORS)]
    return f"{base_desc}, {flavor}" if flavor else base_desc


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
        f"{STATIONARY_CAMERA_LOCK_EN}, with strict frame-to-frame temporal stability, ready as a clean compositing plate."
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

    # Propagate nearest known user mask to unsampled/unconfirmed sample frames so
    # size, tilt, and shot-distance measurements stay consistent when only Frame 0 is cut.
    known_pairs = [(i, m) for i, m in enumerate(loaded_masks) if m is not None and np.any(m > 0)]
    if known_pairs:
        for i in range(len(loaded_masks)):
            if loaded_masks[i] is None or not np.any(loaded_masks[i] > 0):
                _, nearest_m = min(known_pairs, key=lambda pair: abs(pair[0] - i))
                loaded_masks[i] = nearest_m

    base = analyze_scene_telemetry(loaded_frames[0], fg_mask=loaded_masks[0])
    hand_w_list: List[float] = []
    hand_h_list: List[float] = []
    fg_occ_list: List[float] = []
    orient_deg_list: List[float] = []
    gx_list: List[float] = []
    gy_list: List[float] = []
    contrast_list: List[float] = []
    luma_list: List[float] = []
    black_p5_list: List[float] = []
    white_p95_list: List[float] = []
    sat_med_list: List[float] = []
    kelvin_list: List[float] = []
    tint_a_list: List[float] = []
    fg_sharp_list: List[float] = []
    bg_blur_list: List[float] = []
    grain_list: List[float] = []
    chroma_noise_list: List[float] = []
    entry_scores = {"left": 0.0, "right": 0.0, "center": 0.0}
    has_meaningful_skin = False
    fg_spans_portrait = False
    green_bg_votes = 0

    from src.relight_harmonizer import detect_emissive_screen_mask

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
        bg_bool = ~fg_bool
        if int(np.sum(bg_bool)) < 150:
            bg_bool = np.ones((ah, aw), dtype=bool)

        # Detect green-screen backdrop on exterior pixels so chroma cloth never skews Kelvin/noise/blur
        b_bg = act[..., 0][bg_bool].astype(np.float32)
        g_bg = act[..., 1][bg_bool].astype(np.float32)
        r_bg = act[..., 2][bg_bool].astype(np.float32)
        is_gs_frame = float(np.mean((g_bg - np.maximum(r_bg, b_bg)) > 16.0)) >= 0.22
        if is_gs_frame:
            green_bg_votes += 1

        # Exclude glowing UI screens when measuring scene illumination, color tone, sharpness & sensor noise
        scr_act = detect_emissive_screen_mask(act, fg_act) > 0
        fg_non_scr = fg_bool & (~scr_act)
        if int(np.sum(fg_non_scr)) < 120:
            fg_non_scr = fg_bool

        skin = _extract_skin_roi_mask(act, fg_act) & (~scr_act)
        skin_frac_frame = float(np.mean(skin)) if np.any(skin) else 0.0
        fg_frac_frame = max(1e-4, float(np.mean(fg_bool)))
        use_skin_roi = (skin_frac_frame >= 0.02) and (skin_frac_frame / fg_frac_frame >= 0.15)
        if use_skin_roi:
            has_meaningful_skin = True
        measure_roi = skin if use_skin_roi else fg_non_scr

        fg_occ_list.append(float(np.mean(fg_bool)) * 100.0)
        fy, fx = np.where(fg_bool)
        if len(fx) > 0:
            hand_w_list.append(float(np.max(fx) - np.min(fx) + 1) / float(max(aw, 1)) * 100.0)
            f_h_ratio = float(np.max(fy) - np.min(fy) + 1) / float(max(ah, 1))
            hand_h_list.append(f_h_ratio * 100.0)
            if f_h_ratio >= 0.72 and float(np.min(fy)) <= float(ah) * 0.24 and not use_skin_roi:
                fg_spans_portrait = True
            elif f_h_ratio >= 0.80 and float(np.min(fy)) <= float(ah) * 0.18:
                fg_spans_portrait = True
            # Principal axis tilt angle of foreground subject (deg from vertical; + tilts right, - tilts left)
            if len(fx) >= 60:
                step_p = max(1, len(fx) // 1500)
                pts = np.stack([fx[::step_p].astype(np.float32), fy[::step_p].astype(np.float32)], axis=1)
                pts_c = pts - np.mean(pts, axis=0, keepdims=True)
                cov = np.cov(pts_c.T)
                eigvals, eigvecs = np.linalg.eigh(cov)
                major = eigvecs[:, int(np.argmax(eigvals))]
                # Ensure vector points upward (-y) so tilt left/right sign is intuitive
                if major[1] > 0:
                    major = -major
                tilt_deg = float(np.degrees(np.arctan2(major[0], -major[1])))
                orient_deg_list.append(tilt_deg)
            bot = measure_roi[int(ah * 0.70):ah, :]
            if np.any(bot):
                bw = bot.shape[1]
                entry_scores["left"] += float(np.mean(bot[:, :max(1, bw // 3)]))
                entry_scores["center"] += float(np.mean(bot[:, bw // 3:max(bw // 3 + 1, (2 * bw) // 3)]))
                entry_scores["right"] += float(np.mean(bot[:, (2 * bw) // 3:]))

        gray = cv2.cvtColor(act, cv2.COLOR_BGR2GRAY).astype(np.float32)
        lab_act = cv2.cvtColor(act, cv2.COLOR_BGR2LAB).astype(np.float32)
        hsv_act = cv2.cvtColor(act, cv2.COLOR_BGR2HSV).astype(np.float32)

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
            p5, p15, p50, p90, p95 = np.percentile(vals, [5, 15, 50, 90, 95])
            contrast_list.append(float(p90 - p15) / max(float(p50), 12.0))
            luma_list.append(float(np.mean(vals)))
            black_p5_list.append(float(p5))
            white_p95_list.append(float(p95))
        else:
            contrast_list.append(0.45)
            luma_list.append(float(np.mean(gray)))
            black_p5_list.append(float(np.percentile(gray, 5)))
            white_p95_list.append(float(np.percentile(gray, 95)))

        # Color tone, saturation & Kelvin estimated from real subject lighting (never corrupted by green screen!)
        fg_lab_vals = lab_act[fg_non_scr]
        fg_hsv_s = hsv_act[..., 1][fg_non_scr]
        sat_med_list.append(float(np.median(fg_hsv_s)) / 255.0 if len(fg_hsv_s) else 0.35)

        if is_gs_frame:
            # Prefer neutral bright/midtone pixels on foreground (white bezels, paper, neutral clothing)
            # to read true studio lamp Kelvin & green-magenta tint without skin or green-screen bias.
            chroma_mag = np.hypot(fg_lab_vals[:, 1] - 128.0, fg_lab_vals[:, 2] - 128.0)
            neut_px = fg_lab_vals[(fg_lab_vals[:, 0] >= 75.0) & (chroma_mag <= 24.0)]
            if len(neut_px) >= 32:
                b_shift = float(np.median(neut_px[:, 2]) - 128.0)
                a_shift = float(np.median(neut_px[:, 1]) - 128.0)
                kelvin_list.append(float(np.clip(5400.0 - b_shift * 95.0, 3200.0, 6800.0)))
                tint_a_list.append(a_shift)
            else:
                # Reference skin baseline (~Lab a=142, b=144 under neutral 5300K light)
                skin_vals = lab_act[measure_roi]
                b_rel = float(np.median(skin_vals[:, 2]) - 144.0)
                a_rel = float(np.median(skin_vals[:, 1]) - 142.0)
                kelvin_list.append(float(np.clip(5300.0 - b_rel * 75.0, 3400.0, 6600.0)))
                tint_a_list.append(a_rel * 0.5)
        else:
            b_m = float(np.mean(act[..., 0][bg_bool]))
            r_m = float(np.mean(act[..., 2][bg_bool]))
            rb_diff = r_m - b_m
            kelvin_list.append(float(np.clip(5400.0 - rb_diff * 24.0, 3200.0, 6800.0)))
            tint_a_list.append(float(np.median(lab_act[..., 1][bg_bool]) - 128.0))

        # True Subject Optical Sharpness & Background DoF Blur
        lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
        fg_s = float(np.percentile(lap[fg_non_scr], 90)) if np.any(fg_non_scr) else 25.0
        fg_sharp_list.append(fg_s)
        if is_gs_frame:
            # Physically calibrated DoF softness from subject distance rather than flat green cloth
            est_blur = 2.0 if base["subject_distance_cm"] <= 36 else (1.4 if base["subject_distance_cm"] <= 50 else 1.0)
            bg_blur_list.append(est_blur)
        else:
            bg_s = float(np.percentile(lap[bg_bool], 90))
            ratio = fg_s / max(bg_s, 1.0)
            bg_blur_list.append(float(np.clip(0.4 + max(0.0, ratio - 1.1) * 0.75, 0.4, 3.8)))

        # Sensor Luma & Chroma Noise measured on smooth interior patches of the FOREGROUND subject
        # (never misreading flat compressed green-screen cloth as camera noise!)
        smooth_roi = fg_non_scr & (lap <= float(np.percentile(lap[fg_non_scr], 60))) if np.any(fg_non_scr) else fg_bool
        if int(np.sum(smooth_roi)) < 80:
            smooth_roi = fg_non_scr
        blur_ref = cv2.GaussianBlur(gray, (3, 3), 0.8)
        hp = np.abs(gray[smooth_roi] - blur_ref[smooth_roi])
        mad_sigma = float(np.median(hp)) * 1.4826 / 255.0
        grain_list.append(mad_sigma)

        a_ch = lab_act[..., 1]
        b_ch_l = lab_act[..., 2]
        c_hp = 0.5 * (
            np.abs(a_ch[smooth_roi] - cv2.GaussianBlur(a_ch, (3, 3), 0.8)[smooth_roi])
            + np.abs(b_ch_l[smooth_roi] - cv2.GaussianBlur(b_ch_l, (3, 3), 0.8)[smooth_roi])
        )
        chroma_noise_list.append(float(np.median(c_hp)) * 1.4826 / 255.0)

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
    chroma_noise = round(float(np.mean(chroma_noise_list)), 4) if chroma_noise_list else 0.002
    fg_sharp_val = round(float(np.mean(fg_sharp_list)), 1) if fg_sharp_list else 20.0
    orient_deg = round(float(np.mean(orient_deg_list)), 1) if orient_deg_list else 0.0
    black_p5 = round(float(np.mean(black_p5_list)), 1) if black_p5_list else 18.0
    white_p95 = round(float(np.mean(white_p95_list)), 1) if white_p95_list else 210.0
    sat_med = round(float(np.mean(sat_med_list)), 2) if sat_med_list else 0.35
    tint_a = round(float(np.mean(tint_a_list)), 1) if tint_a_list else 0.0
    is_green_screen_multi = green_bg_votes >= max(1, (len(loaded_frames) + 1) // 2)

    # Subject orientation / tilt description
    if orient_deg >= 6.0:
        orient_vi = f"nghiêng ~{abs(orient_deg):.0f}° sang PHẢI"
        orient_en = f"tilted ~{abs(orient_deg):.0f} degrees to the right"
    elif orient_deg <= -6.0:
        orient_vi = f"nghiêng ~{abs(orient_deg):.0f}° sang TRÁI"
        orient_en = f"tilted ~{abs(orient_deg):.0f} degrees to the left"
    else:
        orient_vi = "thẳng đứng cân đối (~0°)"
        orient_en = "upright vertical orientation (~0° tilt)"

    # Optical Sharpness / Blur profile of the source footage
    if fg_sharp_val >= 34.0:
        sharp_vi = f"Nét cao sắc sảo (Laplacian {fg_sharp_val:.0f})"
        sharp_en = "crisp natural 1080p smartphone optical sharpness on in-focus planes without artificial over-sharpening"
    elif fg_sharp_val >= 14.0:
        sharp_vi = f"Nét tự nhiên kiểu điện thoại (Laplacian {fg_sharp_val:.0f})"
        sharp_en = "authentic everyday smartphone optical softness (organic micro-contrast, slightly soft lens resolution — never overly sharp CGI)"
    else:
        sharp_vi = f"Nét mềm nhẹ / hơi mờ chuyển động (Laplacian {fg_sharp_val:.0f})"
        sharp_en = "soft organic handheld lens rendering with gentle motion softness and smooth roll-off (strictly avoid hyper-crisp AI textures)"

    # Sensor Noise / Grain / Equivalent ISO classification
    if grain_sigma >= 0.015:
        iso_est = 1200
        noise_vi = f"Nhiễu hạt rõ (ISO ~{iso_est}, grain {grain_sigma:.3f}, chroma {chroma_noise:.3f})"
        noise_en = f"visible organic mobile low-light sensor noise (equivalent ~ISO {iso_est}, subtle luma + fine chroma grain matching real handheld compression)"
    elif grain_sigma >= 0.007:
        iso_est = 400
        noise_vi = f"Nhiễu hạt cảm biến nhẹ tự nhiên (ISO ~{iso_est}, grain {grain_sigma:.3f})"
        noise_en = f"subtle authentic consumer smartphone sensor grain (equivalent ~ISO {iso_est}, fine organic filmic luminance noise, no sterile AI plastic smoothness)"
    else:
        iso_est = 160
        noise_vi = f"Rất sạch, hạt cảm biến siêu mịn (ISO ~{iso_est}, grain {grain_sigma:.3f})"
        noise_en = f"clean daylight smartphone sensor texture (equivalent ~ISO {iso_est}, ultra-fine natural micro-grain, realistic mobile H.264 video texture)"

    # Color Tone, Saturation & Black-Point Grading profile
    if kelvin_est <= 4400:
        tone_temp_vi = "Tông ấm (vàng dịu)"
        tone_temp_en = f"warm indoor ~{kelvin_est}K color balance"
    elif kelvin_est >= 5800:
        tone_temp_vi = "Tông lạnh / trắng hiện đại"
        tone_temp_en = f"clean cool-neutral ~{kelvin_est}K daylight/LED color balance"
    else:
        tone_temp_vi = "Tông trung tính tự nhiên"
        tone_temp_en = f"neutral true-to-life ~{kelvin_est}K daylight color balance"

    if sat_med <= 0.24:
        sat_vi = "bão hòa màu dịu nhẹ"
        sat_en = "muted natural low-to-moderate color saturation"
    elif sat_med >= 0.48:
        sat_vi = "màu sắc tươi rõ"
        sat_en = "moderately rich, natural smartphone color saturation (non-oversaturated)"
    else:
        sat_vi = "bão hòa cân bằng trung thực"
        sat_en = "balanced true-to-life consumer smartphone color saturation (no HDR oversaturation)"

    if black_p5 >= 28.0:
        bp_vi = f"vùng tối nâng nhẹ kiểu matte (P5={int(black_p5)})"
        bp_en = "slightly lifted matte shadow floor with gentle contrast roll-off (avoid crushed pitch blacks)"
    elif black_p5 <= 14.0:
        bp_vi = f"vùng tối sâu rõ (P5={int(black_p5)}, P95={int(white_p95)})"
        bp_en = "rich natural black-point depth and realistic dynamic range without harsh HDR clipping"
    else:
        bp_vi = f"tương phản & điểm đen cân bằng (P5={int(black_p5)}, P95={int(white_p95)})"
        bp_en = "balanced smartphone contrast curve with natural shadow floor and soft highlight roll-off"

    color_tone_vi = f"{tone_temp_vi}, {sat_vi}, {bp_vi}" + (" (đã tách bỏ sai lệch từ phông xanh)" if is_green_screen_multi else "")
    color_tone_en = f"{tone_temp_en}, {sat_en}, {bp_en}" + (", completely free of any green-screen tint" if is_green_screen_multi else "")

    # HARD CAMERA LOCK for foreground compositing plates:
    # Even when the foreground hands/subject move in the source video, the background camera
    # MUST stand completely still in one fixed spot with only imperceptible handheld micro-breathing.
    motion_vi = STATIONARY_CAMERA_LOCK_VI
    motion_en = STATIONARY_CAMERA_LOCK_EN

    if shot_kind == "portrait_medium":
        bg_scale_rule_vi = (
            f"Chủ thể chiếm ~{hand_w_pct}%×{hand_h_pct}% khung ({orient_vi}) ở khoảng trung cảnh {est_dist_label} -> "
            f"nội thất/ghế/kệ ngay phía sau (1.8–3m) và cảnh xa (4–7m) phải đúng tỷ lệ góc nhìn {base['focal_length_mm']}mm."
        )
        bg_scale_rule_en = (
            f"Because the medium-framed foreground subject occupies ~{hand_w_pct}%x{hand_h_pct}% of the frame ({orient_en}) at {est_dist_label} "
            f"({wrist_en}), scale mid-ground furniture/shelves (1.8–3m away) and distant architectural elements (4–7m away) "
            f"naturally to a {base['focal_length_mm']}mm perspective with matching eye-level horizon geometry."
        )
    else:
        bg_scale_rule_vi = (
            f"Chủ thể/tay tiền cảnh chiếm ~{hand_w_pct}%×{hand_h_pct}% khung ({orient_vi}) ở cự ly gần {est_dist_label} -> "
            f"vật thể trung cảnh (1.5–2.5m) và người/nội thất hậu cảnh (3–5m) phải có kích thước đủ gần tự nhiên đúng ống kính {base['focal_length_mm']}mm (tránh quá nhỏ xa tít)."
        )
        bg_scale_rule_en = (
            f"Because the close-up foreground subject/hand occupies ~{hand_w_pct}%x{hand_h_pct}% of the frame ({orient_en}) at {est_dist_label} distance "
            f"({wrist_en}), place real-world mid-ground surfaces/furniture (~1.5–2.5m away) and background people/architecture (~3–5m away) "
            f"at natural medium-close {base['focal_length_mm']}mm smartphone scale — neither tiny distant wide-angle miniatures nor blocking the center foreground."
        )

    vision_notes = _estimate_vision_notes(loaded_frames) if use_vision_ai else ""
    analysis_mode = f"Multi-Frame CV ({len(loaded_frames)} khung)" + (" + Vision AI" if vision_notes else " · Offline 0 Token")

    horizon_pct = int(round(np.clip(52.0 - (base["camera_pitch_deg"] + 10) * 0.6, 38.0, 58.0)))
    telemetry_cards = [
        {
            "id": "lens_scale",
            "icon": "📐",
            "title": "Kích thước, Hướng & Tỷ lệ chủ thể",
            "value": f"Rộng {hand_w_pct}% × Cao {hand_h_pct}% · {orient_vi} · {est_dist_label}",
            "detail": f"{wrist_vi}. {bg_scale_rule_vi}",
        },
        {
            "id": "camera_angle",
            "icon": "🎥",
            "title": "Tiêu cự, Góc máy & Chuyển động",
            "value": f"{base['focal_length_mm']}mm · Nghiêng {base['camera_pitch_deg']}° · Chân trời ~{horizon_pct}%",
            "detail": f"{base['active_aspect_ratio']}. {motion_vi}.",
        },
        {
            "id": "key_light",
            "icon": "☀️",
            "title": "Hướng sáng & Chiều đổ bóng",
            "value": light_dir_vi,
            "detail": f"{shadow_vi}. {hard_vi}.",
        },
        {
            "id": "color_tone",
            "icon": "🎨",
            "title": "Màu sắc, Tone & Điểm đen",
            "value": f"~{kelvin_est}K · {tone_temp_vi} · Luma {int(mean_luma)}",
            "detail": f"{color_tone_vi}. {exp_vi}.",
        },
        {
            "id": "sharp_dof",
            "icon": "🔍",
            "title": "Độ nét, Độ mờ nền & Nhiễu hạt (ISO)",
            "value": f"{aperture_est} · Mờ ~{bg_blur_sigma}px · ISO ~{iso_est} · Grain {grain_sigma:.3f}",
            "detail": f"{sharp_vi}. {noise_vi}.",
        },
        {
            "id": "mode",
            "icon": "🤖",
            "title": "Phân tích đa khung hình",
            "value": analysis_mode,
            "detail": vision_notes or f"Đã đo đồng bộ {len(loaded_frames)} frame ({sampled_indices}) trên vùng chủ thể thật để tạo nền khớp màu & độ nhiễu ngay từ gốc.",
        },
    ]

    profile = dict(base)
    profile.update({
        "sampled_indices": sampled_indices,
        "is_green_screen": is_green_screen_multi,
        "foreground_occupancy_pct": fg_occ_pct,
        "hand_width_pct": hand_w_pct,
        "hand_height_pct": hand_h_pct,
        "subject_orientation_deg": orient_deg,
        "subject_orientation_vi": orient_vi,
        "subject_orientation_en": orient_en,
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
        "black_point_p5": black_p5,
        "white_point_p95": white_p95,
        "saturation_med": sat_med,
        "tint_a_shift": tint_a,
        "color_tone_vi": color_tone_vi,
        "color_tone_en": color_tone_en,
        "fg_sharpness": fg_sharp_val,
        "sharpness_vi": sharp_vi,
        "sharpness_en": sharp_en,
        "bg_blur_sigma_px": bg_blur_sigma,
        "aperture_est": aperture_est,
        "grain_sigma": grain_sigma,
        "chroma_noise_sigma": chroma_noise,
        "iso_est": iso_est,
        "noise_vi": noise_vi,
        "noise_en": noise_en,
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
    variation_seed: int = 0,
) -> Optional[str]:
    v_seed = max(0, int(variation_seed))
    sys_msg = (
        "You are a candid real-world smartphone video supervisor (NOT a commercial cinema director). "
        "Translate the user's Vietnamese background description into 2-3 plain, unglamorous, raw everyday English sentences "
        "describing ONLY the empty real-world environment as captured casually on an ordinary mid-range smartphone. "
        "CRITICAL CAMERA LOCK: The person holding the phone stands in one fixed spot with a very gentle, slow natural handheld sway — "
        "STRICTLY FORBID walking, moving forward/backward, dolly, panning, tilting, or tracking shots. "
        "ANTI-CINEMATIC RULE: STRICTLY FORBID words like 'cinematic', 'luxury', 'sleek', 'sophisticated', 'dreamy', 'creamy bokeh', "
        "'softly blurred', 'shallow depth of field', 'dramatic lighting', or 'glossy'. Keep it ordinary, real, and unpolished. "
        "Never include hands, phones, or foreground objects. Output ONLY English prose."
    )
    var_note = (
        f"Variation #{v_seed + 1}: craft a fresh alternative camera framing/composition and subtle background layout for this location while keeping the camera 100% stationary in one fixed spot."
        if v_seed > 0
        else "Primary clean composition viewed from a single stationary standing spot."
    )
    color_tone_str = profile.get("color_tone_en") or f"~{profile.get('kelvin_est', 5400)}K"
    user_msg = (
        f"Vietnamese Background Idea: {target_background_vi}\n"
        f"Foreground Subject to Composite Later: {subject_description_vi}\n"
        f"Composition Request: {var_note}\n"
        f"Optical & Color Profile: {profile['aspect_ratio_en']}, {profile['focal_length_mm']}mm smartphone lens, "
        f"{profile['camera_angle_en']}, {profile['bg_scale_rule_en']}, "
        f"{profile['lighting_en']}, {profile['shadow_direction_en']}, {profile['light_hardness_en']}, "
        f"Color & Tone: {color_tone_str}, {profile['exposure_en']}, "
        f"Sharpness & DoF: {profile.get('sharpness_en') or ''}, {profile['aperture_est']} subtle DoF (~{profile['bg_blur_sigma_px']}px), "
        f"Noise: {profile.get('noise_en') or 'fine organic sensor grain'}, "
        f"Camera Lock: {STATIONARY_CAMERA_LOCK_EN}. Vision note: {profile.get('vision_notes') or 'none'}."
    )
    payload = {
        "model": LOCAL_ROUTER_MODEL,
        "messages": [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.25 if v_seed == 0 else 0.55,
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
    variation_seed: int = 0,
) -> Dict[str, Any]:
    """
    Generate 3 optics-matched background prompts (Clean Video Plate, Google Flow V2V,
    and Static Photo Plate) with HARD STATIONARY CAMERA LOCK for compositing stability.
    """
    v_seed = max(0, int(variation_seed))
    bg_idea = (target_background_vi or "").strip() or "Phòng khách hoặc không gian tự nhiên có chiều sâu chân thực"
    subj_vi = (subject_description_vi or "").strip() or "Chủ thể chính đã khoanh vùng trong video gốc"
    subj_en = (
        f"the exact foreground subject described as ({subj_vi})"
        if (subject_description_vi or "").strip()
        else "the exact foreground subject(s), pose, clothing/skin texture, and held items from the source video"
    )

    ai_bg_en = _craft_matched_environment_via_llm(bg_idea, subj_vi, profile, variation_seed=v_seed) if use_local_llm else None
    if not ai_bg_en:
        ai_bg_en = _offline_bg_translation(bg_idea, variation_seed=v_seed)

    scale_en = profile.get("bg_scale_rule_en") or profile["scale_calibration_en"]
    shadow_en = profile.get("shadow_direction_en") or "consistent soft contact shadows"
    hard_en = profile.get("light_hardness_en") or "natural semi-diffused light"
    exp_en = profile.get("exposure_en") or "balanced natural auto-exposure"
    kelvin_est = profile.get("kelvin_est", 5400)
    color_tone_en = profile.get("color_tone_en") or f"neutral ~{kelvin_est}K daylight color tone"
    sharp_en = profile.get("sharpness_en") or "natural consumer smartphone optical sharpness"
    noise_en = profile.get("noise_en") or "subtle organic smartphone sensor grain"
    aperture_est = profile.get("aperture_est", "f/2.2")
    bg_blur = profile.get("bg_blur_sigma_px", 1.2)
    dist_focus = profile.get("subject_distance_label") or f"~{profile['subject_distance_cm']}cm"

    clean_plate_prompt = (
        f"Raw unedited everyday smartphone background video in {profile['aspect_ratio_en']}, "
        f"{profile['scale_calibration_en']}. "
        f"{RAW_SMARTPHONE_REALISM_EN}. "
        f"Camera Motion: {STATIONARY_CAMERA_LOCK_EN}. "
        f"Scene: {ai_bg_en}, with a completely empty, unobstructed foreground (strictly NO people, NO hands, NO held objects in the center foreground). "
        f"Scale & Framing: {scale_en} "
        f"Ungraded Lighting, Color & Tone: match {profile['lighting_en']} with {color_tone_en}, {hard_en}, {shadow_en}, and {exp_en} using ordinary unedited smartphone auto-exposure and auto white balance so real phone footage composites seamlessly. "
        f"Everyday Phone Optics, Sharpness & Noise: small consumer phone sensor ({profile['focal_length_mm']}mm, deep natural phone DoF — NOT DSLR f/1.4 bokeh, background stays recognizable with only mild ~{bg_blur}px everyday lens unsharpness), matching {sharp_en} and {noise_en} with authentic mobile H.264 video texture and strict frame-to-frame stability."
    )

    v2v_prompt = (
        f"Raw everyday smartphone video-to-video background replacement in {profile['aspect_ratio_en']}, {profile['scale_calibration_en']}. "
        f"{RAW_SMARTPHONE_REALISM_EN}. "
        f"KEEP {subj_en} 100% intact with identical silhouette, details, surface texture, and original motion. "
        f"Camera Motion: {STATIONARY_CAMERA_LOCK_EN}. "
        f"CHANGE only the environment behind the foreground into {ai_bg_en}, strictly obeying the perspective rule: {scale_en} "
        f"Match plain ungraded lighting and color with {profile['lighting_en']}, {color_tone_en}, {hard_en}, {shadow_en}, {exp_en}. "
        f"Render background with ordinary small-sensor phone optics (deep natural focus, only mild ~{bg_blur}px optical unsharpness — NO creamy DSLR bokeh or smooth AI sheen), matching {sharp_en} and {noise_en}, with strict temporal frame-to-frame stability."
    )

    still_bg_prompt = (
        f"Raw unedited candid smartphone photo of an empty everyday background scene in {profile['aspect_ratio_en']}, "
        f"{profile['scale_calibration_en']}. {RAW_SMARTPHONE_REALISM_EN}. "
        f"Scene: {ai_bg_en}, completely unobstructed empty foreground, viewed from a single fixed standing spot. "
        f"{scale_en} Illuminated by {profile['lighting_en']} with {color_tone_en}, {hard_en}, {shadow_en}, and {exp_en}. "
        f"Captured casually on a mid-range {profile['focal_length_mm']}mm smartphone camera (deep natural mobile focus, only mild ~{bg_blur}px optical unsharpness — NO DSLR portrait blur), matching {sharp_en} and {noise_en}."
    )

    full_markdown = (
        f"### Phân Tích Quang Học & Màu Sắc ({profile.get('analysis_mode', 'Multi-Frame')})\n"
        f"- **Kích thước, Hướng & Tỷ lệ chủ thể:** `{profile['focal_length_mm']}mm`, rộng ~`{profile.get('hand_width_pct', profile['foreground_occupancy_pct'])}%` × cao ~`{profile.get('hand_height_pct', 50)}%` ({profile.get('subject_orientation_vi', 'thẳng')}) ở `{dist_focus}` — {profile.get('bg_scale_rule_vi', profile['scale_calibration_vi'])}\n"
        f"- **Khóa chuyển động Camera:** {STATIONARY_CAMERA_LOCK_VI}\n"
        f"- **Hướng sáng & Bóng đổ:** `{profile['lighting_direction']}` — {profile.get('shadow_direction_vi', '')} ({profile.get('light_hardness_vi', '')})\n"
        f"- **Màu sắc, Tone & Điểm đen:** `~{kelvin_est}K` | {profile.get('color_tone_vi', '')} | {profile.get('exposure_vi', '')}\n"
        f"- **Độ nét, Độ mờ nền & Nhiễu hạt:** `{aperture_est}` (~`{bg_blur}px` blur) | {profile.get('sharpness_vi', '')} | {profile.get('noise_vi', '')}\n"
        + (f"- **Nhận xét Vision AI:** {profile['vision_notes']}\n" if profile.get("vision_notes") else "")
    )

    return {
        "clean_plate_prompt": clean_plate_prompt,
        "v2v_prompt": v2v_prompt,
        "still_bg_prompt": still_bg_prompt,
        "ai_bg_en": ai_bg_en,
        "variation_seed": v_seed,
        "telemetry_cards": profile.get("telemetry_cards", []),
        "profile": profile,
        "full_markdown": full_markdown,
    }
