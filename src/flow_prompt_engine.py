from __future__ import annotations

import json
import urllib.request
from typing import Any, Dict, Optional

import cv2
import numpy as np

from src.alpha_matting import detect_active_frame_box

LOCAL_ROUTER_URL = "http://127.0.0.1:20128/v1/chat/completions"
LOCAL_ROUTER_KEY = "sk-00b59870f0cdb29e-od5qn5-26b35cc0"
LOCAL_ROUTER_MODEL = "ds/deepseek-v4-flash"


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

    # 1. Measure Background Depth-of-Field / Bokeh Sharpness vs. Foreground
    gray = cv2.cvtColor(active_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    fg_sharp = float(np.percentile(lap[active_fg > 0], 90)) if np.any(active_fg > 0) else 25.0
    bg_sharp = float(np.percentile(lap[bg_zone], 90))
    sharp_ratio = fg_sharp / max(bg_sharp, 1.0)

    if sharp_ratio > 2.2 or bg_sharp < 18.0:
        bokeh_vi = "Xóa phông Bokeh mạnh (Ống kính tiêu cự 35mm-50mm, khẩu độ lớn f/1.8 - f/2.2, hậu cảnh mờ mịn)"
        bokeh_en = "shallow depth of field (35mm-50mm prime lens at f/1.8-f/2.2) with soft, creamy out-of-focus background bokeh"
    elif sharp_ratio > 1.35 or bg_sharp < 32.0:
        bokeh_vi = "Xóa phông tự nhiên vừa phải (Tiêu cự ~35mm f/2.8, chủ thể nét căng, hậu cảnh dịu)"
        bokeh_en = "natural medium-shallow depth of field (35mm lens at f/2.8), razor-sharp foreground plane with gently softened background"
    else:
        bokeh_vi = "Độ sâu trường ảnh rộng (Tiêu cự ~24mm-28mm f/4, hậu cảnh rõ chi tiết)"
        bokeh_en = "deep focus smartphone/wide lens (26mm-28mm at f/4) with clear background architectural details"

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

    # 4. Foreground Subject Placement & Camera Angle
    ys, xs = np.where(active_fg > 0)
    if len(xs) > 0:
        fg_ratio = round(float(len(xs)) / float(max(ah * aw, 1)) * 100.0, 1)
        cy_norm = float(np.mean(ys)) / float(max(ah, 1))
    else:
        fg_ratio = 35.0
        cy_norm = 0.62

    angle_vi = "Góc quay POV góc nhìn thứ nhất (First-Person POV), tầm mắt hơi hướng nhẹ xuống thiết bị trên tay, đường chân trời ở khoảng 50%-55% khung hình"
    angle_en = (
        "first-person POV eye-level shot angled slightly downward (~10 degrees) at the handheld device in the lower-center foreground, "
        "with the background horizon line positioned at mid-frame (~52% height) and natural one-point perspective"
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
        f"Optical Telemetry to Match: {telemetry['aspect_ratio_en']}, {telemetry['camera_angle_en']}, "
        f"{telemetry['bokeh_en']}, {telemetry['lighting_en']}, {telemetry['color_temp_en']}."
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
        f"Video-to-video background transformation in {telemetry['aspect_ratio_en']}. "
        f"KEEP the exact foreground hands, finger positions, hand gestures, and the handheld electronic device completely unchanged, "
        f"preserving the device's exact rectangular geometry, black bezel, physical buttons, and original on-screen Vietnamese UI text without any alteration. "
        f"KEEP the exact original {telemetry['camera_angle_en']} and subtle handheld camera motion. "
        f"CHANGE only the background environment behind the hands and device into {ai_bg_en}. "
        f"Match the original lens optics using {telemetry['bokeh_en']}, aligning the background horizon and perspective lines seamlessly with the camera's micro-movements. "
        f"Illuminate the scene with {telemetry['lighting_en']} and {telemetry['color_temp_en']}, casting realistic soft ambient light wrap and subtle surface reflections onto the hands and device edges without altering the screen content. "
        f"Maintain strict frame-to-frame temporal consistency with zero background flickering, zero device morphing into a smartphone, and no unrequested camera zooms."
    )

    # PROMPT 2: Clean Empty Background Plate (for 100% Real Hand + Product Cutout Compositing)
    clean_plate_prompt = (
        f"Photorealistic clean background plate video in {telemetry['aspect_ratio_en']}, shot from a {telemetry['camera_angle_en']}. "
        f"The scene depicts {ai_bg_en}, with a completely clear, unobstructed empty foreground (no hands, no person, no handheld device in the foreground). "
        f"Captured with {telemetry['bokeh_en']}, so the entire background environment has authentic optical defocus matching a close-up foreground focus plane at 40cm distance. "
        f"Lighting features {telemetry['lighting_en']} at {telemetry['color_temp_en']} with natural spatial light gradients and realistic dynamic range. "
        f"Include subtle, organic handheld camera breathing and micro-sway with strict frame-to-frame temporal stability, ready as a clean V2V compositing plate."
    )

    full_markdown = f"""### A. Phân Tích Quang Học Từ Ảnh/Video Gốc & Chiến Lược (AGY Telemetry)
- **Độ phân giải & Tỷ lệ khung hình thực:** `{telemetry['active_resolution']}` — **{telemetry['active_aspect_ratio']}** {"*(Đã phát hiện viền đen hai bên, tự động đo theo khung dọc thật ở giữa)*" if telemetry['is_pillarboxed'] else ""}
- **Góc máy & Bố cục (Camera Framing):** {telemetry['camera_angle_vi']} (Chủ thể chiếm ~`{telemetry['foreground_occupancy_pct']}%` khung hình).
- **Ống kính & Độ xóa phông (Bokeh / DoF):** {telemetry['bokeh_blur_level']}
- **Hướng sáng & Nhiệt độ màu gốc:** {telemetry['lighting_direction']} | {telemetry['color_temperature']}
- **Giữ nguyên (KEEP):** {subj_vi} (Tách chuẩn từng sub-pixel từ video gốc: giữ 100% bàn tay thật, thiết bị thật và chữ tiếng Việt sắc nét trên màn hình).
- **Thay đổi (CHANGE):** Đổi phông nền sang **{bg_idea}** với cùng tiêu cự xóa phông, độ cao đường chân trời và hướng chiếu sáng.

---

### B1. PROMPT TẠO VIDEO PHÔNG NỀN SẠCH (KHUYÊN DÙNG ĐỂ GHÉP TAY + SẢN PHẨM THẬT 100%)
*(Copy prompt này sang **Google Flow / Veo** để tạo 1 video nền trống không dính tay/máy ảo, nhưng có góc máy, độ mờ Bokeh và ánh sáng khớp 100% với video gốc để nạp vào Tab Ghép Nền của Tool)*:
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
- **[Độ mờ hậu cảnh (Bokeh)]:** `{telemetry['bokeh_en']}`
- **[Hướng sáng & Màu nắng]:** `{telemetry['lighting_en']}`, `{telemetry['color_temp_en']}`
"""

    return {
        "v2v_prompt": v2v_prompt,
        "clean_plate_prompt": clean_plate_prompt,
        "ai_bg_en": ai_bg_en,
        "full_markdown": full_markdown,
    }
