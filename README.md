# Tool Tách Ghép Nền Chất Lượng Cao — AGY Universal AI Roto-Brush & Matting Studio

Ứng dụng chạy **tối ưu GPU CUDA trên máy cá nhân** ở độ phân giải gốc giúp tách & ghép nền chuyên nghiệp cho **mọi loại chủ thể bất kỳ** (người ngồi đọc sách, tay cầm sách/sản phẩm, chân dung, thú cưng, thiết bị có màn hình phát sáng...), tích hợp sẵn:

- **Web Studio độc lập (`studio_server.py` / FastAPI + HTML5 Canvas)**: hỗ trợ upload phân mảnh từ xa (`4 MB/mảnh`), **Zoom/Pan**, **Box-Select (`⬜ Khoanh vùng`)** bằng kéo chuột cho bất kỳ vật thể nào, **Cọ Xanh/Đỏ tự bắt nét (`SAM 2.1`)**, **Live Split-Wipe Preview** kéo thanh trượt thấy hiệu ứng tức thì, và **tự động xóa sạch dữ liệu tạm trên máy chủ ngay sau khi tải video về máy**.
- **Tách Nền AI Hybrid Đa Tầng (`BiRefNet_HR-matting` + `MatAnyone 2` + `SAM 2.1` + `PyMatting`)**:
  - **`✨ Tự tách AI 1-Click` / `🎯 AI Toàn diện`**: kết hợp `BiRefNet_HR-matting` với kiểm tra hoàn chỉnh hình khối `SAM 2.1` để giữ trọn ngón tay, trang sách, tay áo, và viền kim loại mảnh.
  - **`⬜ Khoanh vùng (Box-Select)` + `⚡ Tự bắt nét` Roto-Brush**: kéo hình chữ nhật quanh đúng đối tượng muốn tách hoặc tô nét Xanh/Đỏ để thêm/trừ thông minh ở mức cục bộ, bảo toàn alpha mượt ban đầu và có `↩ Hoàn tác` / `Ctrl+Z`.
- **Tinh Chỉnh Viền Bảo Toàn Pixel & Ánh Sáng Dual-Zone (`refine_alpha_edge_controls` + `composite_with_relighting`)**:
  - Điều chỉnh trực tiếp trên Live Preview: **Bảo toàn pixel viền gốc**, **Co/Giãn biên (`0.5 px`)**, **Guided Filter**, **Khử màu viền PyMatting**, **Làm mềm mép**, **Khớp nhiệt độ màu Lab**, **Cân bằng điểm đen**, **Ánh sáng bọc viền Linear-RGB**, **Bảo vệ vùng phát sáng**, và **4 chế độ soi lỗi (`Gốc ⟷ Ghép`, `Halo Đen`, `Halo Trắng`, `Vùng phát sáng`)**.
- **Hậu Kiểm Tự Động QA 10% (`src/matte_qa.py`)**:
  - Tự động quét đường cong diện tích/IoU trên `100%` video, kiểm tra chéo **~10% khung hình rải rác từ đầu đến cuối** (cộng thêm các frame biến động bất thường), vá lỗ thủng/mép bị khuyết, loại bỏ đốm nền thừa và lan truyền sửa sang các khung hình lân cận trước khi nén **NVENC GPU H.264**.
- **Phân Tích Quang Học Đa Khung Hình & Tạo Prompt Bối Cảnh (`src/flow_prompt_engine.py`)**:
  - Tự đo tỷ lệ chủ thể, góc máy, hướng sáng, độ mờ hậu cảnh, nhiệt độ màu Kelvin và chuyển động camera để xuất **3 Prompt chuẩn phối cảnh** cho Google Flow / Veo / Omni 1.2 V2V / ảnh nền tĩnh.

---

## Cài Đặt & Khởi Chạy Nhanh

### 1. Cài đặt môi trường (`Python 3.11+`, khuyến nghị GPU NVIDIA `VRAM >= 6 GB`)
```powershell
git clone https://github.com/hoangphuong302/tool-tach-ghep-nen-chat-luong-cao.git
cd tool-tach-ghep-nen-chat-luong-cao
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
python download_checkpoints.py
```

*(Tùy chọn cho engine MatAnyone 2)*: clone `https://github.com/pq-yang/MatAnyone2` vào `third_party/MatAnyone2`.

### 2. Chạy Web Studio Hiện Đại (Khuyên Dùng)
```powershell
.\.venv\Scripts\python.exe studio_server.py --host 127.0.0.1 --port 7883
```
Sau đó mở trình duyệt tại `http://127.0.0.1:7883`.

### 3. Chạy Giao Diện Gradio Cổ Điển / CLI
```powershell
.\.venv\Scripts\python.exe app.py
# Hoặc CLI:
.\.venv\Scripts\python.exe cli_test.py --src input.mp4 --ai bg.mp4 --out output.mp4 --points_json points.json
```

---

## Cấu Trúc Mã Nguồn & Kiểm Thử

- [studio_server.py](studio_server.py) + [web/](web/): FastAPI microservice upload phân mảnh, Live Preview, Render NVENC & tự dọn phiên tạm.
- [src/segmentation.py](src/segmentation.py): SAM 2.1 Box-Select, Universal Roto-Brush, và `auto_cut_universal_hybrid`.
- [src/neural_matting.py](src/neural_matting.py): BiRefNet_HR-matting, ViTMatte, MatAnyone 2 GPU pipelining & PyMatting foreground estimation.
- [src/alpha_matting.py](src/alpha_matting.py): `refine_alpha_edge_controls`, Pass-2 pixel verification & temporal alpha stabilization.
- [src/matte_qa.py](src/matte_qa.py): `heal_universal_structural_defects` & `audit_and_heal_video_mattes` (QA rà soát 10% & tự vá lỗi).
- [src/relight_harmonizer.py](src/relight_harmonizer.py): Linear-RGB compositing, CIE Lab ambient harmonize, black-point match, LUTs & Pass-3 AA.
- [src/flow_prompt_engine.py](src/flow_prompt_engine.py): Phân tích quang học đa khung hình & tạo Prompt Google Flow cho mọi chủ thể.

Chạy toàn bộ bộ kiểm thử tự động:
```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -q
```
