# Tool Tách Ghép Nền Chất Lượng Cao — AGY 3D Roto-Brush & Google Flow Studio

Ứng dụng chạy **100% Offline trên GPU máy cá nhân ở độ phân giải gốc (không giảm scale)** giúp tách chủ thể (bàn tay cầm sản phẩm, thiết bị điện tử, vật thể bất kỳ trên phông xanh hoặc phông nền đời thực) ở cấp độ **Sub-Pixel**, tự động **Kiểm Định & Bù Đắp Pixel 3 Vòng (Pass-1 ➔ Pass-2 ➔ Pass-3 Khử Răng Cưa Tiếp Tuyến & Khôi Phục Pixel Gốc)**, **Khớp Ánh Sáng Trung Tính Mép Viền** không ám đỏ/viền trắng, **Khóa Chuyển Động Camera** và tự động **Đo Tiêu Cự Ống Kính (`mm`)** viết **Prompt Google Flow** phong cách điện thoại bình dân chân thực.

---

## 1. Tính Năng Cốt Lõi

1. **Chọn Vùng Thông Minh 1 Nét Kiểu Photoshop (`1-Stroke Object Selection` + `SAM 2.1`)**
   - Bôi **1 nét Cọ Xanh Lá** lên đối tượng ➔ công cụ tự động nhận diện và phủ xanh lá ngọc lục bảo toàn bộ vật thể cộng dồn (nhiều vật thể chạm nhẹ mỗi thứ 1 nét); dùng **Cọ Đỏ** để trừ vùng thừa, hoặc bấm **Tự Động Tách Tay + Sản Phẩm 1-Click** / **Tách Phông Xanh 1-Click**.
   - Tự động phát hiện và cắt bỏ viền đen 2 bên (ví dụ video dọc `9:16` nằm trong khung `16:9`), đồng thời cắt sẵn 6 ảnh tham chiếu từ video gốc.
2. **Hệ Thống Kiểm Định & Tinh Chỉnh 3 Vòng Khép Kín Ở Độ Phân Giải Gốc (`100% Full-Res 3-Pass`)**
   - **Vòng 1 (`SAM 2.1 + Temporal Alpha Stabilization`):** Theo dõi chuỗi khung hình bằng `SAM2VideoPredictor` (`add_new_mask`) kết hợp làm mượt biên & ổn định alpha thích ứng chuyển động (`stabilize_alpha_temporal`) khử sạch hiện tượng nhấp nháy giữa các frame.
   - **Vòng 2 (`Pass-2 Pixel Verification & Neutral White-Rim De-Glare`):** Đối chiếu với khung hình gốc và keyframe tham chiếu để **bù đắp pixel bị thiếu**, **cắt bỏ khe nền thừa**, **khử spill trung tính không ám đỏ lên da tay**, và dùng **Highlight-Clamped Near-Boundary Inpainting** bóc sạch viền trắng đèn studio cũ.
   - **Vòng 3 (`Pass-3 Post-Composite Audit, Source-Pixel Copy-Back & 5-Tap Tangential AA`):** Quét trực tiếp từng khung hình thành phẩm so với khung hình gốc để khôi phục 100% chi tiết da/chữ màn hình bị sai lệch và chạy khử răng cưa **5-tap dọc tiếp tuyến đường biên + edge bilateral polish** cho mép ngón tay/thiết bị siêu mịn màng.
3. **Ánh Sáng Mép Viền Tự Nhiên & Bảo Vệ Màn Hình Phát Sáng (`Dual-Zone Relighting`)**
   - Lấy mẫu bức xạ ánh sáng nền mới theo vector pháp tuyến ngoài đường biên ở độ phân giải gốc kết hợp chuyển màu CIE Lab giới hạn kênh `a` (ngăn da bị đỏ gắt), hòa trộn mép viền tự nhiên.
   - Tự động nhận diện và giữ nguyên 100% độ sắc nét chữ/giao diện trên màn hình thiết bị, chỉ phủ phản quang mặt kính nhẹ.
4. **Đo Tiêu Cự Quang Trắc (`mm`) & Tạo Prompt Google Flow Chất Lượng Điện Thoại Chân Thực**
   - Từ khung hình gốc tự động suy luận tiêu cự ống kính điện thoại (`24mm/26mm/28mm/35mm`), góc nghiêng `pitch/roll`, khoảng cách tay-máy (`cm`) và tương quan tỷ lệ tiền cảnh/hậu cảnh.
   - Viết Prompt Google Flow (Prompt nền sạch & Prompt Omni 1.2 V2V) mô phỏng **chất lượng quay từ điện thoại bình dân** (chỉ mờ nhòe rất nhẹ ở hậu cảnh xa + nhiễu cảm biến tự nhiên, không xóa phông giả DSLR `f/1.8` gây lộ viền ghép).
5. **Khóa Chuyển Động Camera**
   - Đo rung/lắc/nghiêng máy quay của nền video gốc bằng Optical Flow RANSAC và áp vào video nền mới để chủ thể và hậu cảnh khóa chặt thành một khối.

---

## 2. Yêu Cầu Hệ Thống

- **Hệ điều hành:** Windows 10/11 hoặc Linux
- **Python:** Python `3.11` (khuyên dùng) hoặc `3.10+`
- **GPU:** NVIDIA GPU hỗ trợ CUDA (khuyến nghị `VRAM >= 6 GB`, đã kiểm chứng chạy mượt trên `RTX 4060 8GB`)
- **FFmpeg:** Tự động dùng `imageio-ffmpeg` đi kèm hoặc `ffmpeg` hệ thống trong `PATH`

---

## 3. Cấu Hình & Cài Đặt

### Bước 1: Clone mã nguồn
```bash
git clone https://github.com/hoangphuong302/tool-tach-ghep-nen-chat-luong-cao.git
cd tool-tach-ghep-nen-chat-luong-cao
```

### Bước 2: Tạo môi trường ảo và cài đặt thư viện
Trên **Windows (PowerShell / CMD)**:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

> **Lưu ý:** Nếu dùng CPU hoặc CUDA phiên bản khác, chọn lệnh cài `torch` tương ứng tại [pytorch.org](https://pytorch.org/get-started/locally/).

### Bước 3: Tải trọng số Model Offline (`SAM 2.1`)
Chạy script tải checkpoint `sam2.1_hiera_small.pt` (~184 MB) vào thư mục `checkpoints/`:
```powershell
python download_checkpoints.py
```
*(Nếu dùng chế độ **ViTMatte GPU**, lần chạy đầu tiên cần có mạng để HuggingFace tải model `hustvl/vitmatte-small-composition-1k` về cache, sau đó chạy 100% offline.)*

### Bước 4: (Tùy chọn) Cấu hình AGY Local AI Router cho Prompt Studio
Mặc định, nếu máy có chạy AGY Local Router tại `http://127.0.0.1:20128/v1/chat/completions`, tool sẽ tự dùng model local để làm giàu câu văn điện ảnh tiếng Anh cho Prompt Google Flow. Nếu không chạy router, tool vẫn hoạt động **100% offline** bằng bộ suy luận quang học tích hợp sẵn.

---

## 4. Hướng Dẫn Sử Dụng

### Cách 1: Dùng Giao Diện Web Trực Quan (Khuyên Dùng)
- Chạy nhanh trên Windows bằng cách bấm đúp file **`run.bat`**, hoặc chạy lệnh:
```powershell
.\.venv\Scripts\python.exe app.py
```
- Mở trình duyệt truy cập địa chỉ: **`http://127.0.0.1:7860`**

**Quy trình 3 bước trên giao diện Web:**
1. **Bước 1 — Tách Chủ Thể Thông Minh:**
   - Chọn Video gốc (`.mp4`, `.mov`) hoặc Ảnh gốc (`.jpg`, `.png`) ➔ Bấm **`1. Nạp Video & Phân Tích`**.
   - Bấm **`✨ Tự Động Tách Tay + Sản Phẩm`** (hoặc dùng **Cọ Xanh Lá** bôi lên vật cần giữ, **Cọ Đỏ** bôi lên nền cần xóa rồi bấm **`🖌️ Xác Nhận Nét Cọ`**; nếu quay phông xanh bấm **`🟩 Tách Phông Xanh`**).
   - Kiểm tra kết quả tách viền Sub-Pixel ở cột bên phải (nền caro) hoặc qua tab **Tách Toàn Bộ Video** để xem trước toàn clip.
2. **Bước 2 — Viết Prompt Tạo Video Bối Cảnh Mới (Google Flow):**
   - Chọn nhanh bối cảnh gợi ý (Sảnh Sân Bay, Quán Cafe Tokyo, Khách Sạn 5 Sao, Phố Đi Bộ, Ghế Sau Taxi) hoặc gõ mô tả tiếng Việt bất kỳ ➔ Bấm **`2. Viết Prompt Tạo Video Nền Mới (Google Flow)`**.
   - Copy **PROMPT 1** mang sang **Google Flow / Veo** để tạo video nền sạch khớp 100% góc máy, độ mờ Bokeh và ánh sáng gốc.
3. **Bước 3 — Ghép Nền Mới & Tự Khớp Ánh Sáng 3D:**
   - Nạp video nền mới tải từ Google Flow vào ô **Nạp Video Bối Cảnh Mới**.
   - Bấm **`👁️ Xem Trước 1 Khung Hình (Before / After)`** để kéo thanh trượt đối chiếu trực tiếp.
   - Bấm **`🎬 3. Xuất Video Ghép Hoàn Chỉnh (MP4 Full HD)`** để render toàn bộ video MP4 chất lượng cao kèm âm thanh gốc.
   - *(Nếu cần tinh chỉnh sâu hơn: mở ngăn **`⚙️ Tùy chỉnh kỹ thuật nâng cao`** ở cuối trang để điều chỉnh Trimap, Đổ bóng 3D mép viền, Bảo vệ màn hình, Độ mờ Bokeh nền và Chế độ Camera Tracking.)*

### Cách 2: Chạy Qua Dòng Lệnh (CLI)
```powershell
.\.venv\Scripts\python.exe cli_test.py --src path\to\source.mp4 --ai path\to\bg.mp4 --out output.mp4 --points_json points.json
```

---

## 5. Cấu Trúc Thư Mục & Kiểm Thử

```text
tool-tach-ghep-nen-chat-luong-cao/
├── app.py                        # Giao diện Web Gradio 3 bước + Logic điều phối
├── cli_test.py                   # Pipeline CLI xử lý hàng loạt
├── download_checkpoints.py       # Script tự động tải checkpoint SAM 2.1
├── requirements.txt              # Danh sách thư viện Python cần thiết
├── run.bat                       # Script khởi động nhanh trên Windows
├── src/
│   ├── alpha_matting.py          # Pass-2 bù đắp pixel, SDF smoothing, khử viền trắng & ổn định alpha
│   ├── relight_harmonizer.py     # Khớp sáng Lab, Edge wrap & Pass-3 khử răng cưa tiếp tuyến + copy pixel gốc
│   ├── segmentation.py           # Roto-Brush SAM 2.1, Multi-cluster, Color Edge Snap & Full-res Temporal Smooth
│   ├── spatial_align.py          # Đo & khóa chuyển động Camera nền (Optical Flow RANSAC)
│   ├── flow_prompt_engine.py     # Đo tiêu cự quang trắc (mm) & tạo Prompt Google Flow điện thoại thật
│   ├── temporal_sync.py          # Đồng bộ thời gian (Auto DTW / Manual / 1:1)
│   ├── compositor.py             # Compositing 3 lớp truyền thống
│   └── video_io.py               # Đọc/xuất frame & đóng gói MP4 H.264 CRF 16
└── tests/
    ├── test_matting_relight_prompt.py  # Bộ 10 test định lượng & tích hợp End-to-End
    ├── test_alignment.py               # Test căn khớp không gian
    └── smoke_test.py                   # Smoke test tổng thể
```

### Chạy Bộ Kiểm Thử Tự Động
```powershell
.\.venv\Scripts\python.exe tests\test_matting_relight_prompt.py
```
Kết quả chuẩn sẽ hiển thị: `ALL TESTS PASSED!`.
