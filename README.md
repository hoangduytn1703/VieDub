# VieDub

Công cụ **lồng tiếng Việt tự động cho video tiếng Trung** (Douyin, TikTok, YouTube, Bilibili… hoặc file từ máy). Chạy hoàn toàn trên máy của bạn, giao diện web mở bằng trình duyệt.

VieDub được xây dựng trên [pyVideoTrans](https://github.com/jianchang512/pyvideotrans) (nhận dạng giọng nói, dịch, tạo giọng, ghép video) và thêm giao diện web, trình sửa câu thoại, xoá chữ gốc bằng AI, chế độ tải video và đóng dấu logo/chữ.

## Tính năng

- **Lồng tiếng Việt**: nhận dạng giọng nói (Whisper) → dịch → tạo giọng Việt (Edge-TTS) → ghép lại, giữ nhạc nền, chèn phụ đề tiếng Việt.
- **Trình sửa kiểu CapCut**: xem video, dòng thời gian, bảng câu thoại; sửa chữ từng câu, tạo lại giọng, nghe thử rồi mới xuất.
- **Xoá chữ gốc bằng AI**: tự tìm phụ đề tiếng Trung in sẵn trong video và vẽ lại nền (OCR + LaMa). Hoặc làm mờ / che nền tối.
- **Chế độ "Chỉ tải video"**: tải video từ link về máy (tối đa 1080p), không lồng tiếng.
- **Đóng dấu logo và/hoặc chữ** ở vị trí cố định, dùng cho cả hai chế độ.
- **Nhiều nền tảng**: Douyin, TikTok, YouTube, Bilibili, Facebook, X, Instagram (cần cookie).
- **Xử lý hàng loạt**: dán nhiều link / thả nhiều file, đặt tên từng video (là tên file khi xuất).
- Nút **Dừng ngay**, nút **Dọn dẹp** (xoá file tạm, nhật ký để lấy lại dung lượng).

## Yêu cầu

| | Yêu cầu |
|---|---|
| Hệ điều hành | **Windows 10 / 11** (64-bit) |
| Card màn hình | **NVIDIA, từ 6 GB VRAM** khuyến nghị (đã chạy tốt trên RTX 3060 Laptop 6 GB). Driver NVIDIA mới (từ 570 trở lên, vì dùng CUDA 12.8) |
| Ổ cứng trống | Khoảng **15 GB** (thư viện ~8,5 GB + model ~2 GB + chỗ cho video) |
| RAM | 16 GB khuyến nghị |
| Mạng | Cần Internet: cài đặt, tải model lần đầu, dịch (Google), tạo giọng (Edge-TTS), tải video từ link |
| Phần mềm | `uv` (quản lý Python) và `ffmpeg`. Cách cài ở bước dưới |

**Máy không có card NVIDIA?** Vẫn chạy được lồng tiếng, nhưng chậm hơn (nhận dạng giọng nói khoảng 0,7 lần độ dài video trên CPU 16 luồng). **Riêng xoá chữ AI gần như không dùng được** trên CPU (chậm hơn card khoảng 19 lần, video 7 phút có thể mất vài giờ). Hãy tắt công tắc "Xoá chữ gốc bằng AI" và dùng "Làm mờ" / "Che nền tối" trong trình sửa. Chế độ "Chỉ tải video" và đóng dấu không cần card.

## Cài đặt (làm 1 lần)

Mở **PowerShell** và chạy lần lượt:

**1. Cài `uv`** (tự tải đúng Python 3.10 cho dự án, bạn không cần cài Python):

```powershell
winget install --id=astral-sh.uv -e
```

**2. Cài `ffmpeg`:**

```powershell
winget install --id=yt-dlp.FFmpeg -e
```

**3. Đóng PowerShell, mở lại** để PATH mới có hiệu lực. Kiểm tra:

```powershell
uv --version
ffmpeg -version
```

**4. Tải mã nguồn:**

```powershell
git clone git@github.com:hoangduytn1703/VieDub.git
cd VieDub
```

(Chưa có git? `winget install --id=Git.Git -e`. Hoặc bấm nút **Code → Download ZIP** trên GitHub rồi giải nén.)

## Chạy

Cách đơn giản nhất, nhấp đúp file **`run_vi_dub.bat`** (hoặc chạy trong PowerShell):

```powershell
.\run_vi_dub.bat
```

- **Lần đầu** mất khoảng 5–15 phút (tuỳ mạng): tự cài Python 3.10 và tải ~8,5 GB thư viện (PyTorch + CUDA…). Các lần sau mở trong vài chục giây.
- Khi thấy dòng `Running on local URL: http://127.0.0.1:7861`, trình duyệt tự mở. Nếu không mở, tự vào địa chỉ đó.
- **Lần đầu chạy lồng tiếng**, app tự tải thêm model (Whisper ~1,6 GB, nhạc nền, LaMa ~0,2 GB khi dùng xoá chữ AI). Chờ một chút, không phải làm gì.
- Tắt app: bấm `Ctrl + C` trong cửa sổ chạy, hoặc đóng cửa sổ đó.

Chạy bằng tay (không dùng file bat), hoặc đổi cổng:

```powershell
uv run --extra webui vi_dub_web.py --port 7861
```

Tuỳ chọn: `--port <số>` đổi cổng, `--host 0.0.0.0` cho máy khác trong mạng LAN truy cập.

## Cách dùng nhanh

### Lồng tiếng

1. Vào trang **Lồng tiếng**. Ở khung **1. Thêm video**: chọn nền tảng (hoặc để "Tự nhận"), dán link vào ô (dán nhiều link được, mỗi link một ô), bấm **Kiểm tra tất cả**. Hoặc qua tab **File từ máy** để thả file mp4.
2. Khung **2. Danh sách video**: tích ô để chọn video, có thể sửa tên (là tên file khi xuất).
3. Khung **3. Cài đặt**: chọn ngôn ngữ gốc, giọng đọc, bật/tắt giữ nhạc nền, **Xoá chữ gốc bằng AI**…
4. (Tuỳ chọn) khung **Đóng dấu logo / chữ**: bật logo và/hoặc chữ, chọn vị trí.
5. Bấm **🚀 Lồng tiếng** ở thanh dưới cùng. Xong, **Trình sửa** tự mở.
6. Trong **Trình sửa**: bấm một dòng ở bảng câu thoại để nghe và sửa chữ ở khung bên phải, **Tạo lại giọng** nếu sửa. Chỉnh phụ đề, che chữ gốc ở khung bên phải. Đặt tên và thư mục xuất rồi bấm **Xuất video này** (hoặc **Xuất tất cả**).
7. Video kết quả hiện ở tab **Kết quả** để xem thử. Không ưng thì bấm **Không ưng, xoá & về trang chủ** để xoá hết file liên quan.

File xuất nằm ở `<thư mục xuất>/<tên>/<tên>.mp4`, kèm phụ đề `.vi.srt` và `.zh-cn.srt`.

### Chỉ tải video

Bấm nút **⬇️ Chỉ tải video** ở hàng **Chế độ**. Dán link, kiểm tra, tích chọn, chọn **Thư mục lưu**, (tuỳ chọn) bật đóng dấu, bấm **Tải**. File lưu thẳng thành `<tên>.mp4`.

## Thư mục dữ liệu

| Thư mục | Nội dung | Xoá được không |
|---|---|---|
| `output/` | Video đã xuất (nếu để thư mục xuất mặc định) | Đây là thành quả của bạn |
| `output/_work/` | File làm việc của từng video đang sửa | Được, hoặc dùng nút **Dọn dẹp** |
| `downloads/` | Video tải từ link, cookie (`cookies.txt`), logo đã chọn | Được (video phải tải lại) |
| `models/` | Model AI đã tải | Được, nhưng sẽ phải tải lại |
| `tmp/`, `logs/` | File tạm, nhật ký | Được, hoặc dùng nút **Dọn dẹp** |

Cài đặt (giọng, đóng dấu…) được lưu trong trình duyệt, không nằm trong thư mục dự án.

## Cookie (khi Douyin / TikTok / Instagram chặn)

Một số nền tảng đòi đăng nhập. Cách làm:

1. Cài tiện ích **Get cookies.txt LOCALLY** cho Chrome / Edge.
2. Mở trang web nền tảng đó (đã đăng nhập), bấm tiện ích → **Export**.
3. Trong VieDub vào **Cài đặt → Nâng cao → Chọn cookies.txt**. Cookie được lưu trên máy, lần sau không cần chọn lại.

Nên để ô "lấy cookie thẳng từ trình duyệt" là **Không dùng**. Chrome và Edge đang mở sẽ khoá file cookie nên hay lỗi.

## Xử lý lỗi thường gặp

| Hiện tượng | Cách xử lý |
|---|---|
| `uv` hoặc `ffmpeg` "không được nhận ra" | Cài như bước 1–2, **đóng và mở lại PowerShell** |
| Cài lần đầu báo lỗi mạng giữa chừng | Chạy lại `run_vi_dub.bat`, uv tải tiếp phần còn thiếu |
| Hết VRAM / `CUDA out of memory` | Đóng chương trình khác dùng card (game, trình duyệt nhiều tab), thử lại. Hoặc tắt "Xoá chữ gốc bằng AI" |
| Tạo giọng báo `No audio was received` | Máy chủ Edge-TTS thỉnh thoảng trả rỗng. App tự thử lại nhiều lần. Nếu vẫn lỗi, đợi vài phút rồi bấm **Tạo lại giọng** |
| Douyin báo bị chặn tạm thời | Đợi 10–30 phút, hoặc dùng cookie (phần trên) |
| Link YouTube/Facebook không dùng được | Cập nhật yt-dlp: `uv lock --upgrade-package yt-dlp` rồi chạy lại. Một số video cần cookie |
| Kuaishou, Xiaohongshu | Chưa hỗ trợ (yt-dlp chưa tải được các trang này) |
| Không tải được model (không vào được HuggingFace) | App tự thử nguồn dự phòng. Nếu vẫn lỗi, kiểm tra mạng / VPN rồi chạy lại |
| Cổng 7861 đã bị dùng | Chạy với cổng khác: `uv run --extra webui vi_dub_web.py --port 7862` |
| Muốn xem chi tiết lỗi | Mở thư mục `logs/` hoặc mục **Nhật ký chi tiết** ở khung Tiến trình |

## Cập nhật

```powershell
git pull
uv sync --extra webui
```

Hoặc chỉ cần `git pull` rồi chạy lại `run_vi_dub.bat`, file bat tự đồng bộ thư viện.

## Cấu trúc chính

| File | Vai trò |
|---|---|
| `vi_dub_web.py` | Giao diện web và điều phối (lồng tiếng, trình sửa, tải video, đóng dấu) |
| `vi_dub_erase.py` | Xoá chữ gốc bằng AI (RapidOCR tìm chữ + LaMa vẽ lại nền) |
| `videotrans/` | Lõi pyVideoTrans (nhận dạng, dịch, TTS, ghép video) |
| `run_vi_dub.bat` | Chạy app trên Windows |
| `docs/viedub_dd.md` | Tài liệu thiết kế và kế hoạch phát triển |
| `docs/README_pyVideoTrans.md` | README gốc của pyVideoTrans |

## Lưu ý về dịch vụ bên thứ ba

Dịch dùng Google Dịch và tạo giọng dùng Edge-TTS qua đường **không chính thức, miễn phí**: phù hợp cho cá nhân dùng, có thể chậm hoặc bị chặn khi dùng nhiều. Video tải từ các nền tảng thuộc bản quyền của người tạo ra chúng, hãy tuân thủ điều khoản của nền tảng và pháp luật khi sử dụng.

## Giấy phép

[GPL-3.0](LICENSE). VieDub dựa trên [pyVideoTrans](https://github.com/jianchang512/pyvideotrans) (GPL-3.0) của jianchang512. Mọi bản phân phối lại phải kèm mã nguồn theo cùng giấy phép.
