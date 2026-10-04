# VieDub — Tài liệu thiết kế: kế hoạch phát triển bản dùng trên máy

| | |
|---|---|
| Trạng thái | Bản nháp, chờ duyệt |
| Ngày | 03/10/2026 |
| Phạm vi | Bản chạy trên máy cá nhân (Windows, RTX 3060 Laptop 6GB), một người dùng |
| Mã liên quan | `vi_dub_web.py` (giao diện + điều phối), `vi_dub_erase.py` (xoá chữ AI), pipeline `videotrans/` của pyVideoTrans |

## 1. Mục tiêu

1. **Nhanh là ưu tiên số 1.** Giảm thời gian từ lúc bấm *Lồng tiếng* tới lúc mở được trình sửa, và thời gian tới khi có file xuất.
2. **Đủ tính năng để dùng hằng ngày**, lấy theo các tính năng của ViralCrawl mà ta đánh giá là làm được trên máy.
3. **Ổn định**: chạy hàng loạt video không phải ngồi canh, lỗi mạng tự thử lại, dừng giữa chừng không để lại rác.

**Ngoài phạm vi tài liệu này** (xem mục 8): bán gói và chạy trên server, tự đăng bài, cào video theo từ khoá hoặc theo kênh, lịch chạy 24/7.

## 2. Hiện trạng và số đo

Luồng hiện tại cho mỗi video, chạy tuần tự:

```
tải video → [xoá chữ AI] → tách nhạc nền + chuẩn bị → Whisper → dịch → tạo giọng → MỞ TRÌNH SỬA
                                                                   → (sửa) → căn khớp → ghép → file xuất
```

Ở chế độ nhiều video, trình sửa chỉ mở khi **tất cả** video đã chạy xong.

Số đo lấy từ nhật ký các lần chạy thật ngày 03/10/2026 (bật giữ nhạc nền, Whisper `large-v3-turbo` chạy GPU, Edge-TTS):

| Bước | Video 40 giây, 9 câu | Video ~7 phút (403s và 456s), 104–114 câu |
|---|---|---|
| Xoá chữ AI | chưa ghi | **ước tính 12–21 phút** (đo được 10–18 khung/giây, video ~30 khung/giây) |
| Tách nhạc nền + chuẩn bị | 9–11 giây | 22–29 giây |
| Nhận dạng Whisper | 19–25 giây | 55–98 giây |
| Dịch (Google) | ~1 giây | 7–8 giây lần đầu, ~1 giây khi có cache |
| Tạo giọng (Edge-TTS) | 1–2 giây | **160–224 giây** lần đầu (1–4 câu lỗi), 7–10 giây khi có cache |
| Căn khớp + ghép khi xuất | 19 + 6 giây | chưa ghi |

**Kết luận:**

- **Xoá chữ AI chiếm phần lớn thời gian**, và hiện nó chặn mọi bước sau. Video 7 phút có bật xoá AI phải chờ khoảng 17–26 phút mới vào được trình sửa.
- **Tạo giọng là bước chậm thứ hai.** Máy chủ Edge-TTS trả rỗng theo đợt: khi thử ngày 03/10, hơn 1/3 số lần gọi bị trả rỗng. Mỗi lần thử lại của pyVideoTrans phải chờ 5 giây.
- **Whisper chậm hơn cần thiết.** Mỗi video nạp lại mô hình trong một tiến trình mới. Bộ chạy theo lô `BatchedInferencePipeline` có import nhưng không dùng. `beam_size` đang là 5.
- **Chạy lại cùng video** nhanh hơn hẳn nhờ cache dịch và cache giọng. Whisper vẫn chạy lại từ đầu. Xoá chữ AI thì đã có cache theo file.

## 3. Chỉ số và mục tiêu

Đo trên hai video chuẩn cố định: clip 40 giây (áo mưa, 9 câu) và một video Douyin ~7 phút. Đo ở lần chạy đầu, không có cache.

| Chỉ số | Hiện tại (7 phút, bật xoá AI) | Mục tiêu |
|---|---|---|
| Thời gian tới khi mở trình sửa | ~17–26 phút | **≤ 3 phút** |
| Thời gian tới file xuất (không sửa gì) | ~18–27 phút | **≤ 10 phút** |
| Tốc độ xoá chữ AI | 10–18 khung/giây | **≥ 30 khung/giây** |
| Câu tạo giọng lỗi sau cùng | 1–4 câu/video | 0 |
| Chạy lại cùng video, cùng cài đặt | vài phút | ≤ 30 giây tới trình sửa |

## 4. Nguyên tắc thiết kế

- **Đo trước, tối ưu sau.** Mỗi thay đổi về tốc độ phải có số trước/sau trên hai video chuẩn.
- **Không bắt người dùng chờ thứ chưa cần.** Bước nào chỉ cần cho lúc xuất thì chạy nền trong khi người dùng sửa.
- **Ngân sách VRAM 6GB.** Không để hai mô hình GPU lớn cùng nằm trong bộ nhớ khi chưa đo là vừa.
- **Cache mọi kết quả trung gian theo nội dung file** (hash + cài đặt liên quan), để chạy lại hoặc đổi một cài đặt không phải làm lại từ đầu.
- **Hạn chế sửa mã gốc pyVideoTrans.** Ưu tiên bọc ngoài từ `vi_dub_*.py` để còn cập nhật được bản pyVideoTrans mới.

## 5. Giai đoạn 1: Tốc độ

### 5.1. Đo đạc sẵn trong app (làm đầu tiên)

- Ghi thời gian từng bước của từng video vào `output/_work/<video>/timings.json`, và hiện trong "Nhật ký chi tiết".
- Thêm script `tools/bench_viedub.py` chạy hai video chuẩn và in bảng so sánh.
- **Xong khi:** script in ra được bảng như mục 2 cho cả hai video.

### 5.2. Xoá chữ AI chạy nền, không chặn trình sửa

- **Ý tưởng:** Whisper chạy trên âm thanh gốc, không cần video đã xoá chữ. Bản đã xoá chữ chỉ cần cho lúc xuất.
- **Luồng mới:** tìm vị trí sub gốc (vài giây), chạy Whisper, dịch, tạo giọng, rồi mở trình sửa. Xoá chữ AI chạy trong một luồng nền riêng, bắt đầu sau khi Whisper xong để không tranh VRAM.
- Trong lúc chờ, trình sửa phát video gốc kèm thanh tiến độ "Đang xoá chữ: 45%". Xong thì tự đổi sang bản đã xoá.
- Khi xuất mà bản xoá chưa xong thì chờ tiếp và báo tiến độ.
- Nút Dừng và nút Xoá dự án phải huỷ được cả việc chạy nền này.
- **Xong khi:** video 7 phút mở được trình sửa mà không phải chờ bước xoá chữ.

### 5.3. Tăng tốc chính bước xoá chữ AI

Đo riêng từng phần trước: giải mã video, nhận chữ (OCR, đang chạy CPU), LaMa (GPU), mã hoá. Sau đó làm theo thứ tự lợi ích:

1. **LaMa chạy FP16** (nửa độ chính xác) trên GPU. So ảnh trước/sau để chắc chất lượng không giảm thấy được.
2. **Bỏ qua khung không có chữ** trong dải sub, chép thẳng mà không qua LaMa. Phần này đã có một phần; cần đo tỉ lệ khung thực sự phải vẽ lại.
3. **Dùng lại hộp chữ giữa các khung.** Một câu phụ đề thường đứng yên 1–3 giây, nên chỉ cần OCR lại khi dải sub thay đổi đáng kể.
4. **Giải mã bằng GPU** (`-hwaccel cuda`), cho OCR và LaMa chạy song song qua hàng đợi.
5. Nếu OCR vẫn là nút thắt, thử chạy OCR bằng onnxruntime trên GPU.

**Xong khi:** đạt ≥ 30 khung/giây trên video Douyin 720p, và ảnh kết quả không kém bản hiện tại.

### 5.4. Tạo giọng nhanh và chắc

- Thay cơ chế thử lại cố định 5 giây của pyVideoTrans bằng cơ chế riêng của VieDub:
  - chạy song song 4–6 câu;
  - khi bị trả rỗng thì giảm số luồng và chờ tăng dần;
  - cuối cùng quét lại các câu còn lỗi.
- Cơ chế này đã có cho nút tạo giọng trong trình sửa, giờ áp dụng cả cho lần lồng tiếng đầu.
- Thêm kênh giọng chạy trên máy (Supertonic3, xem 6.1) để không phụ thuộc mạng, và đo tốc độ của nó trên 3060.
- **Xong khi:** video 7 phút tạo giọng xong trong ≤ 60 giây với Edge-TTS và không còn câu lỗi.

### 5.5. Whisper nhanh hơn

1. Dùng `BatchedInferencePipeline` của faster-whisper (đã có sẵn trong thư viện).
2. Giữ mô hình trong một tiến trình chạy lâu, dùng lại cho các video sau thay vì nạp lại. Giải phóng khi cần VRAM cho LaMa.
3. Thử `beam_size` 1–2 và đo xem chữ nhận dạng có kém đi không.

**Xong khi:** video 7 phút nhận dạng xong trong ≤ 25 giây, tỉ lệ sai chữ không tăng rõ.

### 5.6. Nhiều video: xong video nào mở video đó

- Mở trình sửa ngay khi video đầu tiên xong. Các video sau tiếp tục chạy nền và hiện dần trong danh sách "Video đang sửa".
- Tải trước video kế tiếp trong lúc xử lý video hiện tại. Việc tải chỉ dùng mạng, không đụng GPU.

### 5.7. Cache theo nội dung

- Lưu kết quả Whisper (srt gốc), bản dịch và vị trí sub gốc theo hash file + cài đặt liên quan.
- Chạy lại cùng video thì chỉ làm lại các bước có cài đặt đã đổi.
- Nút Dọn dẹp xoá được nhóm cache này.

### 5.8. Xuất nhanh hơn

- Gộp mọi xử lý hình (che sub, logo, hiệu ứng ở 6.4) vào **một** lượt mã hoá NVENC duy nhất, thay vì mỗi bước mã hoá lại một lần.
- Kiểm tra bước ghép cuối của pyVideoTrans đã dùng NVENC chưa. Nếu chưa thì truyền tham số để dùng.
- Thêm nút **Xem nhanh**: dựng bản 480p chất lượng thấp trong vài giây để kiểm tra trước khi xuất bản đầy đủ.

### 5.9. Khởi động app nhanh hơn

- App hiện mất khoảng 40 giây mới mở. Chuyển các thư viện nặng (torch, mô hình) sang nạp khi cần lần đầu.

## 6. Giai đoạn 2: Tính năng

### 6.1. Thêm giọng

- Thêm ô **Kênh giọng** trong Cài đặt. Danh sách giọng đổi theo kênh, có nút **Nghe thử** một câu mẫu.
- **Supertonic3:** 10 giọng (5 nữ, 5 nam), có tiếng Việt, chạy trên máy.
- **F5-TTS bản tiếng Việt:** nhân bản giọng từ đoạn ghi âm mẫu 5–10 giây, có chế độ dùng chính giọng người nói gốc. Cần chỗ tải lên và quản lý giọng mẫu.
- **ElevenLabs** (tuỳ chọn, cần khoá API riêng): giọng Việt tự nhiên nhất, tính tiền theo ký tự.
- Trình sửa vẫn tạo lại giọng bằng đúng kênh của video đó, vì đã dùng `tts_type` của dự án.

### 6.2. Giọng theo từng nhân vật

- Bật bước tách người nói của pyVideoTrans (VieDub đang tắt).
- Tự đoán nam/nữ cho từng người nói: thử bằng cao độ giọng trước, cần thì dùng mô hình phân loại.
- Trong trình sửa có bảng "Người nói → Giọng" để đổi giọng cho cả một người nói một lần.

### 6.3. Kiểu phụ đề

- Chọn font (Arial, Impact, Bangers, Tahoma…), màu chữ, màu viền, độ dày viền, bóng.
- **Karaoke từng từ.** Whisper đã có mốc thời gian từng từ (`word_timestamps=True`) cho câu gốc. Câu tiếng Việt thì chia đều theo độ dài giọng đã tạo. Hiển thị bằng hiệu ứng karaoke của định dạng phụ đề ASS.
- Lưu thành mẫu, và xem trước ngay trên ảnh như hiện tại.

### 6.4. Hiệu ứng video

- Logo hoặc chữ đóng dấu (chọn vị trí, độ trong), lật ngang, phóng to nhẹ, đổi tốc độ, chỉnh màu, cắt bỏ đầu/đuôi, cắt mép.
- Tất cả gộp vào một chuỗi bộ lọc ffmpeg ở bước xuất (xem 5.8).

### 6.5. Bộ cấu hình

- Lưu nhiều bộ cấu hình (ví dụ "Douyin review", "Phim ngắn"), mỗi bộ gồm cài đặt chung, phụ đề và hiệu ứng. Chọn bộ trước khi lồng tiếng.

### 6.6. Thêm nguồn video

- YouTube, Bilibili, Facebook, Instagram qua yt-dlp (app đã có sẵn yt-dlp).
- Kuaishou, Xiaohongshu: làm ở mức cố gắng, báo lỗi rõ khi bị chặn.

### 6.7. Chỉnh thời gian câu thoại trong trình sửa

- Kéo hai mép đoạn trên dòng thời gian, hoặc sửa số ở cột Từ/Đến.
- Tách một câu làm hai, gộp hai câu làm một.

## 7. Thứ tự làm

| Mốc | Nội dung | Ghi chú |
|---|---|---|
| M1 | 5.1 đo đạc | Mọi việc sau dựa vào số đo này |
| M2 | 5.2 xoá AI chạy nền + 5.6 mở trình sửa sớm | Lợi ích cảm nhận lớn nhất |
| M3 | 5.3 tăng tốc xoá AI | Lợi ích tổng thời gian lớn nhất |
| M4 | 5.4 tạo giọng + 6.1 thêm giọng (Supertonic3 trước) | Hai việc dùng chung phần chọn kênh giọng |
| M5 | 5.5 Whisper + 5.7 cache + 5.9 khởi động | |
| M6 | 5.8 xuất một lượt + 6.4 hiệu ứng + 6.5 bộ cấu hình | Cùng chạm vào bước xuất |
| M7 | 6.3 kiểu phụ đề + karaoke | |
| M8 | 6.2 giọng theo nhân vật, 6.6 nguồn video, 6.7 chỉnh thời gian | |

Mỗi mốc kết thúc bằng ba việc: chạy script đo trên hai video chuẩn, chạy trọn luồng trên trình duyệt, và cập nhật bảng số ở mục 2.

## 8. Để sau (khi tính chuyện bán gói)

Không làm trong kế hoạch này, nhưng các quyết định ở trên không được chặn đường làm sau.

- **Tự đăng bài, cào video theo từ khoá hoặc theo kênh, lịch chạy 24/7.**
- **Chạy trên server cho nhiều người dùng.** App hiện giữ dự án, nút Dừng và file kiểu phụ đề ở dạng dùng chung cho cả tiến trình. Khi làm 5.2 và 5.6, nên gom trạng thái theo từng dự án để sau này tách người dùng dễ hơn.
- **Giấy phép trước khi bán:**
  - đổi Edge-TTS và Google Dịch miễn phí sang dịch vụ trả phí có giấy phép;
  - không dùng F5-TTS bản gốc, vì giấy phép cấm thương mại;
  - đọc kỹ giấy phép OpenRAIL của Supertonic3;
  - pyVideoTrans dùng GPL-3.0: bán dạng web thì không phải công khai mã, bán dạng app cài đặt thì phải giao kèm mã nguồn.

## 9. Rủi ro

| Rủi ro | Cách xử lý |
|---|---|
| Xoá AI chạy nền tranh VRAM với Whisper hoặc mô hình giọng chạy trên máy | Chỉ bắt đầu sau khi Whisper xong; đo VRAM; có khoá dùng GPU chung |
| FP16 làm LaMa để lại vệt | So ảnh trước/sau trên các khung khó; có công tắc quay về FP32 |
| Máy chủ Edge-TTS chặn mạnh hơn | Tự giảm số luồng; có kênh giọng chạy trên máy để thay |
| Douyin chặn IP khi tải nhiều | Giữ giãn cách hiện có; dùng cookie |
| Sửa vào mã pyVideoTrans làm khó cập nhật | Bọc ngoài trong `vi_dub_*.py`; buộc phải sửa thì ghi lại từng chỗ |

## 10. Câu hỏi còn mở

1. Có chấp nhận nhận dạng kém đi một chút (beam nhỏ hơn) để Whisper nhanh hơn không?
2. Video kết quả cần độ phân giải nào: giữ như gốc, hay cho chọn 720p/1080p?
3. Có trả phí ElevenLabs cho giọng không, hay chỉ dùng các kênh miễn phí?
