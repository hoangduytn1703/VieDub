"""
Xoá phụ đề cứng (chữ in sẵn trong video) bằng AI cho vi_dub_web.py. Tự tìm phụ đề, không cần người dùng chỉ vùng.

Cách làm:
  1. find_sub_bands: nhận diện chữ (DBNet qua RapidOCR, CPU) trên cả khung của ~24–48 khung rải đều,
     gom các dòng chữ ngang LẶP LẠI ở cùng độ cao qua nhiều khung -> đó là các dải phụ đề (có thể nhiều dải,
     vd. tiêu đề phía trên + sub phía dưới). Chữ trong cảnh (biển hiệu, chữ trên sản phẩm) chỉ xuất hiện lẻ tẻ
     nên không thành dải và không bị xoá.
  2. erase_video: quét ~10 lần mỗi giây, chỉ xoá các hộp chữ chạm vào 1 dải phụ đề; các khung ở giữa hai lần quét
     dùng hợp của hai kết quả lân cận, nhờ vậy lúc sub đổi câu không bị hở chữ.
  3. LaMa (big-lama, TorchScript, GPU) vẽ lại nền tại các hộp chữ, mỗi dải xử lý riêng kèm ngữ cảnh trên/dưới.
"""
import json
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

LAMA_URL = 'https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt'
DET_PER_SEC = 10        # số lần nhận diện chữ mỗi giây video
CTX_RATIO = 0.08        # ngữ cảnh thêm phía trên/dưới các hộp chữ cho LaMa (theo chiều cao video)
SEARCH_EXTRA = 0.5      # dải đưa vào bộ nhận diện = vùng tìm nới thêm 50% chiều cao vùng mỗi phía
LAMA_WIDTH = 640        # LaMa chạy ở bề rộng <= chừng này (thu nhỏ theo bề rộng video)
LAMA_QUANT = 32         # chiều cao vùng LaMa xử lý (sau thu nhỏ) là bội số của số này
BATCH_FRAMES = 12       # số khung gom lại cho 1 lượt LaMa
# cuDNN chọn thuật toán theo kích thước tensor; vài kích thước bị chọn rất dở (chậm 2–3 lần). Bật benchmark để
# tự dò thuật toán tốt nhất cho từng kích thước — vì vậy kích thước vùng xử lý phải cố định (xem _crop_rows).

_model = None
_detector = None
_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Mô hình
# ---------------------------------------------------------------------------
def model_path(root_dir) -> Path:
    return Path(root_dir) / 'models' / 'lama' / 'big-lama.pt'


def _ensure_model(root_dir) -> Path:
    p = model_path(root_dir)
    if p.exists() and p.stat().st_size > 100 << 20:
        return p
    import requests
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix('.part')
    with requests.get(LAMA_URL, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(tmp, 'wb') as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    tmp.replace(p)
    return p


def _device():
    import torch
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def load_model(root_dir):
    global _model
    with _lock:
        if _model is None:
            import torch
            _model = torch.jit.load(str(_ensure_model(root_dir)), map_location=_device()).eval()
        return _model


def release_model():
    """Trả VRAM cho các bước sau (Whisper, TTS...)."""
    global _model
    with _lock:
        _model = None
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


def _get_detector():
    global _detector
    with _lock:
        if _detector is None:
            from rapidocr_onnxruntime import RapidOCR
            # mặc định RapidOCR phóng ảnh sao cho cạnh NHỎ >= 736 -> dải cắt 1280x200 bị phóng gần 4 lần, rất chậm.
            # Giới hạn theo cạnh LỚN: ảnh <= 1280 giữ nguyên kích thước.
            _detector = RapidOCR(det_limit_type='max', det_limit_side_len=1280)
        return _detector


# ---------------------------------------------------------------------------
# Nhận diện chữ
# ---------------------------------------------------------------------------
def detect_boxes(rgb: np.ndarray) -> list:
    """Hộp chữ (x0, y0, x1, y1) trên ảnh RGB HxWx3."""
    if rgb.shape[0] < 8 or rgb.shape[1] < 8:
        return []
    res, _ = _get_detector()(np.ascontiguousarray(rgb[:, :, ::-1]), use_det=True, use_cls=False, use_rec=False)
    boxes = []
    for quad in res or []:
        xs = [float(p[0]) for p in quad]
        ys = [float(p[1]) for p in quad]
        boxes.append((int(min(xs)), int(min(ys)), int(max(xs)) + 1, int(max(ys)) + 1))
    return boxes


def _rows(H, top, size):
    """Dải (y0,y1) theo % chiều cao và vùng đưa vào bộ nhận diện (sy0,sy1) = dải nới thêm SEARCH_EXTRA mỗi phía."""
    y0 = max(0, min(H - 2, int(H * top / 100)))
    y1 = max(y0 + 2, min(H, int(H * (top + size) / 100)))
    extra = int((y1 - y0) * SEARCH_EXTRA)
    return y0, y1, max(0, y0 - extra), min(H, y1 + extra)


def _lama_scale(W):
    return min(1.0, LAMA_WIDTH / W)


def _crop_rows(boxes, W, H):
    """Vùng LaMa xử lý: bao các hộp chữ (đã nới) + ngữ cảnh trên/dưới; chiều cao làm tròn lên sao cho sau khi
    thu nhỏ là bội số của LAMA_QUANT (ít kích thước khác nhau -> cuDNN benchmark chỉ dò vài lần)."""
    pads = [_pad_box(b, W, H) for b in boxes]
    ctx = int(H * CTX_RATIO)
    cy0 = max(0, min(p[1] for p in pads) - ctx)
    cy1 = min(H, max(p[3] for p in pads) + ctx)
    q = max(2, round(LAMA_QUANT / _lama_scale(W)))
    want = min(H // 2 * 2, max(q, -(-(cy1 - cy0) // q) * q))
    extra = want - (cy1 - cy0)
    cy0 = max(0, cy0 - extra // 2)
    cy1 = min(H, cy0 + want)
    cy0 = max(0, cy1 - want)
    return cy0, cy1


class _cudnn_benchmark:
    """Bật cudnn.benchmark trong lúc xoá chữ, xong trả lại như cũ (mô hình khác của pyVideoTrans không bị ảnh hưởng)."""

    def __enter__(self):
        import torch
        self.old = torch.backends.cudnn.benchmark
        torch.backends.cudnn.benchmark = True

    def __exit__(self, *a):
        import torch
        torch.backends.cudnn.benchmark = self.old


def _pad_box(box, W, H):
    x0, y0, x1, y1 = box
    bh = max(1, y1 - y0)
    px, py = max(6, int(bh * 0.35)), max(4, int(bh * 0.3))   # phủ viền đen + bóng đổ quanh chữ
    return max(0, x0 - px), max(0, y0 - py), min(W, x1 + px), min(H, y1 + py)


def _touches_band(box, W, H, y0, y1) -> bool:
    """Hộp chữ (tính cả viền / bóng đổ quanh chữ) chạm vào dải [y0, y1)."""
    _, py0, _, py1 = _pad_box(box, W, H)
    return py1 > y0 and py0 < y1


def _sub_like(box, W, H) -> bool:
    """Trông giống 1 dòng phụ đề: chữ ngang, đủ rộng, không quá cao (bỏ logo / watermark nhỏ)."""
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    return w >= 0.08 * W and 0.012 * H <= h <= 0.1 * H and w >= 1.5 * h


def boxes_in_bands(rgb: np.ndarray, bands) -> list:
    """[(chỉ số dải, hộp chữ)] cho các hộp chạm vào 1 trong các dải phụ đề (bands: [(top%, size%), ...])."""
    H, W = rgb.shape[:2]
    out = []
    for k, (top, size) in enumerate(bands):
        y0, y1, sy0, sy1 = _rows(H, top, size)
        for x0, by0, x1, by1 in detect_boxes(rgb[sy0:sy1]):
            box = (x0, by0 + sy0, x1, by1 + sy0)
            if _touches_band(box, W, H, y0, y1):
                out.append((k, box))
    return out


def _bands_from_lines(lines, H, n_frames, min_frames):
    """Gom các dòng chữ (y0, y1, chỉ số khung) theo độ cao -> các dải xuất hiện ở >= min_frames khung."""
    if not lines:
        return []
    lines = sorted(lines, key=lambda t: (t[0] + t[1]) / 2)
    clusters, cur = [], [lines[0]]
    for ln in lines[1:]:
        ref = sum((a + b) / 2 for a, b, _ in cur) / len(cur)
        if abs((ln[0] + ln[1]) / 2 - ref) < 0.03 * H:
            cur.append(ln)
        else:
            clusters.append(cur)
            cur = [ln]
    clusters.append(cur)
    bands = []
    for c in clusters:
        hits = len({i for _, _, i in c})
        if hits < min_frames:
            continue
        heights = sorted(b - a for a, b, _ in c)
        pad = int(0.6 * heights[len(heights) // 2])
        y0 = max(0, min(a for a, _, _ in c) - pad)
        y1 = min(H, max(b for _, b, _ in c) + pad)
        top = y0 / H * 100
        bands.append({'top': round(top, 1), 'size': round(max(3.0, y1 / H * 100 - top), 1), 'hits': hits})
    # gộp các dải chồng lên nhau (vd. sub 2 dòng bị tách thành 2 cụm)
    bands.sort(key=lambda b: b['top'])
    merged = []
    for b in bands:
        if merged and b['top'] <= merged[-1]['top'] + merged[-1]['size']:
            m = merged[-1]
            end = max(m['top'] + m['size'], b['top'] + b['size'])
            m.update(size=round(end - m['top'], 1), hits=max(m['hits'], b['hits']))
        else:
            merged.append(dict(b))
    return sorted(merged, key=lambda b: -b['hits'])  # dải xuất hiện nhiều nhất đứng đầu


_band_cache = {}


def find_sub_bands(src, n_samples=None):
    """Tự tìm mọi dải phụ đề trong video. Trả về (bands, info): bands = [{'top','size','hits'}] theo % chiều cao,
    dải xuất hiện nhiều nhất đứng đầu; [] nếu không thấy. Có cache theo file."""
    key = (str(Path(src).resolve()), Path(src).stat().st_mtime)
    if key in _band_cache:
        return _band_cache[key]
    W, H, _, _, dur, _ = probe(src)
    if not W or dur <= 0:
        return [], 'Không đọc được video.'
    n = n_samples or int(min(48, max(24, dur / 5)))
    times = [dur * (0.03 + 0.94 * i / max(1, n - 1)) for i in range(n)]
    with ThreadPoolExecutor(4) as ex:
        frames = list(ex.map(lambda t: grab_frame(src, t), times))
    lines = [(b[1], b[3], i) for i, f in enumerate(frames) for b in detect_boxes(f) if _sub_like(b, W, H)]
    # phụ đề = dòng chữ lặp lại ở cùng độ cao trong >= 12% số khung (tối thiểu 3)
    bands = _bands_from_lines(lines, H, n, max(3, int(0.12 * n)))
    if bands:
        desc = ', '.join(f'{b["top"]:.0f}–{b["top"] + b["size"]:.0f}%' for b in bands)
        info = f'Tìm thấy phụ đề gốc ở {desc} (quét {n} khung).'
    else:
        info = f'Không thấy phụ đề cứng nào (quét {n} khung).'
    _band_cache[key] = (bands, info)
    return bands, info


def scan_subtitle_band(src, n_samples=None):
    """Dải phụ đề chính (xuất hiện nhiều nhất), cho nút 'Tự tìm vị trí sub gốc'. Trả về (top%, size%, thông tin)."""
    bands, info = find_sub_bands(src, n_samples)
    if not bands:
        return None, None, info
    b = bands[0]
    return int(b['top']), max(4, int(np.ceil(b['size']))), info


def bands_from_image(rgb: np.ndarray) -> list:
    """Khi chỉ có 1 ảnh (vd. ảnh bìa của link chưa tải): coi mọi dòng giống phụ đề là 1 dải."""
    H, W = rgb.shape[:2]
    lines = [(b[1], b[3], 0) for b in detect_boxes(rgb) if _sub_like(b, W, H)]
    return _bands_from_lines(lines, H, 1, 1)


def _pct_bands(bands):
    return [(b['top'], b['size']) for b in bands]


# ---------------------------------------------------------------------------
# Xoá chữ bằng LaMa
# ---------------------------------------------------------------------------
def _inpaint(frames, masks, cy0, cy1, root_dir):
    """frames: B,H,W,3 uint8 (GPU); masks: B,1,h,w float (h = cy1-cy0). Sửa tại chỗ, trả về số khung đã xử lý."""
    import torch.nn.functional as F
    model = load_model(root_dir)
    has = masks.flatten(1).sum(1) > 0
    if not has.any():
        return 0
    idx = has.nonzero().flatten()
    c = frames[idx, cy0:cy1].permute(0, 3, 1, 2).float() / 255
    mm = masks[idx]
    h, w = c.shape[-2:]
    scale = _lama_scale(w)
    sh = max(LAMA_QUANT, round(h * scale / LAMA_QUANT) * LAMA_QUANT)
    sw = max(32, round(w * scale / 32) * 32)
    cs = F.interpolate(c, (sh, sw), mode='bilinear', align_corners=False)
    ms = (F.interpolate(mm, (sh, sw), mode='bilinear', align_corners=False) > 0.05).float()
    out = F.interpolate(model(cs, ms).clamp(0, 1), (h, w), mode='bilinear', align_corners=False)
    soft = F.avg_pool2d(F.max_pool2d(mm, 7, 1, 3), 7, 1, 3)   # mép mềm ra phía ngoài, bên trong = 1
    frames[idx, cy0:cy1] = ((c * (1 - soft) + out * soft) * 255).round().byte().permute(0, 2, 3, 1)
    return len(idx)


def _mask_from_boxes(boxes, W, H, cy0, cy1, device):
    import torch
    m = torch.zeros((1, 1, cy1 - cy0, W), device=device)
    for b in boxes:
        x0, y0, x1, y1 = _pad_box(b, W, H)
        y0, y1 = max(y0, cy0) - cy0, min(y1, cy1) - cy0
        if y1 > y0 and x1 > x0:
            m[:, :, y0:y1, x0:x1] = 1
    return m


def _erase_batch(frames, per_frame_boxes, n_bands, W, H, root_dir):
    """frames: B,H,W,3 (GPU). per_frame_boxes[i] = [(chỉ số dải, hộp)]. Mỗi dải 1 lượt LaMa riêng
    (dải trên + dải dưới không bị gộp thành 1 vùng cao gần hết khung)."""
    import torch
    for k in range(n_bands):
        boxes_k = [[b for kk, b in fb if kk == k] for fb in per_frame_boxes]
        flat = [b for bs in boxes_k for b in bs]
        if not flat:
            continue
        cy0, cy1 = _crop_rows(flat, W, H)
        masks = torch.cat([_mask_from_boxes(bs, W, H, cy0, cy1, frames.device) for bs in boxes_k])
        _inpaint(frames, masks, cy0, cy1, root_dir)


def erase_image(img, root_dir, bands=None):
    """Xoá phụ đề trên 1 ảnh PIL (xem trước). bands: dải tìm được từ cả video; None -> tự tìm trên chính ảnh.
    Trả về (ảnh kết quả, hộp đã xoá, dòng chữ khác được giữ nguyên)."""
    import torch
    from PIL import Image
    rgb = np.array(img.convert('RGB'))
    H, W = rgb.shape[:2]
    pct = _pct_bands(bands if bands is not None else bands_from_image(rgb))
    found = boxes_in_bands(rgb, pct)
    erased = [b for _, b in found]
    bands_px = [_rows(H, t, s)[:2] for t, s in pct]
    kept = [b for b in detect_boxes(rgb) if _sub_like(b, W, H)
            and not any(_touches_band(b, W, H, y0, y1) for y0, y1 in bands_px)]
    if not found:
        return img.convert('RGB'), [], kept
    with torch.inference_mode(), _cudnn_benchmark():
        t = torch.from_numpy(rgb).unsqueeze(0).to(_device())
        _erase_batch(t, [found], len(pct), W, H, root_dir)
        return Image.fromarray(t[0].cpu().numpy()), erased, kept


# ---------------------------------------------------------------------------
# Video
# ---------------------------------------------------------------------------
def probe(src):
    r = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                        'stream=width,height,r_frame_rate,nb_frames:stream_side_data=rotation:format=duration',
                        '-of', 'json', src], capture_output=True, text=True, encoding='utf-8')
    d = json.loads(r.stdout or '{}')
    st = (d.get('streams') or [{}])[0]
    w, h = int(st.get('width') or 0), int(st.get('height') or 0)
    rot = 0
    for sd in st.get('side_data_list') or []:
        rot = int(sd.get('rotation') or 0)
    if abs(rot) % 180 == 90:  # ffmpeg tự xoay khi giải mã -> khung thực tế bị đổi chiều
        w, h = h, w
    fps = st.get('r_frame_rate') or '30/1'
    num, den = (fps.split('/') + ['1'])[:2]
    fps_f = float(num) / float(den or 1) or 30.0
    dur = float((d.get('format') or {}).get('duration') or 0)
    total = int(st.get('nb_frames') or 0) or int(dur * fps_f)
    return w, h, fps, fps_f, dur, max(total, 1)


def grab_frame(src, t: float):
    """1 khung hình RGB (numpy) tại giây t."""
    W, H, *_ = probe(src)
    r = subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-ss', f'{max(0.0, t):.3f}', '-i', src,
                        '-frames:v', '1', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], capture_output=True)
    if len(r.stdout) < W * H * 3:
        raise RuntimeError('Không lấy được khung hình từ video.')
    return np.frombuffer(r.stdout[:W * H * 3], np.uint8).reshape(H, W, 3)


_encoder = None


def _video_encoder():
    """h264_nvenc nếu dùng được, không thì libx264."""
    global _encoder
    if _encoder is None:
        r = subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'color=s=256x256:d=0.1',
                            '-c:v', 'h264_nvenc', '-f', 'null', '-'], capture_output=True)
        _encoder = (['-c:v', 'h264_nvenc', '-cq', '20', '-preset', 'p4'] if r.returncode == 0
                    else ['-c:v', 'libx264', '-crf', '18', '-preset', 'veryfast'])
    return _encoder


class EraseStopped(Exception):
    """Người dùng bấm Dừng giữa chừng: file dở dang đã bị xoá."""


def erase_video(src: str, dst: str, bands, root_dir, stop=None):
    """Xoá chữ thuộc các dải phụ đề `bands` (từ find_sub_bands) trên cả video. Generator: yield tỉ lệ xong 0..1.
    `stop`: hàm trả True khi cần dừng ngay (kiểm tra sau mỗi lượt GPU) -> raise EraseStopped."""
    import torch
    W, H, fps, fps_f, _, total = probe(src)
    if not W or not H:
        raise RuntimeError('Không đọc được kích thước video.')
    pct = _pct_bands(bands)
    load_model(root_dir)
    _get_detector()
    stride = max(1, round(fps_f / DET_PER_SEC))
    dev = _device()
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(dst).with_name(Path(dst).stem + '.part.mp4')
    dec = subprocess.Popen(['ffmpeg', '-v', 'error', '-nostdin', '-i', src, '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
                           stdout=subprocess.PIPE)
    enc = subprocess.Popen(['ffmpeg', '-v', 'error', '-nostdin', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                            '-s', f'{W}x{H}', '-r', fps, '-i', '-', '-i', src, '-map', '0:v', '-map', '1:a?',
                            '-c:a', 'copy', *_video_encoder(), '-pix_fmt', 'yuv420p', str(tmp)],
                           stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    fsz, done = W * H * 3, 0

    def read_chunk():
        out = []
        for _ in range(stride):
            b = dec.stdout.read(fsz)
            if len(b) < fsz:
                break
            out.append(np.frombuffer(b, np.uint8).reshape(H, W, 3))
        return out

    def flush(batch):
        """batch: list of (frames_np_list, [(dải, hộp)]). Xử lý 1 lượt trên GPU rồi ghi ra."""
        frames = torch.from_numpy(np.stack([f for fr, _ in batch for f in fr])).to(dev)
        per_frame = [fb for fr, fb in batch for _ in fr]
        if any(per_frame):
            _erase_batch(frames, per_frame, len(pct), W, H, root_dir)
        enc.stdin.write(frames.cpu().numpy().tobytes())
        return frames.shape[0]

    try:
        with torch.inference_mode(), _cudnn_benchmark():
            cur = read_chunk()
            cur_boxes = boxes_in_bands(cur[0], pct) if cur else []
            batch, n_batch = [], 0
            while cur:
                nxt = read_chunk()
                nxt_boxes = boxes_in_bands(nxt[0], pct) if nxt else []
                # hợp 2 lần quét lân cận: không hở chữ lúc sub đổi câu
                batch.append((cur, cur_boxes + nxt_boxes))
                n_batch += len(cur)
                if n_batch >= BATCH_FRAMES or not nxt:
                    done += flush(batch)
                    batch, n_batch = [], 0
                    yield min(done / total, 1.0)
                    if stop and stop():
                        raise EraseStopped()
                cur, cur_boxes = nxt, nxt_boxes
        enc.stdin.close()
        err = enc.stderr.read().decode('utf-8', 'ignore')
        if enc.wait() != 0 or dec.wait() != 0 or not tmp.exists():
            raise RuntimeError(f'Xoá chữ lỗi (ffmpeg): {err[-600:]}')
        tmp.replace(dst)
    finally:
        for p in (dec, enc):
            if p.poll() is None:
                p.kill()
        tmp.unlink(missing_ok=True)


def cut_clip(src: str, start: float, duration: float, dst: str) -> str:
    """Cắt 1 đoạn ngắn (có tiếng) để xem thử."""
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-y', '-ss', f'{max(0.0, start):.3f}', '-i', src,
                        '-t', f'{duration:.3f}', '-c:v', 'libx264', '-crf', '18', '-preset', 'veryfast',
                        '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-movflags', '+faststart', dst],
                       capture_output=True, text=True, encoding='utf-8', errors='ignore')
    if r.returncode != 0 or not Path(dst).exists():
        raise RuntimeError(f'Cắt đoạn xem thử lỗi: {r.stderr[-400:]}')
    return dst
