"""
Web GUI đơn giản: dán link TikTok/Douyin (hoặc tải file mp4) -> dịch + lồng tiếng Việt.

Dùng lại pipeline TransCreate của pyVideoTrans với các mặc định cho Trung -> Việt:
  Faster-Whisper (nhận dạng) -> Google Translate (dịch) -> Edge-TTS (giọng vi-VN) -> ffmpeg (ghép).

Chạy:
    uv sync --extra webui
    uv run vi_dub_web.py
"""
import copy
import hashlib
import html
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import asdict
from pathlib import Path

# pipeline in chữ Trung/Việt ra console (chế độ cli); console Windows mặc định cp1252 sẽ văng UnicodeEncodeError
for _stream in (sys.stdout, sys.stderr):
    if _stream and hasattr(_stream, 'reconfigure'):
        _stream.reconfigure(encoding='utf-8', errors='replace')

import gradio as gr

import vi_dub_erase

os.environ['PYVIDEOTRANS_LANG'] = 'en'

from videotrans.configure import config

config.init_run()

from videotrans.configure.config import ROOT_DIR, TEMP_DIR, app_cfg, logger
from videotrans.configure.constants import FASTER_MODELS_DICT
from videotrans.util import tools
from videotrans.util.gpus import getset_gpu
from videotrans.util.help_role import role_menu

RECOGN_FASTER_WHISPER = 0
TRANSLATE_GOOGLE = 0
TTS_EDGE = 0

DOWNLOAD_DIR = Path(ROOT_DIR) / 'downloads'

SOURCE_LANGS = {'Tiếng Trung': 'zh-cn', 'Tiếng Anh': 'en', 'Tiếng Nhật': 'ja', 'Tiếng Hàn': 'ko'}
SUBTITLE_TYPES = {
    'Sub cứng': 1,
    'Sub mềm (bật/tắt được)': 2,
    'Không chèn sub': 0,
}
MODELS = [m for m in ('large-v3-turbo', 'large-v3', 'medium', 'small') if m in FASTER_MODELS_DICT]
BROWSERS = ['Không dùng', 'chrome', 'edge', 'firefox']
MOBILE_UA = ('Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 '
             '(KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1')


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _vi_roles() -> list:
    try:
        roles = [r for r in role_menu(TTS_EDGE, langcode='vi') if r and r != 'No']
    except Exception:
        roles = []
    return roles or ['HoaiMy(Female/VN)', 'NamMinh(Male/VN)']


def extract_url(text: str) -> str:
    """Lấy URL đầu tiên trong đoạn text (vd. đoạn 'chia sẻ' của Douyin/TikTok có kèm tiêu đề, hashtag)."""
    m = re.search(r'https?://[A-Za-z0-9\-._~:/?#\[\]@!$&\'()*+,;=%]+', text or '')
    if not m:
        raise RuntimeError('Không tìm thấy link (http/https) trong nội dung đã dán.')
    return m.group(0).rstrip('.,;:!?)\'"')


def _is_douyin(url: str) -> bool:
    return any(d in url for d in ('douyin.com', 'iesdouyin.com'))


def _douyin_item(page: str):
    m = re.search(r'window\._ROUTER_DATA\s*=\s*(\{.*?\})\s*</script>', page, re.S)
    if not m:
        return None
    for v in json.loads(m.group(1)).get('loaderData', {}).values():
        try:
            return v['videoInfoRes']['item_list'][0]
        except (KeyError, IndexError, TypeError):
            continue
    return None


def _load_cookies(cookie_file):
    """Đọc file cookies.txt (định dạng Netscape, xuất từ extension trình duyệt)."""
    if not cookie_file:
        return None
    from http.cookiejar import MozillaCookieJar
    jar = MozillaCookieJar(cookie_file)
    try:
        jar.load(ignore_discard=True, ignore_expires=True)
    except Exception as e:
        raise RuntimeError(f'File cookie không đúng định dạng cookies.txt (Netscape): {e}')
    return jar


def _douyin_info(url: str, cookie_file=None) -> dict:
    """Đọc thông tin video Douyin qua trang chia sẻ bản mobile (cookie là tuỳ chọn)."""
    import requests
    s = requests.Session()
    s.headers['User-Agent'] = MOBILE_UA
    jar = _load_cookies(cookie_file)
    if jar is not None:
        s.cookies.update(jar)
    r = s.get(url, allow_redirects=True, timeout=15)
    m = re.search(r'/(?:video|note|slides)/(\d+)', r.url) or re.search(r'(\d{15,})', r.url)
    if not m:
        raise RuntimeError(f'Không đọc được ID video từ link: {r.url}')
    vid = m.group(1)
    # Lượt đầu của mỗi phiên Douyin thường chỉ cấp cookie, không kèm dữ liệu -> gọi lại trong cùng phiên.
    item, page = None, r.text
    for attempt in range(4):
        if attempt:
            time.sleep(1)
            page = s.get(f'https://www.iesdouyin.com/share/video/{vid}/', timeout=15).text
        item = _douyin_item(page)
        if item:
            break
    if not item:
        if '_ROUTER_DATA' not in page:
            raise RuntimeError('Douyin đang chặn tạm thời (trang kiểm tra chống bot). '
                               'Đợi 10–30 phút rồi thử lại, hoặc dùng file cookie trong mục Nâng cao.')
        raise RuntimeError('Douyin không trả về dữ liệu: video đã bị xoá, riêng tư, hoặc cần đăng nhập.')
    video = item.get('video') or {}
    urls = (video.get('play_addr') or {}).get('url_list') or []
    if not urls:
        raise RuntimeError('Link này không phải video (có thể là bài đăng ảnh).')
    covers = (video.get('cover') or {}).get('url_list') or []
    return {
        'id': vid,
        'title': item.get('desc') or vid,
        'uploader': (item.get('author') or {}).get('nickname', ''),
        'duration': (video.get('duration') or 0) / 1000,
        'thumbnail': covers[0] if covers else None,
        'direct_url': urls[0].replace('playwm', 'play'),  # bản không watermark
        'extractor': 'douyin',
    }


def _ytdlp_url(url: str) -> str:
    """yt-dlp không nhận link rút gọn v.douyin.com / iesdouyin.com/share/... -> đổi về douyin.com/video/<id>."""
    if not _is_douyin(url):
        return url
    if 'v.douyin.com' in url:
        import requests
        try:  # chỉ đọc header Location của redirect, không tải trang
            url = requests.head(url, headers={'User-Agent': MOBILE_UA}, allow_redirects=False,
                                timeout=15).headers.get('Location', url)
        except Exception:
            return url
    m = re.search(r'/(?:share/)?video/(\d+)', url)
    return f'https://www.douyin.com/video/{m.group(1)}' if m else url


class _YtdlpQuiet:
    """Logger im lặng cho yt-dlp: lỗi vẫn ném ra exception (app tự xử lý / báo trên giao diện), chỉ không in rác ra
    terminal (vd. 'ERROR: Could not copy Chrome cookie database' khi app đang tự lùi về không dùng cookie)."""
    def debug(self, msg): pass
    def info(self, msg): pass
    def warning(self, msg): pass
    def error(self, msg): pass


def _ytdlp_opts(browser: str, cookie_file=None) -> dict:
    # fetch_pot=never: không gọi plugin lấy mã PO token của YouTube (bgutil chạy Deno, lần đầu hay quá 20 giây -> lỗi
    # TimeoutExpired). YouTube hiện vẫn cho tải 1080p không cần mã; cần thì _ytdlp_run tự thử lại có plugin.
    opts = {'noplaylist': True, 'quiet': True, 'no_warnings': True, 'no_color': True, 'logger': _YtdlpQuiet(),
            'extractor_args': {'youtube': {'fetch_pot': ['never']}}}
    if cookie_file:
        opts['cookiefile'] = cookie_file
    elif browser != 'Không dùng' and browser not in BROWSER_COOKIE_BROKEN:
        opts['cookiesfrombrowser'] = (browser,)
    return opts


_ANSI = re.compile(r'\x1b\[[0-9;]*m')
# trình duyệt đã đọc cookie lỗi trong phiên này (Chrome/Edge đang mở khoá file cookie, hoặc cookie bị mã hoá kiểu mới)
# -> các link sau bỏ qua luôn, khỏi mất thời gian thử lại
BROWSER_COOKIE_BROKEN: set = set()


def _clean_err(e) -> str:
    """Thông báo lỗi dễ đọc: bỏ mã màu ANSI và các tiền tố 'ERROR:' lặp lại của yt-dlp."""
    msg = _ANSI.sub('', str(e))
    return re.sub(r'(ERROR:\s*)+', '', msg).strip()


def _ytdlp_run(url: str, opts: dict, download: bool):
    """extract_info có lùi bước: đọc cookie trình duyệt lỗi -> tự thử lại không dùng cookie.
    Trả về (info, đường dẫn file đã tải hoặc None)."""
    import yt_dlp
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(_ytdlp_url(url), download=download)
            path = None
            if download:
                downloads = info.get('requested_downloads') or []
                path = downloads[0]['filepath'] if downloads else ydl.prepare_filename(info)
            return info, path
    except Exception as e:
        msg = _clean_err(e)
        browser = (opts.get('cookiesfrombrowser') or (None,))[0]
        if browser and re.search(r'cookie|dpapi|decrypt|keyring', msg, re.I):
            BROWSER_COOKIE_BROKEN.add(browser)
            logger.warning(f'[VieDub] Không đọc được cookie {browser}, thử lại không dùng cookie: {msg}')
            return _ytdlp_run(url, {k: v for k, v in opts.items() if k != 'cookiesfrombrowser'}, download)
        pot = ((opts.get('extractor_args') or {}).get('youtube') or {}).get('fetch_pot')
        if pot == ['never'] and re.search(r'po.?token|forbidden|403|sign in to confirm|not a bot', msg, re.I):
            # YouTube đòi mã PO token -> thử lại có plugin (lần đầu plugin khởi động Deno chậm: cho 2 lần)
            logger.warning(f'[VieDub] YouTube đòi PO token, thử lại có plugin: {msg}')
            retry = {**opts, 'extractor_args': {'youtube': {'fetch_pot': ['auto']}}}
            for k in range(2):
                try:
                    return _ytdlp_run(url, retry, download)
                except Exception as e2:
                    if k == 1 or 'Timeout' not in type(e2.__cause__ or e2).__name__ + str(e2):
                        raise
        raise RuntimeError(msg) from e


def _ytdlp_info(url: str, browser: str, cookie_file=None) -> dict:
    info, _ = _ytdlp_run(url, _ytdlp_opts(browser, cookie_file), download=False)
    return {
        'id': info.get('id'),
        'title': info.get('title') or info.get('description') or info.get('id'),
        'uploader': info.get('uploader') or info.get('channel') or '',
        'duration': info.get('duration') or 0,
        'thumbnail': info.get('thumbnail'),
        'extractor': info.get('extractor_key') or info.get('extractor'),
    }


def probe_link(text: str, browser: str = 'Không dùng', cookie_file=None) -> dict:
    """Chỉ kiểm tra link và lấy thông tin video, KHÔNG tải."""
    url = extract_url(text)
    errors = []
    if _is_douyin(url):
        try:
            return {'url': url, **_douyin_info(url, cookie_file)}
        except Exception as e:
            errors.append(f'Douyin: {e}')
    try:
        return {'url': url, **_ytdlp_info(url, browser, cookie_file)}
    except Exception as e:
        errors.append(f'yt-dlp: {_clean_err(e)}')
    raise RuntimeError('\n'.join(errors))


def _download_direct(direct_url: str, dest: Path) -> None:
    import requests
    with requests.get(direct_url, headers={'User-Agent': MOBILE_UA}, stream=True, timeout=30) as r:
        r.raise_for_status()
        tmp = dest.with_suffix('.part')
        with open(tmp, 'wb') as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
        tmp.replace(dest)


def download_video(text: str, browser: str = 'Không dùng', cookie_file=None) -> str:
    """Tải link về file mp4, trả về đường dẫn file. Douyin: trang chia sẻ mobile, còn lại: yt-dlp."""
    url = extract_url(text)
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    errors = []

    if _is_douyin(url):
        try:
            info = _douyin_info(url, cookie_file)
            dest = DOWNLOAD_DIR / f'douyin-{info["id"]}.mp4'
            if not dest.exists():
                _download_direct(info['direct_url'], dest)
            return dest.as_posix()
        except Exception as e:
            errors.append(f'Douyin: {e}')

    opts = _ytdlp_opts(browser, cookie_file) | {
        'format': 'bv*[height<=1080]+ba/b[height<=1080]/bv*+ba/b',
        'merge_output_format': 'mp4',
        'outtmpl': (DOWNLOAD_DIR / '%(extractor)s-%(id)s.%(ext)s').as_posix(),
    }
    # yt-dlp cần ffmpeg để ghép hình + tiếng (YouTube tách riêng 2 luồng). Thư mục ffmpeg/ của dự án có thể chỉ có
    # .gitignore -> chỉ dùng khi có ffmpeg.exe thật, không thì lấy ffmpeg trong PATH.
    local = Path(ROOT_DIR) / 'ffmpeg' / ('ffmpeg.exe' if os.name == 'nt' else 'ffmpeg')
    ffmpeg_bin = local.as_posix() if local.exists() else shutil.which('ffmpeg')
    if ffmpeg_bin:
        opts['ffmpeg_location'] = ffmpeg_bin
    try:
        _, path = _ytdlp_run(url, opts, download=True)
    except Exception as e:
        errors.append(f'yt-dlp: {_clean_err(e)}')
        raise RuntimeError('\n'.join(errors))
    if not Path(path).exists():
        raise RuntimeError(f'Không tìm thấy file sau khi tải: {path}')
    return Path(path).as_posix()


def extract_urls(text: str) -> list:
    """Lấy tất cả URL (không trùng, giữ thứ tự) trong đoạn text."""
    urls = []
    for u in re.findall(r'https?://[A-Za-z0-9\-._~:/?#\[\]@!$&\'()*+,;=%]+', text or ''):
        u = u.rstrip('.,;:!?)\'"')
        if u not in urls:
            urls.append(u)
    return urls


def _fmt_duration(sec: float) -> str:
    sec = int(sec or 0)
    return f'{sec // 60}:{sec % 60:02d}' if sec else '?'


def _short_title(title: str, n: int = 90) -> str:
    t = re.sub(r'#\S+', '', title or '').strip() or (title or '').strip()
    return t if len(t) <= n else t[:n - 1] + '…'


# ---------------------------------------------------------------------------
# Ô nhập link (mỗi link một ô)
# ---------------------------------------------------------------------------
MAX_LINKS = 30


def _pad(vals: list) -> list:
    return list(vals[:MAX_LINKS]) + [''] * (MAX_LINKS - len(vals))


def _row_classes(i: int, n: int) -> list:
    # Ẩn/hiện hàng bằng class CSS: với visible=False, Gradio 6 chỉ dựng hàng ở lần cập nhật đầu mà chưa
    # hiện ra (giao diện trễ 1 nhịp), còn visible='hidden' thì không áp dụng cho Row lúc tải trang.
    return ['lrow'] if i < n else ['lrow', 'lhide']


def _rows_update(n: int) -> list:
    import gradio as gr
    return [gr.update(elem_classes=_row_classes(i, n)) for i in range(MAX_LINKS)]


def add_link_row(n):
    n = min(int(n) + 1, MAX_LINKS)
    return [n, *_rows_update(n)]


def _box_updates(old: list, new: list) -> list:
    """Chỉ gửi giá trị cho ô thật sự đổi: gửi lại cả 30 ô sẽ kích hoạt 30 sự kiện .change dây chuyền."""
    import gradio as gr
    return [v if v != o else gr.update() for o, v in zip(old, _pad(new))]


def remove_link_row(i):
    def _remove(n, *vals):
        rest = [v for j, v in enumerate(vals[:int(n)]) if j != i] or ['']
        return [len(rest), *_box_updates(list(vals), rest), *_rows_update(len(rest))]
    return _remove


def on_link_input(i):
    """Dán nhiều link vào 1 ô -> tách ra nhiều ô; gõ vào ô cuối -> tự mở thêm 1 ô trống."""
    def _input(n, *vals):
        import gradio as gr
        n = int(n)
        vals = list(vals)
        urls = extract_urls(vals[i])
        if len(urls) >= 2:
            new = [v for v in vals[:i] if v.strip()] + urls + [v for v in vals[i + 1:n] if v.strip()]
            new = new[:MAX_LINKS]
            if len(new) < MAX_LINKS:
                new.append('')
            return [len(new), *_box_updates(vals, new), *_rows_update(len(new))]
        if i == n - 1 and vals[i].strip() and n < MAX_LINKS:
            return [n + 1, *[gr.update()] * MAX_LINKS, *_rows_update(n + 1)]
        return [gr.update()] * (1 + 2 * MAX_LINKS)
    return _input


def _links_from_boxes(n, vals) -> list:
    urls = []
    for v in vals[:int(n)]:
        for u in extract_urls(v):
            if u not in urls:
                urls.append(u)
    return urls


# ---------------------------------------------------------------------------
# File từ máy (cộng dồn qua nhiều lần thả)
# ---------------------------------------------------------------------------
def _file_items(files) -> list:
    items = []
    for p in files or []:
        p = Path(p)
        if not p.exists():
            continue
        items.append({
            'key': f'F:{p.as_posix()}', 'kind': 'file', 'src': p.as_posix(),
            'title': p.name, 'meta': f'File từ máy · {p.stat().st_size / 1048576:.1f} MB',
            'thumbnail': None, 'status': 'ok', 'error': '',
        })
    return items


def add_files(files, dropped, link_items, selected):
    files = list(files or [])
    seen = {(Path(f).name, Path(f).stat().st_size) for f in files if Path(f).exists()}
    added = []
    for f in dropped or []:
        key = (Path(f).name, Path(f).stat().st_size)
        if key not in seen:
            seen.add(key)
            files.append(Path(f).as_posix())
            added.append(Path(f).as_posix())
    items = _all_items(link_items, files)
    new_keys = [it['key'] for it in _file_items(added)]   # chỉ tick file vừa thả
    sel = list(dict.fromkeys([*(selected or []), *new_keys]))
    return files, None, _selector_update(items, sel)


def clear_files(link_items, selected):
    items = _all_items(link_items, [])
    sel = [k for k in (selected or []) if not k.startswith('F:')]
    return [], _selector_update(items, sel)


def remove_file(path):
    """Nút ✕ của 1 file (ở tab File từ máy hoặc danh sách bước ②)."""
    def _remove(files, link_items, selected):
        files = [f for f in (files or []) if f != path]
        return files, _selector_update(_all_items(link_items, files), selected)
    return _remove


def remove_link_item(url):
    """Nút ✕ của 1 link ở bước ②: bỏ khỏi danh sách và cả ô nhập link chứa nó (để kiểm tra lại không hiện lại)."""
    def _remove(link_items, files, selected, n, *vals):
        link_items = [it for it in (link_items or []) if it['src'] != url]
        rest = [v for v in vals[:int(n)] if url not in extract_urls(v)] or ['']
        return [link_items, _selector_update(_all_items(link_items, files), selected), len(rest),
                *_box_updates(list(vals), rest), *_rows_update(len(rest))]
    return _remove


# ---------------------------------------------------------------------------
# Tên video (đặt ngay ở danh sách, dùng làm tên file xuất)
# ---------------------------------------------------------------------------
NAME_MAX = 120
_WIN_RESERVED = {'CON', 'PRN', 'AUX', 'NUL', *[f'COM{i}' for i in range(1, 10)], *[f'LPT{i}' for i in range(1, 10)]}


def safe_name(name, fallback: str = 'video') -> str:
    """Tên dùng được làm tên file/thư mục trên Windows: bỏ ký tự cấm, khoảng trắng thừa, dấu chấm cuối."""
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]', ' ', str(name or ''))
    s = re.sub(r'\s+', ' ', s).strip().rstrip('. ')
    s = s[:NAME_MAX].rstrip('. ')
    if s and s.split('.')[0].upper() in _WIN_RESERVED:
        s = f'{s}_'
    return s or fallback


def _default_name(it: dict) -> str:
    if it['kind'] == 'file':
        return safe_name(Path(it['src']).stem)
    t = re.sub(r'#\S+', '', it.get('title') or '').strip() or (it.get('title') or '')
    return safe_name(t[:60], fallback=f'video-{it.get("vid") or "link"}')


def _item_name(names, it: dict) -> str:
    """Tên người dùng đã đặt; chưa đặt (hoặc xoá trắng) thì lấy tên mặc định."""
    return safe_name((names or {}).get(it['key']) or '', fallback=_default_name(it))


# ---------------------------------------------------------------------------
# Danh sách video ở bước ②
# ---------------------------------------------------------------------------
def _all_items(link_items, files) -> list:
    return list(link_items or []) + _file_items(files)



def _selector_update(items, selected=None) -> list:
    """Danh sách key đang chọn (chỉ giữ video 'Sẵn sàng'); selected=None -> chọn tất cả."""
    keys = [it['key'] for it in items if it['status'] == 'ok']
    return keys if selected is None else [k for k in dict.fromkeys(selected) if k in keys]


BADGES = {
    'pending': ('b-mute', 'Chờ kiểm tra'),
    'checking': ('b-info', 'Đang kiểm tra…'),
    'ok': ('b-ok', 'Sẵn sàng'),
    'error': ('b-err', 'Không dùng được'),
}


EMPTY_ITEMS = ('<div class="empty"><div class="empty-ico">🎬</div>'
               'Chưa có video. Dán link ở mục <b>1. Thêm video</b> rồi bấm <b>Kiểm tra tất cả</b>, '
               'hoặc thả file vào tab <b>File từ máy</b>.</div>')


def _check_summary(items) -> str:
    links = [it for it in items if it['kind'] == 'link']
    if not links:
        return ''
    ok = sum(it['status'] == 'ok' for it in links)
    bad = sum(it['status'] == 'error' for it in links)
    left = len(links) - ok - bad
    s = f'**{len(links)} link** · ✅ {ok} dùng được · ❌ {bad} lỗi'
    return s + (f' · ⏳ còn {left}' if left else '')


def check_links(plat, browser, cookie_file, files, selected, n, *vals):
    """Kiểm tra lần lượt từng link (chỉ đọc thông tin, không tải). `plat`: nền tảng đang chọn ('auto' = tự nhận)."""
    urls = _links_from_boxes(n, vals)
    if not urls:
        items = _file_items(files)
        yield [], _selector_update(items, selected), '⚠️ Chưa có link nào.'
        return
    link_items = [{'key': f'L:{u}', 'kind': 'link', 'src': u, 'title': u, 'meta': '', 'thumbnail': None,
                   'status': 'pending', 'error': ''} for u in urls]
    prev_douyin = False
    for it in link_items:
        it['status'] = 'checking'
        items = _all_items(link_items, files)
        # bản sao: danh sách ở bước ② là gr.render theo State, chỉ vẽ lại khi giá trị State thực sự đổi
        yield copy.deepcopy(link_items), gr.update(), _check_summary(items)   # gr.update(): State giữ nguyên
        pk = it['platform'] = _platform_of(it['src'])
        if plat != 'auto' and pk != plat:
            it.update(status='error', error=f'Link này của {_plat_label(pk)}, nhưng đang chọn nền tảng '
                                            f'{_plat_label(plat)}. Chọn "Tự nhận" hoặc đúng nền tảng rồi kiểm tra lại.')
            continue
        if PLAT.get(pk, PLAT['auto'])[3] == 'soon':
            it.update(status='error', error=f'{_plat_label(pk)}: chưa hỗ trợ tải.')
            continue
        if prev_douyin and _is_douyin(it['src']):
            time.sleep(1.5)  # giãn cách để Douyin không chặn IP
        prev_douyin = _is_douyin(it['src'])
        try:
            info = probe_link(it['src'], browser, cookie_file)
            it.update(status='ok', title=info['title'], thumbnail=info.get('thumbnail'),
                      direct_url=info.get('direct_url'), vid=info.get('id'),
                      meta=' · '.join(x for x in (info.get('uploader'), _fmt_duration(info['duration']),
                                                  _plat_label(pk) if pk != 'other' else str(info.get('extractor') or ''))
                                      if x))
            if info['duration'] and info['duration'] > 600:
                it['meta'] += ' · ⏱️ dài, xử lý lâu'
        except Exception as e:
            it.update(status='error', error=_clean_err(e))
    items = _all_items(link_items, files)
    if browser in BROWSER_COOKIE_BROKEN and not cookie_file:
        gr.Warning(f'Không đọc được cookie của {browser} (trình duyệt đang mở hoặc cookie bị mã hoá), app đã tự bỏ qua '
                   f'cookie. Link cần đăng nhập thì dùng file cookies.txt ở Cài đặt → Nâng cao, hoặc chọn "Không dùng".',
                   duration=15)
    yield copy.deepcopy(link_items), gr.update(), _check_summary(items)


def select_after_check(link_items, files, selected):
    """Sau khi kiểm tra link: giữ lựa chọn hiện tại của file (đọc lúc này, không phải lúc bắt đầu kiểm tra)
    và tick mọi link dùng được."""
    keep = [k for k in (selected or []) if k.startswith('F:')]
    ok_links = [it['key'] for it in (link_items or []) if it['status'] == 'ok']
    return _selector_update(_all_items(link_items, files), keep + ok_links)


# ---------------------------------------------------------------------------
# Cookie (lưu lại trên máy để lần sau khỏi chọn)
# ---------------------------------------------------------------------------
COOKIE_PATH = DOWNLOAD_DIR / 'cookies.txt'


def _cookie_status() -> str:
    if not COOKIE_PATH.exists():
        return '<span class="hint">Chưa có cookie. Chỉ cần khi Douyin/TikTok chặn.</span>'
    t = time.strftime('%d/%m/%Y %H:%M', time.localtime(COOKIE_PATH.stat().st_mtime))
    return f'<span class="chip ok">🍪 Đã lưu cookie · {t}</span>'


def save_cookie(path):
    import gradio as gr
    try:
        _load_cookies(path)
    except Exception as e:
        gr.Warning(str(e))
        return gr.update(), _cookie_status()
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, COOKIE_PATH)
    gr.Info('Đã lưu cookie, lần sau không cần chọn lại.')
    return COOKIE_PATH.as_posix(), _cookie_status()


def delete_cookie():
    COOKIE_PATH.unlink(missing_ok=True)
    return None, _cookie_status()


def init_cookie():
    return (COOKIE_PATH.as_posix() if COOKIE_PATH.exists() else None), _cookie_status()


# ---------------------------------------------------------------------------
# Che phụ đề gốc (chữ Trung in sẵn trong video) + vị trí sub tiếng Việt
# ---------------------------------------------------------------------------
COVER_MODES = ['Không che', 'Làm mờ', 'Che nền tối']
OLD_ERASE_MODE = 'Xoá chữ (AI)'  # trước đây là 1 lựa chọn của "Cách che", nay là công tắc riêng (ai_erase)
COVER_DIR = DOWNLOAD_DIR / '_covered'
PREVIEW_DIR = Path(TEMP_DIR) / 'viedub_preview'
ASS_JSON = Path(ROOT_DIR) / 'videotrans' / 'ass.json'
ASS_PLAYRES_Y = 288  # ffmpeg chuyển SRT -> ASS với PlayResY mặc định 288


def _video_size(path: str):
    r = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                        'stream=width,height:format=duration', '-of', 'json', path],
                       capture_output=True, text=True, encoding='utf-8')
    d = json.loads(r.stdout or '{}')
    st = (d.get('streams') or [{}])[0]
    return int(st.get('width') or 0), int(st.get('height') or 0), float((d.get('format') or {}).get('duration') or 0)


def _band(height: int, top: float, size: float):
    y = int(height * top / 100) // 2 * 2
    h = max(2, int(height * size / 100) // 2 * 2)
    return y, min(h, height - y)


def _sub_place(opts: dict, erase_band) -> dict:
    """Chỉ xoá AI (không che): sub Việt đặt vào đúng dải phụ đề gốc mà AI tìm được, không theo thanh 'Vùng sub gốc'."""
    if opts.get('ai_erase') and opts['cover_mode'] == 'Không che' and erase_band:
        opts = dict(opts, cover_top=erase_band['top'], cover_size=erase_band['size'])
    return opts


def _region_active(mode, erase) -> bool:
    """Vùng sub gốc có được xử lý không (xoá bằng AI hoặc che)."""
    return bool(erase) or mode != 'Không che'


def _cache_out(tag: str, src: str) -> Path:
    """File kết quả trong cache: giữ nguyên tên file gốc để thư mục output không đổi."""
    key = hashlib.md5(f'{tag}|{Path(src).resolve()}|{Path(src).stat().st_mtime}'.encode()).hexdigest()[:10]
    return COVER_DIR / key / Path(src).name


def cover_original_subs(src: str, mode: str, top: float, size: float, erase: bool = False, bands=None):
    """Xử lý sub gốc: (1) xoá phụ đề bằng AI nếu bật — tự tìm vị trí phụ đề (bands), KHÔNG dùng vùng top/size —
    rồi (2) che dải [top, top+size]% theo `mode` nếu khác 'Không che'.
    Generator: yield (tỉ lệ xong, None), lần cuối (1.0, file mới). Mỗi bước có cache riêng theo thông số."""
    path = src
    if erase:
        if bands is None:
            bands, _ = vi_dub_erase.find_sub_bands(src)
        if bands:
            out = _cache_out(f'erase-v2|{[(b["top"], b["size"]) for b in bands]}', src)
            if not (out.exists() and out.stat().st_size > 0):
                share = 0.95 if mode != 'Không che' else 1.0  # phần tiến độ dành cho bước xoá chữ
                try:
                    for frac in vi_dub_erase.erase_video(src, out.as_posix(), bands, ROOT_DIR, stop=STOP.is_set):
                        yield frac * share, None
                finally:
                    vi_dub_erase.release_model()  # trả VRAM cho Whisper / TTS ở các bước sau
            path = out.as_posix()
    if mode == 'Không che':
        yield 1.0, path
        return
    yield from _cover_band(path, mode, top, size)


def _cover_band(src: str, mode: str, top: float, size: float):
    """Làm mờ / phủ đen dải sub gốc bằng ffmpeg."""
    w, h, _ = _video_size(src)
    if not w or not h:
        raise RuntimeError('Không đọc được kích thước video để che sub gốc.')
    y, bh = _band(h, top, size)
    # 'v3': đổi khi đổi bộ lọc để không dùng lại bản che cũ trong cache
    out = _cache_out(f'v3|{mode}|{top}|{size}', src)
    if out.exists() and out.stat().st_size > 0:
        yield 1.0, out.as_posix()
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    if mode == 'Làm mờ':
        # boxblur yêu cầu bán kính < 1/2 cạnh nhỏ của từng kênh (kênh màu yuv420 chỉ cao bh/2)
        lr, cr = max(1, min(bh // 2 - 1, 40)), max(1, min(bh // 4 - 1, 20))
        fc = (f'[0:v]split[m][c];[c]crop={w}:{bh}:0:{y},'
              f'boxblur=luma_radius={lr}:luma_power=3:chroma_radius={cr}:chroma_power=3,'
              f'drawbox=x=0:y=0:w=iw:h=ih:color=black@0.25:t=fill[b];[m][b]overlay=0:{y}[v]')
    else:
        fc = f'[0:v]drawbox=x=0:y={y}:w=iw:h={bh}:color=black:t=fill[v]'  # đen đặc, không lộ chữ gốc
    base = ['ffmpeg', '-y', '-nostdin', '-i', src, '-filter_complex', fc, '-map', '[v]', '-map', '0:a?',
            '-c:a', 'copy', '-pix_fmt', 'yuv420p']
    last_err = ''
    for enc in (['-c:v', 'h264_nvenc', '-cq', '21', '-preset', 'p4'],
                ['-c:v', 'libx264', '-crf', '19', '-preset', 'veryfast']):
        r = subprocess.run(base + enc + [out.as_posix()], capture_output=True, text=True,
                           encoding='utf-8', errors='ignore')
        if r.returncode == 0 and out.exists() and out.stat().st_size > 0:
            yield 1.0, out.as_posix()
            return
        last_err = r.stderr[-800:]
    out.unlink(missing_ok=True)
    raise RuntimeError(f'Che sub gốc lỗi (ffmpeg): {last_err}')


def _ass_style(opts: dict) -> dict:
    """Style sub cứng tiếng Việt. Vị trí: đặt vào vùng sub gốc (nếu đang xoá/che nó) hoặc theo thanh 'Vị trí sub'."""
    if _region_active(opts['cover_mode'], opts.get('ai_erase')) and opts['sub_follow']:
        margin_v = round((1 - (opts['cover_top'] + opts['cover_size']) / 100) * ASS_PLAYRES_Y) + 3
    else:
        margin_v = round(opts['sub_pos'] / 100 * ASS_PLAYRES_Y)
    box = opts['sub_box']
    return {
        'Name': 'Default', 'Fontname': 'Arial', 'Fontsize': opts['sub_size'],
        'PrimaryColour': '&H00FFFFFF&', 'SecondaryColour': '&H00FFFFFF&',
        'OutlineColour': '&H80000000&' if box else '&H00000000&',
        'BackColour': '&H80000000&',
        'Bold': 1, 'Italic': 0, 'Underline': 0, 'StrikeOut': 0, 'ScaleX': 100, 'ScaleY': 100,
        'Spacing': 0, 'Angle': 0, 'BorderStyle': 3 if box else 1, 'Outline': 3 if box else 1.2,
        'Shadow': 0 if box else 0.6, 'Alignment': 2, 'MarginL': 12, 'MarginR': 12,
        'MarginV': max(0, margin_v), 'Encoding': 1,
        'Bottom_Fontname': 'Arial', 'Bottom_Fontsize': opts['sub_size'] - 2,
        'Bottom_PrimaryColour': '&H00FFFFFF&',
    }


def _font(px: int):
    from PIL import ImageFont
    for name in ('arialbd.ttf', 'arial.ttf', 'segoeuib.ttf', 'DejaVuSans-Bold.ttf'):
        try:
            return ImageFont.truetype(name, px)
        except OSError:
            continue
    return ImageFont.load_default()


def _local_path(item: dict):
    """File video đã có trên máy của 1 mục (file từ máy, hoặc link Douyin đã tải), không thì None."""
    if item['kind'] == 'file':
        return item['src']
    if item.get('vid'):
        p = DOWNLOAD_DIR / f'douyin-{item["vid"]}.mp4'
        if p.exists():
            return p.as_posix()
    return None


def _fetch(item: dict, browser, cookie_file) -> str:
    """Lấy file cục bộ cho 1 mục: file từ máy giữ nguyên, link thì tải (dùng lại link trực tiếp nếu đã có)."""
    if item['kind'] == 'file':
        return item['src']
    if item.get('direct_url') and item.get('vid'):
        dest = DOWNLOAD_DIR / f'douyin-{item["vid"]}.mp4'
        if dest.exists():
            return dest.as_posix()
        try:
            DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
            _download_direct(item['direct_url'], dest)
            return dest.as_posix()
        except Exception:
            pass  # link trực tiếp hết hạn -> tải lại từ đầu
    return download_video(item['src'], browser, cookie_file)


def _is_hard_sub(subtitle_name) -> bool:
    """Chỉ sub cứng mới dùng style/vị trí của app; sub mềm do trình phát tự vẽ."""
    return SUBTITLE_TYPES.get(subtitle_name, 1) == 1


def _audio_seconds(path) -> float:
    r = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', str(path)],
                       capture_output=True, text=True, encoding='utf-8')
    try:
        return float(r.stdout.strip() or 0)
    except ValueError:
        return 0.0


def _ms(ms) -> str:
    """mm:ss.c cho dòng thời gian / bảng."""
    s = max(0, int(ms)) / 1000
    return f'{int(s // 60)}:{s % 60:04.1f}'


# ---------------------------------------------------------------------------
# Xem trước sub Việt trên 1 khung hình (dùng trong trình sửa)
# ---------------------------------------------------------------------------
def _draw_sub_preview(img, mode, top, size, opts, subtitle_name, text):
    """Vẽ vùng che (nếu có) + 1 dòng sub tiếng Việt thật lên ảnh. Trả về (ảnh, ghi chú)."""
    from PIL import Image, ImageDraw, ImageFilter
    note = ''
    scale = 1.0
    if img.width > 720:
        scale = 720 / img.width
        img = img.resize((720, round(img.height * scale)))
    W, H = img.size
    y, bh = _band(H, top, size)
    if mode == 'Làm mờ':
        band = img.crop((0, y, W, y + bh)).filter(ImageFilter.GaussianBlur(max(2, bh // 6)))
        img.paste(band, (0, y))
        img.paste(img.crop((0, y, W, y + bh)).point(lambda p: int(p * 0.75)), (0, y))
    d = ImageDraw.Draw(img)
    if mode == 'Che nền tối':
        d.rectangle((0, y, W, y + bh), fill=(0, 0, 0))
    if mode != 'Không che':
        d.rectangle((1, y, W - 2, y + bh), outline=(167, 139, 250), width=2)
    if not _is_hard_sub(subtitle_name):
        note = (' · không chèn sub tiếng Việt' if SUBTITLE_TYPES.get(subtitle_name) == 0
                else ' · sub mềm: chữ và vị trí do trình phát video quyết định')
        return img, note
    style = _ass_style(opts)
    # libass: cỡ chữ theo chiều cao (PlayResY=288), tự xuống dòng trong khoảng giữa lề trái/phải
    px = max(10, round(opts['sub_size'] / ASS_PLAYRES_Y * H))
    font = _font(px)
    max_w = W - 2 * style['MarginL'] / 384 * W
    lines, cur = [], ''
    for word in (text or 'Đây là phụ đề tiếng Việt mẫu').split():
        test = f'{cur} {word}'.strip()
        if cur and d.textlength(test, font=font) > max_w:
            lines.append(cur)
            cur = word
        else:
            cur = test
    lines.append(cur)
    line_h = px * 1.2
    bottom = H - style['MarginV'] / ASS_PLAYRES_Y * H
    if opts['sub_box']:  # nền chữ trong video thật là đen trong suốt ~50%
        shade = Image.new('RGBA', img.size, (0, 0, 0, 0))
        sd = ImageDraw.Draw(shade)
        for k, line in enumerate(reversed(lines)):
            tw = d.textlength(line, font=font)
            tx, ty = (W - tw) / 2, bottom - line_h * (k + 1)
            sd.rectangle((tx - px * 0.2, ty, tx + tw + px * 0.2, ty + line_h), fill=(0, 0, 0, 128))
        img.paste(Image.alpha_composite(img.convert('RGBA'), shade).convert('RGB'))
        d = ImageDraw.Draw(img)
    for k, line in enumerate(reversed(lines)):
        tw = d.textlength(line, font=font)
        tx, ty = (W - tw) / 2, bottom - line_h * (k + 1)
        if opts['sub_box']:
            d.text((tx, ty + px * 0.05), line, font=font, fill=(255, 255, 255))
        else:
            d.text((tx, ty + px * 0.05), line, font=font, fill=(255, 255, 255), stroke_width=max(1, px // 14),
                   stroke_fill=(0, 0, 0))
    return img, note


# ---------------------------------------------------------------------------
# Dự án: 1 video đã lồng tiếng xong, đang chờ sửa chữ / xuất
# ---------------------------------------------------------------------------
PROJECTS: dict = {}   # pid -> {'trk','title','src','opts','items','edits','dur','erase_band','result','exported'}
STOP = threading.Event()
CURRENT = {'uuid': None}   # uuid của video đang lồng tiếng, để nút Dừng báo cho pipeline thoát sớm
WORK_DIR = (Path(ROOT_DIR) / 'output' / '_work').as_posix()   # thư mục làm việc của pyVideoTrans (srt, cache)


class Stopped(Exception):
    """Người dùng bấm Dừng: video đang dở bị huỷ, file tạm của nó đã xoá."""

STAGES = [
    ('Chuẩn bị video/âm thanh', 'prepare'),
    ('Nhận dạng giọng nói', 'recogn'),
    ('Tách người nói', 'diariz'),
    ('Dịch phụ đề sang tiếng Việt', 'trans'),
    ('Tạo giọng lồng tiếng', 'dubbing'),
]


def build_project(file_path: str, opts: dict, title: str, name: str = ''):
    """Pha 1: chạy pyVideoTrans tới hết bước lồng tiếng rồi dừng (chưa ghép video).
    Yield (số bước, tên bước); lần cuối ('done', pid)."""
    app_cfg.exit_soft = False
    app_cfg.current_status = 'ing'
    app_cfg.exec_mode = 'cli'
    getset_gpu()

    file_obj = tools.format_video(file_path)
    nospace = file_obj['basename'].replace(' ', '-').replace('.', '-')
    pid = file_obj['uuid']
    if pid in PROJECTS:  # chạy lại cùng video: bỏ dự án cũ
        drop_project(pid)
    cache_folder = f'{TEMP_DIR}/{pid}'
    app_cfg.rm_uuid(pid)
    target_dir = f'{WORK_DIR}/{nospace}'
    file_obj['target_dir'] = target_dir
    Path(cache_folder).mkdir(parents=True, exist_ok=True)
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    for f in Path(target_dir).rglob('*'):
        if f.is_file() and f.suffix.lower() in ('.mp4', '.mkv'):
            f.unlink(missing_ok=True)

    params = {'name': file_path, 'cache_folder': cache_folder, **asdict(file_obj)}
    params.update({
        'source_language_code': opts['source'],
        'target_language_code': 'vi',
        'detect_language': opts['source'],
        'recogn_type': RECOGN_FASTER_WHISPER,
        'model_name': opts['model'],
        'is_cuda': opts['cuda'],
        'remove_noise': False,
        'enable_diariz': False,
        'nums_diariz': -1,
        'rephrase': 0,
        'fix_punc': 0,
        'translate_type': TRANSLATE_GOOGLE,
        'tts_type': TTS_EDGE,
        'voice_role': opts['voice'],
        'voice_rate': '+0%',
        'volume': '+0%',
        'pitch': '+0Hz',
        'voice_autorate': opts['voice_autorate'],
        'video_autorate': opts['video_autorate'],
        'align_sub_audio': True,
        'is_separate': opts['keep_bgm'],
        'embed_bgm': opts['keep_bgm'],
        'loop_backaudio': 0,
        'backaudio_volume': 0.8,
        'background_music': '',
        'recogn2pass': False,
        'subtitle_type': opts['subtitle'],
        'clear_cache': True,
    })
    from videotrans.task.trans_create import TransCreate
    from videotrans.task.taskcfg import TaskCfgVTT
    trk = TransCreate(cfg=TaskCfgVTT(**params))
    CURRENT['uuid'] = trk.uuid
    try:
        for i, (stage_title, method) in enumerate(STAGES, 1):
            if STOP.is_set():
                raise Stopped()
            yield i, stage_title
            getattr(trk, method)()   # khi Dừng: các bước tự thoát sớm vì uuid nằm trong stoped_uuid_set
        if STOP.is_set():
            raise Stopped()
    except BaseException:
        # dở dang (lỗi hoặc Dừng): bỏ hết file tạm của video này, không để lại dự án nửa vời
        app_cfg.rm_uuid(trk.uuid)
        shutil.rmtree(cache_folder, ignore_errors=True)
        shutil.rmtree(target_dir, ignore_errors=True)
        raise
    finally:
        CURRENT['uuid'] = None
    items = copy.deepcopy(trk.queue_tts)
    for it in items:
        it['dubbing_s'] = _audio_seconds(it['filename']) if Path(it['filename']).exists() else 0.0
    # video không tiếng do prepare tạo; giữ 1 bản gốc để mỗi lần xuất (che / làm chậm) đều bắt đầu từ bản sạch
    shutil.copy2(trk.cfg.novoice_mp4, f'{cache_folder}/novoice.orig.mp4')
    _, _, dur = _video_size(file_path)
    PROJECTS[pid] = {'trk': trk, 'title': title, 'name': safe_name(name, fallback=nospace), 'src': file_path,
                     'opts': dict(opts), 'items': items, 'edits': {},
                     'dur': dur, 'erase_band': opts.get('erase_band'), 'result': None, 'exported': [],
                     'nospace': nospace}
    yield 'done', pid


def drop_project(pid):
    """Bỏ dự án và xoá mọi thứ liên quan: cache tmp, thư mục làm việc, file đã xuất (nếu có)."""
    p = PROJECTS.pop(pid, None)
    if not p:
        return
    app_cfg.rm_uuid(p['trk'].uuid)
    shutil.rmtree(p['trk'].cfg.cache_folder, ignore_errors=True)
    shutil.rmtree(p['trk'].cfg.target_dir, ignore_errors=True)
    left = []
    for f in p.get('exported') or []:
        # trình duyệt vừa phát file này: Windows giữ khoá thêm một lúc -> thử lại vài lần
        for k in range(8):
            try:
                Path(f).unlink(missing_ok=True)
                break
            except PermissionError:
                time.sleep(0.4)
        else:
            left.append(f)
            continue
        parent = Path(f).parent
        try:
            if parent.exists() and not any(parent.iterdir()):   # chỉ xoá thư mục đã rỗng (không đụng file khác)
                parent.rmdir()
        except OSError:
            pass
    return left


def _line_text(p, i) -> str:
    return p['edits'].get(i, p['items'][i]['text'])


def _file_url(path) -> str:
    """URL Gradio phát 1 file trên máy. Mã hoá đường dẫn: tên có '#', '%', '?' sẽ hỏng nếu để nguyên."""
    from urllib.parse import quote
    return '/gradio_api/file=' + quote(Path(path).resolve().as_posix(), safe='/:')


def _audio_html(path, pending=False) -> str:
    """Thẻ <audio> gốc của trình duyệt (trình phát của Gradio giải mã bằng WebAudio, không ổn định).
    Thêm ?v=mtime vì tạo lại giọng giữ nguyên tên file -> trình duyệt sẽ không dùng bản cũ trong cache."""
    if pending:
        return '<div class="hint">Dòng này đã sửa chữ, bấm <b>Tạo lại giọng</b> để nghe bản mới.</div>'
    if not path or not Path(path).exists():
        return '<div class="hint">Chưa có giọng cho dòng này.</div>'
    src = f'{_file_url(path)}?v={int(Path(path).stat().st_mtime)}'
    return f'<audio controls preload="metadata" src="{html.escape(src)}" style="width:100%"></audio>'


def _line_status(p, i) -> str:
    it = p['items'][i]
    if i in p['edits']:
        return '✎ chờ tạo giọng'
    if not it.get('dubbing_s'):
        return '⚠ chưa có giọng'
    slot = (it['end_time'] - it['start_time']) / 1000
    over = it['dubbing_s'] - slot
    return f'{it["dubbing_s"]:.1f}s ▲ +{over:.1f}s' if over > 0.3 else f'{it["dubbing_s"]:.1f}s ✓'


def _df_rows(p) -> list:
    return [[i + 1, _ms(it['start_time']), _ms(it['end_time']), it.get('ref_text') or '', _line_text(p, i),
             _line_status(p, i)] for i, it in enumerate(p['items'])]


def _render_status(p) -> str:
    n, pend = len(p['items']), len(p['edits'])
    over = sum(1 for i in range(n) if '▲' in _line_status(p, i))
    s = f'<b>{n}</b> dòng thoại'
    s += f' · <span class="warn">✎ {pend} dòng đã sửa, chờ tạo giọng</span>' if pend else ''
    s += f' · ▲ {over} dòng giọng dài hơn chỗ trống' if over else ''
    if p.get('result'):
        s += f' · ✅ đã xuất: <code>{html.escape(p["result"])}</code>'
    return f'<div class="hint">{s}</div>'


def _render_timeline(p, sel=-1) -> str:
    """Dòng thời gian: mỗi đoạn thoại 1 khối; bấm vào để tua video tới đó (JS vdSeek trong head)."""
    dur = max(p['dur'], 0.1)
    segs = []
    for i, it in enumerate(p['items']):
        left = it['start_time'] / 1000 / dur * 100
        width = max(0.25, (it['end_time'] - it['start_time']) / 1000 / dur * 100)
        cls = 'sel' if i == sel else ('pend' if i in p['edits'] else ('over' if '▲' in _line_status(p, i) else 'ok'))
        tip = html.escape(f'{i + 1} · {_ms(it["start_time"])} → {_ms(it["end_time"])} · {_line_text(p, i)}')
        segs.append(f'<div class="tl-seg {cls}" style="left:{left:.2f}%;width:{width:.2f}%" title="{tip}" '
                    f'onclick="vdSeek({it["start_time"] / 1000:.2f})"></div>')
    ticks = ''.join(f'<span>{_ms(dur * 1000 * k / 6)}</span>' for k in range(7))
    return (f'<div class="tl"><div class="tl-track">{"".join(segs)}<div id="ed-playhead" class="tl-ph"></div></div>'
            f'<div class="tl-ticks">{ticks}</div></div>')


# Edge-TTS (máy chủ Microsoft) thỉnh thoảng trả rỗng "No audio was received" (đo được ~1/3 số lần gọi liên tiếp),
# lỗi theo đợt chứ không theo câu -> thử lại với thời gian chờ tăng dần, rồi quét lại các dòng còn lỗi thêm 1 lượt.
REDUB_RETRY_WAIT = (1.5, 3, 6, 10)   # giây chờ trước mỗi lần thử lại 1 dòng
REDUB_GAP = 0.8                      # giãn cách giữa các dòng khi tạo hàng loạt
REDUB_SECOND_PASS_WAIT = 10          # chờ trước lượt quét lại các dòng lỗi


def _redub_many(p, lines, on_line=None):
    """Tạo giọng nhiều dòng; dòng lỗi được quét lại 1 lượt sau khi chờ. Trả về {dòng: lỗi} của các dòng vẫn hỏng."""
    failed = {}
    for k, i in enumerate(lines):
        if on_line:
            on_line(k, i, False)
        if k:
            time.sleep(REDUB_GAP)
        try:
            _redub(p, i, p['edits'][i])
        except Exception as e:
            failed[i] = e
    if failed:
        time.sleep(REDUB_SECOND_PASS_WAIT)
        for k, i in enumerate(sorted(failed)):
            if on_line:
                on_line(k, i, True)
            try:
                _redub(p, i, p['edits'][i])
                failed.pop(i)
            except Exception as e:
                failed[i] = e
    return failed


def _redub(p, i, text):
    """Tạo lại giọng 1 dòng bằng đúng kênh TTS của dự án (cách làm của hộp thoại sửa lồng tiếng bản desktop)."""
    from videotrans.tts import run as run_tts
    trk, it = p['trk'], p['items'][i]
    d = dict(it, text=text)
    wav = Path(it['filename'])
    mp3 = Path(it['filename'] + '.mp3')
    bak = wav.with_name(wav.name + '.bak')
    if wav.exists():
        wav.replace(bak)   # giữ giọng cũ: tạo mới thất bại thì trả lại
    last_err = None
    try:
        # Edge-TTS với 1 dòng không tự thử lại; gọi liên tiếp nhiều dòng (tạo hàng loạt) hay bị Microsoft trả rỗng
        # "No audio was received" -> tự thử lại, giãn cách tăng dần.
        for attempt in range(len(REDUB_RETRY_WAIT) + 1):
            if attempt:
                time.sleep(REDUB_RETRY_WAIT[attempt - 1])
            mp3.unlink(missing_ok=True)
            try:
                run_tts(queue_tts=[dict(d)], language=trk.cfg.target_language_code, uuid=trk.uuid,
                        tts_type=trk.cfg.tts_type, is_cuda=trk.cfg.is_cuda)
            except Exception as e:
                last_err = e
            if wav.exists() and wav.stat().st_size > 0:
                break
            wav.unlink(missing_ok=True)
        else:
            raise RuntimeError(f'thử {len(REDUB_RETRY_WAIT) + 1} lần vẫn lỗi ({last_err or "không nhận được âm thanh"}). '
                               'Giọng cũ được giữ nguyên, thử lại sau ít phút.')
    except BaseException:
        if bak.exists():
            bak.replace(wav)
        raise
    bak.unlink(missing_ok=True)
    it['text'] = text
    it['dubbing_s'] = _audio_seconds(it['filename']) if Path(it['filename']).exists() else 0.0
    p['edits'].pop(i, None)


# ---------------------------------------------------------------------------
# Pha 1: lồng tiếng các video đã chọn -> mở trình sửa
# ---------------------------------------------------------------------------
JOB_BADGES = {
    'queued': ('b-mute', 'Đang chờ'),
    'downloading': ('b-info', 'Đang tải…'),
    'covering': ('b-info', 'Xoá sub gốc…'),
    'processing': ('b-info', 'Đang xử lý'),
    'stamping': ('b-info', 'Đóng dấu…'),
    'done': ('b-ok', 'Xong'),
    'error': ('b-err', 'Lỗi'),
    'stopped': ('b-mute', 'Đã huỷ'),
    'skipped': ('b-mute', 'Đã bỏ qua'),
}


QUEUE_STATE = {'jobs': []}


def _render_queue(jobs, mode='dub') -> str:
    QUEUE_STATE['jobs'] = jobs
    if not jobs:
        return ('<div class="empty"><div class="empty-ico">⏳</div>'
                + ('Tích chọn video ở mục <b>2. Danh sách video</b> rồi bấm <b>Lồng tiếng ngay</b>. Lồng tiếng xong, '
                   'trình sửa sẽ mở để bạn chỉnh từng câu thoại trước khi xuất.' if mode == 'dub' else
                   'Tích chọn video ở mục <b>2. Danh sách video</b>, chọn cách đóng dấu (nếu cần) rồi bấm <b>Tải</b>. '
                   'File được lưu thẳng vào thư mục đã chọn.') + '</div>')
    n = len(jobs)
    finished = sum(j['status'] in ('done', 'error', 'skipped', 'stopped') for j in jobs)
    part = sum(j.get('stage', 0) / len(STAGES) for j in jobs if j['status'] == 'processing')
    pct = int((finished + part) / n * 100)
    done = sum(j['status'] == 'done' for j in jobs)
    rows = []
    for i, j in enumerate(jobs, 1):
        cls, text = JOB_BADGES[j['status']]
        if j['status'] == 'processing':
            text = f'Bước {j["stage"]}/{len(STAGES)}'
        note = html.escape(j.get('note') or '')
        rows.append(f'<div class="qitem q-{j["status"]}"><span class="qdot"></span>'
                    f'<div class="vbody"><div class="vtitle one">{i}. {html.escape(_short_title(j.get("name") or j["title"], 60))}</div>'
                    f'<div class="vmeta">{note}</div></div><span class="badge {cls}">{text}</span></div>')
    return (f'<div class="qhead"><span><b>{done}</b>/{n} video xong</span><span>{pct}%</span></div>'
            f'<div class="pbar"><div style="width:{pct}%"></div></div>'
            f'<div class="qlist">{"".join(rows)}</div>')


def _proj_choices():
    return [(p.get('name') or _short_title(p['title'], 50), pid) for pid, p in PROJECTS.items()]


def start_projects(selected, names, link_items, files, source_name, model_name, voice_role_name, subtitle_name,
                   keep_bgm, voice_autorate, video_autorate, use_cuda, browser, cookie_file, ai_erase):
    """Outputs: queue, log, run_btn, stop_btn, proj_dd, *NAV (7), stepper."""
    run_busy = gr.update(value='⏳ Đang xử lý…', interactive=False)
    stop_on = gr.update(interactive=True, value=STOP_LABEL)
    stop_off = gr.update(interactive=False, value=STOP_LABEL)
    keep = gr.update()

    items = _all_items(link_items, files)
    chosen = [it for it in items if it['key'] in (selected or []) and it['status'] == 'ok']
    if not chosen:
        gr.Warning('Chưa chọn video nào: tích ô ở đầu mỗi video trong danh sách.')
        yield _render_queue([]), '', gr.update(), stop_off, keep, *_nav_keep(), keep
        return

    opts = {
        'source': SOURCE_LANGS[source_name], 'model': model_name, 'voice': voice_role_name,
        'subtitle': SUBTITLE_TYPES[subtitle_name], 'subtitle_name': subtitle_name, 'keep_bgm': bool(keep_bgm),
        'voice_autorate': bool(voice_autorate), 'video_autorate': bool(video_autorate),
        'cuda': bool(use_cuda) and _cuda_available(), 'ai_erase': bool(ai_erase),
    }
    STOP.clear()
    jobs = [{'title': it['title'], 'name': _item_name(names, it), 'item': it, 'status': 'queued', 'note': ''}
            for it in chosen]
    lines, new_pids = [], []

    def log(msg):
        lines.append(f"[{time.strftime('%H:%M:%S')}] {msg}")

    def emit(run_btn=run_busy, stop_btn=stop_on, dd=keep, nav=None, step=_stepper(2)):
        return (_render_queue(jobs), '\n'.join(lines), run_btn, stop_btn, dd, *(nav or _nav_keep()), step)

    yield emit()
    for idx, job in enumerate(jobs, 1):
        if STOP.is_set():
            job.update(status='skipped', note='Đã dừng theo yêu cầu')
            continue
        tag = f'[{idx}/{len(jobs)}]'
        try:
            job.update(status='downloading', note='Đang tải video về máy…')
            log(f'{tag} Bắt đầu: {job["name"]}')
            yield emit()
            path = _fetch(job['item'], browser, cookie_file)
            log(f'{tag} Có file: {Path(path).name} ({Path(path).stat().st_size / 1048576:.1f} MB)')

            job_opts = dict(opts)
            if STOP.is_set():
                raise Stopped()
            if opts['ai_erase']:
                job.update(status='covering', note='Đang tìm vị trí phụ đề gốc trong video…')
                yield emit()
                bands, info = vi_dub_erase.find_sub_bands(path)
                log(f'{tag} {info}')
                if not bands:
                    gr.Warning(f'{Path(path).name}: {info} Bỏ qua bước xoá chữ.')
                else:
                    job_opts['erase_band'] = bands[0]
                    t_cover, last_pct = time.time(), -1
                    for frac, out_path in cover_original_subs(path, 'Không che', 0, 0, erase=True, bands=bands):
                        pct = int(frac * 100)
                        if out_path is None and pct != last_pct:
                            last_pct = pct
                            eta = (time.time() - t_cover) / max(frac, 1e-3) * (1 - frac)
                            job.update(note=f'Xoá chữ (AI)… {pct}% · còn ~{int(eta // 60)}:{int(eta % 60):02d}')
                            yield emit()
                        if out_path:
                            path = out_path
                    log(f'{tag} Xong xoá chữ ({time.time() - t_cover:.0f}s)')
                vi_dub_erase.release_model()  # nhường VRAM cho Whisper

            job.update(status='processing', stage=0)
            for step, title in build_project(path, job_opts, job['title'], job['name']):
                if step == 'done':
                    new_pids.append(title)
                    break
                job.update(stage=step, note=title + '…')
                log(f'{tag} Bước {step}/{len(STAGES)}: {title}')
                yield emit()
            job.update(status='done', note=f'{len(PROJECTS[new_pids[-1]]["items"])} dòng thoại · chờ bạn sửa trong trình sửa')
            log(f'{tag} ✅ Lồng tiếng xong, mở trình sửa')
        except (Stopped, vi_dub_erase.EraseStopped):
            job.update(status='stopped', note='Bạn đã bấm Dừng: phần đang làm dở đã bị huỷ và xoá file tạm')
            log(f'{tag} ⏹ Đã dừng, huỷ video đang xử lý dở')
        except Exception as e:
            if STOP.is_set():   # bước nào đó thoát sớm vì Dừng rồi báo lỗi thiếu file -> coi là đã huỷ
                job.update(status='stopped', note='Bạn đã bấm Dừng: phần đang làm dở đã bị huỷ và xoá file tạm')
                log(f'{tag} ⏹ Đã dừng ({str(e)[:80]})')
            else:
                job.update(status='error', note=str(e)[:200])
                log(f'{tag} ❌ Lỗi: {e}\n{traceback.format_exc()}')
        yield emit()

    stopped = STOP.is_set()
    STOP.clear()
    log(('Đã dừng' if stopped else 'Hoàn tất') + f': {len(new_pids)}/{len(jobs)} video sẵn sàng để sửa.')
    run_idle = gr.update(value=_run_label(selected), interactive=True)
    if new_pids and not stopped:
        gr.Info(f'Đã lồng tiếng {len(new_pids)} video. Sửa chữ rồi bấm Xuất video.')
        yield emit(run_idle, stop_off, gr.update(choices=_proj_choices(), value=new_pids[0]), _nav('editor'),
                   _stepper(3))
    else:
        if stopped:
            gr.Info(f'Đã dừng. {len(new_pids)} video xong trước đó vẫn còn trong trình sửa.' if new_pids
                    else 'Đã dừng, không có video nào hoàn tất.')
        choices = _proj_choices()
        yield emit(run_idle, stop_off,
                   gr.update(choices=choices, value=new_pids[0] if new_pids else (choices[0][1] if choices else None)),
                   step=_stepper(1))


# ---------------------------------------------------------------------------
# Nền tảng video (chọn ở tab Link). Trạng thái đo thực tế ngày 03/10/2026 với yt-dlp 2026.08:
# ok = tải được, cookie = cần cookie đăng nhập, soon = yt-dlp chưa tải được.
# ---------------------------------------------------------------------------
PLATFORMS = [
    # key, tên, tên miền, trạng thái, ghi chú hiện dưới hàng chọn
    ('auto', 'Tự nhận', (), 'ok', 'Dán link bất kỳ, app tự nhận nền tảng theo tên miền.'),
    ('douyin', 'Douyin', ('douyin.com', 'iesdouyin.com'), 'ok',
     'Dán link hoặc nguyên đoạn chia sẻ. Bị chặn thì dùng cookie ở Cài đặt → Nâng cao.'),
    ('tiktok', 'TikTok', ('tiktok.com',), 'ok', 'Link video hoặc link rút gọn vm.tiktok.com.'),
    ('youtube', 'YouTube', ('youtube.com', 'youtu.be'), 'ok',
     'Video và Shorts. Tải tối đa 1080p.'),
    ('bilibili', 'Bilibili', ('bilibili.com', 'b23.tv'), 'ok', 'Chất lượng từ 1080p trở lên cần cookie.'),
    ('facebook', 'Facebook', ('facebook.com', 'fb.watch', 'fb.com'), 'ok',
     'Video công khai; video riêng tư hoặc trong nhóm cần cookie.'),
    ('x', 'X', ('x.com', 'twitter.com'), 'ok', 'Bài đăng có video trên X (Twitter).'),
    ('instagram', 'Instagram', ('instagram.com',), 'cookie',
     'Hầu như luôn cần cookie đăng nhập (chọn file cookies.txt ở Cài đặt → Nâng cao).'),
    ('kuaishou', 'Kuaishou', ('kuaishou.com',), 'soon', 'Chưa hỗ trợ (yt-dlp chưa tải được).'),
    ('xiaohongshu', 'Xiaohongshu', ('xiaohongshu.com', 'xhslink.com'), 'soon', 'Chưa hỗ trợ (yt-dlp chưa tải được).'),
]
PLAT = {p[0]: p for p in PLATFORMS}


def _platform_of(url: str) -> str:
    host = (re.sub(r'^https?://', '', url or '').split('/')[0] or '').lower()
    for key, _, domains, _, _ in PLATFORMS:
        if any(host == d or host.endswith('.' + d) for d in domains):
            return key
    return 'other'


def _plat_label(key: str) -> str:
    return PLAT[key][1] if key in PLAT else 'Khác'


def _plat_hint(key: str) -> str:
    _, label, _, status, note = PLAT.get(key, PLAT['auto'])
    badge = {'ok': '', 'cookie': ' <span class="badge b-mute">cần cookie</span>',
             'soon': ' <span class="badge b-err">chưa hỗ trợ</span>'}[status]
    return f'<div class="hint plat-hint"><b>{html.escape(label)}</b>{badge} · {html.escape(note)}</div>'


def _plat_placeholder(key: str, i: int) -> str:
    if key == 'auto':
        return f'Link {i + 1}: dán link Douyin, TikTok, YouTube, Bilibili… hoặc nguyên đoạn chia sẻ'
    return f'Link {i + 1}: dán link {_plat_label(key)}'


# ---------------------------------------------------------------------------
# Chế độ "Chỉ tải video": tải nguyên bản (+ đóng dấu logo / chữ) rồi lưu thẳng, không lồng tiếng
# ---------------------------------------------------------------------------
MODES = {'dub': '🎙️ Lồng tiếng', 'dl': '⬇️ Chỉ tải video'}
MODE_HINTS = {
    'dub': 'Nhận dạng → dịch → lồng tiếng Việt → mở trình sửa để chỉnh từng câu rồi xuất.',
    'dl': 'Chỉ tải video gốc về máy (tối đa 1080p), có thể gắn logo hoặc chữ ở một vị trí cố định. Không lồng tiếng.',
}
STEPS_DL = ['Thêm video', 'Tải & đóng dấu', 'Hoàn tất']
WM_POS = ['↖', '↑', '↗', '←', '●', '→', '↙', '↓', '↘']   # lưới 3×3: trên-trái … dưới-phải
WM_FONT = next((f for f in ('C:/Windows/Fonts/arialbd.ttf', 'C:/Windows/Fonts/segoeuib.ttf',
                            'C:/Windows/Fonts/arial.ttf') if Path(f).exists()), None)
WM_LOGO_DIR = DOWNLOAD_DIR / '_wm'   # logo đã chọn được giữ lại cho lần sau
DOWNLOADS: list = []           # file đã tải / đóng dấu xong trong phiên này
DL_STATE = {'done': False}     # lượt tải gần nhất có video xong -> thanh bước hiện "Hoàn tất"
_FRAME_CACHE: dict = {}
# thứ tự giá trị đóng dấu đi qua các hàm (cũng là thứ tự component trong giao diện)
WM_FIELDS = ['wm_logo_on', 'wm_logo', 'wm_logo_pos', 'wm_logo_size', 'wm_logo_opacity',
             'wm_text_on', 'wm_text', 'wm_color', 'wm_text_pos', 'wm_text_size', 'wm_text_opacity', 'wm_margin']


def _pos_rc(pos) -> tuple:
    i = WM_POS.index(pos) if pos in WM_POS else 8
    return i // 3, i % 3


def _wm_opts(logo_on=False, logo=None, logo_pos='↖', logo_size=18, logo_op=100, text_on=False, text='',
             color='#FFFFFF', text_pos='↘', text_size=5, text_op=85, margin=3) -> dict:
    """Gom giá trị đóng dấu thành {'items': [lớp logo?, lớp chữ?], 'margin'}. Lớp nào bật mà thiếu dữ liệu thì bỏ."""
    items = []
    if logo_on and logo and Path(str(logo)).exists():
        r, c = _pos_rc(logo_pos)
        items.append({'type': 'logo', 'path': str(logo), 'row': r, 'col': c, 'size': float(logo_size or 18),
                      'opacity': min(1.0, max(0.05, float(logo_op or 100) / 100))})
    if text_on and (text or '').strip():
        r, c = _pos_rc(text_pos)
        col = str(color or '').strip()
        items.append({'type': 'text', 'text': text.strip(), 'row': r, 'col': c, 'size': float(text_size or 6),
                      'opacity': min(1.0, max(0.05, float(text_op or 85) / 100)),
                      'color': col.upper() if re.fullmatch(r'#[0-9a-fA-F]{6}', col) else '#FFFFFF'})
    return {'items': items, 'margin': float(margin or 0)}


def _wm_active(logo_on, logo, text_on, text) -> bool:
    return bool((logo_on and logo) or (text_on and (text or '').strip()))


def _wm_summary(wm) -> str:
    """'Logo ↘ + Chữ "@kenh" ↗' hoặc '' nếu không đóng dấu."""
    parts = []
    for it in wm['items']:
        pos = WM_POS[it['row'] * 3 + it['col']]
        parts.append(f'Logo {pos}' if it['type'] == 'logo' else f'Chữ "{it["text"][:20]}" {pos}')
    return ' + '.join(parts)


def _wm_xy(it, margin, W, H, w, h):
    m = int(W * margin / 100)
    x = m if it['col'] == 0 else ((W - w) // 2 if it['col'] == 1 else W - w - m)
    y = m if it['row'] == 0 else ((H - h) // 2 if it['row'] == 1 else H - h - m)
    return max(0, int(x)), max(0, int(y))


def _wm_text_px(it, W) -> int:
    """Cỡ chữ = % bề rộng video (cỡ 6 trên video 1080 px rộng ≈ 65 px)."""
    return max(10, int(W * it['size'] / 100))


def _logo_size(it, W):
    from PIL import Image
    with Image.open(it['path']) as im:
        lw = max(8, int(W * it['size'] / 100))
        return lw, max(1, round(im.height * lw / im.width))


def stamp_video(src: str, dst: str, wm: dict) -> None:
    """Đóng dấu các lớp logo / chữ lên video trong 1 lượt ffmpeg (NVENC, dự phòng libx264). Âm thanh giữ nguyên."""
    W, H, _ = _video_size(src)
    if not W or not H:
        raise RuntimeError('Không đọc được kích thước video.')
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    inputs, parts, cur = ['-i', src], [], '[0:v]'
    for k, it in enumerate(wm['items']):
        if it['type'] == 'logo':
            lw, lh = _logo_size(it, W)
            x, y = _wm_xy(it, wm['margin'], W, H, lw, lh)
            inputs += ['-i', it['path']]
            idx = len(inputs) // 2 - 1
            parts.append(f'[{idx}:v]scale={lw}:{lh},format=rgba,colorchannelmixer=aa={it["opacity"]:.2f}[lg{k}]')
            parts.append(f'{cur}[lg{k}]overlay={x}:{y}:format=auto[s{k}]')
        else:
            px = _wm_text_px(it, W)
            m = int(W * wm['margin'] / 100)
            x = str(m) if it['col'] == 0 else ('(w-text_w)/2' if it['col'] == 1 else f'w-text_w-{m}')
            y = str(m) if it['row'] == 0 else ('(h-text_h)/2' if it['row'] == 1 else f'h-text_h-{m}')
            # chữ đưa qua file (textfile) để khỏi thoát ký tự đặc biệt của filtergraph; ffmpeg chạy trong PREVIEW_DIR
            tf = PREVIEW_DIR / f'wm-{hashlib.md5(it["text"].encode()).hexdigest()[:8]}.txt'
            tf.write_text(it['text'], encoding='utf-8')
            font = f":fontfile='{WM_FONT.replace(':', chr(92) + ':')}'" if WM_FONT else ''
            parts.append(f"{cur}drawtext=textfile='{tf.name}'{font}:fontsize={px}"
                         f":fontcolor=0x{it['color'][1:]}@{it['opacity']:.2f}:borderw={max(1, px // 14)}"
                         f":bordercolor=0x000000@{min(1.0, it['opacity'] + 0.15):.2f}:x={x}:y={y}[s{k}]")
        cur = f'[s{k}]'
    base = ['ffmpeg', '-y', '-nostdin', *inputs, '-filter_complex', ';'.join(parts), '-map', cur, '-map', '0:a?',
            '-c:a', 'copy', '-pix_fmt', 'yuv420p', '-movflags', '+faststart']
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    last_err = ''
    for enc in (['-c:v', 'h264_nvenc', '-cq', '21', '-preset', 'p4'],
                ['-c:v', 'libx264', '-crf', '19', '-preset', 'veryfast']):
        r = subprocess.run(base + enc + [str(Path(dst).resolve())], capture_output=True, text=True, encoding='utf-8',
                           errors='ignore', cwd=str(PREVIEW_DIR))
        if r.returncode == 0 and Path(dst).exists() and Path(dst).stat().st_size > 0:
            return
        last_err = r.stderr[-800:]
    Path(dst).unlink(missing_ok=True)
    raise RuntimeError(f'Đóng dấu lỗi (ffmpeg): {last_err}')


def _wm_draw(img, wm):
    """Vẽ các lớp dấu lên ảnh PIL để xem trước, cùng tỉ lệ với ffmpeg."""
    from PIL import Image, ImageDraw, ImageFont
    if not wm['items']:
        return img
    W, H = img.size
    out = img.convert('RGBA')
    for it in wm['items']:
        if it['type'] == 'logo':
            logo = Image.open(it['path']).convert('RGBA')
            lw, lh = _logo_size(it, W)
            logo = logo.resize((lw, lh))
            if it['opacity'] < 1:
                logo.putalpha(logo.getchannel('A').point(lambda a, o=it['opacity']: int(a * o)))
            out.alpha_composite(logo, _wm_xy(it, wm['margin'], W, H, lw, lh))
        else:
            px = _wm_text_px(it, W)
            font = ImageFont.truetype(WM_FONT, px) if WM_FONT else ImageFont.load_default()
            layer = Image.new('RGBA', out.size, (0, 0, 0, 0))
            d = ImageDraw.Draw(layer)
            bbox = d.textbbox((0, 0), it['text'], font=font)
            x, y = _wm_xy(it, wm['margin'], W, H, bbox[2] - bbox[0], bbox[3] - bbox[1])
            a = int(255 * it['opacity'])
            col = tuple(int(it['color'][i:i + 2], 16) for i in (1, 3, 5))
            d.text((x - bbox[0], y - bbox[1]), it['text'], font=font, fill=(*col, a), stroke_width=max(1, px // 14),
                   stroke_fill=(0, 0, 0, min(255, a + 40)))
            out = Image.alpha_composite(out, layer)
    return out.convert('RGB')


def _frame_for(path):
    """Khung hình (PIL) để xem trước: lấy ở 30% video, thu về bề rộng 540; không có file thì khung mẫu 9:16."""
    from PIL import Image
    import numpy as np
    if path and Path(path).exists():
        key = (path, Path(path).stat().st_mtime)
        if key not in _FRAME_CACHE:
            try:
                _, _, dur = _video_size(path)
                img = Image.fromarray(vi_dub_erase.grab_frame(path, max(0.0, dur * 0.3)))
                if img.width > 540:
                    img = img.resize((540, round(img.height * 540 / img.width)))
                _FRAME_CACHE.clear()
                _FRAME_CACHE[key] = img
            except Exception:
                _FRAME_CACHE[key] = None
        if _FRAME_CACHE.get(key) is not None:
            return _FRAME_CACHE[key].copy(), True
    g = np.linspace(28, 70, 960, dtype=np.uint8)[:, None].repeat(540, 1)
    arr = np.stack([g // 2 + 10, g // 2 + 12, g + 20], -1).astype(np.uint8)
    return Image.fromarray(arr), False


def wm_preview(link_items, files, selected, *wm_vals):
    """Ảnh xem trước dấu trên video đầu tiên đang chọn (nếu đã có file trên máy)."""
    wm = _wm_opts(*wm_vals)
    items = _all_items(link_items, files)
    ok = [it for it in items if it['status'] == 'ok']
    chosen = [it for it in ok if it['key'] in (selected or [])] or ok
    path = next((p for p in (_local_path(it) for it in chosen) if p), None)
    img, real = _frame_for(path)
    img = _wm_draw(img, wm)
    if not wm['items']:
        cap = 'Bật <b>Gắn logo</b> hoặc <b>Gắn chữ</b> (hoặc cả hai) để xem trước.'
    elif real:
        cap = f'Xem trước trên <b>{html.escape(Path(path).name)}</b> · video thật sẽ y như vậy.'
    else:
        cap = 'Xem trước trên khung mẫu 9:16 (video từ link chỉ hiện sau khi tải về).'
    return img, f'<div class="hint">{cap}</div>'


def _wm_boxes(logo_on, text_on):
    """Hiện/ẩn khối Logo, khối Chữ, phần chung (cách mép + xem trước) theo 2 công tắc."""
    return (gr.update(elem_classes=['wm-sub'] if logo_on else ['wm-sub', 'lhide']),
            gr.update(elem_classes=['wm-sub'] if text_on else ['wm-sub', 'lhide']),
            gr.update(elem_classes=[] if (logo_on or text_on) else ['lhide']))


def save_logo(path):
    """Giữ bản sao logo trong downloads/_wm để lần sau mở app vẫn còn."""
    if not path or not Path(path).exists():
        return gr.update()
    WM_LOGO_DIR.mkdir(parents=True, exist_ok=True)
    for old in WM_LOGO_DIR.glob('logo.*'):
        old.unlink(missing_ok=True)
    dst = WM_LOGO_DIR / f'logo{Path(path).suffix.lower() or ".png"}'
    shutil.copy2(path, dst)
    return dst.as_posix()


def clear_logo():
    for old in WM_LOGO_DIR.glob('logo.*') if WM_LOGO_DIR.exists() else []:
        old.unlink(missing_ok=True)


def saved_logo():
    found = sorted(WM_LOGO_DIR.glob('logo.*')) if WM_LOGO_DIR.exists() else []
    return found[0].as_posix() if found else None


def wm_export_note(*wm_vals) -> str:
    """Dòng tóm tắt trong khung Xuất của trình sửa: video xuất ra sẽ được đóng dấu gì."""
    s = _wm_summary(_wm_opts(*wm_vals)) if wm_vals else ''
    if not s:
        return ('<div class="hint wm-note">🏷️ Không đóng dấu. Muốn gắn logo / chữ: bật ở khung '
                '<b>Đóng dấu logo / chữ</b> trang Lồng tiếng.</div>')
    return (f'<div class="hint wm-note on">🏷️ Video xuất sẽ được đóng dấu: <b>{html.escape(s)}</b> '
            '· đổi ở trang Lồng tiếng.</div>')


def _unique_flat(out_root: Path, name: str, ext: str = '.mp4') -> Path:
    for k in range(1, 1000):
        p = out_root / (f'{name}{ext}' if k == 1 else f'{name} ({k}){ext}')
        if not p.exists():
            return p
    raise RuntimeError(f'Quá nhiều file trùng tên "{name}" trong {out_root}')


def download_jobs(selected, names, link_items, files, browser, cookie_file, out_dir, wm):
    """Chế độ 'Chỉ tải video': tải (hoặc lấy file từ máy) -> đóng dấu nếu có -> lưu <thư mục>/<tên>.mp4.
    Outputs giống start_projects: queue, log, run_btn, stop_btn, proj_dd, *NAV, stepper."""
    run_busy = gr.update(value='⏳ Đang tải…', interactive=False)
    stop_on = gr.update(interactive=True, value=STOP_LABEL)
    stop_off = gr.update(interactive=False, value=STOP_LABEL)
    keep = gr.update()
    items = _all_items(link_items, files)
    chosen = [it for it in items if it['key'] in (selected or []) and it['status'] == 'ok']
    if not chosen:
        gr.Warning('Chưa chọn video nào: tích ô ở đầu mỗi video trong danh sách.')
        yield _render_queue([], 'dl'), '', keep, stop_off, keep, *_nav_keep(), keep
        return
    out_root = _out_root(out_dir)
    try:
        out_root.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        gr.Warning(f'Không tạo được thư mục lưu {out_root}: {e}')
        yield _render_queue([], 'dl'), '', keep, stop_off, keep, *_nav_keep(), keep
        return
    _allow_path(out_root)
    STOP.clear()
    DL_STATE['done'] = False
    jobs = [{'title': it['title'], 'name': _item_name(names, it), 'item': it, 'status': 'queued', 'note': ''}
            for it in chosen]
    lines = []

    def log(msg):
        lines.append(f"[{time.strftime('%H:%M:%S')}] {msg}")

    def emit(run_btn=run_busy, stop_btn=stop_on, step=None):
        return (_render_queue(jobs, 'dl'), '\n'.join(lines), run_btn, stop_btn, keep, *_nav_keep(),
                step if step is not None else _stepper(2, 'dl'))

    what = f'đóng dấu {_wm_summary(wm)}' if wm['items'] else None
    gr.Info(f'Bắt đầu tải {len(jobs)} video' + (f' và {what}' if what else '') + f' vào {out_root}', duration=6)
    yield emit()
    for idx, job in enumerate(jobs, 1):
        if STOP.is_set():
            job.update(status='skipped', note='Đã dừng theo yêu cầu')
            continue
        tag = f'[{idx}/{len(jobs)}]'
        try:
            is_link = job['item']['kind'] == 'link'
            job.update(status='downloading', note='Đang tải video về máy…' if is_link else 'Đang đọc file…')
            log(f'{tag} Bắt đầu: {job["name"]}')
            yield emit()
            path = _fetch(job['item'], browser, cookie_file)
            log(f'{tag} Có file: {Path(path).name} ({Path(path).stat().st_size / 1048576:.1f} MB)')
            if STOP.is_set():
                raise Stopped()
            dst = _unique_flat(out_root, job['name'], '.mp4' if wm['items'] else Path(path).suffix.lower())
            if wm['items']:
                job.update(status='stamping', note=f'Đang {what}…')
                yield emit()
                t0 = time.time()
                stamp_video(path, dst.as_posix(), wm)
                log(f'{tag} Xong {what} ({time.time() - t0:.0f}s)')
            else:
                shutil.copy2(path, dst)
            DOWNLOADS.append(dst.as_posix())
            job.update(status='done', note=f'Đã lưu: {dst}')
            log(f'{tag} ✅ {dst}')
        except (Stopped, vi_dub_erase.EraseStopped):
            job.update(status='stopped', note='Bạn đã bấm Dừng')
            log(f'{tag} ⏹ Đã dừng')
        except Exception as e:
            if STOP.is_set():
                job.update(status='stopped', note='Bạn đã bấm Dừng')
                log(f'{tag} ⏹ Đã dừng ({str(e)[:80]})')
            else:
                job.update(status='error', note=str(e)[:200])
                log(f'{tag} ❌ Lỗi: {e}')
                logger.exception(f'[VieDub] Tải video lỗi ({job["name"]}): {e}')
        yield emit()
    stopped = STOP.is_set()
    STOP.clear()
    n_ok = sum(j['status'] == 'done' for j in jobs)
    bad = [j for j in jobs if j['status'] == 'error']
    DL_STATE['done'] = n_ok > 0
    log(('Đã dừng' if stopped else 'Hoàn tất') + f': {n_ok}/{len(jobs)} video đã lưu vào {out_root}')
    if bad:
        gr.Warning(f'Xong {n_ok}/{len(jobs)} video. Lỗi: ' + '; '.join(f'{j["name"]} ({j["note"][:100]})' for j in bad),
                   duration=20)
    elif stopped:
        gr.Info(f'Đã dừng. {n_ok}/{len(jobs)} video đã lưu vào {out_root}', duration=10)
    else:
        gr.Info(f'✅ Đã lưu {n_ok} video vào {out_root}', duration=12)
    idle = gr.update(value=_run_label(selected, 'dl', bool(wm['items'])), interactive=True)
    yield emit(idle, stop_off, _stepper(3 if n_ok else 1, 'dl'))


def start_run(mode, selected, names, link_items, files, source_name, model_name, voice_role_name, subtitle_name,
              keep_bgm, voice_autorate, video_autorate, use_cuda, browser, cookie_file, ai_erase, dl_out_dir, *wm_vals):
    """Nút chạy chính: theo chế độ đang chọn. Chế độ lồng tiếng: dấu được gắn lúc xuất (trong trình sửa)."""
    if mode == 'dl':
        wm = _wm_opts(*wm_vals)
        yield from download_jobs(selected, names, link_items, files, browser, cookie_file, dl_out_dir, wm)
    else:
        yield from start_projects(selected, names, link_items, files, source_name, model_name, voice_role_name,
                                  subtitle_name, keep_bgm, voice_autorate, video_autorate, use_cuda, browser,
                                  cookie_file, ai_erase)


PAGE_HEADS = {
    'dub': ('⚡', 'Lồng tiếng', 'Thêm video, đặt tên, chọn cài đặt rồi bấm Lồng tiếng. Xong sẽ mở trình sửa để chỉnh '
                               'từng câu trước khi xuất.'),
    'dl': ('⬇️', 'Tải video', 'Thêm video, đặt tên, chọn cách đóng dấu rồi bấm Tải. File lưu thẳng vào thư mục đã chọn, '
                             'không lồng tiếng.'),
}


def _add_title(mode) -> str:
    sub = ('Dán link Douyin, TikTok, YouTube, Bilibili… hoặc thả file từ máy' if mode == 'dub'
           else 'Dán link Douyin, TikTok, YouTube, Bilibili…')
    return _ptitle('🔗', '1. Thêm video', sub, 'pink')


def set_mode(mode, selected, logo_on, logo, text_on, text):
    """Đổi chế độ: tiêu đề trang, hàng đợi, chip, khối cài đặt, nhãn nút chạy, ghi chú, thanh bước, tiêu đề tiến trình,
    tiêu đề khung 1, tab Link/File (chế độ chỉ tải: ẩn tab, chỉ dán link)."""
    wm_kind = _wm_active(logo_on, logo, text_on, text)
    on, off = ['chipbtn', 'active'], ['chipbtn']
    return (_page_head(*PAGE_HEADS[mode]), gr.update() if QUEUE_STATE['jobs'] else _render_queue([], mode), mode, gr.update(elem_classes=on if mode == 'dub' else off), gr.update(elem_classes=on if mode == 'dl' else off),
            gr.update(elem_classes=[] if mode == 'dub' else ['lhide']), gr.update(elem_classes=[] if mode == 'dl' else ['lhide']),
            gr.update(value=_run_label(selected, mode, wm_kind)),
            f'<div class="hint mode-hint">{MODE_HINTS[mode]}</div>',
            _stepper(3 if mode == 'dl' and DL_STATE['done'] else 1, mode),
            _ptitle('⏳', '4. Tiến trình', 'Lần lượt từng video · xong sẽ mở trình sửa' if mode == 'dub'
                    else 'Lần lượt từng video · file lưu thẳng vào thư mục đã chọn', 'org'),
            _add_title(mode),
            gr.update(selected='link', elem_classes=['seg'] if mode == 'dub' else ['seg', 'notabs']))


def set_plat(key):
    """Chọn nền tảng: chip, ghi chú, placeholder của 30 ô link."""
    def _f():
        return (key, *[gr.update(elem_classes=['chipbtn', 'plat'] + (['active'] if k == key else [])
                                 + (['soon'] if PLAT[k][3] == 'soon' else [])) for k in PLAT],
                _plat_hint(key), *[gr.update(placeholder=_plat_placeholder(key, i)) for i in range(MAX_LINKS)])
    return _f


# ---------------------------------------------------------------------------
# Trình sửa
# ---------------------------------------------------------------------------
def load_project(pid):
    """Nạp 1 dự án vào trình sửa."""
    p = PROJECTS.get(pid)
    if not p:
        return (_video_html(None), '', '', gr.update(value=[]), 0,
                gr.update(choices=[], value=None), gr.update(), *_result_html(None), gr.update(selected='orig'),
                gr.update(elem_classes=['lhide']), gr.update(elem_classes=[]), gr.update(value=''))
    return (_video_html(p['src']), _render_status(p), _render_timeline(p, 0), gr.update(value=_df_rows(p)), 0,
            gr.update(choices=_proj_choices(), value=pid), gr.update(value=p['opts']['subtitle_name']),
            *_result_html(p), gr.update(selected='result' if p.get('result') else 'orig'),
            gr.update(elem_classes=[]), gr.update(elem_classes=['lhide']), gr.update(value=p.get('name') or ''))


def _video_html(path, elem_id='ed-native', empty='Chưa có video.') -> str:
    """Thẻ <video> gốc của trình duyệt: nút âm lượng như YouTube (bấm = tắt tiếng, rê chuột = hiện thanh âm lượng),
    trình phát của Gradio thì phải bấm mới hiện thanh âm lượng. ?v=mtime để xuất lại cùng tên không bị cache."""
    if not path or not Path(path).exists():
        return f'<div class="empty">{empty}</div>'
    src = f'{_file_url(path)}?v={int(Path(path).stat().st_mtime)}'
    return (f'<video id="{elem_id}" class="ed-player" controls preload="metadata" playsinline '
            f'src="{html.escape(src)}"></video>')


RESULT_EMPTY = ('Chưa dựng video. Sửa chữ xong, bấm <b>Xuất video</b> ở cột phải: video kết quả (giọng Việt + phụ đề) '
                'sẽ hiện ở đây để xem thử.')


def _result_html(p) -> str:
    """Khung 'Kết quả': video đã xuất của dự án + đường dẫn."""
    if not p or not p.get('result') or not Path(p['result']).exists():
        return _video_html(None, 'ed-result', RESULT_EMPTY), '<div class="hint">Chưa có video kết quả.</div>'
    folder = Path(p['result']).parent
    files = ' · '.join(html.escape(Path(f).name) for f in p.get('exported') or [p['result']] if Path(f).parent == folder)
    info = (f'<div class="hint">📁 <code>{html.escape(str(Path(p["result"]).parent))}</code><br>{files}</div>')
    return _video_html(p['result'], 'ed-result', RESULT_EMPTY), info


def select_line(pid, i):
    """Nạp dòng i vào ô 'Dòng đang chọn' + tua video tới đó."""
    p = PROJECTS.get(pid)
    if not p or not p['items']:
        return gr.update(), '', gr.update(value=''), _audio_html(None), '', gr.update()
    i = max(0, min(int(i or 0), len(p['items']) - 1))
    it = p['items'][i]
    title = (f'<div class="sub-head">Dòng {i + 1} / {len(p["items"])} · {_ms(it["start_time"])} → '
             f'{_ms(it["end_time"])} · chỗ trống {(it["end_time"] - it["start_time"]) / 1000:.1f}s · '
             f'giọng {it.get("dubbing_s", 0):.1f}s</div>')
    zh = f'<div class="zh">{html.escape(it.get("ref_text") or "")}</div>'
    audio = _audio_html(it['filename'], pending=i in p['edits'])
    return i, title, gr.update(value=_line_text(p, i)), audio, zh, f'{it["start_time"] / 1000:.2f}#{time.time()}'


def on_df_select(evt: gr.SelectData, pid):
    i = evt.index[0] if isinstance(evt.index, (list, tuple)) else int(evt.index)
    return (*select_line(pid, i), gr.update(value=_render_timeline(PROJECTS[pid], i)) if pid in PROJECTS else gr.update())


def on_line_submit(text, pid, sel):
    """Sửa trong ô 'Dòng đang chọn' (Enter / rời ô): cập nhật bảng + danh sách chờ."""
    p = PROJECTS.get(pid)
    if not p:
        return gr.update(), gr.update(), gr.update(), gr.update()
    i = int(sel or 0)
    new = (text or '').strip()
    if new and new != p['items'][i]['text']:
        p['edits'][i] = new
    else:
        p['edits'].pop(i, None)
    return (gr.update(value=_df_rows(p)), _render_status(p), _render_timeline(p, i),
            _audio_html(p['items'][i]['filename'], pending=i in p['edits']))


def redub_line(text, pid, sel):
    p = PROJECTS.get(pid)
    if not p:
        return (gr.update(),) * 4
    i = int(sel or 0)
    new = (text or '').strip() or p['items'][i]['text']
    try:
        _redub(p, i, new)
    except Exception as e:
        gr.Warning(f'Tạo giọng lỗi: {e}')
        return (gr.update(),) * 4
    return gr.update(value=_df_rows(p)), _render_status(p), _render_timeline(p, i), _audio_html(p['items'][i]['filename'])


def redub_all(pid, sel, progress=gr.Progress()):
    p = PROJECTS.get(pid)
    if not p:
        return (gr.update(),) * 3
    pend = sorted(p['edits'])

    def on_line(k, i, again):
        progress((k, len(pend)), desc=f'{"Thử lại" if again else "Tạo giọng"} dòng {i + 1}…')
    failed = _redub_many(p, pend, on_line)
    if failed:   # gom thành 1 thông báo; các dòng lỗi vẫn ở trạng thái chờ tạo giọng để bấm lại
        gr.Warning(f'Chưa tạo được giọng dòng {", ".join(str(i + 1) for i in failed)}: '
                   f'{list(failed.values())[-1]}')
    elif pend:
        gr.Info(f'Đã tạo giọng {len(pend)} dòng.')
    return gr.update(value=_df_rows(p)), _render_status(p), _render_timeline(p, int(sel or 0))


def remove_project(pid):
    """Nút 'Xoá video này': bỏ dự án, xoá mọi file liên quan (tạm, làm việc, đã xuất) rồi về bảng điều khiển."""
    p = PROJECTS.get(pid)
    left = drop_project(pid)
    if p and left:
        gr.Warning(f'Đã bỏ "{p.get("name") or _short_title(p["title"], 40)}", nhưng {len(left)} file đang được mở '
                   f'nên chưa xoá được: {Path(left[0]).parent}')
    elif p:
        gr.Info(f'Đã xoá "{p.get("name") or _short_title(p["title"], 40)}" cùng file tạm và file đã xuất của nó.')
    choices = _proj_choices()
    nxt = choices[0][1] if choices else None
    return (gr.update(choices=choices, value=nxt), *_nav('dash'))


def editor_preview(pid, sel, subtitle_name, mode, top, size, sub_size, sub_pos, sub_follow, sub_box, *wm_vals):
    """Ảnh xem trước: khung hình tại dòng đang chọn + đúng câu tiếng Việt của dòng đó."""
    from PIL import Image
    p = PROJECTS.get(pid)
    if not p:
        return gr.update(), ''
    i = max(0, min(int(sel or 0), len(p['items']) - 1)) if p['items'] else 0
    t = (p['items'][i]['start_time'] + p['items'][i]['end_time']) / 2000 if p['items'] else p['dur'] * 0.3
    try:
        img = Image.fromarray(vi_dub_erase.grab_frame(p['src'], t))
    except Exception:
        img = Image.new('RGB', (720, 1280), (40, 44, 60))
    opts = _sub_place({'cover_mode': mode, 'cover_top': top, 'cover_size': size, 'sub_size': sub_size,
                       'sub_pos': sub_pos, 'sub_follow': sub_follow, 'sub_box': sub_box,
                       'ai_erase': p['opts']['ai_erase']}, p.get('erase_band'))
    img, note = _draw_sub_preview(img, mode, top, size, opts, subtitle_name, _line_text(p, i) if p['items'] else '')
    wm = _wm_opts(*wm_vals) if wm_vals else {'items': []}
    img = _wm_draw(img, wm)
    if wm['items']:
        note += f' · đóng dấu: {_wm_summary(wm)}'
    return img, f'<div class="hint">Khung hình tại dòng {i + 1} ({_ms(t * 1000)}){html.escape(note)}</div>'


def auto_find_band(pid):
    """Nút 'Tự tìm vị trí sub gốc' (cho Che sub gốc): quét video của dự án, đặt lại 2 thanh trượt."""
    p = PROJECTS.get(pid)
    if not p:
        return gr.update(), gr.update(), ''
    try:
        top, size, info = vi_dub_erase.scan_subtitle_band(p['src'])
    except Exception as e:
        gr.Warning(str(e))
        return gr.update(), gr.update(), f'<div class="hint">❌ {html.escape(str(e))}</div>'
    if top is None:
        return gr.update(), gr.update(), f'<div class="hint">⚠️ {html.escape(info)}</div>'
    return top, size, f'<div class="hint">✅ {html.escape(info)}</div>'


# ---------------------------------------------------------------------------
# Pha 2: xuất video (căn khớp + ghép) từ chữ đã sửa
# ---------------------------------------------------------------------------
DEFAULT_OUT_DIR = (Path(ROOT_DIR) / 'output').as_posix()
_APP = None  # Blocks đang chạy, để thêm thư mục xuất vào allowed_paths


def _out_root(out_dir) -> Path:
    p = Path((out_dir or '').strip().strip('"') or DEFAULT_OUT_DIR).expanduser()
    if not p.is_absolute():
        p = Path(ROOT_DIR) / p
    return p


def _allow_path(path: Path):
    """Gradio chỉ phát file trong allowed_paths; danh sách này được đọc lại mỗi lần nên thêm lúc chạy được."""
    if _APP is not None and str(path) not in _APP.allowed_paths:
        _APP.allowed_paths.append(str(path))


def export_project(pid, sub, out_root: Path, log, wm=None):
    """Generator: yield (ghi chú tiến độ); trả về đường dẫn mp4 kết quả qua StopIteration.value."""
    p = PROJECTS[pid]
    trk, cache = p['trk'], p['trk'].cfg.cache_folder
    if p['edits']:
        yield f'Tạo giọng {len(p["edits"])} dòng đã sửa…'
        failed = _redub_many(p, sorted(p['edits']))
        if failed:
            raise RuntimeError(f'chưa tạo được giọng dòng {", ".join(str(i + 1) for i in failed)}: '
                               f'{list(failed.values())[-1]}')
    app_cfg.exit_soft = False
    app_cfg.current_status = 'ing'
    app_cfg.exec_mode = 'cli'
    app_cfg.rm_uuid(trk.uuid)
    trk.queue_tts = copy.deepcopy(p['items'])
    trk.cfg.subtitle_type = SUBTITLE_TYPES[sub['subtitle_name']]
    # video không tiếng: bắt đầu từ bản sạch của prepare (lần xuất trước có thể đã che / làm chậm nó)
    shutil.copy2(f'{cache}/novoice.orig.mp4', trk.cfg.novoice_mp4)
    if sub['cover_mode'] != 'Không che':
        yield f'{sub["cover_mode"]} vùng sub gốc…'
        covered = None
        for _, out_path in _cover_band(p['src'], sub['cover_mode'], sub['cover_top'], sub['cover_size']):
            covered = out_path or covered
        r = subprocess.run(['ffmpeg', '-y', '-nostdin', '-i', covered, '-an', '-c:v', 'copy', trk.cfg.novoice_mp4],
                           capture_output=True, text=True, encoding='utf-8', errors='ignore')
        if r.returncode != 0:
            raise RuntimeError(f'Tạo video đã che lỗi: {r.stderr[-300:]}')
    style_opts = _sub_place(dict(sub, ai_erase=p['opts']['ai_erase']), p.get('erase_band'))
    old_ass = ASS_JSON.read_text(encoding='utf-8') if ASS_JSON.exists() else None
    ASS_JSON.write_text(json.dumps(_ass_style(style_opts), indent=4, ensure_ascii=False), encoding='utf-8')
    try:
        yield 'Căn khớp giọng với hình…'
        trk.align()
        yield 'Ghép video, giọng và phụ đề…'
        trk.assembling()   # không gọi task_done(): nó xoá cache -> không sửa / xuất lại được
    finally:
        if old_ass is None:
            ASS_JSON.unlink(missing_ok=True)
        else:
            ASS_JSON.write_text(old_ass, encoding='utf-8')
    mp4s = sorted(Path(trk.cfg.target_dir).glob('*.mp4'), key=lambda f: f.stat().st_mtime)
    if not mp4s:
        raise RuntimeError(f'Không thấy video kết quả trong {trk.cfg.target_dir}')
    dest, base = _unique_dest(out_root, safe_name(p.get('name'), fallback=p['nospace']), p)
    dest.mkdir(parents=True, exist_ok=True)
    final = dest / f'{base}.mp4'
    if wm and wm['items']:
        yield f'Đóng dấu {_wm_summary(wm)}…'
        stamp_video(mp4s[-1].as_posix(), final.as_posix(), wm)
    else:
        shutil.copy2(mp4s[-1], final)
    exported = [final.as_posix()]
    for srt in Path(trk.cfg.target_dir).glob('*.srt'):   # vi.srt, zh-cn.srt -> <tên>.vi.srt, <tên>.zh-cn.srt
        tgt = dest / f'{base}.{srt.name}'
        shutil.copy2(srt, tgt)
        exported.append(tgt.as_posix())
    p['result'] = final.as_posix()
    p['exported'] = sorted(set((p.get('exported') or []) + exported))   # để 'Xoá video này' dọn được
    log(f'✅ {p["title"][:50]} → {final.as_posix()}')
    return final.as_posix()


def _unique_dest(out_root: Path, name: str, p) -> tuple:
    """Thư mục + tên file xuất: <thư mục xuất>/<tên>/<tên>.mp4. Trùng với file của video khác thì thêm (2), (3)…;
    file do chính video này xuất lần trước thì ghi đè."""
    # lần xuất trước của chính video này với cùng tên (kể cả bản "(k)") trong cùng thư mục xuất -> ghi đè
    pat = re.compile(rf'{re.escape(name)}( \(\d+\))?', re.I)
    for f in p.get('exported') or []:
        f = Path(f)
        if (f.suffix.lower() == '.mp4' and f.exists() and pat.fullmatch(f.stem) and f.parent.name == f.stem
                and str(f.parent.parent.resolve()).lower() == str(out_root.resolve()).lower()):
            return f.parent, f.stem
    for k in range(1, 1000):
        base = name if k == 1 else f'{name} ({k})'
        if not (out_root / base / f'{base}.mp4').exists():
            return out_root / base, base
    raise RuntimeError(f'Quá nhiều video trùng tên "{name}" trong {out_root}')


def _export(pids, cur_pid, subtitle_name, mode, top, size, sub_size, sub_pos, sub_follow, sub_box, out_dir, *wm_vals):
    wm = _wm_opts(*wm_vals) if wm_vals else {'items': [], 'margin': 0}
    sub = {'subtitle_name': subtitle_name, 'cover_mode': mode, 'cover_top': float(top), 'cover_size': float(size),
           'sub_size': int(sub_size), 'sub_pos': float(sub_pos), 'sub_follow': bool(sub_follow),
           'sub_box': bool(sub_box)}
    out_root = _out_root(out_dir)
    try:
        out_root.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        gr.Warning(f'Không tạo được thư mục xuất {out_root}: {e}')
        return
    _allow_path(out_root)
    lines, results = [], []
    busy = gr.update(value='⏳ Đang xuất…', interactive=False)

    def log(msg):
        lines.append(f"[{time.strftime('%H:%M:%S')}] {msg}")

    def emit(btn=busy, done=False):
        p = PROJECTS.get(cur_pid)
        # video kết quả hiện trong tab 'Kết quả' bên trái, tự chuyển tab khi xuất xong
        res_video, res_info = _result_html(p) if done else (gr.update(), gr.update())
        tabs = gr.update(selected='result') if done and p and p.get('result') else gr.update()
        return ('<div class="explog">' + '<br>'.join(html.escape(l) for l in lines) + '</div>',
                res_video, res_info, tabs, btn, btn)

    pids = [pid for pid in pids if pid in PROJECTS]
    pname = lambda pid: PROJECTS[pid].get('name') or _short_title(PROJECTS[pid]['title'], 40)
    total = len(pids)
    # báo ngay khi bấm: xuất video dài mất vài phút, không có thông báo người dùng tưởng lỗi rồi bấm lại
    gr.Info(f'Đang xuất {total} video… Xong sẽ báo ở đây.' if total > 1 else
            f'Đang xuất "{pname(pids[0])}"… Xong sẽ báo ở đây.' if pids else 'Không có video nào để xuất.', duration=6)
    yield emit()
    failed = {}   # pid -> lỗi

    def run_one(pid, tag):
        gen = export_project(pid, sub, out_root, log, wm)
        while True:
            try:
                note = next(gen)
            except StopIteration as st:
                return st.value
            log(f'{tag} {note}'.strip())
            yield None

    for attempt in (1, 2):   # lượt 2: thử lại các video lỗi ở lượt 1 (lỗi tạm thời: file đang bị khoá, mạng TTS…)
        todo = pids if attempt == 1 else list(failed)
        if attempt == 2 and todo:
            log(f'Thử lại {len(todo)} video bị lỗi…')
        for n, pid in enumerate(todo, 1):
            tag = (f'[{n}/{len(todo)}]' if len(todo) > 1 else '') + (' (thử lại)' if attempt == 2 else '')
            try:
                gen = run_one(pid, tag)
                while True:
                    try:
                        next(gen)
                    except StopIteration as st:
                        results.append(st.value)
                        failed.pop(pid, None)
                        break
                    yield emit()
            except Exception as e:
                failed[pid] = e
                log(f'{tag} ❌ {pname(pid)}: {e}')
                # ghi cả traceback vào logs/ để tra cứu sau (khung log trên giao diện không được lưu lại)
                logger.exception(f'[VieDub] Xuất video lỗi ({pname(pid)}, lượt {attempt}): {e}')
            yield emit()
    done_names = [pname(pid) for pid in pids if pid not in failed]
    if failed and done_names:
        gr.Warning(f'Đã xuất {len(done_names)}/{total} video. Chưa xuất được: '
                   + '; '.join(f'{pname(pid)} ({str(e)[:120]})' for pid, e in failed.items()), duration=20)
    elif failed:
        gr.Warning('Xuất video lỗi: ' + '; '.join(f'{pname(pid)} ({str(e)[:160]})' for pid, e in failed.items()),
                   duration=20)
    elif done_names:
        gr.Info(f'✅ Đã xuất xong {len(done_names)} video: {", ".join(done_names)}. Xem thử ở tab Kết quả bên trái.'
                if len(done_names) > 1 else f'✅ Đã xuất xong "{done_names[0]}". Xem thử ở tab Kết quả bên trái.',
                duration=12)
    idle = gr.update(value='🎬 Xuất video này', interactive=True)
    out = emit(idle, done=True)
    yield out[:5] + (gr.update(value='Xuất tất cả', interactive=True),)


def _apply_name(pid, name):
    """Ô 'Tên video' trong khung Xuất: đổi tên dự án (để trống thì giữ tên cũ)."""
    p = PROJECTS.get(pid)
    if p and (name or '').strip():
        p['name'] = safe_name(name, fallback=p.get('name') or p['nospace'])


def export_current(pid, name, *args):
    if not pid or pid not in PROJECTS:
        gr.Warning('Chưa có video nào trong trình sửa.')
        return
    _apply_name(pid, name)
    yield from _export([pid], pid, *args)


def export_all(pid, name, *args):
    _apply_name(pid, name)
    yield from _export(list(PROJECTS), pid, *args)


def rename_project(pid, name):
    """Rời ô tên trong khung Xuất: lưu tên mới, cập nhật danh sách 'Video đang sửa', trả lại tên đã làm sạch."""
    _apply_name(pid, name)
    p = PROJECTS.get(pid)
    if not p:
        return gr.update(), gr.update()
    return gr.update(choices=_proj_choices(), value=pid), gr.update(value=p['name'])


# ---------------------------------------------------------------------------
# Dọn dẹp: xoá log + mọi file sinh ra khi xử lý (không đụng video đã xuất ở thư mục xuất, cookie, model)
# ---------------------------------------------------------------------------
def _dir_size(p: Path) -> int:
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob('*') if f.is_file()) if p.is_dir() else 0


def _fmt_size(n: int) -> str:
    return f'{n / 1073741824:.2f} GB' if n >= 1073741824 else f'{n / 1048576:.0f} MB'


def _clean_groups(include_downloads: bool) -> list:
    """[(tên nhóm, [đường dẫn])]. tmp/<pid> là cache từng tiến trình (kể cả tiến trình này: dự án đang sửa)."""
    from gradio.utils import get_upload_folder
    tmp_root = Path(TEMP_DIR).parent
    tmp_misc = [p for p in tmp_root.iterdir() if p.is_file()] if tmp_root.is_dir() else []
    tmp_pids = [p for p in tmp_root.iterdir() if p.is_dir() and p.name.isdigit()] if tmp_root.is_dir() else []
    groups = [
        ('Cache xử lý (tmp: âm thanh, nhận dạng, giọng đã tạo)',
         tmp_pids + [tmp_root / 'dubbing_cache', tmp_root / 'translate_cache', PREVIEW_DIR] + tmp_misc),
        ('Thư mục làm việc (output/_work: srt, video ghép)', [Path(WORK_DIR)]),
        ('Video đã xoá / che sub gốc (downloads/_covered)', [COVER_DIR]),
        ('File tải lên trình duyệt (bản copy của Gradio)', [Path(get_upload_folder())]),
        ('Nhật ký (logs)', [Path(ROOT_DIR) / 'logs']),
    ]
    if include_downloads:
        groups.append(('Video đã tải về từ link (downloads/*.mp4)',
                       [p for p in DOWNLOAD_DIR.glob('*') if p.is_file() and p.name != COOKIE_PATH.name]))
    return [(name, [p for p in paths if p.exists()]) for name, paths in groups]


def clean_report(include_downloads):
    rows, total = [], 0
    for name, paths in _clean_groups(bool(include_downloads)):
        n = sum(_dir_size(p) for p in paths)
        total += n
        rows.append(f'<tr><td>{html.escape(name)}</td><td class="num">{_fmt_size(n)}</td></tr>')
    note = (f'<div class="warn">⚠️ Có {len(PROJECTS)} video đang trong trình sửa, dọn dẹp sẽ bỏ luôn (chưa xuất thì '
            f'phải lồng tiếng lại).</div>' if PROJECTS else '')
    return (f'<div class="clean-box"><table class="clean-tbl">{"".join(rows)}'
            f'<tr class="tot"><td>Tổng sẽ xoá</td><td class="num">{_fmt_size(total)}</td></tr></table>{note}'
            f'<div class="hint">Không đụng tới video đã xuất ở thư mục xuất, cookie và model AI.</div></div>')


def do_cleanup(include_downloads, link_items):
    """Xoá thật. Log hôm nay đang được ghi (file đang mở) nên chỉ làm rỗng; tạo lại các thư mục pipeline cần."""
    for pid in list(PROJECTS):
        drop_project(pid)
    freed, errors = 0, []
    for _, paths in _clean_groups(bool(include_downloads)):
        for p in paths:
            try:
                freed += _dir_size(p)
                if p.is_dir():
                    for f in sorted(p.rglob('*'), reverse=True):
                        try:
                            f.unlink() if f.is_file() else f.rmdir()
                        except OSError:
                            if f.is_file() and f.suffix == '.log':
                                open(f, 'w').close()   # file log đang mở: làm rỗng
                            else:
                                raise
                    if p.exists() and not any(p.iterdir()):
                        p.rmdir()
                else:
                    p.unlink()
            except OSError as e:
                errors.append(f'{p.name}: {e}')
    for d in (Path(TEMP_DIR), Path(TEMP_DIR).parent / 'translate_cache', Path(ROOT_DIR) / 'logs', Path(WORK_DIR)):
        d.mkdir(parents=True, exist_ok=True)
    msg = f'Đã dọn {_fmt_size(freed)}.' + (f' Không xoá được {len(errors)} mục (đang được dùng).' if errors else '')
    gr.Info(msg)
    items = _all_items(link_items, [])   # file tải lên đã xoá -> bỏ khỏi danh sách; link giữ nguyên
    report = clean_report(include_downloads).replace(
        '<div class="clean-box">', f'<div class="clean-box"><div class="clean-ok">✅ {html.escape(msg)}</div>', 1)
    return ([], _selector_update(items, []), gr.update(choices=[], value=None), report,
            gr.update(value=_edit_label()))


def open_output_dir(out_dir):
    out = _out_root(out_dir)
    try:
        out.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        gr.Warning(f'Không mở được {out}: {e}')
        return
    if hasattr(os, 'startfile'):
        os.startfile(out)


def pick_out_dir(current):
    """Mở hộp chọn thư mục của Windows (chạy trên máy đang chạy app, tức máy của bạn)."""
    import tkinter
    from tkinter import filedialog
    root = tkinter.Tk()
    root.withdraw()
    root.attributes('-topmost', True)  # hiện lên trên trình duyệt
    try:
        start = _out_root(current)
        chosen = filedialog.askdirectory(parent=root, title='Chọn thư mục xuất video',
                                         initialdir=str(start if start.exists() else ROOT_DIR))
    finally:
        root.destroy()
    return Path(chosen).as_posix() if chosen else gr.update()


STOP_LABEL = '⏹ Dừng ngay'


def request_stop():
    """Dừng ngay: video đang làm dở bị huỷ (xoá file tạm), các video đã xong vẫn giữ trong trình sửa."""
    STOP.set()
    if CURRENT['uuid']:
        app_cfg.stoped_uuid_set.add(CURRENT['uuid'])   # Whisper / dịch / TTS kiểm tra cờ này và thoát sớm
    gr.Info('Đang dừng… video đang làm dở sẽ bị huỷ, video đã xong vẫn giữ.')
    return gr.update(interactive=False, value='⏹ Đang dừng…')


def _run_label(selected, mode='dub', stamping=False) -> str:
    n = len(selected or [])
    if mode == 'dl':
        act = '⬇️ Tải & đóng dấu' if stamping else '⬇️ Tải'
        return f'{act} {n} video' if n else f'{act} video'
    return f'🚀 Lồng tiếng {n} video' if n else '🚀 Lồng tiếng ngay'


# ---------------------------------------------------------------------------
# Lưu cài đặt trên trình duyệt (localStorage)
# ---------------------------------------------------------------------------
SETTINGS_KEY = 'viedub-settings-v1'
# Khoá cố định: mặc định Gradio sinh khoá ngẫu nhiên mỗi lần chạy -> mở lại app sẽ không đọc được cài đặt cũ
SETTINGS_SECRET = 'viedub-local-settings'
SETTING_NAMES = ['source', 'voice', 'subtitle', 'keep_bgm', 'voice_autorate', 'video_autorate', 'model', 'cuda',
                 'browser', 'cover_mode', 'cover_top', 'cover_size', 'sub_size', 'sub_pos', 'sub_follow', 'sub_box',
                 'out_dir', 'ai_erase',
                 'dl_out_dir', 'wm_logo_on', 'wm_logo_pos', 'wm_logo_size', 'wm_logo_opacity', 'wm_text_on', 'wm_text',
                 'wm_color', 'wm_text_pos', 'wm_text_size', 'wm_text_opacity', 'wm_margin', 'plat']


def _settings_defaults(roles, has_cuda) -> dict:
    return {
        'source': 'Tiếng Trung', 'voice': roles[0], 'subtitle': 'Sub cứng', 'keep_bgm': True,
        'voice_autorate': True, 'video_autorate': False, 'model': MODELS[0], 'cuda': has_cuda,
        'browser': 'Không dùng', 'cover_mode': 'Không che', 'cover_top': 68, 'cover_size': 14,
        'sub_size': 16, 'sub_pos': 5, 'sub_follow': True, 'sub_box': False, 'out_dir': DEFAULT_OUT_DIR,
        'ai_erase': True,
        'dl_out_dir': DEFAULT_OUT_DIR, 'wm_logo_on': False, 'wm_logo_pos': '↖', 'wm_logo_size': 18,
        'wm_logo_opacity': 100, 'wm_text_on': False, 'wm_text': '', 'wm_color': '#FFFFFF', 'wm_text_pos': '↘',
        'wm_text_size': 5, 'wm_text_opacity': 85, 'wm_margin': 3, 'plat': 'auto',
    }


def load_settings(saved, roles, has_cuda):
    d = _settings_defaults(roles, has_cuda)
    choices = {'source': list(SOURCE_LANGS), 'voice': roles, 'subtitle': list(SUBTITLE_TYPES), 'model': MODELS,
               'browser': BROWSERS, 'cover_mode': COVER_MODES, 'wm_logo_pos': WM_POS, 'wm_text_pos': WM_POS,
               'plat': [k for k in PLAT if PLAT[k][3] != 'soon']}
    saved = dict(saved or {})
    if saved.get('cover_mode') == OLD_ERASE_MODE:  # cài đặt cũ: "Xoá chữ (AI)" từng là 1 cách che
        saved.update(cover_mode='Không che', ai_erase=True)
    if 'wm_kind' in saved and 'wm_logo_on' not in saved:   # cài đặt cũ: chỉ chọn được 1 trong 2 (logo hoặc chữ)
        kind, pos = saved.get('wm_kind'), saved.get('wm_pos', '↘')
        if kind == 'Logo (ảnh)':
            saved.update(wm_logo_on=True, wm_logo_pos=pos, wm_logo_size=saved.get('wm_size', 18))
        elif kind == 'Chữ':
            saved.update(wm_text_on=True, wm_text_pos=pos, wm_text_size=round(saved.get('wm_size', 12) * 0.45, 1))
    for k, v in saved.items():
        if k in d and (k not in choices or v in choices[k]):
            d[k] = v
    if not has_cuda:
        d['cuda'] = False
    if not str(d['out_dir'] or '').strip():
        d['out_dir'] = DEFAULT_OUT_DIR
    if not str(d['dl_out_dir'] or '').strip():
        d['dl_out_dir'] = DEFAULT_OUT_DIR
    return [d[k] for k in SETTING_NAMES]


# ---------------------------------------------------------------------------
# Giao diện
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Khung app: thanh bên, thanh các bước, thẻ thống kê, thẻ video
# ---------------------------------------------------------------------------
PAGES = ('dash', 'editor', 'clean')
STEPS = ['Thêm video', 'Lồng tiếng', 'Sửa câu thoại', 'Xuất video']
NAV_OUT_N = 7   # page_state, 3 trang, 3 nút điều hướng


def _edit_label() -> str:
    n = len(PROJECTS)
    return f'Trình sửa · {n}' if n else 'Trình sửa'


def _nav_cls(me: str, page: str) -> list:
    return ['nav-item', 'active'] if me == page else ['nav-item']


def _page_cls(me: str, page: str) -> list:
    # ẩn bằng class chứ không dùng visible=False: Gradio 6 dựng khối ẩn ở lần cập nhật đầu nhưng chưa hiện ra
    return ['page'] if me == page else ['page', 'lhide']


def _nav(page: str) -> tuple:
    """Chuyển trang: [page_state, trang Lồng tiếng, trang Trình sửa, trang Dọn dẹp, 3 nút điều hướng]."""
    return (page, *[gr.update(elem_classes=_page_cls(pg, page)) for pg in PAGES],
            gr.update(elem_classes=_nav_cls('dash', page)),
            gr.update(value=_edit_label(), elem_classes=_nav_cls('editor', page)),
            gr.update(elem_classes=_nav_cls('clean', page)))


def _nav_keep() -> tuple:
    """Giữ nguyên trang, chỉ cập nhật số video trên nút Trình sửa."""
    k = gr.update()
    return (k, k, k, k, k, gr.update(value=_edit_label()), k)


def _stepper(active: int, mode: str = 'dub') -> str:
    parts = []
    for k, name in enumerate(STEPS if mode == 'dub' else STEPS_DL, 1):
        cls = 'done' if k < active else ('cur' if k == active else '')
        parts.append(f'<div class="st {cls}"><i>{"✓" if k < active else k}</i><span>{name}</span></div>')
    return '<div class="stepper">' + '<b class="st-line"></b>'.join(parts) + '</div>'


def _stepper_for(page, pid, mode='dub'):
    if page == 'editor':
        p = PROJECTS.get(pid)
        return _stepper(4 if p and p.get('result') else 3)
    if page == 'dash':
        return _stepper(3 if mode == 'dl' and DL_STATE['done'] else 1, mode)
    return gr.update()


def _page_head(icon: str, title: str, sub: str) -> str:
    return f'<div class="page-head"><h1><span class="ph-ico">{icon}</span>{title}</h1><p>{sub}</p></div>'


def _ptitle(icon: str, title: str, sub: str = '', tone: str = 'vio') -> str:
    small = f'<small>{sub}</small>' if sub else ''
    return f'<div class="ptitle"><span class="pico t-{tone}">{icon}</span><div><b>{title}</b>{small}</div></div>'


def _stat(tone, icon, label, value, sub, sub_cls='') -> str:
    return (f'<div class="stat"><span class="sico t-{tone}">{icon}</span><div class="slbl">{label}</div>'
            f'<div class="sval">{value}</div><div class="ssub {sub_cls}">{sub}</div></div>')


def stats_html(lis, files, selected, mode='dub') -> str:
    items = _all_items(lis, files)
    n = len(items)
    links = sum(it['kind'] == 'link' for it in items)
    ok = sum(it['status'] == 'ok' for it in items)
    bad = sum(it['status'] == 'error' for it in items)
    sel = len(_selector_update(items, selected or []))
    ed = len(PROJECTS)
    ex = sum(1 for p in PROJECTS.values() if p.get('result')) + len(DOWNLOADS)
    return '<div class="stats">' + ''.join([
        _stat('vio', '🎬', 'Video đã thêm', n, f'{links} link · {n - links} file' if n else 'chưa có video nào'),
        _stat('grn', '✅', 'Sẵn sàng', ok, f'{bad} link lỗi' if bad else ('dùng được ngay' if ok else '—'),
              'bad' if bad else ('ok' if ok else '')),
        _stat('pink', '☑️', 'Đã chọn', sel, ('sẽ lồng tiếng' if mode == 'dub' else 'sẽ tải về') if sel
              else 'tích ô ở mỗi video'),
        _stat('blu', '✏️', 'Trong trình sửa', ed, 'chờ sửa và xuất' if ed else 'chưa có video nào'),
        _stat('org', '🎉', 'Đã xuất / tải', ex, 'video hoàn tất' if ex else 'chưa có video nào', 'ok' if ex else ''),
    ]) + '</div>'


def runbar_html(mode, selected, source, voice, ai_erase, dl_out_dir, *wm_vals) -> str:
    n = len(selected or [])
    wm = _wm_opts(*wm_vals)
    stamp = ' + '.join('Logo' if it['type'] == 'logo' else f'Chữ "{it["text"][:14]}"' for it in wm['items']) or 'Không'
    if mode == 'dl':
        folder = str(_out_root(dl_out_dir))
        cells = [('vio', '🎬', 'Đã chọn', f'{n} video'),
                 ('pink' if stamp != 'Không' else 'mute', '🏷️', 'Đóng dấu', stamp),
                 ('blu', '📁', 'Lưu vào', folder if len(folder) <= 36 else '…' + folder[-34:])]
    else:
        cells = [('vio', '🎬', 'Đã chọn', f'{n} video'),
                 ('blu', '🌐', 'Ngôn ngữ', f'{(source or "").replace("Tiếng ", "")} → Việt'),
                 ('pink', '🗣️', 'Giọng đọc', (voice or '').split('(')[0]),
                 ('grn' if ai_erase else 'mute', '🤖', 'Xoá chữ AI', 'Bật' if ai_erase else 'Tắt')]
        if wm['items']:
            cells.append(('pink', '🏷️', 'Đóng dấu khi xuất', stamp))
    return '<div class="runsum">' + ''.join(
        f'<div class="rs"><span class="sico t-{t}">{i}</span><div><small>{lbl}</small><b>{html.escape(v)}</b></div></div>'
        for t, i, lbl, v in cells) + '</div>'


def _item_thumb(it: dict) -> str:
    thumb = (f'<img class="vthumb" src="{html.escape(it["thumbnail"])}" referrerpolicy="no-referrer" '
             f'loading="lazy" onerror="this.replaceWith(Object.assign(document.createElement(\'div\'),'
             f'{{className:\'vthumb ph\',textContent:\'▶\'}}))">'
             if it.get('thumbnail') else '<div class="vthumb ph">▶</div>')
    return f'<div class="vth">{thumb}</div>'


def _item_info(it: dict, named: bool) -> str:
    """Dòng thông tin dưới ô tên (video Sẵn sàng) hoặc tiêu đề + lỗi (video chưa dùng được)."""
    cls, text = BADGES[it['status']]
    meta = html.escape(it.get('meta') or (it.get('src') if it['kind'] == 'link' else ''))
    title = it.get('title') or ''
    head = '' if named else f'<div class="vtitle">{html.escape(_short_title(title))}</div>'
    orig = (f'<span class="vorig" title="{html.escape(title[:500])}">· Gốc: {html.escape(_short_title(title, 48))}</span>'
            if named and it['kind'] == 'link' and title else '')
    err = (f'<div class="verr" title="{html.escape(it["error"][:1000])}">{html.escape(it["error"][:300])}</div>'
           if it.get('error') else '')
    plat = (f'<span class="badge b-plat p-{it["platform"]}">{html.escape(_plat_label(it["platform"]))}</span>'
            if it.get('platform') and it['platform'] != 'other' else '')
    return (f'{head}<div class="vmeta"><span class="badge {cls}">{text}</span>{plat}<span class="vm">{meta}</span>{orig}</div>'
            f'{err}')


def _toggle_sel(key: str):
    def _t(selected, on):
        rest = [k for k in (selected or []) if k != key]
        return [*rest, key] if on else rest
    return _t


def _set_name(key: str):
    def _s(names, value):
        return {**(names or {}), key: value}
    return _s


# dừng và bỏ nguồn mọi <video>/<audio> của trình sửa: trình duyệt nhả file để Windows cho xoá / ghi đè
RELEASE_MEDIA_JS = ("() => { document.querySelectorAll('#ed-result, #ed-native, .pnl audio').forEach(m => { "
                    "try { m.pause(); m.removeAttribute('src'); m.load(); } catch (e) {} }); }")

EDITOR_EMPTY = ('<div class="empty big"><div class="empty-ico">✏️</div><b>Chưa có video nào trong trình sửa</b>'
                '<p>Vào trang <b>Lồng tiếng</b>, tích chọn video rồi bấm <b>Lồng tiếng ngay</b>. '
                'Lồng tiếng xong, video sẽ hiện ở đây để bạn sửa từng câu trước khi xuất.</p></div>')


def _icon_css() -> str:
    """Biểu tượng nét (kiểu lucide) cho thanh bên, nhúng thẳng bằng data URI."""
    from urllib.parse import quote
    paths = {
        'nav-dash': "<path d='M13 2 3 14h9l-1 8 10-12h-9l1-8z'/>",
        'nav-edit': ("<path d='M12 3H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7'/>"
                     "<path d='M18.4 2.6a2.1 2.1 0 0 1 3 3L12 15l-4 1 1-4z'/>"),
        'nav-clean': ("<path d='M3 6h18'/><path d='M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6'/>"
                      "<path d='M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2'/><path d='M10 11v6M14 11v6'/>"),
    }
    out = []
    for eid, body in paths.items():
        svg = ("<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' "
               f"stroke-width='2' stroke-linecap='round' stroke-linejoin='round'>{body}</svg>")
        out.append(f'#{eid} {{ --ico: url("data:image/svg+xml,{quote(svg)}"); }}')
    return '\n'.join(out)


HEAD_JS = """
<script>
// Giao diện chỉ có bản tối: ép Gradio vào chế độ tối (thêm ?__theme=dark 1 lần khi mở).
(function () {
  try {
    const u = new URL(location.href);
    if (u.searchParams.get('__theme') !== 'dark') { u.searchParams.set('__theme', 'dark'); location.replace(u.toString()); }
  } catch (e) {}
})();
window.vdSeek = function (t) {
  const v = document.getElementById('ed-native'); if (!v) return;
  try { v.currentTime = t; v.play().catch(() => {}); } catch (e) {}
};
setInterval(function () {
  const v = document.getElementById('ed-native'), ph = document.getElementById('ed-playhead');
  if (!v || !ph || !v.duration) return;
  ph.style.left = (v.currentTime / v.duration * 100) + '%';
}, 200);
</script>
"""

CSS = """
/* ================= nền + khung trang ================= */
:root { color-scheme: dark; }
body, gradio-app {
  background: radial-gradient(circle at 100% 0%, rgba(139,92,246,.18), transparent 38%),
              radial-gradient(900px 560px at 6% -10%, rgba(99,102,241,.12), transparent 55%),
              linear-gradient(135deg, #070711, #111325) fixed !important;
}
/* Gradio đặt overflow:hidden ở đây -> position:sticky (thanh bên, thanh trên, thanh nút cuối) không dính.
   overflow-x:clip vẫn chặn cuộn ngang mà không tạo khung cuộn. */
.gradio-container { width: 100% !important; max-width: none !important; margin: 0 !important; padding: 0 !important;
  background: transparent !important; overflow: visible !important; overflow-x: clip !important; }
.gradio-container > .main { padding: 0 !important; max-width: none !important; width: 100% !important; margin: 0 !important; }
footer { display: none !important; }
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,.12); border-radius: 10px; border: 2px solid transparent;
  background-clip: padding-box; }
::-webkit-scrollbar-track { background: transparent; }

/* khối .form của Gradio có nền = màu viền để vẽ đường kẻ giữa các ô: bỏ đi, các ô cách nhau bằng khoảng trống */
.form { background: transparent !important; border: none !important; box-shadow: none !important; gap: 14px !important;
  overflow: visible !important; }
#shell { gap: 0 !important; flex-wrap: nowrap !important; align-items: stretch !important; min-height: 100vh; }
#main { min-width: 0 !important; padding: 0 26px 26px !important; gap: 18px !important; }
.page { gap: 18px !important; }
.grow { flex: 1 1 auto !important; min-width: 0 !important; }

/* ================= thanh bên ================= */
#side { flex: 0 0 236px !important; width: 236px !important; max-width: 236px !important; min-width: 236px !important;
  position: sticky !important; top: 0; height: 100vh; overflow-y: auto; padding: 18px 14px !important; gap: 6px !important;
  background: rgba(15,16,32,.72) !important; border-right: 1px solid rgba(255,255,255,.06);
  backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px); }
.brand { display: flex; align-items: center; gap: 11px; padding: 4px 6px 20px; }
.brand .logo { width: 40px; height: 40px; border-radius: 12px; display: grid; place-items: center; flex: none;
  font-weight: 800; font-size: 14px; color: #fff; letter-spacing: -.02em;
  background: linear-gradient(135deg, #8b5cf6, #ec4899); box-shadow: 0 0 24px rgba(139,92,246,.45); }
.brand b { font-size: 18px; font-weight: 800; letter-spacing: -.01em; color: #fff; display: block; }
.brand small { display: block; font-size: 11.5px; color: #8a8fa6; font-weight: 500; margin-top: 1px; }
.nav-item { justify-content: flex-start !important; text-align: left !important; gap: 12px !important;
  width: 100% !important; min-height: 44px; padding: 11px 12px !important; border-radius: 11px !important;
  font-size: 14px !important; font-weight: 600 !important; color: #a1a1aa !important; background: transparent !important;
  border: 1px solid transparent !important; box-shadow: none !important; transition: background .15s, color .15s; }
.nav-item:hover { color: #fff !important; background: rgba(255,255,255,.05) !important; }
.nav-item.active { color: #fff !important; border-color: rgba(139,92,246,.4) !important;
  background: linear-gradient(135deg, rgba(139,92,246,.35), rgba(99,102,241,.22)) !important;
  box-shadow: 0 0 30px rgba(139,92,246,.25) !important; }
.nav-item::before { content: ''; width: 18px; height: 18px; flex: none; background: currentColor;
  -webkit-mask: var(--ico) center / contain no-repeat; mask: var(--ico) center / contain no-repeat; }
.side-bottom { margin-top: auto !important; }
.side-card { padding: 14px; border-radius: 14px; background: linear-gradient(135deg, #1c2742, #141d33);
  border: 1px solid rgba(255,255,255,.08); font-size: 12px; color: #a9b0c6; line-height: 1.5; }
.side-card b { display: block; color: #fff; font-size: 13px; margin-bottom: 4px; }
.side-card .gpu { color: #4ade80; font-weight: 600; }
.side-card .gpu.off { color: #fbbf24; }

/* ================= thanh trên: các bước ================= */
#topbar { position: sticky !important; top: 0; z-index: 200; margin: 0 -26px !important; padding: 14px 26px !important;
  width: calc(100% + 52px) !important; max-width: none !important;
  align-items: center !important; flex-wrap: nowrap !important; gap: 12px !important;
  background: rgba(9,10,22,.72) !important; border-bottom: 1px solid rgba(255,255,255,.06);
  backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px); }
.tb-left { flex: 1 1 auto !important; min-width: 0 !important; }
.tb-right-cell { flex: 0 0 auto !important; width: auto !important; }
.stepper { display: inline-flex; align-items: center; gap: 12px; padding: 7px 16px; border-radius: 14px;
  background: rgba(255,255,255,.03); border: 1px solid rgba(255,255,255,.07); max-width: 100%; overflow: hidden; }
.st { display: flex; align-items: center; gap: 8px; color: #8a8fa6; font-size: 13.5px; font-weight: 600; white-space: nowrap; }
.st i { font-style: normal; width: 24px; height: 24px; border-radius: 50%; display: grid; place-items: center;
  font-size: 12px; font-weight: 700; border: 1px solid rgba(255,255,255,.18); color: #cfd3e3; }
.st.cur { color: #fff; }
.st.cur i { background: linear-gradient(135deg, #8b5cf6, #ec4899); border-color: transparent; color: #fff;
  box-shadow: 0 0 16px rgba(236,72,153,.45); }
.st.done { color: #cfd3e3; }
.st.done i { background: rgba(34,197,94,.16); border-color: rgba(34,197,94,.5); color: #4ade80; }
.st-line { width: 26px; height: 1px; background: rgba(255,255,255,.14); flex: none; }
.tb-right { display: flex; justify-content: flex-end; gap: 8px; }
.tb-pill { display: inline-flex; align-items: center; gap: 8px; padding: 7px 13px; border-radius: 999px;
  font-size: 12.5px; font-weight: 600; white-space: nowrap; color: #d6d9e6;
  background: rgba(255,255,255,.04); border: 1px solid rgba(255,255,255,.08); }
.tb-pill .dot { width: 8px; height: 8px; border-radius: 50%; background: #22c55e; box-shadow: 0 0 10px #22c55e; }
.tb-pill .dot.off { background: #f59e0b; box-shadow: 0 0 10px #f59e0b; }

/* ================= tiêu đề trang + thống kê ================= */
.page-head h1 { margin: 4px 0 4px; font-size: 24px; font-weight: 800; letter-spacing: -.02em; color: #fff;
  display: flex; align-items: center; gap: 10px; }
.page-head p { margin: 0; color: #9096ab; font-size: 14px; }
.page-row { align-items: flex-end !important; gap: 12px !important; flex-wrap: nowrap !important; }
.page-row > .grow { flex: 1 1 0% !important; }
.stats { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 14px; }
.stat { padding: 16px; border-radius: 16px; border: 1px solid rgba(255,255,255,.086);
  background: linear-gradient(160deg, rgba(255,255,255,.075), rgba(255,255,255,.03));
  box-shadow: 0 6px 22px rgba(0,0,0,.25), inset 0 1px 0 rgba(255,255,255,.05); }
.sico { width: 34px; height: 34px; border-radius: 10px; display: grid; place-items: center; font-size: 16px; flex: none; }
.stat .sico { margin-bottom: 12px; }
.t-vio { background: rgba(139,92,246,.18); } .t-grn { background: rgba(34,197,94,.16); }
.t-pink { background: rgba(236,72,153,.16); } .t-blu { background: rgba(61,139,255,.16); }
.t-org { background: rgba(245,151,43,.16); } .t-mute { background: rgba(255,255,255,.06); filter: grayscale(1); }
.slbl { font-size: 12px; color: #a1a1aa; }
.sval { font-size: 26px; font-weight: 800; color: #fff; margin-top: 2px; line-height: 1.15; }
.ssub { font-size: 11.5px; color: #7c8197; margin-top: 3px; }
.ssub.ok { color: #22c55e; } .ssub.bad { color: #f0506e; }
@media (max-width: 1300px) { .stats { grid-template-columns: repeat(3, minmax(0, 1fr)); } }

/* ================= bảng (panel) ================= */
.pnl { background: #141627 !important; border: 1px solid rgba(255,255,255,.10) !important; border-radius: 16px !important;
  padding: 16px 18px !important; gap: 14px !important; box-shadow: 0 6px 22px rgba(0,0,0,.22) !important; }
.ptitle { display: flex; align-items: center; gap: 12px; }
.pico { width: 36px; height: 36px; border-radius: 11px; display: grid; place-items: center; font-size: 16px; flex: none; }
.ptitle b { display: block; font-size: 15.5px; font-weight: 750; color: #fff; }
.ptitle small { display: block; font-size: 12.5px; color: #9096ab; margin-top: 1px; }
.phead { align-items: center !important; flex-wrap: nowrap !important; gap: 8px !important; }
.sub-head { font-weight: 700; font-size: 14px; color: #fff; }
.sub-label { font-weight: 600; font-size: 13px; color: #9096ab; margin-top: 2px; }
.hint { font-size: 13px; color: #9096ab; }
.hint b, .hint code { color: #d6d9e6; }
.warn { font-size: 13px; color: #f5972b; font-weight: 600; }

/* ================= tab dạng viên thuốc ================= */
.seg { gap: 14px !important; }
.seg .tab-wrapper { padding: 0 !important; border: none !important; height: auto !important; }
.seg [role=tablist] { display: flex !important; gap: 6px; padding: 5px; border-radius: 13px; width: 100%;
  background: #0d0f1c; border: 1px solid rgba(255,255,255,.08); }
.seg [role=tab] { flex: 1 1 0; justify-content: center !important; border: none !important; border-radius: 10px !important;
  padding: 9px 12px !important; margin: 0 !important; color: #a1a1aa !important; font-weight: 650 !important;
  font-size: 13.5px !important; background: transparent !important; height: auto !important; }
.seg [role=tab]:hover { color: #fff !important; background: rgba(255,255,255,.05) !important; }
.seg [role=tab][aria-selected=true] { color: #fff !important;
  background: linear-gradient(135deg, #8b5cf6, #ec4899) !important; box-shadow: 0 6px 18px rgba(236,72,153,.25); }
.seg [role=tab]::after, .seg [role=tab][aria-selected=true]::after { display: none !important; }
.seg .tabitem, .seg [role=tabpanel] { padding: 14px 0 0 !important; border: none !important; background: transparent !important; }

/* ================= ô nhập link ================= */
.lrows { gap: 8px !important; }
.lhide { display: none !important; }
.lrow { gap: 8px !important; align-items: center !important; flex-wrap: nowrap !important; }
.lrow textarea, .lrow input { font-size: 13.5px !important; }
.rm { min-width: 36px !important; max-width: 40px; height: 36px; padding: 0 !important; }
#drop { border: 1.5px dashed rgba(236,72,153,.45) !important; border-radius: 14px !important;
  background: rgba(236,72,153,.05) !important; }
#drop:hover { border-color: rgba(236,72,153,.75) !important; background: rgba(236,72,153,.08) !important; }
#drop button { width: 100% !important; color: #d6d9e6 !important; }
#drop .icon-wrap { color: #f472b6 !important; }

/* ================= thẻ video (danh sách) ================= */
.vlist { gap: 10px !important; max-height: 560px; overflow: auto; flex-wrap: nowrap !important; padding-right: 4px; }
.vcard { align-items: center !important; flex-wrap: nowrap !important; gap: 12px !important; padding: 10px 12px !important;
  border-radius: 14px !important; background: #10121f !important; border: 1px solid rgba(255,255,255,.08) !important;
  transition: border-color .2s, background .2s; }
.vcard:hover { border-color: rgba(255,255,255,.16) !important; }
.vcard:has(.vchk input:checked) { border-color: rgba(139,92,246,.6) !important;
  background: linear-gradient(135deg, rgba(139,92,246,.14), rgba(16,18,31,.9) 60%) !important; }
.vcard.s-error { border-color: rgba(240,80,110,.35) !important; }
.vcard.s-checking { border-color: rgba(139,92,246,.45) !important; }
.vchk { flex: 0 0 22px !important; min-width: 22px !important; max-width: 22px !important; }
.vchk label { padding: 0 !important; background: transparent !important; border: none !important; }
.vchk input[type=checkbox] { width: 20px !important; height: 20px !important; border-radius: 6px !important; margin: 0 !important; }
.vchk-ph { display: block; width: 22px; }
.vthc { flex: 0 0 auto !important; width: auto !important; min-width: 0 !important; }
.vth { position: relative; }
.vthumb { width: 54px; height: 72px; border-radius: 10px; object-fit: cover; background: #0b0c18; display: block; }
.vthumb.ph { display: grid; place-items: center; color: #6b7088; font-size: 18px; }
.vmain { gap: 6px !important; min-width: 0 !important; }
.vmain > * { min-width: 0 !important; }
.vname input, .vname textarea { font-weight: 650 !important; font-size: 14px !important; padding: 8px 11px !important; }
.vinfo { min-width: 0; }
.vtitle { font-weight: 650; font-size: 14px; line-height: 1.35; color: #fff; word-break: break-word;
  display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }
.vtitle.one { -webkit-line-clamp: 1; }
.vmeta { display: flex; align-items: center; gap: 8px; font-size: 12.5px; color: #8a8fa6; margin-top: 3px; min-width: 0;
  white-space: nowrap; overflow: hidden; }
.vm, .vorig { overflow: hidden; text-overflow: ellipsis; min-width: 0; }
.vorig { color: #6b7088; flex: 1 1 auto; }
.verr { font-size: 12px; color: #f0506e; margin-top: 4px; word-break: break-word; display: -webkit-box;
  -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; }
.badge { font-size: 11.5px; padding: 3px 9px; border-radius: 999px; font-weight: 650; white-space: nowrap; flex: none; }
.b-ok { color: #4ade80; background: rgba(34,197,94,.14); }
.b-err { color: #ff7a93; background: rgba(240,80,110,.14); }
.b-info { color: #c4b5fd; background: rgba(139,92,246,.18); }
.b-mute { color: #a1a1aa; background: rgba(255,255,255,.06); }

/* ================= công tắc (checkbox kiểu toggle) ================= */
.sw label { gap: 10px !important; align-items: center !important; background: transparent !important; border: none !important;
  padding: 2px 0 !important; }
.sw input[type=checkbox] { -webkit-appearance: none; appearance: none; flex: none; position: relative; margin: 0 !important;
  width: 38px !important; height: 22px !important; border-radius: 999px !important; cursor: pointer;
  background: #2a2d45 !important; background-image: none !important; border: 1px solid rgba(255,255,255,.14) !important;
  box-shadow: none !important; transition: background .2s, border-color .2s; }
.sw input[type=checkbox]::after { content: ''; position: absolute; top: 2px; left: 2px; width: 16px; height: 16px;
  border-radius: 50%; background: #fff; transition: left .2s; box-shadow: 0 1px 3px rgba(0,0,0,.4); }
.sw input[type=checkbox]:checked { background: #22c55e !important; border-color: #22c55e !important; }
.sw input[type=checkbox]:checked::after { left: 18px; }
/* Gradio gom các ô tick liền nhau vào 1 khối .form -> chia lưới ngay trên khối đó */
.sw-grid > .form { display: grid !important; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px 18px !important; }
.sw-grid > .form > * { min-width: 0 !important; width: 100% !important; margin: 0 !important; justify-self: stretch; }
.ai-card { padding: 14px !important; border-radius: 14px !important; gap: 6px !important;
  background: linear-gradient(135deg, rgba(34,197,94,.08), rgba(16,18,31,.6)) !important;
  border: 1px solid rgba(34,197,94,.28) !important; }
#ai-erase label span { font-weight: 700; font-size: 14px; color: #fff; }
.subcard { border-radius: 14px; padding: 14px !important; gap: 10px !important;
  background: #10121f !important; border: 1px solid rgba(255,255,255,.08) !important; }
.vn-part { gap: 10px !important; border-top: 1px dashed rgba(255,255,255,.10); padding-top: 12px !important; }
.auto-status { align-self: center; }
.chip { font-size: 12.5px; padding: 5px 11px; border-radius: 999px; font-weight: 600; display: inline-block;
  background: rgba(255,255,255,.05); border: 1px solid rgba(255,255,255,.10); }
.chip.ok { color: #4ade80; border-color: rgba(34,197,94,.35); background: rgba(34,197,94,.10); }

/* ================= tiến trình ================= */
.qlist { display: flex; flex-direction: column; gap: 8px; max-height: 360px; overflow: auto; padding-right: 2px; }
.qitem { display: grid; grid-template-columns: 10px 1fr auto; align-items: center; gap: 12px; padding: 10px 12px;
  border-radius: 12px; background: #10121f; border: 1px solid rgba(255,255,255,.08); }
.qitem.q-processing, .qitem.q-downloading, .qitem.q-covering { border-color: rgba(139,92,246,.55); }
.vbody { min-width: 0; }
.qdot { width: 10px; height: 10px; border-radius: 50%; background: rgba(255,255,255,.18); }
.q-done .qdot { background: #22c55e; }
.q-error .qdot { background: #f0506e; }
.q-processing .qdot, .q-downloading .qdot, .q-covering .qdot { background: #8b5cf6;
  box-shadow: 0 0 0 0 rgba(139,92,246,.6); animation: pulse 1.4s infinite; }
@keyframes pulse { 70% { box-shadow: 0 0 0 8px rgba(139,92,246,0); } 100% { box-shadow: 0 0 0 0 rgba(139,92,246,0); } }
.qhead { display: flex; justify-content: space-between; font-size: 14px; margin-bottom: 6px; color: #d6d9e6; }
.pbar { height: 8px; border-radius: 99px; background: #0b0c18; overflow: hidden; margin-bottom: 12px; }
.pbar > div { height: 100%; border-radius: 99px; background: linear-gradient(90deg, #8b5cf6, #ec4899); transition: width .5s; }
.empty { text-align: center; padding: 26px 16px; border-radius: 14px; font-size: 14px; color: #8a8fa6;
  border: 1.5px dashed rgba(255,255,255,.12); background: rgba(255,255,255,.015); }
.empty b { color: #e6e8f2; }
.empty.big { padding: 70px 20px; }
.empty.big p { max-width: 520px; margin: 8px auto 0; line-height: 1.6; }
.empty-ico { font-size: 28px; margin-bottom: 8px; }

/* ================= thanh nút cuối trang ================= */
#runbar { position: sticky !important; bottom: 14px; z-index: 150; align-items: center !important; flex-wrap: nowrap !important;
  gap: 16px !important; padding: 12px 14px 12px 18px !important; border-radius: 18px !important;
  background: rgba(20,22,39,.92) !important; border: 1px solid rgba(255,255,255,.10) !important;
  backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px); box-shadow: 0 18px 40px rgba(0,0,0,.45) !important; }
.runsum { display: flex; gap: 24px; flex-wrap: wrap; row-gap: 10px; }
.rs { display: flex; align-items: center; gap: 10px; min-width: 0; }
.rs small { display: block; font-size: 10.5px; letter-spacing: .06em; text-transform: uppercase; color: #8a8fa6; font-weight: 700; }
.rs b { display: block; font-size: 15px; color: #fff; white-space: nowrap; }
#run-btn, #export-btn { min-height: 54px; font-size: 16px !important; font-weight: 800 !important; letter-spacing: .01em;
  border-radius: 14px !important; border: none !important; color: #fff !important;
  background: linear-gradient(135deg, #ec4899, #db2f86) !important; box-shadow: 0 6px 18px rgba(236,72,153,.35) !important; }
#run-btn:hover, #export-btn:hover { filter: brightness(1.08); }
#run-btn:disabled, #export-btn:disabled { opacity: .6; filter: none; }

/* ================= trình sửa ================= */
.ed-player { display: block; width: 100%; max-height: 520px; background: #000; border-radius: 12px; }
.ed-vlabel { font-size: 13px; color: #9096ab; margin-bottom: 8px; }
#ed-df tbody tr { cursor: pointer; }
#ed-df table { font-size: 13.5px; }
#ed-df table, #ed-df td, #ed-df th, #ed-df td *, #ed-df th * { font-family: inherit !important; }
#ed-df tbody td:nth-child(-n+3), #ed-df tbody td:nth-child(-n+3) * { white-space: nowrap !important; }
#ed-df th { color: #b8bdd0 !important; font-weight: 700 !important; background: #0d0f1c !important; }
#ed-df .cell-wrap { white-space: pre-wrap !important; }
.tl { margin: 10px 0 2px; }
.tl-track { position: relative; height: 44px; border-radius: 10px; background: #0b0c18;
  border: 1px solid rgba(255,255,255,.08); overflow: hidden; }
.tl-seg { position: absolute; top: 8px; height: 28px; border-radius: 5px; cursor: pointer; min-width: 2px;
  background: rgba(34,197,94,.55); border: 1px solid rgba(34,197,94,.8); }
.tl-seg:hover { filter: brightness(1.25); }
.tl-seg.pend { background: rgba(245,151,43,.6); border-color: #f5972b; }
.tl-seg.over { background: rgba(240,80,110,.5); border-color: #f0506e; }
.tl-seg.sel { background: rgba(167,139,250,.85); border-color: #fff; box-shadow: 0 0 0 2px #a78bfa; z-index: 2; }
.tl-ph { position: absolute; top: 0; bottom: 0; width: 2px; background: #ec4899; left: 0; pointer-events: none; z-index: 3; }
.tl-ticks { display: flex; justify-content: space-between; font-size: 11px; color: #7c8197; padding: 3px 2px 0; }
.zh { font-size: 15px; padding: 10px 12px; border-radius: 10px; background: #0b0c18;
  border: 1px solid rgba(255,255,255,.08); min-height: 44px; color: #e6e8f2; }
.explog { font-size: 13px; line-height: 1.55; max-height: 220px; overflow: auto; padding: 10px 12px; border-radius: 10px;
  background: #0b0c18; border: 1px solid rgba(255,255,255,.08); color: #cfd3e3; }
.ed-status { display: flex; }

/* ================= dọn dẹp ================= */
.clean-tbl { width: 100%; border-collapse: collapse; font-size: 14px; border: none !important; }
.clean-tbl tr, .clean-tbl td { border: none !important; background: transparent !important; }
.clean-tbl td { padding: 10px 6px; border-bottom: 1px solid rgba(255,255,255,.06) !important; color: #d6d9e6; }
.clean-tbl td.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; font-weight: 650; }
.clean-tbl tr.tot td { font-weight: 800; border-bottom: none; color: #fff; font-size: 15px; }
.clean-box { display: flex; flex-direction: column; gap: 10px; }
.clean-ok { padding: 10px 12px; border-radius: 10px; color: #4ade80; background: rgba(34,197,94,.10);
  border: 1px solid rgba(34,197,94,.3); font-weight: 600; }

/* ================= chế độ / nền tảng / đóng dấu ================= */
.modebar { align-items: center !important; flex-wrap: wrap !important; gap: 10px !important; padding: 10px 14px !important;
  border-radius: 14px; background: rgba(255,255,255,.03); border: 1px solid rgba(255,255,255,.07); }
.mode-lblc { flex: 0 0 auto !important; width: auto !important; min-width: 0 !important; }
.mode-lbl { font-size: 13px; font-weight: 700; color: #a1a1aa; padding: 0 4px; }
.mode-hint { padding-left: 6px; }
/* flex: 0 0 auto: Gradio đặt flex-basis 0% + min-width 0 cho nút scale=0 -> chip co lại còn phần đệm */
.chipbtn { flex: 0 0 auto !important; width: auto !important; min-width: 0 !important;
  border-radius: 999px !important; padding: 8px 16px !important; font-weight: 650 !important; font-size: 13.5px !important;
  color: #cfd3e3 !important; background: rgba(255,255,255,.05) !important; border: 1px solid rgba(255,255,255,.12) !important;
  box-shadow: none !important; min-height: 36px; }
.chipbtn:hover { color: #fff !important; background: rgba(255,255,255,.09) !important; }
.chipbtn.active { color: #fff !important; border-color: transparent !important;
  background: linear-gradient(135deg, #8b5cf6, #ec4899) !important; box-shadow: 0 6px 18px rgba(236,72,153,.25) !important; }
.chipbtn.soon { opacity: .45; cursor: not-allowed !important; }
.chipbtn.soon::after { content: ' · sắp có'; font-weight: 500; font-size: 11.5px; }
.platrow { flex-wrap: wrap !important; gap: 8px !important; }
/* chế độ chỉ tải: ẩn hàng tab Link / File, chỉ còn phần dán link */
.notabs .tab-wrapper, .notabs [role=tablist] { display: none !important; }
.notabs .tabitem, .notabs [role=tabpanel] { padding-top: 0 !important; }
.wm-sub { padding: 12px 14px !important; gap: 10px !important; border-radius: 14px !important;
  background: #10121f !important; border: 1px solid rgba(255,255,255,.08) !important; }
.wm-note.on { color: #f9a8d4; }
.plat { padding: 6px 12px 6px 8px !important; min-height: 32px; font-size: 13px !important; gap: 8px !important; }
.plat::before { content: attr(data-l); width: 20px; height: 20px; border-radius: 6px; display: inline-grid; place-items: center;
  font-size: 11px; font-weight: 800; color: #fff; flex: none; }
#plat-auto::before { content: '✦'; background: #8b5cf6; }
#plat-douyin::before { content: 'D'; background: #fe2c55; }
#plat-tiktok::before { content: 'T'; background: #000; box-shadow: 0 0 0 1px rgba(255,255,255,.35) inset; }
#plat-youtube::before { content: '▶'; background: #ff0000; font-size: 9px; }
#plat-bilibili::before { content: 'B'; background: #00a1d6; }
#plat-facebook::before { content: 'f'; background: #1877f2; }
#plat-x::before { content: 'X'; background: #fff; color: #000; }
#plat-instagram::before { content: 'I'; background: linear-gradient(135deg, #f58529, #dd2a7b 55%, #8134af); }
#plat-kuaishou::before { content: 'K'; background: #ff5000; }
#plat-xiaohongshu::before { content: '红'; background: #ff2442; font-size: 10px; }
.badge.b-plat { color: #c4b5fd; background: rgba(139,92,246,.14); }
/* radio dạng chip (Đóng dấu) và lưới 3×3 (Vị trí) */
.chips .wrap, .posgrid .wrap { gap: 6px !important; background: transparent !important; border: none !important; }
.chips label, .posgrid label { border-radius: 999px !important; padding: 7px 14px !important; cursor: pointer;
  background: rgba(255,255,255,.05) !important; border: 1px solid rgba(255,255,255,.12) !important; color: #cfd3e3 !important;
  font-weight: 600 !important; box-shadow: none !important; }
.chips label:has(input:checked), .posgrid label:has(input:checked) { color: #fff !important; border-color: transparent !important;
  background: linear-gradient(135deg, #8b5cf6, #ec4899) !important; }
.chips input[type=radio], .posgrid input[type=radio] { display: none !important; }
.posgrid .wrap { display: grid !important; grid-template-columns: repeat(3, 46px); }
.posgrid label { width: 46px; height: 40px; justify-content: center !important; padding: 0 !important; border-radius: 10px !important;
  font-size: 16px !important; }
.posgrid label span { padding: 0 !important; }

/* ================= chi tiết nhỏ ================= */
button { white-space: nowrap !important; text-overflow: ellipsis; }
button, [role=tab], [role=option], select, summary, .tl-seg, input[type=checkbox], input[type=radio], input[type=range],
label:has(> input[type=checkbox]), label:has(> input[type=radio]), .wrap-inner, .wrap-inner input,
[data-testid=dropdown] input, .icon-wrap, .dropdown-arrow, #drop, #drop * { cursor: pointer !important; }
button:disabled, input:disabled, label:has(> input:disabled) { cursor: not-allowed !important; }
"""


def _clean_stale_tmp():
    """pyVideoTrans tạo tmp/<pid> cho mỗi tiến trình và chỉ bản desktop tự dọn khi thoát. App web chạy nhiều lần
    sẽ để lại hàng trăm thư mục (có cái hàng GB) -> xoá thư mục của các tiến trình không còn chạy."""
    try:
        import psutil
    except ImportError:
        return
    root = Path(TEMP_DIR).parent
    for d in root.iterdir() if root.is_dir() else []:
        if d.is_dir() and d.name.isdigit() and int(d.name) != os.getpid() and not psutil.pid_exists(int(d.name)):
            shutil.rmtree(d, ignore_errors=True)


def _gpu_name() -> str:
    try:
        import torch
        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''
    except Exception:
        return ''


def _sub_controls(subtitle_name, mode, follow, pid):
    """Khối tuỳ chọn phụ đề chỉ hiện khi chèn sub cứng; thanh vị trí bị khoá khi sub đặt vào chỗ sub gốc."""
    hard = _is_hard_sub(subtitle_name)
    erase = bool(PROJECTS.get(pid, {}).get('opts', {}).get('ai_erase'))
    note = '' if hard else (
        '<div class="hint">Đang chọn <b>Không chèn sub</b>: video xuất ra chỉ có giọng lồng tiếng.</div>'
        if SUBTITLE_TYPES.get(subtitle_name) == 0 else
        '<div class="hint">Đang chọn <b>Sub mềm</b>: chữ và vị trí do trình phát video quyết định.</div>')
    return (gr.update(visible=hard), gr.update(interactive=not (_region_active(mode, erase) and follow)), note)


def build_ui():
    global _APP
    roles = _vi_roles()
    has_cuda = _cuda_available()
    gpu = _gpu_name()
    gpu_short = gpu.replace('NVIDIA ', '').replace('GeForce ', '') if gpu else ''
    dflt = _settings_defaults(roles, has_cuda)

    with gr.Blocks(title='VieDub · Lồng tiếng Việt') as app:
        _APP = app
        link_items = gr.State([])
        files_state = gr.State([])
        n_links = gr.State(1)
        cookie_state = gr.State(None)
        sel_i = gr.State(0)
        selector = gr.State([])        # key các video đang được tích chọn
        names_state = gr.State({})     # key -> tên người dùng đặt
        page_state = gr.State('dash')
        mode_state = gr.State('dub')       # 'dub' = lồng tiếng, 'dl' = chỉ tải video
        plat_state = gr.State('auto')      # nền tảng đang chọn ở tab Link
        saved = gr.BrowserState({}, storage_key=SETTINGS_KEY, secret=SETTINGS_SECRET)

        with gr.Row(elem_id='shell', equal_height=False):
            # ============================ THANH BÊN ============================
            with gr.Column(elem_id='side', scale=0, min_width=236):
                gr.HTML('<div class="brand"><span class="logo">VD</span><div><b>VieDub</b>'
                        '<small>Lồng tiếng Việt tự động</small></div></div>', padding=False)
                nav_dash = gr.Button('Lồng tiếng', elem_id='nav-dash', elem_classes=_nav_cls('dash', 'dash'))
                nav_edit = gr.Button(_edit_label(), elem_id='nav-edit', elem_classes=_nav_cls('editor', 'dash'))
                nav_clean = gr.Button('Dọn dẹp', elem_id='nav-clean', elem_classes=_nav_cls('clean', 'dash'))
                gpu_line = (f'<span class="gpu">⚡ {html.escape(gpu_short)}</span>' if gpu
                            else '<span class="gpu off">🐢 Không có GPU, chạy bằng CPU</span>')
                gr.HTML(f'<div class="side-card"><b>Chạy trên máy của bạn</b>{gpu_line}<br>'
                        f'Whisper · Google Dịch · Edge-TTS</div>', padding=False, elem_classes='side-bottom')

            # ============================ VÙNG CHÍNH ============================
            with gr.Column(elem_id='main', scale=1):
                with gr.Row(elem_id='topbar', equal_height=True):
                    stepper = gr.HTML(_stepper(1), padding=False, elem_classes='tb-left')
                    gr.HTML('<div class="tb-right">'
                            + ('<span class="tb-pill"><span class="dot"></span>GPU sẵn sàng</span>' if gpu else
                               '<span class="tb-pill"><span class="dot off"></span>Chạy bằng CPU</span>')
                            + '</div>', padding=False, elem_classes='tb-right-cell')

                # ------------------------- TRANG 1: LỒNG TIẾNG -------------------------
                with gr.Column(elem_classes=_page_cls('dash', 'dash')) as dash_col:
                    page_head = gr.HTML(_page_head(*PAGE_HEADS['dub']), padding=False)
                    stats = gr.HTML(stats_html([], [], []), padding=False)
                    with gr.Row(elem_classes='modebar', equal_height=True):
                        gr.HTML('<div class="mode-lbl">Chế độ</div>', padding=False, elem_classes='mode-lblc')
                        mode_dub = gr.Button(MODES['dub'], elem_id='mode-dub', elem_classes=['chipbtn', 'active'],
                                             scale=0, min_width=150)
                        mode_dl = gr.Button(MODES['dl'], elem_id='mode-dl', elem_classes=['chipbtn'], scale=0,
                                            min_width=160)
                        mode_hint = gr.HTML(f'<div class="hint mode-hint">{MODE_HINTS["dub"]}</div>', padding=False,
                                            elem_classes='grow')
                    with gr.Row(equal_height=False):
                        with gr.Column(scale=11, min_width=520):
                            with gr.Column(elem_classes='pnl'):
                                add_title = gr.HTML(_add_title('dub'), padding=False)
                                with gr.Tabs(elem_classes='seg') as add_tabs:
                                    with gr.Tab('🔗  Link', id='link'):
                                        with gr.Row(elem_classes='platrow'):
                                            plat_btns = {}
                                            for key, label, _, status, _ in PLATFORMS:
                                                plat_btns[key] = gr.Button(
                                                    label, elem_id=f'plat-{key}', size='sm', scale=0, min_width=0,
                                                    interactive=status != 'soon',
                                                    elem_classes=['chipbtn', 'plat'] + (['active'] if key == 'auto' else [])
                                                    + (['soon'] if status == 'soon' else []))
                                        plat_hint = gr.HTML(_plat_hint('auto'), padding=False)
                                        link_rows, link_boxes, rm_btns = [], [], []
                                        with gr.Column(elem_classes='lrows'):
                                            for i in range(MAX_LINKS):
                                                with gr.Row(elem_classes=_row_classes(i, 1)) as row:
                                                    box = gr.Textbox(show_label=False, container=False, lines=1,
                                                                     max_lines=3, scale=20,
                                                                     placeholder=_plat_placeholder('auto', i))
                                                    rm = gr.Button('✕', variant='secondary', size='sm', scale=0,
                                                                   min_width=36, elem_classes='rm')
                                                link_rows.append(row)
                                                link_boxes.append(box)
                                                rm_btns.append(rm)
                                        with gr.Row():
                                            add_btn = gr.Button('＋ Thêm link', variant='secondary', scale=1)
                                            check_btn = gr.Button('🔍 Kiểm tra tất cả', variant='primary', scale=2)
                                            clear_btn = gr.Button('Xoá hết link', variant='secondary', scale=1)
                                        check_status = gr.Markdown(elem_classes='hint')
                                    with gr.Tab('📁  File từ máy', id='file'):
                                        drop = gr.File(show_label=False, file_count='multiple', type='filepath',
                                                       file_types=['video'], height=150, elem_id='drop')
                                        with gr.Row(equal_height=True):
                                            gr.HTML('<div class="hint">Thả được nhiều lần, file được cộng dồn vào danh '
                                                    'sách bên dưới.</div>', padding=False, elem_classes='grow')
                                            clear_files_btn = gr.Button('Xoá hết file', variant='secondary', size='sm',
                                                                        scale=0, min_width=120)

                            with gr.Column(elem_classes='pnl'):
                                with gr.Row(equal_height=True, elem_classes='phead'):
                                    gr.HTML(_ptitle('🎬', '2. Danh sách video',
                                                    'Tích ô để chọn · đặt tên ngay trong ô tên (là tên file khi xuất)'),
                                            padding=False, elem_classes='grow')
                                    all_btn = gr.Button('Chọn tất cả', size='sm', variant='secondary', scale=0,
                                                        min_width=110)
                                    none_btn = gr.Button('Bỏ chọn', size='sm', variant='secondary', scale=0,
                                                         min_width=90)

                                def _cards(items, checking, sel, names):
                                    """Vẽ thẻ video (gọi bên trong gr.render)."""
                                    with gr.Column(elem_classes='vlist'):
                                        for it in items:
                                            ok = it['status'] == 'ok'
                                            # link đang kiểm tra: chưa hiện ô tên (lượt kiểm tra vẽ lại liên tục, đang gõ
                                            # sẽ mất chữ); xong mới hiện
                                            named = ok and not (it['kind'] == 'link' and checking)
                                            with gr.Row(elem_classes=['vcard', f's-{it["status"]}'], equal_height=False):
                                                if ok:
                                                    # key: giữ nguyên ô qua các lần vẽ lại; giá trị luôn theo lựa chọn
                                                    # hiện tại (preserved_by_key=None)
                                                    chk = gr.Checkbox(value=it['key'] in sel, show_label=False,
                                                                      container=False, interactive=True,
                                                                      key=f'chk-{it["key"]}', preserved_by_key=None,
                                                                      scale=0, min_width=22, elem_classes='vchk')
                                                    chk.input(_toggle_sel(it['key']), inputs=[selector, chk],
                                                              outputs=selector, show_progress='hidden')
                                                else:
                                                    gr.HTML('<span class="vchk-ph"></span>', padding=False,
                                                            elem_classes='vchk')
                                                gr.HTML(_item_thumb(it), padding=False, elem_classes='vthc')
                                                with gr.Column(scale=1, min_width=0, elem_classes='vmain'):
                                                    if named:
                                                        nm = gr.Textbox(value=_item_name(names, it), show_label=False,
                                                                        container=False, lines=1, max_lines=1,
                                                                        interactive=True, key=f'name-{it["key"]}',
                                                                        placeholder='Tên video (dùng làm tên file xuất)',
                                                                        elem_classes='vname')
                                                        # .change bắt cả gõ lẫn dán (dán bằng Ctrl+V không phát .input)
                                                        nm.change(_set_name(it['key']), inputs=[names_state, nm],
                                                                  outputs=names_state, show_progress='hidden',
                                                                  trigger_mode='always_last')
                                                    gr.HTML(_item_info(it, named), padding=False, elem_classes='vinfo')
                                                if it['kind'] == 'link' and checking:
                                                    continue   # đang kiểm tra: chưa cho xoá (lượt kiểm tra sẽ ghi đè lại)
                                                b = gr.Button('✕', variant='secondary', size='sm', scale=0,
                                                              min_width=36, elem_classes='rm')
                                                if it['kind'] == 'file':
                                                    b.click(remove_file(it['src']),
                                                            inputs=[files_state, link_items, selector],
                                                            outputs=[files_state, selector])
                                                else:
                                                    b.click(remove_link_item(it['src']),
                                                            inputs=[link_items, files_state, selector, n_links,
                                                                    *link_boxes],
                                                            outputs=[link_items, selector, n_links, *link_boxes,
                                                                     *link_rows])

                                # 2 khối vẽ riêng: kiểm tra link chỉ vẽ lại khối link, ô tên của file không bị vẽ lại
                                # giữa lúc đang gõ. concurrency_limit=1: mặc định không giới hạn -> 2 lượt vẽ song
                                # song đụng nhau (KeyError trong Gradio). Tự đặt triggers thì phải có app.load.
                                @gr.render(inputs=[link_items, files_state, selector, names_state],
                                           triggers=[app.load, link_items.change, files_state.change, selector.change],
                                           concurrency_limit=1)
                                def _links_list(lis, files, selected, names):
                                    if not lis and not files:
                                        gr.HTML(EMPTY_ITEMS, padding=False)
                                        return
                                    checking = any(it['status'] in ('pending', 'checking') for it in lis or [])
                                    if lis:
                                        _cards(lis, checking, set(selected or []), names)

                                @gr.render(inputs=[files_state, selector, names_state],
                                           triggers=[app.load, files_state.change, selector.change],
                                           concurrency_limit=1)
                                def _files_list(files, selected, names):
                                    items = _file_items(files)
                                    if items:
                                        _cards(items, False, set(selected or []), names)

                        with gr.Column(scale=9, min_width=440):
                            with gr.Column(elem_classes=['lhide']) as dl_pnl:
                                with gr.Column(elem_classes='pnl'):
                                    gr.HTML(_ptitle('⬇️', '3. Tải về', 'Tải nguyên bản (tối đa 1080p) · đóng dấu ở '
                                                    'khung bên dưới', 'grn'), padding=False)
                                    with gr.Row(equal_height=True):
                                        dl_out_dir = gr.Textbox(value=dflt['dl_out_dir'], label='Thư mục lưu', scale=8,
                                                                placeholder=DEFAULT_OUT_DIR)
                                        dl_pick_btn = gr.Button('Chọn…', variant='secondary', size='sm', scale=0,
                                                                min_width=76)
                                        dl_open_btn = gr.Button('Mở', variant='secondary', size='sm', scale=0,
                                                                min_width=56)
                            with gr.Column() as dub_pnl:
                              with gr.Column(elem_classes='pnl'):
                                  gr.HTML(_ptitle('⚙️', '3. Cài đặt',
                                                  'Áp dụng chung cho các video đã chọn · tự lưu trên trình duyệt', 'blu'),
                                          padding=False)
                                  with gr.Tabs(elem_classes='seg'):
                                      with gr.Tab('🎙️  Giọng & phụ đề'):
                                          with gr.Row():
                                              source = gr.Dropdown(list(SOURCE_LANGS), value=dflt['source'],
                                                                   label='Ngôn ngữ gốc')
                                              voice = gr.Dropdown(roles, value=dflt['voice'], label='Giọng đọc tiếng Việt')
                                          subtitle = gr.Dropdown(list(SUBTITLE_TYPES), value=dflt['subtitle'],
                                                                 label='Phụ đề tiếng Việt')
                                          with gr.Row(elem_classes='sw-grid'):
                                              keep_bgm = gr.Checkbox(value=dflt['keep_bgm'], label='Giữ nhạc nền',
                                                                     elem_classes='sw')
                                              voice_autorate = gr.Checkbox(value=dflt['voice_autorate'],
                                                                           label='Tăng tốc giọng cho vừa câu',
                                                                           elem_classes='sw')
                                              video_autorate = gr.Checkbox(value=dflt['video_autorate'],
                                                                           label='Làm chậm video cho vừa giọng',
                                                                           elem_classes='sw')
                                          with gr.Column(elem_classes='ai-card'):
                                              ai_erase = gr.Checkbox(value=dflt['ai_erase'], elem_id='ai-erase',
                                                                     label='🤖 Xoá chữ gốc bằng AI', elem_classes='sw')
                                              gr.HTML('<div class="hint">Tự tìm phụ đề gốc ở bất kỳ đâu trong video (dòng '
                                                      'chữ lặp lại ở cùng chỗ qua nhiều cảnh) rồi vẽ lại nền bằng AI.</div>',
                                                      padding=False)
                                      with gr.Tab('🛠️  Nâng cao'):
                                          with gr.Row(equal_height=True):
                                              model = gr.Dropdown(MODELS, value=dflt['model'],
                                                                  label='Model nhận dạng (Whisper)')
                                              cuda = gr.Checkbox(value=dflt['cuda'], label='Dùng GPU (CUDA)',
                                                                 interactive=has_cuda, elem_classes='sw')
                                          with gr.Column(elem_classes='subcard'):
                                              gr.HTML('<div class="sub-head">🍪 Cookie (khi Douyin/TikTok chặn)</div>',
                                                      padding=False)
                                              with gr.Row(equal_height=True):
                                                  cookie_info = gr.HTML(_cookie_status(), padding=False,
                                                                        elem_classes='grow')
                                                  cookie_btn = gr.UploadButton('📄 Chọn cookies.txt', file_types=['.txt'],
                                                                               size='sm', variant='secondary', scale=0,
                                                                               min_width=170)
                                                  cookie_del = gr.Button('Xoá cookie', size='sm', variant='secondary',
                                                                         scale=0, min_width=100)
                                              gr.Markdown('Cách lấy: cài extension **Get cookies.txt LOCALLY** cho '
                                                          'Chrome/Edge, mở douyin.com (hoặc tiktok.com), bấm extension → '
                                                          '**Export**, rồi chọn file đó. Cookie được lưu lại trên máy.',
                                                          elem_classes='hint')
                                              browser = gr.Dropdown(BROWSERS, value=dflt['browser'],
                                                                    label='Hoặc lấy cookie thẳng từ trình duyệt '
                                                                          '(Chrome/Edge hay lỗi, Firefox ổn)')

                            with gr.Column(elem_classes='pnl', elem_id='wm-pnl'):
                                gr.HTML(_ptitle('🏷️', 'Đóng dấu logo / chữ',
                                                'Bật logo, chữ hoặc cả hai · gắn khi tải (Chỉ tải video) hoặc khi xuất '
                                                '(Lồng tiếng)', 'pink'), padding=False)
                                with gr.Row(elem_classes='sw-grid'):
                                    wm_logo_on = gr.Checkbox(value=dflt['wm_logo_on'], label='🖼️ Gắn logo',
                                                             elem_classes='sw', elem_id='wm-logo-on')
                                    wm_text_on = gr.Checkbox(value=dflt['wm_text_on'], label='🔤 Gắn chữ',
                                                             elem_classes='sw', elem_id='wm-text-on')
                                with gr.Column(elem_classes=['wm-sub', 'lhide']) as wm_logo_box:
                                    gr.HTML('<div class="sub-label">🖼️ Logo</div>', padding=False)
                                    with gr.Row(equal_height=False):
                                        wm_logo = gr.Image(label='Ảnh logo (PNG nền trong suốt đẹp nhất)', type='filepath',
                                                           sources=['upload'], height=150, image_mode='RGBA', scale=1,
                                                           min_width=180, elem_id='wm-logo')
                                        wm_logo_pos = gr.Radio(WM_POS, value=dflt['wm_logo_pos'], label='Vị trí',
                                                               elem_classes='posgrid', scale=0, min_width=160,
                                                               elem_id='wm-logo-pos')
                                    with gr.Row():
                                        wm_logo_size = gr.Slider(4, 50, value=dflt['wm_logo_size'], step=1,
                                                                 label='Cỡ (% bề rộng video)')
                                        wm_logo_opacity = gr.Slider(10, 100, value=dflt['wm_logo_opacity'], step=5,
                                                                    label='Độ đậm (%)')
                                with gr.Column(elem_classes=['wm-sub', 'lhide']) as wm_text_box:
                                    gr.HTML('<div class="sub-label">🔤 Chữ</div>', padding=False)
                                    with gr.Row(equal_height=True):
                                        wm_text = gr.Textbox(value=dflt['wm_text'], label='Chữ đóng dấu', lines=1,
                                                             max_lines=1, scale=5, placeholder='Ví dụ: @kenh_cua_ban',
                                                             elem_id='wm-text')
                                        wm_color = gr.ColorPicker(value=dflt['wm_color'], label='Màu chữ', scale=1,
                                                                  min_width=90)
                                    with gr.Row(equal_height=False):
                                        wm_text_pos = gr.Radio(WM_POS, value=dflt['wm_text_pos'], label='Vị trí',
                                                               elem_classes='posgrid', scale=0, min_width=160,
                                                               elem_id='wm-text-pos')
                                        with gr.Column(scale=1, min_width=200):
                                            wm_text_size = gr.Slider(2, 20, value=dflt['wm_text_size'], step=0.5,
                                                                     label='Cỡ chữ (% bề rộng video)')
                                            wm_text_opacity = gr.Slider(10, 100, value=dflt['wm_text_opacity'], step=5,
                                                                        label='Độ đậm (%)')
                                with gr.Column(elem_classes=['lhide']) as wm_opts_box:
                                    wm_margin = gr.Slider(0, 15, value=dflt['wm_margin'], step=1,
                                                          label='Cách mép (% bề rộng video)')
                                    wm_prev = gr.Image(show_label=False, interactive=False, height=360, type='pil',
                                                       buttons=[])
                                    wm_cap = gr.HTML('', padding=False)

                            with gr.Column(elem_classes='pnl', elem_id='progress-card'):
                                with gr.Row(equal_height=True, elem_classes='phead'):
                                    prog_title = gr.HTML(_ptitle('⏳', '4. Tiến trình',
                                                                 'Lần lượt từng video · xong sẽ mở trình sửa', 'org'),
                                                         padding=False, elem_classes='grow')
                                    stop_btn = gr.Button(STOP_LABEL, variant='stop', size='sm', interactive=False,
                                                         scale=0, min_width=120)
                                queue_html = gr.HTML(_render_queue([]), padding=False)
                                with gr.Accordion('Nhật ký chi tiết', open=False):
                                    log_box = gr.Textbox(show_label=False, lines=12, max_lines=12, interactive=False,
                                                         autoscroll=True)

                    with gr.Row(elem_id='runbar', equal_height=True):
                        runbar = gr.HTML(runbar_html('dub', [], dflt['source'], dflt['voice'], dflt['ai_erase'],
                                                     dflt['dl_out_dir']), padding=False, elem_classes='grow')
                        run_btn = gr.Button(_run_label([]), elem_id='run-btn', scale=0, min_width=280)

                # ------------------------- TRANG 2: TRÌNH SỬA -------------------------
                with gr.Column(elem_classes=_page_cls('editor', 'dash')) as editor_col:
                    with gr.Row(equal_height=True, elem_classes='page-row'):
                        gr.HTML(_page_head('✏️', 'Trình sửa', 'Bấm 1 câu thoại để nghe và sửa, chỉnh phụ đề rồi xuất '
                                                             'video.'), padding=False, elem_classes='grow')
                        proj_dd = gr.Dropdown(choices=[], label='Video đang sửa', scale=0, min_width=340,
                                              interactive=True, elem_id='proj-dd')
                        drop_btn = gr.Button('🗑 Xoá video này', variant='stop', scale=0, min_width=160)
                    ed_empty = gr.HTML(EDITOR_EMPTY, padding=False)
                    with gr.Column(elem_classes=['lhide']) as ed_body:
                        ed_status = gr.HTML('', padding=False)
                        seek_box = gr.Textbox(visible=False)
                        with gr.Row(equal_height=False):
                            with gr.Column(scale=7):
                                with gr.Column(elem_classes='pnl'):
                                    with gr.Tabs(elem_id='ed-tabs', elem_classes='seg') as ed_tabs:
                                        with gr.Tab('🎞️  Video gốc', id='orig'):
                                            gr.HTML('<div class="ed-vlabel">Video gốc (đã xoá sub nếu bật AI) · bấm '
                                                    'vào đoạn trên dòng thời gian để tua</div>', padding=False)
                                            ed_video = gr.HTML(_video_html(None), padding=False)
                                            ed_timeline = gr.HTML('', padding=False)
                                        with gr.Tab('✅  Kết quả', id='result'):
                                            gr.HTML('<div class="ed-vlabel">Video đã dựng: giọng Việt + phụ đề + nhạc '
                                                    'nền. Không ưng thì sửa chữ rồi xuất lại, hoặc xoá.</div>',
                                                    padding=False)
                                            res_video = gr.HTML(_video_html(None, 'ed-result', RESULT_EMPTY),
                                                                padding=False)
                                            res_info = gr.HTML('', padding=False)
                                            with gr.Row():
                                                open_res_btn = gr.Button('📂 Mở thư mục chứa', variant='secondary',
                                                                         size='sm')
                                                discard_btn = gr.Button('🗑 Không ưng, xoá & về trang chủ',
                                                                        variant='stop', size='sm')
                                with gr.Column(elem_classes='pnl'):
                                    gr.HTML(_ptitle('💬', 'Các câu thoại', 'Bấm 1 dòng để nghe và sửa ở khung bên phải'),
                                            padding=False)
                                    ed_df = gr.Dataframe(headers=['#', 'Từ', 'Đến', 'Gốc', 'Tiếng Việt', 'Giọng'],
                                                         datatype=['number', 'str', 'str', 'str', 'str', 'str'],
                                                         type='array', interactive=False, wrap=True, max_height=480,
                                                         column_widths=['5%', '10%', '10%', '27%', '36%', '12%'],
                                                         show_row_numbers=False, elem_id='ed-df', show_label=False)
                            with gr.Column(scale=5):
                                with gr.Column(elem_classes='pnl'):
                                    gr.HTML(_ptitle('✍️', 'Dòng đang chọn', 'Sửa chữ, nghe lại và tạo lại giọng', 'pink'),
                                            padding=False)
                                    ln_title = gr.HTML('<div class="sub-head">Dòng đang chọn</div>', padding=False)
                                    ln_zh = gr.HTML('<div class="zh"></div>', padding=False)
                                    with gr.Row(equal_height=True):
                                        # 1 dòng (tự giãn tới 5 dòng): Enter = lưu
                                        ln_vi = gr.Textbox(label='Tiếng Việt (Enter hoặc bấm Lưu)', lines=1, max_lines=5,
                                                           scale=6)
                                        save_btn = gr.Button('💾 Lưu', size='sm', variant='secondary', scale=0,
                                                             min_width=80)
                                    gr.HTML('<div class="sub-label">🔊 Giọng của dòng này</div>', padding=False)
                                    ln_audio = gr.HTML(_audio_html(None), padding=False)
                                    with gr.Row():
                                        seek_btn = gr.Button('⏩ Tua tới dòng này', size='sm', variant='secondary')
                                        redub_btn = gr.Button('🔁 Tạo lại giọng', size='sm', variant='primary')
                                    redub_all_btn = gr.Button('🔁 Tạo giọng mọi dòng đã sửa', size='sm',
                                                              variant='secondary')

                                with gr.Column(elem_classes='pnl'):
                                    gr.HTML(_ptitle('🅰️', 'Phụ đề tiếng Việt', 'Kiểu chữ, vị trí và che sub gốc', 'blu'),
                                            padding=False)
                                    ed_subtitle = gr.Dropdown(list(SUBTITLE_TYPES), value=dflt['subtitle'],
                                                              label='Chèn phụ đề')
                                    sub_note = gr.HTML('', padding=False)
                                    with gr.Column(visible=_is_hard_sub(dflt['subtitle'])) as sub_panel:
                                        with gr.Row(equal_height=True):
                                            sub_size = gr.Slider(10, 30, value=dflt['sub_size'], step=1, label='Cỡ chữ')
                                            sub_pos = gr.Slider(0, 60, value=dflt['sub_pos'], step=1,
                                                                label='Vị trí (% từ mép dưới lên)')
                                        with gr.Row(elem_classes='sw-grid'):
                                            sub_follow = gr.Checkbox(value=dflt['sub_follow'],
                                                                     label='Đặt sub vào chỗ sub gốc', elem_classes='sw')
                                            sub_box = gr.Checkbox(value=dflt['sub_box'], label='Nền đen mờ sau chữ',
                                                                  elem_classes='sw')
                                        with gr.Column(elem_classes='vn-part'):
                                            gr.HTML('<div class="sub-label">Che sub gốc (nếu không dùng xoá AI) · khung '
                                                    'tím trên ảnh xem trước</div>', padding=False)
                                            cover_mode = gr.Radio(COVER_MODES, value=dflt['cover_mode'], label='Cách che')
                                            with gr.Row(equal_height=True):
                                                cover_top = gr.Slider(0, 95, value=dflt['cover_top'], step=1,
                                                                      label='Vị trí (% từ mép trên)')
                                                cover_size = gr.Slider(3, 40, value=dflt['cover_size'], step=1,
                                                                       label='Chiều cao (% video)')
                                            with gr.Row(equal_height=True):
                                                auto_btn = gr.Button('🔎 Tự tìm vị trí sub gốc', variant='secondary',
                                                                     size='sm', scale=0, min_width=200)
                                                auto_status = gr.HTML('', padding=False, elem_classes='auto-status')
                                    preview_img = gr.Image(show_label=False, interactive=False, height=360, type='pil',
                                                           buttons=[])
                                    preview_cap = gr.HTML('<div class="hint">Xem trước</div>', padding=False)

                                with gr.Column(elem_classes='pnl'):
                                    gr.HTML(_ptitle('🎬', 'Xuất video', 'Lưu thành &lt;tên&gt;/&lt;tên&gt;.mp4 kèm phụ đề .srt',
                                                    'grn'), padding=False)
                                    exp_name = gr.Textbox(label='Tên video (tên file khi xuất)', lines=1, max_lines=1,
                                                          placeholder='Đặt tên cho video này', elem_id='exp-name')
                                    with gr.Row(equal_height=True):
                                        out_dir = gr.Textbox(value=dflt['out_dir'], label='Thư mục xuất', scale=8,
                                                             placeholder=DEFAULT_OUT_DIR)
                                        pick_btn = gr.Button('Chọn…', variant='secondary', size='sm', scale=0,
                                                             min_width=76)
                                        open_btn = gr.Button('Mở', variant='secondary', size='sm', scale=0, min_width=56)
                                    gr.HTML('<div class="hint">Các dòng đã sửa mà chưa tạo giọng sẽ được tạo tự động '
                                            'khi xuất. Trùng tên với video khác thì tự thêm (2), (3)…</div>',
                                            padding=False)
                                    with gr.Row():
                                        export_btn = gr.Button('🎬 Xuất video này', elem_id='export-btn', scale=3)
                                        export_all_btn = gr.Button('Xuất tất cả', variant='secondary', scale=1,
                                                                   min_width=130)
                                    exp_wm = gr.HTML(wm_export_note(), padding=False)
                                    exp_html = gr.HTML('', padding=False)

                # ------------------------- TRANG 3: DỌN DẸP -------------------------
                with gr.Column(elem_classes=_page_cls('clean', 'dash')) as clean_col:
                    gr.HTML(_page_head('🧹', 'Dọn dẹp', 'Xoá nhật ký và mọi file sinh ra khi xử lý từ trước tới giờ để '
                                                       'lấy lại dung lượng.'), padding=False)
                    with gr.Column(elem_classes='pnl'):
                        gr.HTML(_ptitle('💾', 'Dung lượng có thể giải phóng',
                                        'Không đụng tới video đã xuất ở thư mục xuất, cookie và model AI', 'org'),
                                padding=False)
                        clean_html = gr.HTML('', padding=False)
                        with gr.Row(equal_height=True):
                            clean_dl = gr.Checkbox(value=False, elem_classes='sw', scale=4,
                                                   label='Xoá cả video đã tải về từ link (lần sau phải tải lại)')
                            clean_refresh = gr.Button('🔄 Tính lại', variant='secondary', scale=0, min_width=120)
                            clean_ok = gr.Button('🗑 Xoá hết', variant='stop', scale=0, min_width=140)

        # ============================ SỰ KIỆN ============================
        nav_outputs = [page_state, dash_col, editor_col, clean_col, nav_dash, nav_edit, nav_clean]
        stats_in = [link_items, files_state, selector, mode_state]
        wm_comps = [wm_logo_on, wm_logo, wm_logo_pos, wm_logo_size, wm_logo_opacity, wm_text_on, wm_text, wm_color,
                    wm_text_pos, wm_text_size, wm_text_opacity, wm_margin]   # đúng thứ tự WM_FIELDS / _wm_opts
        runbar_in = [mode_state, selector, source, voice, ai_erase, dl_out_dir, *wm_comps]
        wm_in = [link_items, files_state, selector, *wm_comps]
        wm_out = [wm_prev, wm_cap]
        mode_outs = [page_head, queue_html, mode_state, mode_dub, mode_dl, dub_pnl, dl_pnl, run_btn, mode_hint, stepper,
                     prog_title, add_title, add_tabs]
        run_lbl_in = [selector, mode_state, wm_logo_on, wm_logo, wm_text_on, wm_text]

        def _run_lbl(sel, m, lo, lg, to, t):
            return gr.update(value=_run_label(sel, m, _wm_active(lo, lg, to, t)))
        editor_main = [ed_video, ed_status, ed_timeline, ed_df, sel_i, proj_dd, ed_subtitle, res_video, res_info,
                       ed_tabs, ed_body, ed_empty, exp_name]
        line_outs = [sel_i, ln_title, ln_vi, ln_audio, ln_zh, seek_box]
        pv_inputs = [proj_dd, sel_i, ed_subtitle, cover_mode, cover_top, cover_size, sub_size, sub_pos, sub_follow,
                     sub_box, *wm_comps]
        pv_outputs = [preview_img, preview_cap]
        sub_ctl_inputs = [ed_subtitle, cover_mode, sub_follow, proj_dd]
        sub_ctl_outputs = [sub_panel, sub_pos, sub_note]

        def _enter_editor(pid):
            pid = pid if pid in PROJECTS else next(iter(PROJECTS), None)
            return (*load_project(pid), *select_line(pid, 0))
        enter_outputs = editor_main + line_outs

        def _open_editor(ev):
            """Sau khi chuyển sang trình sửa: nạp video, khối phụ đề, ảnh xem trước, thanh bước."""
            return ev.then(_enter_editor, inputs=proj_dd, outputs=enter_outputs) \
                .then(_sub_controls, inputs=sub_ctl_inputs, outputs=sub_ctl_outputs) \
                .then(editor_preview, inputs=pv_inputs, outputs=pv_outputs) \
                .then(_stepper_for, inputs=[page_state, proj_dd, mode_state], outputs=stepper) \
                .then(wm_export_note, inputs=wm_comps, outputs=exp_wm, show_progress='hidden')

        # ---------------- Điều hướng ----------------
        nav_dash.click(lambda: _nav('dash'), outputs=nav_outputs) \
            .then(_stepper_for, inputs=[page_state, proj_dd, mode_state], outputs=stepper) \
            .then(stats_html, inputs=stats_in, outputs=stats)

        # ---------------- Chế độ / nền tảng / đóng dấu ----------------
        for btn, m in ((mode_dub, 'dub'), (mode_dl, 'dl')):
            btn.click(lambda sel, lo, lg, to, t, m=m: set_mode(m, sel, lo, lg, to, t),
                      inputs=[selector, wm_logo_on, wm_logo, wm_text_on, wm_text], outputs=mode_outs) \
                .then(runbar_html, inputs=runbar_in, outputs=runbar, show_progress='hidden') \
                .then(stats_html, inputs=stats_in, outputs=stats, show_progress='hidden') \
                .then(wm_preview, inputs=wm_in, outputs=wm_out, show_progress='hidden')
        for key, btn in plat_btns.items():
            btn.click(set_plat(key), outputs=[plat_state, *plat_btns.values(), plat_hint, *link_boxes],
                      show_progress='hidden')
        for c in (wm_logo_on, wm_text_on):
            c.change(_wm_boxes, inputs=[wm_logo_on, wm_text_on], outputs=[wm_logo_box, wm_text_box, wm_opts_box],
                     show_progress='hidden')
        for c in (wm_logo_on, wm_logo, wm_logo_pos, wm_text_on, wm_text_pos, wm_color):
            c.change(wm_preview, inputs=wm_in, outputs=wm_out, show_progress='hidden')
        for c in (wm_logo_size, wm_logo_opacity, wm_text_size, wm_text_opacity, wm_margin):
            c.release(wm_preview, inputs=wm_in, outputs=wm_out, show_progress='hidden')
        for ev in (wm_text.submit, wm_text.blur):
            ev(wm_preview, inputs=wm_in, outputs=wm_out, show_progress='hidden')
        for c in (wm_logo_on, wm_logo, wm_logo_pos, wm_text_on, wm_text, wm_text_pos, dl_out_dir):
            c.change(runbar_html, inputs=runbar_in, outputs=runbar, show_progress='hidden')
        for c in (wm_logo_on, wm_logo, wm_text_on, wm_text):
            c.change(_run_lbl, inputs=run_lbl_in, outputs=run_btn, show_progress='hidden')
        for c in (wm_logo_on, wm_logo, wm_logo_pos, wm_text_on, wm_text, wm_text_pos):
            c.change(wm_export_note, inputs=wm_comps, outputs=exp_wm, show_progress='hidden')
        wm_logo.upload(save_logo, inputs=wm_logo, outputs=wm_logo)   # giữ logo cho lần sau
        wm_logo.clear(clear_logo)
        dl_pick_btn.click(pick_out_dir, inputs=dl_out_dir, outputs=dl_out_dir)
        dl_open_btn.click(open_output_dir, inputs=dl_out_dir)
        _open_editor(nav_edit.click(lambda: _nav('editor'), outputs=nav_outputs))
        nav_clean.click(lambda: _nav('clean'), outputs=nav_outputs) \
            .then(clean_report, inputs=clean_dl, outputs=clean_html)

        # ---------------- Link ----------------
        add_btn.click(add_link_row, inputs=n_links, outputs=[n_links, *link_rows])
        for i in range(MAX_LINKS):
            rm_btns[i].click(remove_link_row(i), inputs=[n_links, *link_boxes],
                             outputs=[n_links, *link_boxes, *link_rows])
            # .input bắt thao tác gõ, .blur bắt thao tác dán (Ctrl+V không phát .input) khi rời khỏi ô.
            # Không dùng .change: app tự cập nhật các ô sẽ kích hoạt .change dây chuyền cho cả 30 ô.
            for ev in (link_boxes[i].input, link_boxes[i].blur):
                ev(on_link_input(i), inputs=[n_links, *link_boxes],
                   outputs=[n_links, *link_boxes, *link_rows], show_progress='hidden')
        check_btn.click(check_links, inputs=[plat_state, browser, cookie_state, files_state, selector, n_links, *link_boxes],
                        outputs=[link_items, selector, check_status]) \
            .then(select_after_check, inputs=[link_items, files_state, selector], outputs=selector)

        def _clear_links(files, selected, *vals):
            items = _all_items([], files)
            sel = [k for k in (selected or []) if k.startswith('F:')]
            return [1, *_box_updates(list(vals), []), *_rows_update(1), [], _selector_update(items, sel), '']
        clear_btn.click(_clear_links, inputs=[files_state, selector, *link_boxes],
                        outputs=[n_links, *link_boxes, *link_rows, link_items, selector, check_status])

        # ---------------- File ----------------
        drop.upload(add_files, inputs=[files_state, drop, link_items, selector], outputs=[files_state, drop, selector])
        clear_files_btn.click(clear_files, inputs=[link_items, selector], outputs=[files_state, selector])

        # ---------------- Chọn / thống kê / thanh nút ----------------
        all_btn.click(lambda li, f: _selector_update(_all_items(li, f)), inputs=[link_items, files_state],
                      outputs=selector)
        none_btn.click(lambda: [], outputs=selector)
        selector.change(_run_lbl, inputs=run_lbl_in, outputs=run_btn, show_progress='hidden')
        # đổi video đang chọn -> ảnh xem trước dấu lấy video mới (chỉ khi đang bật đóng dấu)
        selector.change(lambda *a: wm_preview(*a) if _wm_active(a[3], a[4], a[8], a[9]) else (gr.update(), gr.update()),
                        inputs=wm_in, outputs=wm_out, show_progress='hidden')
        for st in (link_items, files_state, selector):
            st.change(stats_html, inputs=stats_in, outputs=stats, show_progress='hidden')
        for c in (selector, source, voice, ai_erase):
            c.change(runbar_html, inputs=runbar_in, outputs=runbar, show_progress='hidden')

        # ---------------- Lồng tiếng ----------------
        run_btn.click(None, js="() => document.getElementById('progress-card')"
                               "?.scrollIntoView({behavior: 'smooth', block: 'center'})")
        run_ev = run_btn.click(start_run,
                               inputs=[mode_state, selector, names_state, link_items, files_state, source, model, voice,
                                       subtitle, keep_bgm, voice_autorate, video_autorate, cuda, browser, cookie_state,
                                       ai_erase, dl_out_dir, *wm_comps],
                               outputs=[queue_html, log_box, run_btn, stop_btn, proj_dd, *nav_outputs, stepper])
        _open_editor(run_ev).then(stats_html, inputs=stats_in, outputs=stats) \
            .then(None, inputs=page_state,
                  js="(p) => { if (p === 'editor') window.scrollTo({top: 0, behavior: 'smooth'}); }")
        stop_btn.click(request_stop, outputs=stop_btn)

        # ---------------- Trình sửa ----------------
        _open_editor(proj_dd.input(lambda: None))
        for b in (drop_btn, discard_btn, export_btn, export_all_btn):
            b.click(None, js=RELEASE_MEDIA_JS)
        for b in (drop_btn, discard_btn):   # xoá dự án + mọi file liên quan, về trang Lồng tiếng
            b.click(remove_project, inputs=proj_dd, outputs=[proj_dd, *nav_outputs]) \
                .then(_enter_editor, inputs=proj_dd, outputs=enter_outputs) \
                .then(_stepper_for, inputs=[page_state, proj_dd, mode_state], outputs=stepper) \
                .then(stats_html, inputs=stats_in, outputs=stats)
        open_res_btn.click(lambda pid: open_output_dir(str(Path(PROJECTS[pid]['result']).parent))
                           if pid in PROJECTS and PROJECTS[pid].get('result') else gr.Warning('Chưa xuất video.'),
                           inputs=proj_dd)
        ed_df.select(on_df_select, inputs=proj_dd, outputs=line_outs + [ed_timeline]) \
            .then(editor_preview, inputs=pv_inputs, outputs=pv_outputs, show_progress='hidden')
        for ev in (ln_vi.submit, ln_vi.blur, save_btn.click):
            ev(on_line_submit, inputs=[ln_vi, proj_dd, sel_i], outputs=[ed_df, ed_status, ed_timeline, ln_audio],
               show_progress='hidden')
        seek_box.change(None, inputs=seek_box,
                        js="(v) => { const t = parseFloat(String(v).split('#')[0]); "
                           "if (!isNaN(t) && window.vdSeek) window.vdSeek(t); }")
        seek_btn.click(lambda pid, i: select_line(pid, i)[-1], inputs=[proj_dd, sel_i], outputs=seek_box,
                       show_progress='hidden')
        redub_btn.click(redub_line, inputs=[ln_vi, proj_dd, sel_i], outputs=[ed_df, ed_status, ed_timeline, ln_audio])
        redub_all_btn.click(redub_all, inputs=[proj_dd, sel_i], outputs=[ed_df, ed_status, ed_timeline]) \
            .then(select_line, inputs=[proj_dd, sel_i], outputs=line_outs)

        for c in (cover_top, cover_size, sub_size, sub_pos):
            c.release(editor_preview, inputs=pv_inputs, outputs=pv_outputs, show_progress='hidden')
        for c in (cover_mode, sub_follow, sub_box, ed_subtitle):
            c.input(editor_preview, inputs=pv_inputs, outputs=pv_outputs, show_progress='hidden')
        for c in (ed_subtitle, cover_mode, sub_follow):
            c.input(_sub_controls, inputs=sub_ctl_inputs, outputs=sub_ctl_outputs, show_progress='hidden')
        auto_btn.click(auto_find_band, inputs=proj_dd, outputs=[cover_top, cover_size, auto_status]) \
            .then(editor_preview, inputs=pv_inputs, outputs=pv_outputs)

        # ---------------- Xuất ----------------
        for ev in (exp_name.blur, exp_name.submit):
            ev(rename_project, inputs=[proj_dd, exp_name], outputs=[proj_dd, exp_name], show_progress='hidden')
        exp_inputs = [ed_subtitle, cover_mode, cover_top, cover_size, sub_size, sub_pos, sub_follow, sub_box, out_dir,
                      *wm_comps]
        exp_outputs = [exp_html, res_video, res_info, ed_tabs, export_btn, export_all_btn]
        for btn, fn in ((export_btn, export_current), (export_all_btn, export_all)):
            # chung concurrency_id: 2 nút xuất không bao giờ chạy song song (pyVideoTrans dùng chung trạng thái)
            btn.click(fn, inputs=[proj_dd, exp_name, *exp_inputs], outputs=exp_outputs, concurrency_id='export') \
                .then(lambda pid: _render_status(PROJECTS[pid]) if pid in PROJECTS else '', inputs=proj_dd,
                      outputs=ed_status) \
                .then(rename_project, inputs=[proj_dd, gr.State('')], outputs=[proj_dd, exp_name]) \
                .then(_stepper_for, inputs=[page_state, proj_dd], outputs=stepper)
        pick_btn.click(pick_out_dir, inputs=out_dir, outputs=out_dir)
        open_btn.click(open_output_dir, inputs=out_dir)

        # ---------------- Dọn dẹp ----------------
        clean_refresh.click(clean_report, inputs=clean_dl, outputs=clean_html)
        clean_dl.input(clean_report, inputs=clean_dl, outputs=clean_html, show_progress='hidden')
        clean_ok.click(do_cleanup, inputs=[clean_dl, link_items],
                       outputs=[files_state, selector, proj_dd, clean_html, nav_edit]) \
            .then(_enter_editor, inputs=proj_dd, outputs=enter_outputs)

        # ---------------- Cookie ----------------
        cookie_btn.upload(save_cookie, inputs=cookie_btn, outputs=[cookie_state, cookie_info])
        cookie_del.click(delete_cookie, outputs=[cookie_state, cookie_info])

        # ---------------- Lưu / nạp cài đặt ----------------
        setting_comps = [source, voice, subtitle, keep_bgm, voice_autorate, video_autorate, model, cuda,
                         browser, cover_mode, cover_top, cover_size, sub_size, sub_pos, sub_follow, sub_box, out_dir,
                         ai_erase, dl_out_dir, wm_logo_on, wm_logo_pos, wm_logo_size, wm_logo_opacity, wm_text_on,
                         wm_text, wm_color, wm_text_pos, wm_text_size, wm_text_opacity, wm_margin, plat_state]
        app.load(lambda s: load_settings(s, roles, has_cuda), inputs=saved, outputs=setting_comps) \
            .then(lambda s: gr.update(value=s), inputs=subtitle, outputs=ed_subtitle) \
            .then(_sub_controls, inputs=sub_ctl_inputs, outputs=sub_ctl_outputs) \
            .then(init_cookie, outputs=[cookie_state, cookie_info]) \
            .then(lambda: (gr.update(value=_edit_label()),
                           gr.update(choices=_proj_choices(), value=next(iter(PROJECTS), None))),
                  outputs=[nav_edit, proj_dd]) \
            .then(runbar_html, inputs=runbar_in, outputs=runbar) \
            .then(stats_html, inputs=stats_in, outputs=stats) \
            .then(_wm_boxes, inputs=[wm_logo_on, wm_text_on], outputs=[wm_logo_box, wm_text_box, wm_opts_box]) \
            .then(saved_logo, outputs=wm_logo) \
            .then(wm_preview, inputs=wm_in, outputs=wm_out) \
            .then(runbar_html, inputs=runbar_in, outputs=runbar) \
            .then(lambda k: set_plat(k)()[1:], inputs=plat_state, outputs=[*plat_btns.values(), plat_hint, *link_boxes]) \
            .then(lambda: _render_queue([], 'dub'), outputs=queue_html)   # trang mới: hàng đợi trống (và nhớ là trống)
        for c in setting_comps:
            c.change(lambda *v: dict(zip(SETTING_NAMES, v)), inputs=setting_comps, outputs=saved,
                     show_progress='hidden')
    return app


def _theme():
    """Theme tối theo phong cách app tham khảo; đặt cùng giá trị cho cả chế độ sáng và tối."""
    import inspect
    vals = dict(
        body_background_fill='#090a16', body_text_color='#eceef5', body_text_color_subdued='#9096ab',
        background_fill_primary='#0f1120', background_fill_secondary='#151829',
        border_color_primary='rgba(255,255,255,0.10)', border_color_accent='#8b5cf6',
        border_color_accent_subdued='rgba(139,92,246,0.4)', color_accent='#8b5cf6',
        color_accent_soft='rgba(139,92,246,0.18)',
        link_text_color='#a78bfa', link_text_color_hover='#c4b5fd', link_text_color_active='#c4b5fd',
        link_text_color_visited='#a78bfa', code_background_fill='#0b0c18',
        block_background_fill='transparent', block_border_width='0px', block_border_color='transparent',
        block_shadow='none', block_padding='0px', block_radius='12px',
        block_label_background_fill='transparent', block_label_text_color='#9096ab', block_label_border_width='0px',
        block_title_background_fill='transparent', block_title_text_color='#b8bdd0', block_title_text_weight='600',
        block_title_text_size='13px', block_title_padding='0 0 6px 0', block_info_text_color='#7c8197',
        panel_background_fill='transparent', panel_border_width='0px', accordion_text_color='#dfe2ee',
        input_background_fill='#0b0c18', input_background_fill_focus='#0d0f1d', input_background_fill_hover='#0d0f1d',
        input_border_color='rgba(255,255,255,0.10)', input_border_color_hover='rgba(255,255,255,0.18)',
        input_border_color_focus='rgba(139,92,246,0.65)', input_border_width='1px', input_radius='10px',
        input_shadow='none', input_shadow_focus='0 0 0 3px rgba(139,92,246,0.18)',
        input_placeholder_color='#5f6478', input_text_size='14px',
        checkbox_background_color='#0b0c18', checkbox_background_color_hover='#0d0f1d',
        checkbox_background_color_focus='#0d0f1d', checkbox_background_color_selected='#8b5cf6',
        checkbox_border_color='rgba(255,255,255,0.22)', checkbox_border_color_hover='rgba(255,255,255,0.35)',
        checkbox_border_color_focus='#8b5cf6', checkbox_border_color_selected='#8b5cf6',
        checkbox_label_background_fill='rgba(255,255,255,0.04)',
        checkbox_label_background_fill_hover='rgba(255,255,255,0.07)',
        checkbox_label_background_fill_selected='rgba(139,92,246,0.18)',
        checkbox_label_border_color='rgba(255,255,255,0.10)',
        checkbox_label_border_color_hover='rgba(255,255,255,0.18)',
        checkbox_label_border_color_selected='rgba(139,92,246,0.55)', checkbox_label_border_width='1px',
        checkbox_label_text_color='#d6d9e6', checkbox_label_text_color_selected='#ffffff',
        slider_color='#8b5cf6', loader_color='#8b5cf6', stat_background_fill='#8b5cf6',
        table_border_color='rgba(255,255,255,0.08)', table_even_background_fill='#10121f',
        table_odd_background_fill='#0d0f1b', table_row_focus='rgba(139,92,246,0.18)', table_text_color='#dfe2ee',
        table_radius='12px',
        button_border_width='1px', button_large_radius='12px', button_medium_radius='10px',
        button_small_radius='9px', button_transform_hover='none', button_transform_active='none',
        button_primary_background_fill='linear-gradient(90deg, #7c5cff, #a47bff)',
        button_primary_background_fill_hover='linear-gradient(90deg, #8a6dff, #b18cff)',
        button_primary_border_color='transparent', button_primary_border_color_hover='transparent',
        button_primary_text_color='#ffffff', button_primary_text_color_hover='#ffffff',
        button_primary_shadow='0 6px 18px rgba(124,92,255,0.28)',
        button_primary_shadow_hover='0 8px 22px rgba(124,92,255,0.36)',
        button_secondary_background_fill='rgba(255,255,255,0.06)',
        button_secondary_background_fill_hover='rgba(255,255,255,0.10)',
        button_secondary_border_color='rgba(255,255,255,0.14)',
        button_secondary_border_color_hover='rgba(255,255,255,0.24)',
        button_secondary_text_color='#e6e8f2', button_secondary_text_color_hover='#ffffff',
        button_secondary_shadow='none', button_secondary_shadow_hover='none',
        button_cancel_background_fill='rgba(240,80,110,0.12)', button_cancel_background_fill_hover='rgba(240,80,110,0.22)',
        button_cancel_border_color='rgba(240,80,110,0.45)', button_cancel_border_color_hover='rgba(240,80,110,0.7)',
        button_cancel_text_color='#ff7a93', button_cancel_text_color_hover='#ffffff',
        button_cancel_shadow='none', button_cancel_shadow_hover='none',
        shadow_drop='0 6px 22px rgba(0,0,0,0.25)', shadow_drop_lg='0 12px 32px rgba(0,0,0,0.35)',
        error_background_fill='rgba(240,80,110,0.12)', error_border_color='rgba(240,80,110,0.4)',
        error_text_color='#ff7a93', error_icon_color='#ff7a93',
        layout_gap='14px', form_gap_width='0px',
    )
    allowed = set(inspect.signature(gr.themes.Base.set).parameters)
    kw = {k: v for k, v in vals.items() if k in allowed}
    kw.update({f'{k}_dark': v for k, v in vals.items() if f'{k}_dark' in allowed})
    theme = gr.themes.Base(primary_hue='violet', secondary_hue='pink', neutral_hue='slate', radius_size='lg',
                           font=[gr.themes.GoogleFont('Inter'), 'Segoe UI', 'system-ui', 'sans-serif'])
    return theme.set(**kw)


if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=7861)
    ap.add_argument('--share', action='store_true')
    a = ap.parse_args()
    _clean_stale_tmp()
    from gradio.utils import get_upload_folder
    ui = build_ui()
    ui.queue(default_concurrency_limit=1)
    ui.launch(server_name=a.host, server_port=a.port, share=a.share, inbrowser=True, theme=_theme(),
              css=CSS + _icon_css(), head=HEAD_JS,
              # get_upload_folder: file người dùng tải lên, để thẻ <video> của trình sửa phát được
              allowed_paths=[str(DOWNLOAD_DIR), f'{ROOT_DIR}/output', f'{ROOT_DIR}/tmp', get_upload_folder()])
