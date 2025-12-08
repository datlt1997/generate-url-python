import os
import uuid
import yt_dlp

def download_subtitles(url, out_dir):
    """
    Kiểm tra video có phụ đề không.
    Nếu có, tải phụ đề dạng srt.
    """
    uid = str(uuid.uuid4())
    out_template = os.path.join(out_dir, f"{uid}.%(ext)s")
    
    ydl_opts = {
        "skip_download": True,
        "writesubtitles": True,
        "subtitlesformat": "srt",
        "subtitleslangs": ["en", "en-US", "vi"],  # ưu tiên en, vi
        "outtmpl": out_template,
        "quiet": True
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        if info.get("subtitles") or info.get("automatic_captions"):
            ydl.download([url])
            for lang in ["en", "en-US", "vi"]:
                path = os.path.join(out_dir, f"{uid}.{lang}.srt")
                if os.path.exists(path):
                    return path
    return None


def download_audio(url, out_dir):
    """
    Tải trực tiếp audio từ YouTube và xuất WAV 16kHz mono.
    """
    uid = str(uuid.uuid4())
    out_template = os.path.join(out_dir, f"{uid}.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "noplaylist": True,
        "quiet": True,
        "outtmpl": out_template,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
            "preferredquality": "128"
        }],
        "postprocessor_args": [
            "-ar", "16000",
            "-ac", "1"
        ]
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        audio_file = ydl.prepare_filename(info)
        audio_file = os.path.splitext(audio_file)[0] + ".wav"

    if not os.path.exists(audio_file) or os.path.getsize(audio_file) == 0:
        raise Exception("Tải audio thất bại hoặc file rỗng")

    return audio_file
