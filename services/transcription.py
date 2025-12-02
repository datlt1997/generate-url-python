from faster_whisper import WhisperModel
from pydub import AudioSegment, silence
import assemblyai as aai

def transcribe_audio(audio_path):
    # ----- 1. Load model -----
    # bạn có thể đổi tiny, base, small, medium, large-v3
    model = WhisperModel("base", device="cpu")   # nếu có GPU: device="cuda"

    # ----- 2. Cắt audio theo im lặng -----
    audio = AudioSegment.from_file(audio_path)

    chunks = silence.split_on_silence(
        audio,
        min_silence_len=250,                # ngắt khi im lặng 0.1s
        silence_thresh=audio.dBFS - 16,
        keep_silence=200
    )

    timestamps = []
    texts = []

    current_time = 0  # thời gian tích lũy

    # ----- 3. Xử lý từng chunk -----
    for chunk in chunks:
        duration = len(chunk) / 1000.0  # ms -> giây

        start = current_time
        end = current_time + duration

        temp_name = "_temp_chunk.wav"
        chunk.export(temp_name, format="wav")

        # ----- 4. faster-whisper: transcribe -----
        segments, info = model.transcribe(
            temp_name,
            beam_size=5,
            vad_filter=False,
            language="vi"
        )

        # faster_whisper trả về nhiều segments, nên phải nối lại
        text = " ".join([seg.text for seg in segments]).strip()

        if text.strip() == "":
            current_time = end
            continue

        timestamps.append({
            "start": round(start, 2),
            "end": round(end, 2),
            "text": text
        })
        texts.append(text)

        current_time = end

    return timestamps, texts

def read_subtitle(sub_file):
    """
    Đọc file srt và trả về text.
    """
    lines = []
    with open(sub_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.isdigit() and "-->" not in line:
                lines.append(line)
    return " ".join(lines)


def format_timestamp(seconds):
    if seconds is None:
        return "00:00"
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"

def transcribe_with_timestamps(audio_path):
    aai.settings.api_key = "248c8e9cf8b847ea8675c72ce9a38ebd"
    transcriber = aai.Transcriber()

    transcript = transcriber.transcribe(
        audio_path,
        config=aai.TranscriptionConfig(
            punctuate=True,
            format_text=True,
            speaker_labels=True,
            filter_disfluencies=True,
            language_code="vi"  # nếu muốn tiếng Việt
        )
    )

    # transcript.words trả về list từ với timestamp
    timestamps = []
    chunk = []
    gap_threshold = 1.0  # giây

    last_end = 0
    for w in transcript.words:
        start_s = w.start / 1000
        end_s = w.end / 1000
        if chunk and start_s - last_end > gap_threshold:
            timestamps.append({
                "start": round(chunk[0]['start'], 2),
                "end": round(chunk[-1]['end'], 2),
                "text": " ".join([c['text'] for c in chunk])
            })
            chunk = []
        chunk.append({"text": w.text, "start": start_s, "end": end_s})
        last_end = end_s

    if chunk:
        timestamps.append({
            "start": round(chunk[0]['start'], 2),
            "end": round(chunk[-1]['end'], 2),
            "text": " ".join([c['text'] for c in chunk])
        })

    return timestamps, transcript.text