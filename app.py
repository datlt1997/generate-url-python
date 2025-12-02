import os
from flask import Flask, render_template, request, jsonify
from services.downloader import download_audio, download_subtitles
from services.transcription import transcribe_audio, transcribe_with_timestamps, format_timestamp
from services.summarizer import chatgpt_summarize, gemini_summarize, gemini_text_to_audio
import base64
import re

app = Flask(__name__)

DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/submit", methods=["POST"])
def submit():
    data = request.get_json()
    text_form = data["text_form"]
    transribe_type = data["transribe_type"]
    model_type = data["model_type"]
    type = data["type"]
    summary = ""

    if not text_form:
        return "Vui lòng nhập nội dung", 400
    file_to_cleanup = None

    if type == "url":
        try:
            audio_file = download_audio(text_form, DOWNLOAD_DIR)
            file_to_cleanup = audio_file
            if transribe_type == "whisper":
                timestamps, text = transcribe_audio(audio_file)
            else:
                timestamps, text = transcribe_with_timestamps(audio_file)
            
            formatted_subs = []
            for item in timestamps:
                start = format_timestamp(item['start'])
                line = f"{start} — {item['text']}"
                formatted_subs.append(line)

            transcript = "\n".join(formatted_subs)
            textSummary = " ".join(text)
                

        except Exception as e:
            return f"Lỗi xử lý video: {e}", 500
        
        try:
            if file_to_cleanup and os.path.exists(file_to_cleanup):
                os.remove(file_to_cleanup)
        except Exception:
            pass

    elif type == "text":
        textSummary = text_form
        transcript = ""
        timestamps = []

    if model_type == "gemini":
        summary = gemini_summarize(textSummary)
    else:
        summary = chatgpt_summarize(textSummary)

    try:
        clean_summary = summary

        clean_summary = re.sub(r'<h2[^>]*>', '\n', clean_summary)
        clean_summary = re.sub(r'</h2>', '\n', clean_summary)
        clean_summary = re.sub(r'<h3[^>]*>', '\n', clean_summary)
        clean_summary = re.sub(r'</h3>', '\n', clean_summary)
        clean_summary = re.sub(r'<p[^>]*>', '\n', clean_summary)
        clean_summary = re.sub(r'</p>', '\n', clean_summary)

        # Xóa mọi thẻ còn lại
        clean_summary = re.sub(r'<[^>]+>', '', clean_summary).strip()
        audio_path, _ = gemini_text_to_audio(clean_summary, "summary_gemini.wav")

        # encode base64 để trả về API
        with open(audio_path, "rb") as f:
            audio_base64 = base64.b64encode(f.read()).decode("utf-8")

    except Exception as e:
        audio_base64 = None
        print("Lỗi tạo audio từ Gemini:", e)

    return jsonify({
        "summary": summary,
        "transcript": transcript,
        "timestamps": timestamps,
        "summary_audio": audio_base64
    })



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
