from transformers import pipeline
import os
from dotenv import load_dotenv
from openai import OpenAI
import google.generativeai as genai
from gtts import gTTS
import base64
from io import BytesIO

load_dotenv()

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Vui lòng đặt OPENAI_API_KEY trong biến môi trường")
client = OpenAI(api_key=api_key)

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def chatgpt_summarize(text, max_chunk_chars=3000):
    """
    Chia text dài thành chunks và tóm tắt bằng ChatGPT
    """
    chunks = [text[i:i+max_chunk_chars] for i in range(0, len(text), max_chunk_chars)]
    summaries = []

    for c in chunks:
        prompt = f"""
Bạn là một trợ lý AI tóm tắt nội dung.

Nhiệm vụ của bạn:


1. Đọc nội dung dưới đây (văn bản hoặc transcript video).
2. Tạo đúng **một tiêu đề nội dung tóm tắt duy nhất** (tối đa 10 từ).
3. Chia nội dung thành các phần chính, tối đa 4-5 phần, mỗi phần 4-5 câu.
4. Trả về kết quả theo đúng định dạng HTML sau:

<h2>Tiêu đề nội dung tóm tắt</h2>

<h4>1. Tiêu đề phần 1</h4>
<p>Nội dung phần 1</p>

<h4>2. Tiêu đề phần 2</h4>
<p>Nội dung phần 2</p>
...

**QUY TẮC BẮT BUỘC:**
- Chỉ được phép tạo **một thẻ <h2> duy nhất** ở đầu.
- Tuyệt đối không được tạo thêm thẻ <h2> ở bất kỳ vị trí nào khác.
- Mọi tiêu đề của phần phải dùng <h4>.
- Mọi nội dung phần phải nằm trong <p>.
- Không thêm mô tả, không thêm thẻ HTML khác.
- Không được tự tạo nhóm nội dung lớn.

Nội dung cần tóm tắt:
\"\"\"{c}\"\"\"
"""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        summaries.append(response.choices[0].message.content.strip())

    # ghép tất cả summary
    return " ".join(summaries)

def gemini_summarize(text, max_chunk_chars=2000):
    """
    Chia text dài thành chunks và tóm tắt bằng Google Gemini
    """
    chunks = [text[i:i+max_chunk_chars] for i in range(0, len(text), max_chunk_chars)]
    summaries = []

    modelGemini = genai.GenerativeModel("gemini-2.0-flash")

    for c in chunks:
        prompt = f"""
Bạn là một trợ lý AI tóm tắt nội dung.

Nhiệm vụ của bạn:

1. Đọc nội dung dưới đây (văn bản hoặc transcript video).
2. Tạo đúng **một tiêu đề nội dung tóm tắt duy nhất** (tối đa 10 từ).
3. Chia nội dung thành các phần chính, tối đa 4-5 phần, mỗi phần 4–5 câu.
4. Trả về kết quả theo đúng định dạng HTML sau:

<h2>Tiêu đề nội dung tóm tắt</h2>

<h4>1. Tiêu đề phần 1</h4>
<p>Nội dung phần 1</p>

<h4>2. Tiêu đề phần 2</h4>
<p>Nội dung phần 2</p>
...

**QUY TẮC BẮT BUỘC:**
- Chỉ được phép tạo **một thẻ <h2> duy nhất** ở đầu.
- Tuyệt đối không được tạo thêm thẻ <h2> ở bất kỳ vị trí nào khác.
- Mọi tiêu đề của phần phải dùng <h4>.
- Mọi nội dung phần phải nằm trong <p>.
- Không thêm mô tả, không thêm thẻ HTML khác.
- Không được tự tạo nhóm nội dung lớn.

Nội dung cần tóm tắt:
\"\"\"{c}\"\"\"
"""

        
        response = modelGemini.generate_content(prompt)
        summaries.append(response.text.strip())

    return " ".join(summaries)


def gemini_text_to_audio(text, output_path="summary_gemini.wav"):
    print("==== Bắt đầu chuyển text sang audio ====", flush=True)
    
    print(f"Text gửi sang TTS:\n{text[:200]}{'...' if len(text) > 200 else ''}", flush=True)
    
    tts = gTTS(text=text, lang='vi')
    fp = BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    
    audio_bytes = fp.read()
    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
    
    with open(output_path, "wb") as f:
        f.write(audio_bytes)
    
    print(f"Đã lưu file audio: {output_path}", flush=True)
    print("==== Hoàn tất chuyển text sang audio ====", flush=True)
    
    return output_path, audio_base64
