FROM python:3.11-slim

# Cài các gói hệ thống cần thiết
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements và cài đặt dependencies
COPY requirements.txt .
RUN python -m pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ source code
COPY . .

# Mở port 5000
EXPOSE 5000

# Command chạy FastAPI app
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]
