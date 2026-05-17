FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    COQUI_TOS_AGREED=1

RUN apt-get update && apt-get install -y \
    git \
    curl \
    gcc \
    g++ \
    build-essential \
    libsndfile1 \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv /root/.local/bin/uv /usr/local/bin/uv

# Clone and install TTS from source
RUN git clone https://github.com/coqui-ai/TTS /tmp/TTS \
    && cd /tmp/TTS \
    && uv pip install --system -e ".[all]" \
    && rm -rf /tmp/TTS/.git

# Install torch and runpod
RUN uv pip install --system \
    --index-url https://download.pytorch.org/whl/cu121 \
    --extra-index-url https://pypi.org/simple \
    torch torchaudio runpod

# Pre-download XTTS-v2 model at build time
RUN python3 -c "from TTS.api import TTS; TTS('tts_models/multilingual/multi-dataset/xtts_v2')"

WORKDIR /app
COPY . /app

EXPOSE 8080
CMD ["python3", "tts_inference.py"]