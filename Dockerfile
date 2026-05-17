# Use CUDA devel image for full compiler toolchain (required by Triton/Unsloth)
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Environment settings
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install Python, curl, and build tools
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-dev \
    curl \
    gcc \
    g++ \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Make python3.10 the default python
RUN ln -sf /usr/bin/python3.10 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python3

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv /root/.local/bin/uv /usr/local/bin/uv

# Set the working directory
WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt /app/
RUN uv pip install --system -r requirements.txt

# Copy remaining project files
COPY . /app

# Expose application port
EXPOSE 8080

# Run the application
CMD ["python", "tts_inference.py"]