# Dockerfile

# 1. Chọn base image chứa Python 3.9.13
FROM nvidia/cuda:12.1.0-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# # # Cập nhật hệ thống và cài đặt một số dependencies cần thiết
RUN apt-get update && apt-get install -y software-properties-common

RUN add-apt-repository ppa:deadsnakes/ppa

RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3.9-distutils \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# # # Tạo liên kết python3 -> python3.9 để dùng lệnh python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1

# # # Thiết lập python3.9 là python
RUN apt-get update && apt-get install -y \
    build-essential \
    zlib1g-dev \
    libncurses5-dev \
    libgdbm-dev \
    libnss3-dev \
    libssl-dev \
    libreadline-dev \
    libffi-dev \
    libsqlite3-dev \
    libbz2-dev \
    curl \
    git \
    wget \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# # Cập nhật pip và cài đặt các thư viện cần thiết    
RUN python3 -m pip install --upgrade pip

# # Cài đặt pip, setuptools và wheel để đảm bảo khả năng cài đặt gói
RUN pip install --upgrade pip setuptools wheel

# libs for OpenCV + ffmpeg decoding
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libgl1 ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 ffmpeg libsm6 libxext6 && rm -rf /var/lib/apt/lists/*

# 2. Thiết lập working directory trong container
WORKDIR /app

# # # Cài đặt các thư viện cần thiết cho dự án
RUN pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118

RUN pip install optimum-quanto

RUN python3.9 -m pip install --upgrade accelerate transformers

# 3. Copy file requirements (nếu có) vào container
COPY requirements.txt .

# # # 5. Cài đặt thư viện Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --ignore-installed -r requirements.txt
    
RUN pip uninstall -y opencv-python-headless
RUN pip install opencv-python

COPY . /app
EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "application.py", "--server.port=8501", "--server.address=0.0.0.0"]