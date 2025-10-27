FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# Basic dependencies
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip python3.10-dev \
    git wget curl libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

WORKDIR /workspace/panda-door-rl

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY . .

CMD ["python3", "src/train.py", "--config", "configs/train_panda_door.yaml"]
