FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV HF_HOME=/work/data/hf_cache

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3.10-dev python3-pip \
        git curl wget ca-certificates \
        ffmpeg libsndfile1 sox \
        libespeak-ng1 espeak-ng-data \
        build-essential \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel

# TensorFlow for microWakeWord training. microWakeWord targets TFLite Micro, so
# the training graph and quantization use TF, not PyTorch. CUDA wheels for
# tensorflow are matched to the CUDA 12.1 base image.
RUN pip install \
        tensorflow==2.16.1 \
        tensorflow-model-optimization==0.8.0 \
        tflite-runtime==2.14.0

# microWakeWord itself (Kevin Ahrendt). The repo is the source of truth for the
# trainer; install from a pinned commit so re-runs are reproducible.
ARG MWW_REF=v1.0.0
RUN git clone --depth 1 --branch ${MWW_REF} \
        https://github.com/kahrendt/microWakeWord.git /opt/microWakeWord \
    && pip install -e /opt/microWakeWord

# Audio + augmentation deps. Some overlap with the openWakeWord image but
# pinned independently — these two trainers should not have to share versions.
RUN pip install \
        audiomentations==0.36.0 \
        torch==2.1.2 torchaudio==2.1.2 \
            --index-url https://download.pytorch.org/whl/cu121 \
    && pip install \
        huggingface_hub==0.24.5 \
        datasets==2.20.0 \
        soundfile==0.12.1 \
        librosa==0.10.2 \
        mutagen==1.47.0 \
        pyyaml==6.0.2 \
        tqdm==4.66.4 \
        pronouncing==0.2.0 \
        onnx==1.16.1

# piper-sample-generator for synthetic positive utterances — same as the
# openWakeWord pipeline so the positive sample distribution stays comparable.
RUN git clone https://github.com/dscripka/piper-sample-generator.git /opt/piper-sample-generator \
    && pip install -r /opt/piper-sample-generator/requirements.txt \
    && mkdir -p /opt/piper-sample-generator/models \
    && wget -q -O /opt/piper-sample-generator/models/en-us-libritts-high.pt \
        https://github.com/rhasspy/piper-sample-generator/releases/download/v1.0.0/en-us-libritts-high.pt

WORKDIR /work
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
