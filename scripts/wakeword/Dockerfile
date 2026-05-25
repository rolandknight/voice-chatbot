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

RUN pip install \
        torch==2.4.1 torchaudio==2.4.1 \
        --index-url https://download.pytorch.org/whl/cu124

RUN pip install \
        openwakeword==0.6.0 \
        torchinfo==1.8.0 \
        torchmetrics==0.11.4 \
        huggingface_hub==0.24.5 \
        datasets==2.20.0 \
        audiomentations==0.36.0 \
        mutagen==1.47.0 \
        speechbrain==0.5.16 \
        onnx==1.16.1 \
        onnxruntime-gpu==1.18.1 \
        tf2onnx==1.16.1 \
        # openwakeword's convert_onnx_to_tflite imports onnx_tf, which on PyPI
        # is only at 1.10.0 (Aug 2022) and breaks against TF >= 2.15. Pin the
        # whole TF stack to the 2.14 line that onnx-tf 1.10.0 was last tested
        # with; tensorflow-addons + tensorflow-probability are transitive onnx-tf
        # deps whose versions must match TF.
        tensorflow-cpu==2.14.1 \
        tensorflow-probability==0.22.1 \
        tensorflow-addons==0.22.0 \
        onnx-tf==1.10.0 \
        pyyaml==6.0.2 \
        tqdm==4.66.4 \
        pronouncing==0.2.0 \
        torch-audiomentations==0.11.1 \
        acoustics==0.2.6

# openwakeword's pip wheel ships Python only; the ONNX feature-extraction
# models (melspectrogram, embedding) must be fetched separately and dropped
# into the package's resources/models/ dir. Bake them in so the image is
# self-contained and the augment phase can find them.
RUN python -c "import openwakeword.utils; openwakeword.utils.download_models()"

RUN git clone https://github.com/dscripka/piper-sample-generator.git /opt/piper-sample-generator \
    && pip install -r /opt/piper-sample-generator/requirements.txt \
    && mkdir -p /opt/piper-sample-generator/models \
    && wget -q -O /opt/piper-sample-generator/models/en-us-libritts-high.pt \
        https://github.com/rhasspy/piper-sample-generator/releases/download/v1.0.0/en-us-libritts-high.pt

WORKDIR /work
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
