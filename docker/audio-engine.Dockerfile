# Service 3 : audio-engine (ONNX Runtime + Librosa)
# Multi-stage build: stage 1 installs the Python libraries, stage 2 is a clean image that only runs the engine.
# Build context = the repo root (docker-compose.yml): docker build -f docker/audio-engine.Dockerfile .

# ---------- stage 1 (builder): install the libraries in /install ----------
FROM python:3.12-slim AS builder
WORKDIR /build
# the same versions of numpy / librosa / soundfile / soxr as in the notebooks -> the same spectrograms
COPY audio_engine/requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---------- stage 2 (runtime): only the libraries, our code and the models ----------
FROM python:3.12-slim
WORKDIR /app
COPY --from=builder /install /usr/local
# our code, benchmark.json (the scores of the models), magma_lut.npy (the colours of the ResNet50 image)
COPY audio_engine/main.py audio_engine/benchmark.json audio_engine/magma_lut.npy ./
# the models exported to ONNX by the notebook 05 (resnet50.onnx is optional: the engine skips a missing file)
COPY audio_engine/*.onnx ./

# the API listens on port 8001 (8000 is the api-gateway)
EXPOSE 8001
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
