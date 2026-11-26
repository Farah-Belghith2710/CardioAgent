# Service 3 : audio-engine (ONNX Runtime + Librosa)

# 1. start from an image that already contains Python
FROM python:3.12-slim

# 2. inside the container, we work in the /app folder
WORKDIR /app

# 3. install the Python libraries (librosa, onnxruntime, fastapi, ...)
#    the same versions of numpy / librosa / soundfile / soxr as in the notebooks -> the same spectrograms
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. copy our code, the models (ONNX files made by the notebook 05) and benchmark.json (their scores)
COPY main.py .
COPY benchmark.json .
COPY *.onnx .

# 5. the API listens on port 8001 (8000 is already used by the api-gateway)
EXPOSE 8001

# 6. command run when the container starts: uvicorn serves the FastAPI app "app" from main.py
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
