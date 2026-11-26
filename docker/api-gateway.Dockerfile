# 1. start from an image that already contains Python
FROM python:3.12-slim

# 2. inside the container, we work in the /app folder
WORKDIR /app

# 3. install the Python libraries (fastapi, uvicorn, ...)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. copy our code and the HTML page into the container
COPY main.py .
COPY index.html .

# 5. the API listens on port 8000
EXPOSE 8000

# 6. command run when the container starts: uvicorn serves the FastAPI app "app" from main.py
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]