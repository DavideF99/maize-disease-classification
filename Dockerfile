# 1. Use an official lightweight Python image
FROM python:3.11-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Install modern system dependencies for OpenCV (Headless)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglx-mesa0 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy and install requirements
COPY requirements_prod.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy your source code and checkpoints
COPY src/ ./src/
COPY checkpoints/ ./checkpoints/

# 6. Expose the port FastAPI runs on
EXPOSE 8000

# 7. Command to run the API
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]