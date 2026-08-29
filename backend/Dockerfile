FROM python:3.11-slim

WORKDIR /app

# Install OpenCV system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create data directories
RUN mkdir -p data/uploads data/heatmaps ml/models

# Train model if not already present (during build)
RUN python ml/generate_dataset.py --seeds 200 && python ml/train.py

# Clean up dataset to reduce image size (model is already saved)
RUN rm -rf data/synthetic_dataset

EXPOSE 8000

ENV PYTHONPATH=/app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
