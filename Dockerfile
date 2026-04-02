FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 wget && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Download real model weights from HuggingFace (bypasses Git LFS pointer issue)
RUN mkdir -p /app/model && \
    wget -q -O /app/model/best.pt \
    "https://huggingface.co/datasets/oliverbunce/id-door-detection-training/resolve/main/models/best_v2.pt?download=true"

ENV MODEL_PATH=/app/model/best.pt

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
