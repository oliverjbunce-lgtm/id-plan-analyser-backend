# ID Plan Analyser — Backend

FastAPI inference server for the Independent Doors door detection portal.

## Setup

```bash
pip install -r requirements.txt
```

Place model weights at `model/best.pt` (copy from `ai-model/final_yolov8l_final2/weights/best.pt`).

## Run locally

```bash
MODEL_PATH=model/best.pt uvicorn main:app --reload --port 8000
```

## Deploy (Hugging Face Spaces)

1. Create a new Space → Gradio SDK → Docker
2. Upload all files in this directory + `model/best.pt`
3. Set `MODEL_PATH=/app/model/best.pt` in Space secrets

## Endpoints

- `GET  /health` — liveness check
- `POST /upload` — upload PDF, get page thumbnails + suggested page
- `POST /analyse` — upload PDF + page number, get door detections + annotated image
