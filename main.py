"""
Independent Doors — Plan Analyser API
FastAPI backend: PDF upload, page thumbnail generation, YOLOv8 inference
"""
import os, uuid, shutil, base64, io
from contextlib import asynccontextmanager
from pathlib import Path
from collections import Counter
from typing import Optional

import fitz  # PyMuPDF
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import HfApi, CommitOperationAdd
from pydantic import BaseModel
from ultralytics import YOLO

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "model/best.pt")
HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_DATASET_REPO = os.getenv("HF_DATASET_REPO", "oliverbunce/id-plan-analyser-training")
UPLOAD_DIR = Path("/tmp/id-uploads")
THUMB_DIR = Path("/tmp/id-thumbs")
TRAINING_DIR = Path("/tmp/id-training")
THUMB_DPI = 72        # low-res for page picker
INFER_DPI = 150       # resolution for model inference

CLASSES = [
    "L_prehung_door", "R_prehung_door", "Double_prehung_door",
    "S_cavity_slider", "D_cavity_slider",
    "Wardrobe_sliding_two_doors_1", "Wardrobe_sliding_two_doors_2",
    "Wardrobe_sliding_four_doors", "Bi_folding_door",
    "Barn_wall_slider", "D_bi_folding_door", "Wardrobe_sliding_three_doors",
]

for d in [UPLOAD_DIR, THUMB_DIR, TRAINING_DIR / "images", TRAINING_DIR / "labels"]:
    d.mkdir(parents=True, exist_ok=True)

# ── Pydantic models ───────────────────────────────────────────────────────────
class CorrectedBox(BaseModel):
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

class FeedbackPayload(BaseModel):
    session_id: str
    image_b64: str  # raw base64, no data: prefix
    boxes: list[CorrectedBox]

# ── App ───────────────────────────────────────────────────────────────────────
model: Optional[YOLO] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model_path = Path(MODEL_PATH)
    # Auto-download if file is missing or is a Git LFS pointer (< 200 bytes)
    if not model_path.exists() or model_path.stat().st_size < 200:
        hf_token = os.getenv("HF_TOKEN", "")
        hf_model = os.getenv("HF_MODEL_FILE", "models/best_v2.pt")
        hf_repo = os.getenv("HF_MODEL_REPO", "oliverbunce/id-door-detection-training")
        if hf_token:
            print(f"Downloading model from HuggingFace ({hf_repo}/{hf_model})…")
            try:
                from huggingface_hub import hf_hub_download
                src = hf_hub_download(repo_id=hf_repo, filename=hf_model, repo_type="dataset", token=hf_token)
                import shutil
                model_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(src, str(model_path))
                print("Model downloaded ✓")
            except Exception as e:
                print(f"⚠️  Model download failed: {e}")
        else:
            print("⚠️  HF_TOKEN not set — cannot download model")
    if model_path.exists() and model_path.stat().st_size > 200:
        print(f"Loading model from {MODEL_PATH}…")
        model = YOLO(MODEL_PATH)
        print("Model ready ✓")
    else:
        print(f"⚠️  Model not found at {MODEL_PATH} — inference will fail")
    yield  # app runs here
    model = None

app = FastAPI(title="ID Plan Analyser", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ───────────────────────────────────────────────────────────────────
def suggest_floor_plan_page(doc: fitz.Document) -> int:
    """
    Return 1-indexed page number most likely to be the floor plan.
    Heuristic: highest vector drawing count, with keyword boost.
    """
    scores = []
    kw = {"floor plan", "ground floor", "level 1", "first floor",
          "floor layout", "room layout", "lighting plan"}
    for i, page in enumerate(doc):
        text = page.get_text().lower()
        drawing_count = len(page.get_drawings())
        keyword_bonus = 500 if any(k in text for k in kw) else 0
        scores.append(drawing_count + keyword_bonus)
    best = scores.index(max(scores))
    return best + 1  # 1-indexed


def pdf_to_thumb_b64(page: fitz.Page, dpi: int = 72) -> str:
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    return base64.b64encode(pix.tobytes("jpeg")).decode()


def read_training_count() -> int:
    count_file = TRAINING_DIR / "count.txt"
    if count_file.exists():
        try:
            return int(count_file.read_text().strip())
        except ValueError:
            return 0
    return 0


def write_training_count(count: int):
    (TRAINING_DIR / "count.txt").write_text(str(count))


def push_to_dataset(session_id: str, image_bytes: bytes, label_text: str):
    """
    Commit one training sample (image + YOLO label) to the HF dataset repo.
    Runs in a background task — failures are logged but don't affect the response.
    """
    if not HF_TOKEN:
        print("⚠️  HF_TOKEN not set — skipping dataset push")
        return
    try:
        api = HfApi(token=HF_TOKEN)
        api.create_repo(HF_DATASET_REPO, repo_type="dataset", exist_ok=True, private=True)
        api.create_commit(
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            commit_message=f"training: add sample {session_id[:8]}",
            operations=[
                CommitOperationAdd(
                    path_in_repo=f"data/images/{session_id}.png",
                    path_or_fileobj=image_bytes,
                ),
                CommitOperationAdd(
                    path_in_repo=f"data/labels/{session_id}.txt",
                    path_or_fileobj=label_text.encode(),
                ),
            ],
        )
        print(f"✓ Training sample {session_id[:8]} pushed to {HF_DATASET_REPO}")
    except Exception as exc:
        print(f"⚠️  Failed to push training sample {session_id[:8]}: {exc}")


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok"}

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Accept a PDF, generate page thumbnails, suggest the floor plan page.
    Returns list of page thumbnail URLs and the suggested page number.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted.")

    session_id = uuid.uuid4().hex
    pdf_path = UPLOAD_DIR / f"{session_id}.pdf"

    with open(pdf_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    doc = fitz.open(str(pdf_path))
    page_count = doc.page_count
    pages = []

    for i in range(min(page_count, 30)):  # cap at 30 pages for speed
        b64 = pdf_to_thumb_b64(doc[i], dpi=THUMB_DPI)
        pages.append({
            "page": i + 1,
            "url": f"data:image/jpeg;base64,{b64}"
        })

    suggested = suggest_floor_plan_page(doc)
    doc.close()

    return {
        "session_id": session_id,
        "total_pages": page_count,
        "suggested_page": suggested,
        "pages": pages,
    }


@app.post("/analyse")
async def analyse_plan(
    file: UploadFile = File(...),
    page: int = Form(1),
):
    """
    Run YOLOv8 inference on a specific page of the uploaded PDF.
    Returns annotated image URL, detection counts, and raw bounding boxes.
    """
    if model is None:
        raise HTTPException(503, "Model not loaded.")

    session_id = uuid.uuid4().hex
    pdf_path = UPLOAD_DIR / f"{session_id}.pdf"

    with open(pdf_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    doc = fitz.open(str(pdf_path))
    if page < 1 or page > doc.page_count:
        raise HTTPException(400, f"Page {page} out of range (1–{doc.page_count}).")

    # Render page at inference resolution
    mat = fitz.Matrix(INFER_DPI / 72, INFER_DPI / 72)
    pix = doc[page - 1].get_pixmap(matrix=mat)
    image_width = pix.width
    image_height = pix.height
    png_path = UPLOAD_DIR / f"{session_id}-p{page}.png"
    pix.save(str(png_path))
    doc.close()

    # Run inference (no file saving — encode annotated image in memory)
    results = model(str(png_path), imgsz=1024, conf=0.25)
    r = results[0]

    # Parse detections
    class_counts: Counter = Counter()
    if r.boxes is not None:
        for cls_id in r.boxes.cls.tolist():
            class_counts[CLASSES[int(cls_id)]] += 1

    detections = [
        {"class": cls, "count": cnt}
        for cls, cnt in sorted(class_counts.items(), key=lambda x: -x[1])
    ]

    # Extract normalized bounding boxes (xyxyn = x1,y1,x2,y2 normalized 0-1)
    boxes = []
    if r.boxes is not None:
        xyxyn = r.boxes.xyxyn.tolist()
        cls_ids = r.boxes.cls.tolist()
        confs = r.boxes.conf.tolist()
        for i, (coords, cls_id, conf) in enumerate(zip(xyxyn, cls_ids, confs)):
            x1, y1, x2, y2 = coords
            boxes.append({
                "id": i,
                "class": CLASSES[int(cls_id)],
                "class_id": int(cls_id),
                "confidence": round(float(conf), 4),
                "x1": round(float(x1), 6),
                "y1": round(float(y1), 6),
                "x2": round(float(x2), 6),
                "y2": round(float(y2), 6),
            })

    # Return the clean rendered page image — the frontend draws its own
    # interactive boxes via Konva, so we must NOT bake the model's annotations
    # into the image (they would be uneditable and overlap the interactive layer).
    image_b64 = base64.b64encode(png_path.read_bytes()).decode()

    return {
        "session_id": session_id,
        "page_used": page,
        "total": sum(class_counts.values()),
        "detections": detections,
        "image_b64": f"data:image/png;base64,{image_b64}",
        "boxes": boxes,
        "image_width": image_width,
        "image_height": image_height,
    }


@app.post("/analyse-stored")
async def analyse_stored(session_id: str = Form(...), page: int = Form(1)):
    """
    Run inference on a previously uploaded PDF without re-uploading.
    Used for multi-page analysis — client calls this once per selected page.
    """
    if model is None:
        raise HTTPException(503, "Model not loaded.")

    safe_id = "".join(c for c in session_id if c.isalnum() or c in "-_")
    pdf_path = UPLOAD_DIR / f"{safe_id}.pdf"
    if not pdf_path.exists():
        raise HTTPException(404, "Upload session not found. Please re-upload the PDF.")

    doc = fitz.open(str(pdf_path))
    if page < 1 or page > doc.page_count:
        raise HTTPException(400, f"Page {page} out of range (1–{doc.page_count}).")

    mat = fitz.Matrix(INFER_DPI / 72, INFER_DPI / 72)
    pix = doc[page - 1].get_pixmap(matrix=mat)
    image_width = pix.width
    image_height = pix.height

    # Unique ID per page so training samples don't overwrite each other
    page_session_id = f"{safe_id}-p{page}"
    png_path = UPLOAD_DIR / f"{page_session_id}.png"
    pix.save(str(png_path))
    doc.close()

    results = model(str(png_path), imgsz=1024, conf=0.25)
    r = results[0]

    class_counts: Counter = Counter()
    if r.boxes is not None:
        for cls_id in r.boxes.cls.tolist():
            class_counts[CLASSES[int(cls_id)]] += 1

    detections = [
        {"class": cls, "count": cnt}
        for cls, cnt in sorted(class_counts.items(), key=lambda x: -x[1])
    ]

    boxes = []
    if r.boxes is not None:
        xyxyn = r.boxes.xyxyn.tolist()
        cls_ids = r.boxes.cls.tolist()
        confs = r.boxes.conf.tolist()
        for i, (coords, cls_id, conf) in enumerate(zip(xyxyn, cls_ids, confs)):
            x1, y1, x2, y2 = coords
            boxes.append({
                "id": i,
                "class": CLASSES[int(cls_id)],
                "class_id": int(cls_id),
                "confidence": round(float(conf), 4),
                "x1": round(float(x1), 6),
                "y1": round(float(y1), 6),
                "x2": round(float(x2), 6),
                "y2": round(float(y2), 6),
            })

    image_b64 = base64.b64encode(png_path.read_bytes()).decode()

    return {
        "session_id": page_session_id,
        "page_used": page,
        "total": sum(class_counts.values()),
        "detections": detections,
        "image_b64": f"data:image/png;base64,{image_b64}",
        "boxes": boxes,
        "image_width": image_width,
        "image_height": image_height,
    }


@app.post("/feedback")
async def feedback(payload: FeedbackPayload, background_tasks: BackgroundTasks):
    """
    Accept corrected bounding boxes and save as YOLO training data.
    Saves locally (ephemeral) and pushes to the HF dataset repo (persistent).
    """
    safe_id = "".join(c for c in payload.session_id if c.isalnum() or c in "-_")
    if not safe_id:
        raise HTTPException(400, "Invalid session_id.")

    try:
        image_bytes = base64.b64decode(payload.image_b64)
    except Exception:
        raise HTTPException(400, "Invalid base64 image data.")

    # Save locally (best-effort; /tmp is ephemeral on HF Spaces)
    (TRAINING_DIR / "images" / f"{safe_id}.png").write_bytes(image_bytes)

    lines = [
        f"{box.class_id} {box.x_center:.6f} {box.y_center:.6f} {box.width:.6f} {box.height:.6f}"
        for box in payload.boxes
    ]
    label_text = "\n".join(lines)
    (TRAINING_DIR / "labels" / f"{safe_id}.txt").write_text(label_text)

    # Persist to HF dataset repo in the background (non-blocking)
    background_tasks.add_task(push_to_dataset, safe_id, image_bytes, label_text)

    count = read_training_count() + 1
    write_training_count(count)

    return {"saved": True, "total_training_samples": count}
