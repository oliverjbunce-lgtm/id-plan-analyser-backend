"""
Independent Doors — Plan Analyser API
FastAPI backend: PDF upload, page thumbnail generation, YOLOv8 inference
"""
import os, uuid, shutil, base64, io
from contextlib import asynccontextmanager
from pathlib import Path
from collections import Counter
from typing import Optional

import cv2
import fitz  # PyMuPDF
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "model/best.pt")
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
    if Path(MODEL_PATH).exists():
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

    # Get annotated image directly from YOLO result (BGR numpy array → PNG bytes → base64)
    annotated_bgr = r.plot()
    success, buffer = cv2.imencode(".png", annotated_bgr)
    if not success:
        raise HTTPException(500, "Failed to encode annotated image.")
    image_b64 = base64.b64encode(buffer.tobytes()).decode()

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


@app.post("/feedback")
async def feedback(payload: FeedbackPayload):
    """
    Accept corrected bounding boxes and save as YOLO training data.
    Saves the image and label file, increments training sample counter.
    """
    # Sanitise session_id to prevent path traversal
    safe_id = "".join(c for c in payload.session_id if c.isalnum() or c in "-_")
    if not safe_id:
        raise HTTPException(400, "Invalid session_id.")

    # Decode and save image
    try:
        image_bytes = base64.b64decode(payload.image_b64)
    except Exception:
        raise HTTPException(400, "Invalid base64 image data.")

    img_path = TRAINING_DIR / "images" / f"{safe_id}.png"
    img_path.write_bytes(image_bytes)

    # Write YOLO label file
    label_path = TRAINING_DIR / "labels" / f"{safe_id}.txt"
    lines = [
        f"{box.class_id} {box.x_center:.6f} {box.y_center:.6f} {box.width:.6f} {box.height:.6f}"
        for box in payload.boxes
    ]
    label_path.write_text("\n".join(lines))

    # Increment counter
    count = read_training_count() + 1
    write_training_count(count)

    return {"saved": True, "total_training_samples": count}
