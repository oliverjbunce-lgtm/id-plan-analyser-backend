"""
Independent Doors — Plan Analyser API
FastAPI backend: PDF upload, page thumbnail generation, YOLOv8 inference
"""
import os, uuid, shutil
from pathlib import Path
from collections import Counter
from typing import Optional

import fitz  # PyMuPDF
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "model/best.pt")
UPLOAD_DIR = Path("/tmp/id-uploads")
RESULTS_DIR = Path("/tmp/id-results")
THUMB_DIR = Path("/tmp/id-thumbs")
THUMB_DPI = 72        # low-res for page picker
INFER_DPI = 150       # resolution for model inference

CLASSES = [
    "L_prehung_door", "R_prehung_door", "Double_prehung_door",
    "S_cavity_slider", "D_cavity_slider",
    "Wardrobe_sliding_two_doors_1", "Wardrobe_sliding_two_doors_2",
    "Wardrobe_sliding_four_doors", "Bi_folding_door",
    "Barn_wall_slider", "D_bi_folding_door", "Wardrobe_sliding_three_doors",
]

for d in [UPLOAD_DIR, RESULTS_DIR, THUMB_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="ID Plan Analyser", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="/tmp"), name="static")

# Load model once at startup
model: Optional[YOLO] = None

@app.on_event("startup")
async def load_model():
    global model
    if Path(MODEL_PATH).exists():
        print(f"Loading model from {MODEL_PATH}…")
        model = YOLO(MODEL_PATH)
        print("Model ready ✓")
    else:
        print(f"⚠️  Model not found at {MODEL_PATH} — inference will fail")


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


def pdf_to_thumb(page: fitz.Page, out_path: Path, dpi: int = 72):
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    pix.save(str(out_path))


# ── Routes ────────────────────────────────────────────────────────────────────
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

    thumb_session_dir = THUMB_DIR / session_id
    thumb_session_dir.mkdir(exist_ok=True)

    for i in range(min(page_count, 30)):  # cap at 30 pages for speed
        thumb_path = thumb_session_dir / f"page-{i+1}.jpg"
        pdf_to_thumb(doc[i], thumb_path, dpi=THUMB_DPI)
        pages.append({
            "page": i + 1,
            "url": f"/static/id-thumbs/{session_id}/page-{i+1}.jpg"
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
    Returns annotated image URL and detection counts.
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
    png_path = UPLOAD_DIR / f"{session_id}-p{page}.png"
    pix.save(str(png_path))
    doc.close()

    # Run inference
    result_dir = RESULTS_DIR / session_id
    result_dir.mkdir(exist_ok=True)

    results = model(
        str(png_path),
        imgsz=1024,
        conf=0.25,
        save=True,
        project=str(result_dir),
        name="out",
        exist_ok=True,
    )
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

    # Annotated image path
    annotated_path = result_dir / "out" / png_path.name
    image_url = f"/static/id-results/{session_id}/out/{png_path.name}"

    return {
        "session_id": session_id,
        "page_used": page,
        "total": sum(class_counts.values()),
        "detections": detections,
        "image_url": image_url,
    }
