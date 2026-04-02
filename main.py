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
from fastapi.responses import HTMLResponse
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

@app.get("/test", response_class=HTMLResponse)
def test_page():
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Model Test — ID Plan Analyser v2</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Inter', sans-serif; background: #f5f5f7; color: #0a0a0a; min-height: 100vh; padding: 40px 24px; }
  .container { max-width: 900px; margin: 0 auto; }
  h1 { font-size: 28px; font-weight: 800; letter-spacing: -0.03em; margin-bottom: 4px; }
  .sub { color: #6b7280; font-size: 14px; margin-bottom: 32px; }
  .badge { display: inline-flex; align-items: center; gap: 6px; background: #dcfce7; color: #16a34a; font-size: 11px; font-weight: 700; padding: 4px 10px; border-radius: 999px; margin-bottom: 12px; letter-spacing: 0.08em; text-transform: uppercase; }
  .drop-zone { border: 2px dashed #d1d5db; border-radius: 16px; background: white; padding: 48px 24px; text-align: center; cursor: pointer; transition: all 0.2s; margin-bottom: 24px; }
  .drop-zone:hover, .drop-zone.drag-over { border-color: #0A84FF; background: #f0f7ff; }
  .drop-zone p { color: #6b7280; font-size: 15px; }
  .drop-zone strong { color: #0a0a0a; }
  .btn { background: #0A84FF; color: white; border: none; border-radius: 12px; padding: 14px 28px; font-size: 15px; font-weight: 700; cursor: pointer; width: 100%; transition: background 0.2s; }
  .btn:hover { background: #0066CC; }
  .btn:disabled { opacity: 0.5; cursor: not-allowed; }
  #status { margin-top: 16px; font-size: 14px; color: #6b7280; text-align: center; min-height: 20px; }
  .results { display: none; margin-top: 32px; }
  .results-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  .card { background: white; border: 1px solid #e5e5e5; border-radius: 16px; padding: 20px; }
  .card h3 { font-size: 13px; font-weight: 700; color: #6b7280; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 14px; }
  .result-img { width: 100%; border-radius: 12px; border: 1px solid #e5e5e5; }
  .detection-row { display: flex; justify-content: space-between; align-items: center; padding: 8px 0; border-bottom: 1px solid #f3f4f6; }
  .detection-row:last-child { border-bottom: none; }
  .det-name { font-size: 13px; color: #374151; }
  .det-conf { font-size: 12px; font-weight: 700; color: #0A84FF; }
  .det-count { font-size: 18px; font-weight: 800; color: #0a0a0a; min-width: 32px; text-align: right; }
  .total-row { display: flex; justify-content: space-between; padding: 12px 0 0; margin-top: 8px; border-top: 2px solid #e5e5e5; }
  .total-row span:first-child { font-size: 13px; font-weight: 700; color: #6b7280; }
  .total-row span:last-child { font-size: 22px; font-weight: 900; color: #0a0a0a; }
  .timing { font-size: 12px; color: #9ca3af; text-align: right; margin-top: 6px; }
  input[type=file] { display: none; }
  .file-name { font-size: 13px; color: #0A84FF; font-weight: 600; margin-top: 8px; }
</style>
</head>
<body>
<div class="container">
  <div class="badge">● Model v2 — 95.1% mAP50</div>
  <h1>Plan Analyser Test</h1>
  <p class="sub">Upload a floor plan PDF to test the new YOLOv8L model before going live.</p>
  <div class="drop-zone" id="dropZone" onclick="document.getElementById('fileInput').click()">
    <p><strong>Drop a floor plan PDF here</strong></p>
    <p style="margin-top:8px;font-size:13px">or click to browse</p>
    <div class="file-name" id="fileName"></div>
  </div>
  <input type="file" id="fileInput" accept=".pdf">
  <button class="btn" id="analyseBtn" onclick="analyse()" disabled>Analyse Plan</button>
  <div id="status"></div>
  <div class="results" id="results">
    <div class="results-grid">
      <div class="card">
        <h3>Detected Components</h3>
        <div id="detectionList"></div>
        <div class="total-row"><span>TOTAL</span><span id="totalCount">0</span></div>
        <div class="timing" id="timingInfo"></div>
      </div>
      <div class="card">
        <h3>Annotated Plan</h3>
        <img id="annotatedImg" class="result-img" src="" alt="Annotated plan">
      </div>
    </div>
  </div>
</div>
<script>
  let sessionId = null, uploadedPage = 1;
  const dropZone = document.getElementById('dropZone');
  const fileInput = document.getElementById('fileInput');
  const btn = document.getElementById('analyseBtn');
  const status = document.getElementById('status');

  fileInput.addEventListener('change', () => {
    if (fileInput.files[0]) {
      document.getElementById('fileName').textContent = fileInput.files[0].name;
      btn.disabled = false;
    }
  });
  dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
  dropZone.addEventListener('drop', e => {
    e.preventDefault(); dropZone.classList.remove('drag-over');
    fileInput.files = e.dataTransfer.files;
    if (fileInput.files[0]) { document.getElementById('fileName').textContent = fileInput.files[0].name; btn.disabled = false; }
  });

  async function analyse() {
    const file = fileInput.files[0];
    if (!file) return;
    btn.disabled = true;
    status.textContent = 'Uploading PDF…';
    document.getElementById('results').style.display = 'none';
    const t0 = Date.now();
    try {
      // Upload
      const fd = new FormData(); fd.append('file', file);
      const upRes = await fetch('/upload', { method: 'POST', body: fd });
      const upData = await upRes.json();
      sessionId = upData.session_id;
      uploadedPage = upData.suggested_page || 1;
      status.textContent = ;
      // Analyse
      const afd = new FormData(); afd.append('session_id', sessionId); afd.append('page', uploadedPage);
      const aRes = await fetch('/analyse-stored', { method: 'POST', body: afd });
      const aData = await aRes.json();
      const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
      renderResults(aData, elapsed);
    } catch(e) {
      status.textContent = 'Error: ' + e.message;
      btn.disabled = false;
    }
  }

  function renderResults(data, elapsed) {
    status.textContent = '';
    btn.disabled = false;
    document.getElementById('results').style.display = 'block';
    // Image
    const img = document.getElementById('annotatedImg');
    img.src = 'data:image/jpeg;base64,' + data.annotated_image;
    // Detections
    const counts = {};
    const confs = {};
    (data.detections || []).forEach(d => {
      counts[d.class_name] = (counts[d.class_name] || 0) + 1;
      if (!confs[d.class_name]) confs[d.class_name] = [];
      confs[d.class_name].push(d.confidence);
    });
    const list = document.getElementById('detectionList');
    list.innerHTML = '';
    let total = 0;
    Object.keys(counts).sort().forEach(name => {
      const avgConf = (confs[name].reduce((a,b)=>a+b,0)/confs[name].length*100).toFixed(0);
      list.innerHTML += '<div class="detection-row"><span class="det-name">'+name+'</span><span class="det-conf">'+avgConf+'%</span><span class="det-count">×'+counts[name]+'</span></div>';
      total += counts[name];
    });
    if (Object.keys(counts).length === 0) {
      list.innerHTML = '<div style="color:#9ca3af;font-size:13px;padding:12px 0">No components detected on this page.</div>';
    }
    document.getElementById('totalCount').textContent = total;
    document.getElementById('timingInfo').textContent = elapsed + 's total';
  }
</script>
</body>
</html>"""

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
