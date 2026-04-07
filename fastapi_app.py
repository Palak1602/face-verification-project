from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import subprocess
import os
import shutil
import sys

app = FastAPI()

# =========================
# CORS
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Paths
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Use current Python interpreter (works locally + Docker)
PYTHON_EXEC = sys.executable

FACE_MATCH_SCRIPT = os.path.join(BASE_DIR, "face_match.py")
VERIFY_SCRIPT = os.path.join(BASE_DIR, "verify_match.py")
MANUAL_CROP_SCRIPT = os.path.join(BASE_DIR, "manual_crop.py")

CAPTURES_DIR = os.path.join(BASE_DIR, "captures")
CARDS_DIR = os.path.join(BASE_DIR, "detected_cards")
MATCHED_DIR = os.path.join(BASE_DIR, "matched_faces")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Create folders if not exist
os.makedirs(CAPTURES_DIR, exist_ok=True)
os.makedirs(CARDS_DIR, exist_ok=True)
os.makedirs(MATCHED_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Serve saved images
app.mount("/captures", StaticFiles(directory=CAPTURES_DIR), name="captures")
app.mount("/detected_cards", StaticFiles(directory=CARDS_DIR), name="detected_cards")
app.mount("/matched_faces", StaticFiles(directory=MATCHED_DIR), name="matched_faces")
app.mount("/results", StaticFiles(directory=RESULTS_DIR), name="results")

# =========================
# Helper
# =========================
def run_script(script_path):
    try:
        result = subprocess.run(
            [PYTHON_EXEC, script_path],
            cwd=BASE_DIR,
            capture_output=True,
            text=True
        )
        return {
            "success": result.returncode == 0,
            "output": result.stdout,
            "error": result.stderr,
            "script": os.path.basename(script_path)
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "script": os.path.basename(script_path)
        }

# =========================
# Routes
# =========================

@app.get("/")
def home():
    return FileResponse(os.path.join(BASE_DIR, "index.html"))

@app.post("/upload-face/{angle}")
async def upload_face(angle: str, file: UploadFile = File(...)):
    allowed = {"front", "left", "right"}
    angle = angle.lower()

    if angle not in allowed:
        return {"success": False, "error": "Invalid angle"}

    save_path = os.path.join(CAPTURES_DIR, f"{angle}.jpg")

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "success": True,
        "message": f"{angle} face saved",
        "path": f"/captures/{angle}.jpg"
    }

@app.post("/upload-id/{num}")
async def upload_id(num: int, file: UploadFile = File(...)):
    if num not in [1, 2]:
        return {"success": False, "error": "Only card 1 or 2 allowed"}

    save_path = os.path.join(CARDS_DIR, f"card_{num}.jpg")

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "success": True,
        "message": f"card_{num}.jpg saved",
        "path": f"/detected_cards/card_{num}.jpg"
    }

@app.get("/extract-id-face")
def extract_face():
    return run_script(FACE_MATCH_SCRIPT)

@app.get("/manual-crop")
def manual_crop():
    return run_script(MANUAL_CROP_SCRIPT)

@app.get("/verify")
def verify():
    return run_script(VERIFY_SCRIPT)

@app.get("/images-status")
def images_status():
    return {
        "faces": {
            "front": os.path.exists(os.path.join(CAPTURES_DIR, "front.jpg")),
            "left": os.path.exists(os.path.join(CAPTURES_DIR, "left.jpg")),
            "right": os.path.exists(os.path.join(CAPTURES_DIR, "right.jpg")),
        },
        "cards": {
            "card_1": os.path.exists(os.path.join(CARDS_DIR, "card_1.jpg")),
            "card_2": os.path.exists(os.path.join(CARDS_DIR, "card_2.jpg")),
        },
        "matched": {
            "auto": os.path.exists(os.path.join(MATCHED_DIR, "id_face_auto.jpg")),
            "manual": os.path.exists(os.path.join(MATCHED_DIR, "id_face_manual.jpg")),
        },
        "result": {
            "verification": os.path.exists(os.path.join(RESULTS_DIR, "verification_result.jpg")),
        }
    }