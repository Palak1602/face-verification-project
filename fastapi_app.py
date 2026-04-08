from fastapi import FastAPI, UploadFile, File, Query, Form
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
PYTHON_EXEC = sys.executable

FACE_MATCH_SCRIPT = os.path.join(BASE_DIR, "face_match.py")
VERIFY_SCRIPT = os.path.join(BASE_DIR, "verify_match.py")
MANUAL_CROP_SCRIPT = os.path.join(BASE_DIR, "manual_crop.py")

CAPTURES_DIR = os.path.join(BASE_DIR, "captures")
CARDS_DIR = os.path.join(BASE_DIR, "detected_cards")
MATCHED_DIR = os.path.join(BASE_DIR, "matched_faces")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(CAPTURES_DIR, exist_ok=True)
os.makedirs(CARDS_DIR, exist_ok=True)
os.makedirs(MATCHED_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# =========================
# Serve folders
# =========================
app.mount("/captures", StaticFiles(directory=CAPTURES_DIR), name="captures")
app.mount("/detected_cards", StaticFiles(directory=CARDS_DIR), name="detected_cards")
app.mount("/matched_faces", StaticFiles(directory=MATCHED_DIR), name="matched_faces")
app.mount("/results", StaticFiles(directory=RESULTS_DIR), name="results")

# =========================
# Helper: session folders
# =========================
def get_session_paths(session_id: str):
    capture_path = os.path.join(CAPTURES_DIR, session_id)
    card_path = os.path.join(CARDS_DIR, session_id)
    matched_path = os.path.join(MATCHED_DIR, session_id)
    result_path = os.path.join(RESULTS_DIR, session_id)

    os.makedirs(capture_path, exist_ok=True)
    os.makedirs(card_path, exist_ok=True)
    os.makedirs(matched_path, exist_ok=True)
    os.makedirs(result_path, exist_ok=True)

    return capture_path, card_path, matched_path, result_path


# =========================
# Helper: run scripts with session_id
# =========================
def run_script(script_path, session_id: str):
    try:
        env = os.environ.copy()
        env["SESSION_ID"] = session_id

        result = subprocess.run(
            [PYTHON_EXEC, script_path],
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            env=env
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
async def upload_face(
    angle: str,
    file: UploadFile = File(...),
    session_id: str = Form(...)
):
    allowed = {"front", "left", "right"}
    angle = angle.lower()

    if angle not in allowed:
        return {"success": False, "error": "Invalid angle"}

    capture_path, _, _, _ = get_session_paths(session_id)
    save_path = os.path.join(capture_path, f"{angle}.jpg")

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "success": True,
        "message": f"{angle} face saved",
        "path": f"/captures/{session_id}/{angle}.jpg"
    }


@app.post("/upload-id/{num}")
async def upload_id(
    num: int,
    file: UploadFile = File(...),
    session_id: str = Form(...)
):
    if num not in [1, 2]:
        return {"success": False, "error": "Only card 1 or 2 allowed"}

    _, card_path, _, _ = get_session_paths(session_id)
    save_path = os.path.join(card_path, f"card_{num}.jpg")

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "success": True,
        "message": f"card_{num}.jpg saved",
        "path": f"/detected_cards/{session_id}/card_{num}.jpg"
    }


@app.get("/extract-id-face")
def extract_face(session_id: str = Query(...)):
    return run_script(FACE_MATCH_SCRIPT, session_id)


@app.get("/manual-crop")
def manual_crop(session_id: str = Query(...)):
    return run_script(MANUAL_CROP_SCRIPT, session_id)


@app.get("/verify")
def verify(session_id: str = Query(...)):
    return run_script(VERIFY_SCRIPT, session_id)


@app.get("/images-status")
def images_status(session_id: str = Query(...)):
    capture_path, card_path, matched_path, result_path = get_session_paths(session_id)

    return {
        "faces": {
            "front": os.path.exists(os.path.join(capture_path, "front.jpg")),
            "left": os.path.exists(os.path.join(capture_path, "left.jpg")),
            "right": os.path.exists(os.path.join(capture_path, "right.jpg")),
        },
        "cards": {
            "card_1": os.path.exists(os.path.join(card_path, "card_1.jpg")),
            "card_2": os.path.exists(os.path.join(card_path, "card_2.jpg")),
        },
        "matched": {
            "auto": os.path.exists(os.path.join(matched_path, "id_face_auto.jpg")),
            "manual": os.path.exists(os.path.join(matched_path, "id_face_manual.jpg")),
        },
        "result": {
            "verification": os.path.exists(os.path.join(result_path, "verification_result.jpg")),
        }
    }