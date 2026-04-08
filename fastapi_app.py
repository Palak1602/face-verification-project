from fastapi import FastAPI, UploadFile, File, Query, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from deepface import DeepFace
import cv2
import numpy as np
import subprocess
import os
import shutil
import sys
import uvicorn

app = FastAPI()

# =========================
# 🚀 LOAD MODEL ONCE (BIG SPEED BOOST)
# =========================
print("Loading FaceNet model...")
FACENET_MODEL = DeepFace.build_model("Facenet")
print("FaceNet model loaded successfully.")

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
# Helper: run scripts
# =========================
def run_script(script_path, session_id: str):
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
        "error": result.stderr
    }


# =========================
# Routes
# =========================

@app.get("/")
def home():
    return FileResponse(os.path.join(BASE_DIR, "index.html"))


@app.post("/upload-face/{angle}")
async def upload_face(angle: str, file: UploadFile = File(...), session_id: str = Form(...)):
    capture_path, _, _, _ = get_session_paths(session_id)
    save_path = os.path.join(capture_path, f"{angle}.jpg")

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {"success": True}


@app.post("/upload-id/{num}")
async def upload_id(num: int, file: UploadFile = File(...), session_id: str = Form(...)):
    _, card_path, _, _ = get_session_paths(session_id)
    save_path = os.path.join(card_path, f"card_{num}.jpg")

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {"success": True}


@app.get("/extract-id-face")
def extract_face(session_id: str = Query(...)):
    return run_script(FACE_MATCH_SCRIPT, session_id)


# =========================
# 🔥 FAST + ACCURATE VERIFY
# =========================
@app.get("/verify")
def verify(session_id: str = Query(...)):
    try:
        capture_path, _, matched_path, result_path = get_session_paths(session_id)

        live_face = os.path.join(capture_path, "front.jpg")
        id_face = os.path.join(matched_path, "id_face_auto.jpg")

        if not os.path.exists(live_face) or not os.path.exists(id_face):
            return {"success": False, "error": "Images missing"}

        live_img = cv2.imread(live_face)
        id_img = cv2.imread(id_face)

        # =========================
        # ⚡ OPTIMIZED PREPROCESSING
        # =========================
        def enhance_face(img):
            img = cv2.resize(img, (192, 192))  # faster + accurate

            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)

            clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(4, 4))
            l = clahe.apply(l)

            img = cv2.merge((l, a, b))
            return cv2.cvtColor(img, cv2.COLOR_LAB2BGR)

        live_img = enhance_face(live_img)
        id_img = enhance_face(id_img)

        # =========================
        # 🚀 FAST VERIFY
        # =========================
        result = DeepFace.verify(
            img1_path=live_img,
            img2_path=id_img,
            model_name="Facenet",
            model=FACENET_MODEL,
            detector_backend="opencv",  # keep accuracy
            enforce_detection=False
        )

        distance = float(result["distance"])
        matched = distance < 0.78
        confidence = int((1 - distance) * 100)

        # =========================
        # RESULT IMAGE (same as yours)
        # =========================
        result_image = os.path.join(result_path, "verification_result.jpg")

        live_show = cv2.resize(live_img, (320, 320))
        id_show = cv2.resize(id_img, (320, 320))

        canvas = np.ones((520, 760, 3), dtype="uint8") * 255
        canvas[80:400, 40:360] = live_show
        canvas[80:400, 400:720] = id_show

        text = "MATCHED" if matched else "NOT MATCHED"
        color = (0, 180, 0) if matched else (0, 0, 255)

        cv2.putText(canvas, text, (220, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        cv2.imwrite(result_image, canvas)

        return {
            "success": True,
            "output": f"""Distance: {distance:.3f}
Confidence: {confidence}%
{text}"""
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)