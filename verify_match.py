from deepface import DeepFace
import cv2
import os
import numpy as np

# =========================
# SESSION SETUP
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SESSION_ID = os.environ.get("SESSION_ID", "default")

CAPTURES_DIR = os.path.join(BASE_DIR, "captures", SESSION_ID)
MATCHED_DIR = os.path.join(BASE_DIR, "matched_faces", SESSION_ID)
RESULT_DIR = os.path.join(BASE_DIR, "results", SESSION_ID)

os.makedirs(CAPTURES_DIR, exist_ok=True)
os.makedirs(MATCHED_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# =========================
# Paths
# =========================
LIVE_FACE = os.path.join(CAPTURES_DIR, "front.jpg")
ID_FACE = os.path.join(MATCHED_DIR, "id_face_auto.jpg")

RESULT_IMAGE = os.path.join(RESULT_DIR, "verification_result.jpg")
TEMP_LIVE = os.path.join(RESULT_DIR, "temp_live.jpg")
TEMP_ID = os.path.join(RESULT_DIR, "temp_id.jpg")

# =========================
# Validate images
# =========================
if not os.path.exists(ID_FACE):
    print("ERROR: No automatically extracted ID face found")
    exit(1)

if not os.path.exists(LIVE_FACE):
    print("ERROR: Live face missing")
    exit(1)

live_img = cv2.imread(LIVE_FACE)
id_img = cv2.imread(ID_FACE)

if live_img is None or id_img is None:
    print("ERROR: Image read failed")
    exit(1)

print("Using auto extracted ID face")

# =========================
# FAST + ACCURATE PREPROCESSING
# =========================
def enhance_face(img):
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6, 6))
    l = clahe.apply(l)

    img = cv2.merge((l, a, b))
    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)

    return img

live_img = enhance_face(live_img)
id_img = enhance_face(id_img)

cv2.imwrite(TEMP_LIVE, live_img)
cv2.imwrite(TEMP_ID, id_img)

# =========================
# VERIFY (BEST BALANCE)
# =========================
print("Running verification...")

result = DeepFace.verify(
    img1_path=TEMP_LIVE,
    img2_path=TEMP_ID,
    model_name="Facenet",   # best speed/accuracy balance
    detector_backend="opencv",
    enforce_detection=False
)

distance = float(result["distance"])

# tuned threshold for Facenet
THRESHOLD = 0.78
matched = distance < THRESHOLD

# confidence mapping
confidence = max(0, int((1 - distance) * 100))

print(f"Distance: {distance}")
print(f"Confidence: {confidence}%")
print("MATCHED" if matched else "NOT MATCHED")

# =========================
# CREATE RESULT IMAGE
# =========================
live_show = cv2.resize(live_img, (320, 320))
id_show = cv2.resize(id_img, (320, 320))

canvas = np.ones((520, 760, 3), dtype="uint8") * 255

canvas[80:400, 40:360] = live_show
canvas[80:400, 400:720] = id_show

result_text = "MATCHED" if matched else "NOT MATCHED"
color = (0, 180, 0) if matched else (0, 0, 255)

cv2.putText(canvas, result_text, (220, 450),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

cv2.putText(canvas, f"Distance: {distance:.3f}", (80, 490),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 50), 2)

cv2.putText(canvas, f"Confidence: {confidence}%", (430, 490),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 50), 2)

cv2.imwrite(RESULT_IMAGE, canvas)

print("Result image saved")