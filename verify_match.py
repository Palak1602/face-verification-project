from deepface import DeepFace
import cv2
import os
import numpy as np

# =========================
# SESSION SETUP (ADDED)
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
# Paths (UPDATED)
# =========================
LIVE_FACE = os.path.join(CAPTURES_DIR, "front.jpg")
ID_FACE = os.path.join(MATCHED_DIR, "id_face_auto.jpg")
RESULT_IMAGE = os.path.join(RESULT_DIR, "verification_result.jpg")

print("\nUsing auto extracted ID face")
print("Live face:", LIVE_FACE)
print("ID face:", ID_FACE)

# =========================
# Validate images
# =========================
if not os.path.exists(LIVE_FACE):
    print("ERROR: Live face image not found")
    exit()

if not os.path.exists(ID_FACE):
    print("ERROR: ID face image not found")
    exit()

live_img = cv2.imread(LIVE_FACE)
id_img = cv2.imread(ID_FACE)

if live_img is None:
    print("ERROR: Cannot read LIVE image")
    exit()

if id_img is None:
    print("ERROR: Cannot read ID image")
    exit()

if live_img.size == 0:
    print("ERROR: LIVE image is empty")
    exit()

if id_img.size == 0:
    print("ERROR: ID image is empty")
    exit()

print("Images loaded successfully")

# =========================
# Improve images for verification
# =========================
def enhance_face(img):
    img = cv2.resize(img, (400, 400), interpolation=cv2.INTER_CUBIC)

    # mild denoise
    img = cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 7, 21)

    # mild contrast improvement
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return img

live_img = enhance_face(live_img)
id_img = enhance_face(id_img)

TEMP_LIVE = os.path.join(RESULT_DIR, "temp_live.jpg")
TEMP_ID = os.path.join(RESULT_DIR, "temp_id.jpg")

cv2.imwrite(TEMP_LIVE, live_img, [cv2.IMWRITE_JPEG_QUALITY, 100])
cv2.imwrite(TEMP_ID, id_img, [cv2.IMWRITE_JPEG_QUALITY, 100])

# =========================
# Confidence Converter
# =========================
def distance_to_confidence(distance):
    if distance <= 0.30:
        return 99
    elif distance <= 0.40:
        return 95
    elif distance <= 0.50:
        return 90
    elif distance <= 0.60:
        return 82
    elif distance <= 0.70:
        return 74
    elif distance <= 0.78:
        return 68
    elif distance <= 0.85:
        return 55
    else:
        return 35

# =========================
# DeepFace Verification
# =========================
distance = None
matched = False
confidence = 0

verification_modes = [
    {"model_name": "Facenet512", "detector_backend": "opencv", "enforce_detection": False},
    {"model_name": "Facenet", "detector_backend": "opencv", "enforce_detection": False},
    {"model_name": "VGG-Face", "detector_backend": "opencv", "enforce_detection": False},
]

verified_successfully = False

for mode in verification_modes:
    try:
        print(f"\nTrying: {mode['model_name']} + {mode['detector_backend']}")

        result = DeepFace.verify(
            img1_path=TEMP_LIVE,
            img2_path=TEMP_ID,
            model_name=mode["model_name"],
            detector_backend=mode["detector_backend"],
            enforce_detection=mode["enforce_detection"]
        )

        distance = float(result["distance"])

        if mode["model_name"] == "VGG-Face":
            MATCH_THRESHOLD = 0.85
        else:
            MATCH_THRESHOLD = 0.78

        matched = distance < MATCH_THRESHOLD
        confidence = distance_to_confidence(distance)

        print("Verification worked with fallback-safe mode")
        print(f"Distance: {distance:.4f}")
        print(f"Confidence: {confidence}%")
        print("MATCHED" if matched else "NOT MATCHED")

        verified_successfully = True
        break

    except Exception as e:
        print(f"Failed with {mode['model_name']}: {e}")

if not verified_successfully:
    print("Verification failed in all modes")
    exit()

# =========================
# Prepare Images for UI
# =========================
live_show = cv2.resize(live_img, (320, 320))
id_show = cv2.resize(id_img, (320, 320))

# =========================
# Create Result UI
# =========================
canvas = np.ones((520, 760, 3), dtype="uint8") * 255

canvas[80:400, 40:360] = live_show
canvas[80:400, 400:720] = id_show

cv2.putText(canvas, "LIVE FACE", (120, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 50, 50), 2)

cv2.putText(canvas, "ID FACE", (500, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 50, 50), 2)

result_text = "MATCHED" if matched else "NOT MATCHED"
result_color = (0, 180, 0) if matched else (0, 0, 255)

cv2.putText(canvas, f"RESULT: {result_text}", (180, 445),
            cv2.FONT_HERSHEY_SIMPLEX, 1.1, result_color, 3)

cv2.putText(canvas, f"Distance: {distance:.3f}", (80, 490),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 60, 60), 2)

cv2.putText(canvas, f"Confidence: {confidence}%", (430, 490),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 60, 60), 2)

# =========================
# Save + Show
# =========================
cv2.imwrite(RESULT_IMAGE, canvas)

# Safe display (Render-safe)
try:
    cv2.imshow("Face Verification Result", canvas)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()
except:
    pass

print(f"\nResult image saved at: {RESULT_IMAGE}")
print(f"RESULT: {'MATCHED' if matched else 'NOT MATCHED'}")
print(f"Distance: {distance:.3f}")
print(f"Confidence: {confidence}%")