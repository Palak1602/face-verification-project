import cv2
import os
import numpy as np

# =========================
# SESSION SETUP (ADDED)
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SESSION_ID = os.environ.get("SESSION_ID", "default")

CARDS_DIR = os.path.join(BASE_DIR, "detected_cards", SESSION_ID)
MATCHED_DIR = os.path.join(BASE_DIR, "matched_faces", SESSION_ID)

os.makedirs(CARDS_DIR, exist_ok=True)
os.makedirs(MATCHED_DIR, exist_ok=True)

# =========================
# Paths
# =========================
card1 = os.path.join(CARDS_DIR, "card_1.jpg")
card2 = os.path.join(CARDS_DIR, "card_2.jpg")
output_folder = MATCHED_DIR
os.makedirs(output_folder, exist_ok=True)

# =========================
# Select best card automatically
# =========================
available_cards = []

if os.path.exists(card1):
    available_cards.append(card1)
if os.path.exists(card2):
    available_cards.append(card2)

if not available_cards:
    print("No detected card images found.")
    exit(1)

# Prefer sharper image
def sharpness_score(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

best_card_path = None
best_score = -1

for path in available_cards:
    img = cv2.imread(path)
    if img is None:
        continue
    score = sharpness_score(img)
    if score > best_score:
        best_score = score
        best_card_path = path

if best_card_path is None:
    print("Could not load card image.")
    exit(1)

print(f"Using best ID card image: {os.path.basename(best_card_path)}")

# =========================
# Load image
# =========================
img = cv2.imread(best_card_path)
if img is None:
    print("Could not load selected card image.")
    exit(1)

orig = img.copy()
img_h, img_w = img.shape[:2]

# =========================
# Focus only on likely photo area
# (ignore top/bottom edges and very outer margins)
# =========================
x1_roi = int(img_w * 0.05)
y1_roi = int(img_h * 0.10)
x2_roi = int(img_w * 0.95)
y2_roi = int(img_h * 0.92)

roi = img[y1_roi:y2_roi, x1_roi:x2_roi].copy()
roi_h, roi_w = roi.shape[:2]

gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

# Improve contrast a bit
gray = cv2.equalizeHist(gray)

# =========================
# Face detector
# =========================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.08,
    minNeighbors=6,
    minSize=(60, 60)
)

# =========================
# Filter only good face candidates
# =========================
valid_faces = []

for (x, y, w, h) in faces:
    face_area = w * h
    roi_area = roi_w * roi_h
    face_ratio = face_area / roi_area

    # Reject too tiny / too huge
    if face_ratio < 0.015 or face_ratio > 0.30:
        continue

    # Reject very edge detections
    margin_x = int(roi_w * 0.03)
    margin_y = int(roi_h * 0.03)

    if x < margin_x or y < margin_y or (x + w) > (roi_w - margin_x) or (y + h) > (roi_h - margin_y):
        continue

    # Slight preference for portrait-ish / square-ish faces
    aspect = w / float(h)
    if aspect < 0.65 or aspect > 1.35:
        continue

    # Face region brightness / texture check
    face_crop_gray = gray[y:y+h, x:x+w]
    if face_crop_gray.size == 0:
        continue

    brightness = np.mean(face_crop_gray)
    texture = cv2.Laplacian(face_crop_gray, cv2.CV_64F).var()

    # Reject very blank / flat / weird areas
    if brightness < 35 or brightness > 240:
        continue
    if texture < 20:
        continue

    valid_faces.append((x, y, w, h, face_ratio, texture))

# =========================
# If no reliable face -> fail
# =========================
if len(valid_faces) == 0:
    print("No reliable ID face detected automatically.")
    exit(1)

# =========================
# Pick best face
# Strategy:
# - prefer larger face
# - prefer sharper face
# =========================
best_face = max(valid_faces, key=lambda f: (f[2] * f[3]) + (f[5] * 5))
x, y, w, h, _, _ = best_face

# Convert back to original image coordinates
x += x1_roi
y += y1_roi

# =========================
# Add safe padding
# =========================
pad_x = int(w * 0.30)
pad_y = int(h * 0.35)

fx1 = max(0, x - pad_x)
fy1 = max(0, y - pad_y)
fx2 = min(img_w, x + w + pad_x)
fy2 = min(img_h, y + h + pad_y)

face_crop = orig[fy1:fy2, fx1:fx2]

# Final safety check
if face_crop.size == 0 or face_crop.shape[0] < 80 or face_crop.shape[1] < 80:
    print("Face crop invalid.")
    exit(1)

# Slight enhancement
face_crop = cv2.resize(face_crop, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

# Save
save_path = os.path.join(output_folder, "id_face_auto.jpg")
cv2.imwrite(save_path, face_crop, [cv2.IMWRITE_JPEG_QUALITY, 100])

print(f"Reliable ID face saved at: {save_path}")

# =========================
# Preview
# =========================
preview = orig.copy()

# Draw ROI box
cv2.rectangle(preview, (x1_roi, y1_roi), (x2_roi, y2_roi), (255, 180, 0), 2)

# Draw chosen face
cv2.rectangle(preview, (fx1, fy1), (fx2, fy2), (0, 255, 0), 3)

cv2.putText(preview, "Detected ID Face", (fx1, max(30, fy1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

# Resize preview if too large
if preview.shape[1] > 1200:
    scale = 1200 / preview.shape[1]
    preview = cv2.resize(preview, None, fx=scale, fy=scale)

# Optional preview (safe for local use; may fail silently on Render)
try:
    cv2.imshow("Automatic ID Face Detection", preview)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()
except:
    pass