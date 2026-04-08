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
# Setup
# =========================
input_folder = CARDS_DIR
output_folder = MATCHED_DIR
os.makedirs(output_folder, exist_ok=True)

# =========================
# Choose which card image to use
# =========================
card1 = os.path.join(input_folder, "card_1.jpg")
card2 = os.path.join(input_folder, "card_2.jpg")

if not os.path.exists(card1):
    print("card_1.jpg not found.")
    exit()

print("\n==============================")
print("AVAILABLE ID CARD IMAGES")
print("==============================")
print("1. card_1.jpg")

if os.path.exists(card2):
    print("2. card_2.jpg")

choice = input("\nEnter which card image to crop (1 or 2): ").strip()

if choice == "2" and os.path.exists(card2):
    input_path = card2
else:
    input_path = card1

print(f"\nUsing: {os.path.basename(input_path)}")

# =========================
# Load ORIGINAL high-res image
# =========================
img = cv2.imread(input_path)

if img is None:
    print("Could not load selected ID card image.")
    exit()

orig_h, orig_w = img.shape[:2]

# =========================
# Resize ONLY for display
# =========================
display_width = 1000
scale = display_width / orig_w
display_height = int(orig_h * scale)

display_img = cv2.resize(img, (display_width, display_height))

# =========================
# ROI selection
# =========================
print("\nSelect the face photo from the ID card")
print("Press ENTER or SPACE after selecting")
print("Press C to cancel\n")

roi = cv2.selectROI("Select Face from ID Card", display_img, False, False)

# Safe close (Render-safe)
try:
    cv2.destroyAllWindows()
except:
    pass

x, y, w, h = roi

if w == 0 or h == 0:
    print("No region selected.")
    exit()

# =========================
# Convert ROI back to ORIGINAL image coordinates
# =========================
x_orig = int(x / scale)
y_orig = int(y / scale)
w_orig = int(w / scale)
h_orig = int(h / scale)

crop = img[y_orig:y_orig + h_orig, x_orig:x_orig + w_orig]

if crop is None or crop.size == 0:
    print("Cropped image is empty.")
    exit()

# =========================
# Safe enhancement
# =========================

# 1. Slight upscale
crop = cv2.resize(crop, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

# 2. Gentle sharpening
sharpen_kernel = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
], dtype=np.float32)

crop = cv2.filter2D(crop, -1, sharpen_kernel)

# 3. Ensure valid pixel range
crop = np.clip(crop, 0, 255).astype(np.uint8)

# =========================
# Save final high-quality crop
# =========================
output_path = os.path.join(output_folder, "id_face_manual.jpg")
cv2.imwrite(output_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 100])

print(f"\nClear ID face saved at: {output_path}")

# =========================
# Preview result safely
# =========================
preview = crop.copy()

if preview.shape[1] > 500:
    scale_preview = 500 / preview.shape[1]
    preview = cv2.resize(preview, None, fx=scale_preview, fy=scale_preview)

print("\nShowing cropped preview...")
print("Press Q / ESC / ENTER to continue")
print("Auto-closing in 3 seconds...")

# Safe display (Render-safe)
try:
    cv2.imshow("Final Cropped ID Face", preview)

    start_time = cv2.getTickCount()

    while True:
        key = cv2.waitKey(100) & 0xFF

        if key == ord('q') or key == 27 or key == 13:
            break

        elapsed = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
        if elapsed >= 3:
            break

        if cv2.getWindowProperty("Final Cropped ID Face", cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()
    cv2.waitKey(1)

except:
    pass