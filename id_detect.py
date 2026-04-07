import cv2
import os
import time
import numpy as np

# =========================
# Setup
# =========================
output_folder = "detected_cards"
os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

for _ in range(10):
    cap.read()

# =========================
# Settings
# =========================
max_captures = 2
cooldown_seconds = 0.8
required_frames = 4   # ~ fast but stable

capture_count = 0
last_capture_time = 0
stable_count = 0

print("\nID CARD SCAN MODE")
print("=" * 40)
print("Place the ID card inside the green box")
print("Keep it straight and fully visible")
print("Auto-capture will happen quickly")
print("Press SPACE to capture manually")
print("Press Q / ESC to quit\n")


# =========================
# Helper: improve crop
# =========================
def enhance_card(img):
    if img is None or img.size == 0:
        return None

    # upscale slightly
    img = cv2.resize(img, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_CUBIC)

    # mild sharpening
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    img = cv2.filter2D(img, -1, kernel)

    return img


# =========================
# Save card
# =========================
def save_card(frame, box, count):
    x1, y1, x2, y2 = box

    crop = frame[y1:y2, x1:x2].copy()
    if crop is None or crop.size == 0:
        print("Invalid crop")
        return False

    crop = enhance_card(crop)
    if crop is None:
        print("Could not enhance crop")
        return False

    save_path = os.path.join(output_folder, f"card_{count}.jpg")
    cv2.imwrite(save_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 100])
    print(f"Saved clean ID card: {save_path}")

    preview = crop.copy()
    if preview.shape[1] > 900:
        scale = 900 / preview.shape[1]
        preview = cv2.resize(preview, None, fx=scale, fy=scale)

    cv2.imshow("Saved Card", preview)
    cv2.waitKey(500)
    cv2.destroyWindow("Saved Card")
    return True


# =========================
# Check if scan zone contains enough "card-like" detail
# =========================
def card_present(scan_roi):
    gray = cv2.cvtColor(scan_roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # edge density
    edges = cv2.Canny(blur, 60, 150)
    edge_pixels = np.count_nonzero(edges)
    total_pixels = edges.shape[0] * edges.shape[1]
    edge_ratio = edge_pixels / float(total_pixels)

    # contrast
    contrast = gray.std()

    # brightness check
    brightness = gray.mean()

    # enough texture / printed content
    return (
        edge_ratio > 0.025 and
        contrast > 35 and
        60 < brightness < 220
    )


# =========================
# Main loop
# =========================
while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        continue

    frame = cv2.flip(frame, 1)
    display = frame.copy()
    H, W = frame.shape[:2]
    now = time.time()

    # =========================
    # FIXED ID SCAN BOX
    # =========================
    box_w = int(W * 0.22)
    box_h = int(H * 0.58)

    x1 = (W - box_w) // 2
    y1 = (H - box_h) // 2 - 20
    x2 = x1 + box_w
    y2 = y1 + box_h

    scan_roi = frame[y1:y2, x1:x2].copy()

    detected = card_present(scan_roi)

    if detected:
        stable_count += 1
    else:
        stable_count = 0

    ready = stable_count >= required_frames

    # =========================
    # Auto capture
    # =========================
    if ready and (now - last_capture_time) > cooldown_seconds:
        capture_count += 1
        ok = save_card(frame, (x1, y1, x2, y2), capture_count)

        if ok:
            print("FAST CAPTURE")

            flash = display.copy()
            cv2.rectangle(flash, (x1, y1), (x2, y2), (255, 255, 255), -1)
            cv2.addWeighted(flash, 0.30, display, 0.70, 0, display)
            cv2.imshow("ID CARD SCAN MODE", display)
            cv2.waitKey(150)

            last_capture_time = now
            stable_count = 0

            if capture_count >= max_captures:
                print("Done! Both ID images captured.")
                break
            else:
                print(f"Next capture {capture_count + 1}/{max_captures}")

        continue

    # =========================
    # UI
    # =========================
    overlay = display.copy()

    cv2.rectangle(overlay, (0, 0), (W, H), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.28, display, 0.72, 0, display)

    display[y1:y2, x1:x2] = frame[y1:y2, x1:x2]

    color = (0, 255, 0) if ready else ((0, 220, 255) if detected else (180, 180, 180))
    label = "Card Ready" if ready else ("Card Detected" if detected else "Place ID Inside Box")

    cv2.rectangle(display, (x1, y1), (x2, y2), color, 3)

    cv2.putText(display, label, (x1, max(35, y1 - 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)

    cv2.putText(display, f"Stable Frames: {stable_count}/{required_frames}",
                (x1, y2 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.putText(display, f"Captured: {capture_count}/{max_captures}",
                (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

    cv2.putText(display, "Align your ID fully inside the box",
                (25, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)

    cv2.putText(display, "SPACE = Capture   |   Q / ESC = Quit",
                (25, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (220, 220, 220), 2)

    cv2.imshow("ID CARD SCAN MODE", display)

    key = cv2.waitKey(1)
    if key != -1:
        key = key & 0xFF

        if key in (ord('q'), ord('Q'), 27):
            print("Quit.")
            break

        elif key == ord(' '):
            capture_count += 1
            ok = save_card(frame, (x1, y1, x2, y2), capture_count)

            if ok:
                print(f"Manual capture {capture_count}/{max_captures}")
                last_capture_time = now
                stable_count = 0

                if capture_count >= max_captures:
                    print("Done! Both ID images captured.")
                    break
                else:
                    print(f"Next capture {capture_count + 1}/{max_captures}")

cap.release()
cv2.destroyAllWindows()
for _ in range(5):
    cv2.waitKey(1)