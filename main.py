import cv2
import os
import time
import mediapipe as mp

# =========================
# Setup
# =========================
output_folder = "captures"
os.makedirs(output_folder, exist_ok=True)

front_path = os.path.join(output_folder, "front.jpg")
left_path = os.path.join(output_folder, "left.jpg")
right_path = os.path.join(output_folder, "right.jpg")

steps = [
    ("FRONT", front_path, "Look straight"),
    ("LEFT", left_path, "Turn your face LEFT"),
    ("RIGHT", right_path, "Turn your face RIGHT"),
]

current_step = 0
step_start_time = None
capture_delay = 1.2   # auto capture after 1.2 sec
cooldown = 0.8
last_capture_time = 0

# =========================
# Camera
# =========================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

for _ in range(10):
    cap.read()

# =========================
# MediaPipe
# =========================
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

print("\n FAST AUTO FACE CAPTURE")
print("=" * 45)
print("FRONT will auto capture")
print("LEFT will auto capture")
print("RIGHT will auto capture")
print("No button needed")
print("Press Q / ESC to quit")


# =========================
# Save face crop
# =========================
def save_face(frame, landmarks, save_path):
    h, w = frame.shape[:2]

    xs = [int(lm.x * w) for lm in landmarks]
    ys = [int(lm.y * h) for lm in landmarks]

    x1 = max(0, min(xs) - 50)
    y1 = max(0, min(ys) - 60)
    x2 = min(w, max(xs) + 50)
    y2 = min(h, max(ys) + 60)

    crop = frame[y1:y2, x1:x2].copy()
    if crop.size == 0:
        return False

    crop = cv2.resize(crop, (320, 320), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(save_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 100])
    return True


# =========================
# Main loop
# =========================
while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        continue

    # Selfie camera preview
    frame = cv2.flip(frame, 1)
    display = frame.copy()
    h, w = frame.shape[:2]
    now = time.time()

    if current_step >= len(steps):
        break

    target_label, target_path, instruction = steps[current_step]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    # ===== UI =====
    cv2.putText(display, f"TARGET: {target_label}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)

    cv2.putText(display, instruction, (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.putText(display, "Selfie Camera Mode", (30, 145),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 255), 2)

    cv2.putText(display, "Q / ESC = Quit", (30, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        lm = face_landmarks.landmark

        # Draw only mesh (NO BOX)
        mp_drawing.draw_landmarks(
            image=display,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_spec
        )

        # Start timer if face is visible
        if step_start_time is None:
            step_start_time = now

        elapsed = now - step_start_time
        remaining = max(0, capture_delay - elapsed)

        cv2.putText(display, f"Capturing in: {remaining:.1f}s", (30, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Auto capture
        if elapsed >= capture_delay and (now - last_capture_time) > cooldown:
            ok = save_face(frame, lm, target_path)

            if ok:
                print(f" Saved {target_label}: {target_path}")

                flash = display.copy()
                cv2.rectangle(flash, (0, 0), (w, h), (255, 255, 255), -1)
                cv2.addWeighted(flash, 0.15, display, 0.85, 0, display)
                cv2.imshow("Fast Auto Face Capture", display)
                cv2.waitKey(250)

                current_step += 1
                step_start_time = None
                last_capture_time = now

                if current_step >= len(steps):
                    print(" All face angles captured successfully!")
                    break
    else:
        step_start_time = None
        cv2.putText(display, "NO FACE DETECTED", (30, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (180, 180, 180), 2)

    cv2.imshow("Fast Auto Face Capture", display)

    key = cv2.waitKey(1)
    if key != -1:
        key = key & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            print(" Quit.")
            break

cap.release()
cv2.destroyAllWindows()
face_mesh.close()