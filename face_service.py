import os
import cv2
import numpy as np
import mediapipe as mp
from deepface import DeepFace

CAPTURES_DIR = "captures"
CARDS_DIR = "detected_cards"
MATCHED_DIR = "matched_faces"
RESULTS_DIR = "results"

os.makedirs(CAPTURES_DIR, exist_ok=True)
os.makedirs(CARDS_DIR, exist_ok=True)
os.makedirs(MATCHED_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def select_best_card(card_paths=None):
    if card_paths is None:
        card_paths = []
        for i in [1, 2]:
            p = os.path.join(CARDS_DIR, f"card_{i}.jpg")
            if os.path.exists(p):
                card_paths.append(p)

    if not card_paths:
        raise FileNotFoundError("No card images available")


def draw_mesh_on_image(image_path, output_path=None, flip=True):
    if not os.path.exists(image_path):
        return None

    img = cv2.imread(image_path)
    if img is None:
        return None

    if flip:
        img = cv2.flip(img, 1)

    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        result = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if result.multi_face_landmarks:
            for face_landmarks in result.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=img,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 200, 255), thickness=1, circle_radius=1)
                )

    if output_path is None:
        output_path = image_path

    cv2.imwrite(output_path, img, [cv2.IMWRITE_JPEG_QUALITY, 100])
    return output_path

    def sharpness_score(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    best_path = None
    best_score = -1

    for path in card_paths:
        img = cv2.imread(path)
        if img is None:
            continue
        score = sharpness_score(img)
        if score > best_score:
            best_score = score
            best_path = path

    if best_path is None:
        raise ValueError("Could not pick a best card image")

    return best_path, best_score


def detect_id_face(card_path, output_path=None, return_crop=False):
    if output_path is None:
        output_path = os.path.join(MATCHED_DIR, "id_face_auto.jpg")

    if not os.path.exists(card_path):
        raise FileNotFoundError(f"Card image not found: {card_path}")

    img = cv2.imread(card_path)
    if img is None:
        raise ValueError("Cannot read card image")

    orig = img.copy()
    img_h, img_w = img.shape[:2]

    x1_roi = int(img_w * 0.05)
    y1_roi = int(img_h * 0.10)
    x2_roi = int(img_w * 0.95)
    y2_roi = int(img_h * 0.92)

    roi = img[y1_roi:y2_roi, x1_roi:x2_roi].copy()
    if roi.size == 0:
        raise ValueError("ROI is empty")

    roi_h, roi_w = roi.shape[:2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.08,
        minNeighbors=6,
        minSize=(60, 60)
    )

    valid_faces = []
    for (x, y, w, h) in faces:
        face_area = w * h
        roi_area = roi_w * roi_h
        face_ratio = face_area / roi_area

        if face_ratio < 0.015 or face_ratio > 0.30:
            continue

        margin_x = int(roi_w * 0.03)
        margin_y = int(roi_h * 0.03)

        if x < margin_x or y < margin_y or (x + w) > (roi_w - margin_x) or (y + h) > (roi_h - margin_y):
            continue

        aspect = w / float(h)
        if aspect < 0.65 or aspect > 1.35:
            continue

        face_crop_gray = gray[y:y+h, x:x+w]
        if face_crop_gray.size == 0:
            continue

        brightness = np.mean(face_crop_gray)
        texture = cv2.Laplacian(face_crop_gray, cv2.CV_64F).var()

        if brightness < 35 or brightness > 240 or texture < 20:
            continue

        valid_faces.append((x, y, w, h, face_ratio, texture))

    if not valid_faces:
        raise ValueError("No reliable face found in card image")

    best_face = max(valid_faces, key=lambda f: (f[2] * f[3]) + (f[5] * 5))
    x, y, w, h, _, _ = best_face

    x += x1_roi
    y += y1_roi

    pad_x = int(w * 0.30)
    pad_y = int(h * 0.35)

    fx1 = max(0, x - pad_x)
    fy1 = max(0, y - pad_y)
    fx2 = min(img_w, x + w + pad_x)
    fy2 = min(img_h, y + h + pad_y)

    face_crop = orig[fy1:fy2, fx1:fx2].copy()
    if face_crop.size == 0 or face_crop.shape[0] < 80 or face_crop.shape[1] < 80:
        raise ValueError("Face crop invalid")

    face_crop = cv2.resize(face_crop, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

    cv2.imwrite(output_path, face_crop, [cv2.IMWRITE_JPEG_QUALITY, 100])

    if return_crop:
        return face_crop
    return output_path


def _enhance_face(img):
    img = cv2.resize(img, (400, 400), interpolation=cv2.INTER_CUBIC)
    img = cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 7, 21)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return img


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


def verify_faces(live_face_path, id_face_path, result_image_path=None):
    if result_image_path is None:
        result_image_path = os.path.join(RESULTS_DIR, "verification_result.jpg")

    if not os.path.exists(live_face_path):
        raise FileNotFoundError(f"Live face not found: {live_face_path}")
    if not os.path.exists(id_face_path):
        raise FileNotFoundError(f"ID face not found: {id_face_path}")

    live_img = cv2.imread(live_face_path)
    id_img = cv2.imread(id_face_path)

    if live_img is None or id_img is None:
        raise ValueError("Could not read input image(s)")

    live_img = _enhance_face(live_img)
    id_img = _enhance_face(id_img)

    temp_live = os.path.join(RESULTS_DIR, "temp_live.jpg")
    temp_id = os.path.join(RESULTS_DIR, "temp_id.jpg")
    cv2.imwrite(temp_live, live_img, [cv2.IMWRITE_JPEG_QUALITY, 100])
    cv2.imwrite(temp_id, id_img, [cv2.IMWRITE_JPEG_QUALITY, 100])

    verification_modes = [
        {"model_name": "Facenet512", "detector_backend": "opencv", "enforce_detection": False},
        {"model_name": "Facenet", "detector_backend": "opencv", "enforce_detection": False},
        {"model_name": "VGG-Face", "detector_backend": "opencv", "enforce_detection": False},
    ]

    distance = None
    matched = False
    confidence = 0
    used_mode = None

    for mode in verification_modes:
        try:
            result = DeepFace.verify(
                img1_path=temp_live,
                img2_path=temp_id,
                model_name=mode["model_name"],
                detector_backend=mode["detector_backend"],
                enforce_detection=mode["enforce_detection"]
            )
            distance = float(result["distance"])
            if mode["model_name"] == "VGG-Face":
                threshold = 0.85
            else:
                threshold = 0.78

            matched = distance < threshold
            confidence = distance_to_confidence(distance)
            used_mode = mode["model_name"]
            break

        except Exception:
            continue

    if used_mode is None:
        raise ValueError("Face verification failed for all models")

    live_show = cv2.resize(live_img, (320, 320))
    id_show = cv2.resize(id_img, (320, 320))
    canvas = np.ones((520, 760, 3), dtype="uint8") * 255
    canvas[80:400, 40:360] = live_show
    canvas[80:400, 400:720] = id_show

    result_text = "MATCHED" if matched else "NOT MATCHED"
    result_color = (0, 180, 0) if matched else (0, 0, 255)

    cv2.putText(canvas, "LIVE FACE", (120, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 50, 50), 2)
    cv2.putText(canvas, "ID FACE", (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 50, 50), 2)
    cv2.putText(canvas, f"RESULT: {result_text}", (180, 445), cv2.FONT_HERSHEY_SIMPLEX, 1.1, result_color, 3)
    cv2.putText(canvas, f"Distance: {distance:.3f}", (80, 490), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 60, 60), 2)
    cv2.putText(canvas, f"Confidence: {confidence}%", (430, 490), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 60, 60), 2)

    cv2.imwrite(result_image_path, canvas)

    return {
        "matched": matched,
        "distance": float(distance),
        "confidence": int(confidence),
        "used_model": used_mode,
        "result_image": result_image_path,
    }


def capture_live_faces(camera_index=0, output_dir=CAPTURES_DIR, required_frames=4, capture_delay=1.2):
    os.makedirs(output_dir, exist_ok=True)
    front_path = os.path.join(output_dir, "front.jpg")
    left_path = os.path.join(output_dir, "left.jpg")
    right_path = os.path.join(output_dir, "right.jpg")

    steps = [
        ("FRONT", front_path),
        ("LEFT", left_path),
        ("RIGHT", right_path),
    ]

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    for _ in range(10):
        cap.read()

    try:
        import mediapipe as mp
        use_mediapipe = True
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    except ImportError:
        use_mediapipe = False
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    current_step = 0
    step_start_time = None
    last_capture_time = 0

    while current_step < len(steps):
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        h, w = frame.shape[:2]

        if use_mediapipe:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            face_found = (results.multi_face_landmarks is not None)
            landmarks = results.multi_face_landmarks[0].landmark if face_found else None
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
            face_found = len(faces) > 0
            landmarks = None
            if face_found:
                x, y, fw, fh = faces[0]
                # create simple landmark rectangle points
                landmarks = []
                for px, py in [(x, y), (x + fw, y), (x, y + fh), (x + fw, y + fh)]:
                    class LM:
                        pass
                    lm = LM()
                    lm.x = px / float(w)
                    lm.y = py / float(h)
                    landmarks.append(lm)

        if face_found and landmarks is not None:
            if step_start_time is None:
                step_start_time = cv2.getTickCount() / cv2.getTickFrequency()

            elapsed = (cv2.getTickCount() / cv2.getTickFrequency()) - step_start_time

            if elapsed >= capture_delay and (cv2.getTickCount() / cv2.getTickFrequency() - last_capture_time) > 0.8:
                if use_mediapipe:
                    face_pts_x = [int(lm.x * w) for lm in landmarks]
                    face_pts_y = [int(lm.y * h) for lm in landmarks]
                else:
                    face_pts_x = [int(lm.x * w) for lm in landmarks]
                    face_pts_y = [int(lm.y * h) for lm in landmarks]

                x1 = max(0, min(face_pts_x) - 50)
                y1 = max(0, min(face_pts_y) - 60)
                x2 = min(w, max(face_pts_x) + 50)
                y2 = min(h, max(face_pts_y) + 60)
                crop = frame[y1:y2, x1:x2]

                if crop.size > 0:
                    crop = cv2.resize(crop, (320, 320), interpolation=cv2.INTER_CUBIC)
                    cv2.imwrite(steps[current_step][1], crop, [cv2.IMWRITE_JPEG_QUALITY, 100])
                    current_step += 1
                    step_start_time = None
                    last_capture_time = cv2.getTickCount() / cv2.getTickFrequency()
                    continue
        else:
            step_start_time = None

    cap.release()
    if use_mediapipe:
        face_mesh.close()

    mesh_front = draw_mesh_on_image(front_path, os.path.join(output_dir, "front_mesh.jpg"), flip=True)
    mesh_left = draw_mesh_on_image(left_path, os.path.join(output_dir, "left_mesh.jpg"), flip=True)
    mesh_right = draw_mesh_on_image(right_path, os.path.join(output_dir, "right_mesh.jpg"), flip=True)

    return {
        "front": front_path,
        "left": left_path,
        "right": right_path,
        "front_mesh": mesh_front,
        "left_mesh": mesh_left,
        "right_mesh": mesh_right,
    }


def capture_id_cards(camera_index=0, output_dir=CARDS_DIR, max_captures=2, min_stable_frames=4):
    import time

    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    for _ in range(10):
        cap.read()

    capture_count = 0
    stable_count = 0
    last_capture_time = 0

    while capture_count < max_captures:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        H, W = frame.shape[:2]

        box_w = int(W * 0.22)
        box_h = int(H * 0.58)
        x1 = (W - box_w) // 2
        y1 = (H - box_h) // 2 - 20
        x2 = x1 + box_w
        y2 = y1 + box_h

        scan_roi = frame[y1:y2, x1:x2]

        gray = cv2.cvtColor(scan_roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 60, 150)
        edge_ratio = np.count_nonzero(edges) / float(edges.size)
        contrast = gray.std()
        brightness = gray.mean()

        detected = (edge_ratio > 0.025 and contrast > 35 and 60 < brightness < 220)

        if detected:
            stable_count += 1
        else:
            stable_count = 0

        if stable_count >= min_stable_frames and (time.time() - last_capture_time) > 0.8:
            card = scan_roi.copy()
            card = cv2.resize(card, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_CUBIC)
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            card = cv2.filter2D(card, -1, kernel)

            capture_count += 1
            save_path = os.path.join(output_dir, f"card_{capture_count}.jpg")
            cv2.imwrite(save_path, card, [cv2.IMWRITE_JPEG_QUALITY, 100])

            last_capture_time = time.time()
            stable_count = 0

    cap.release()

    return {
        "card_1": os.path.join(output_dir, "card_1.jpg"),
        "card_2": os.path.join(output_dir, "card_2.jpg"),
    }
