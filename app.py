import os
import subprocess
import sys

# =========================
# Paths
# =========================
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

SCAN_ENV_PYTHON = os.path.join(PROJECT_DIR, "scan_env", "Scripts", "python.exe")
DEFAULT_PYTHON = os.path.join(PROJECT_DIR, ".venv", "Scripts", "python.exe")

# =========================
# Runner
# =========================
def run(script, python_exec, step):
    print("\n" + "=" * 60)
    print(f" {step}")
    print("=" * 60)

    try:
        subprocess.run([python_exec, script], check=True)
        return True
    except Exception as e:
        print(f" Error in {script}")
        print(e)
        return False


def run_background(command, python_exec, step):
    print("\n" + "=" * 60)
    print(f" {step} (Background)")
    print("=" * 60)

    try:
        process = subprocess.Popen([python_exec] + command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return process
    except Exception as e:
        print(f" Error starting {step}")
        print(e)
        return None


# =========================
# Setup folders
# =========================
os.makedirs("captures", exist_ok=True)
os.makedirs("detected_cards", exist_ok=True)
os.makedirs("matched_faces", exist_ok=True)

# =========================
# Clean old files
# =========================
files_to_delete = [
    "captures/front.jpg",
    "captures/left.jpg",
    "captures/right.jpg",
    "detected_cards/card_1.jpg",
    "detected_cards/card_2.jpg",
    "matched_faces/id_face_auto.jpg"
]

for f in files_to_delete:
    if os.path.exists(f):
        try:
            os.remove(f)
        except:
            pass

# =========================
# Start
# =========================
print("\n FACE VERIFICATION SYSTEM")
print("=" * 60)
input(" Press ENTER to start...")

# =========================
# STEP 0: Start FastAPI Server (Background)
# =========================
fastapi_process = run_background(["-m", "uvicorn", "fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"], DEFAULT_PYTHON, "STEP 0: Start FastAPI Server")
if fastapi_process is None:
    print(" Failed to start FastAPI server")
    sys.exit()

print(" FastAPI server started in background on http://localhost:8000")

# =========================
# STEP 1: Face Capture (scan_env)
# =========================
ok = run("main.py", SCAN_ENV_PYTHON, "STEP 1: Face Capture (MediaPipe)")
if not ok or not os.path.exists("captures/front.jpg"):
    print(" Face capture failed")
    sys.exit()

print(" Face captured")

# =========================
# STEP 2: ID Capture (.venv)
# =========================
ok = run("id_detect.py", DEFAULT_PYTHON, "STEP 2: ID Card Capture")
if not ok:
    sys.exit()

if not os.path.exists("detected_cards/card_1.jpg"):
    print(" ID capture failed")
    sys.exit()

print(" ID captured")

# =========================
# STEP 3: Extract ID Face
# =========================
ok = run("face_match.py", DEFAULT_PYTHON, "STEP 3: Extract Face from ID")
if not ok:
    sys.exit()

if not os.path.exists("matched_faces/id_face_auto.jpg"):
    print(" Face extraction failed")
    sys.exit()

print("✅ ID face extracted")

# =========================
# STEP 4: Verify
# =========================
ok = run("verify_match.py", DEFAULT_PYTHON, "STEP 4: Face Verification")
if not ok:
    sys.exit()

print("\n DONE! FULL PIPELINE COMPLETED")
print("🌐 FastAPI server is still running on http://localhost:8000")
print("Press Ctrl+C to stop the server")