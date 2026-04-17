
import os
# Add ffmpeg to PATH FIRST, before any audio imports
FFMPEG_BIN = r"C:\Users\USER\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1-full_build\bin"
os.environ["PATH"] = FFMPEG_BIN + os.pathsep + os.environ.get("PATH", "")

from flask import Flask, jsonify, request
from flask_cors import CORS
import random
import time
import uuid
import io
import base64
import tempfile
import numpy as np
from datetime import datetime
import speech_recognition as sr
from pydub import AudioSegment
from ultralytics import YOLO
from PIL import Image

# Also set converter paths directly in case PATH doesn't work
AudioSegment.converter = os.path.join(FFMPEG_BIN, "ffmpeg.exe")
AudioSegment.ffprobe = os.path.join(FFMPEG_BIN, "ffprobe.exe")

app = Flask(__name__)
CORS(app)

# ─── Load YOLO ASL Model ───
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best.pt")
print(f"Loading YOLO model from {MODEL_PATH}...")
yolo_model = YOLO(MODEL_PATH)
print(f"Model loaded. Classes: {yolo_model.names}")

# ─── State ───
letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
current_text = ""
current_letter = ""
confidence = 0.0
mode = "asl-to-text"          # asl-to-text | speech-to-asl
communication_logs = []        # timestamped log entries
hardware_status = {
    "arduino": True,
    "camera": True,
    "speaker": True,
    "microphone": True,
    "display": True,
}

# ─── Helpers ───
def _add_log(direction, input_val, output_val):
    entry = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(),
        "direction": direction,
        "input": input_val,
        "output": output_val,
    }
    communication_logs.insert(0, entry)
    # keep last 200 entries
    if len(communication_logs) > 200:
        communication_logs.pop()
    return entry


# ─── 1. ASL → Text/Speech ───
@app.route('/gesture')
def gesture():
    global current_text, current_letter, confidence
    letter = random.choice(letters)
    current_letter = letter
    confidence = round(random.uniform(0.82, 0.99), 2)
    current_text += letter
    _add_log("ASL → Text", f"Gesture: {letter}", current_text)
    return jsonify({
        "letter": letter,
        "text": current_text,
        "confidence": confidence,
        "time": time.time(),
    })


@app.route('/detect', methods=['POST'])
def detect():
    """Receive a detected letter from the frontend camera/ML pipeline."""
    global current_text, current_letter, confidence
    data = request.json or {}
    letter = data.get("letter", "")
    text = data.get("text", "")
    current_letter = letter
    current_text = text
    confidence = 0.95
    _add_log("ASL → Text", f"Gesture: {letter}", text)
    return jsonify({"letter": letter, "text": text, "status": "ok"})


@app.route('/predict', methods=['POST'])
def predict():
    """Receive a base64-encoded camera frame, run YOLO inference, return detected ASL letter."""
    data = request.json or {}
    image_data = data.get("image", "")
    if not image_data:
        return jsonify({"error": "No image data provided"}), 400

    try:
        # Strip data-URL prefix if present (e.g. "data:image/jpeg;base64,...")
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]
        img_bytes = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_w, img_h = img.size
        print(f"[predict] Image size: {img_w}x{img_h}")

        # Run YOLO inference — conf=0.40 balances accuracy vs false positives
        results = yolo_model.predict(source=img, conf=0.40, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = yolo_model.names.get(cls_id, str(cls_id)).upper()
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                # Filter out false positives:
                # 1. Bounding box must not cover >80% of the frame (entire frame = not a hand)
                box_area = (x2 - x1) * (y2 - y1)
                frame_area = img_w * img_h
                if box_area > frame_area * 0.80:
                    continue
                # 2. Bounding box must have a minimum size (tiny boxes = noise)
                if (x2 - x1) < 30 or (y2 - y1) < 30:
                    continue

                detections.append({
                    "letter": label,
                    "confidence": round(conf, 3),
                    "bbox": [round(x1), round(y1), round(x2), round(y2)],
                })

        # Sort by confidence descending, return best detection
        detections.sort(key=lambda d: d["confidence"], reverse=True)
        best = detections[0] if detections else None
        if best:
            all_labels = [f"{d['letter']}({d['confidence']})" for d in detections]
            print(f"[predict] Best: {best['letter']} ({best['confidence']}) | All: {', '.join(all_labels)}")
        else:
            print(f"[predict] No detections")
        return jsonify({
            "detected": best is not None,
            "letter": best["letter"] if best else "",
            "confidence": best["confidence"] if best else 0,
            "bbox": best["bbox"] if best else [],
            "all_detections": detections,
        })
    except Exception as e:
        print(f"Predict error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/clear')
def clear():
    global current_text, current_letter, confidence
    current_text = ""
    current_letter = ""
    confidence = 0.0
    return jsonify({"status": "cleared"})


@app.route('/speak', methods=['POST'])
def speak():
    """Receive text to be spoken by the frontend TTS engine."""
    data = request.json or {}
    text = data.get("text", current_text)
    _add_log("Text → Speech", text, "Audio output")
    return jsonify({"text": text, "status": "spoken"})


# ─── 2. Speech/Text → ASL ───
@app.route('/speech-to-text', methods=['POST'])
def speech_to_text():
    data = request.json or {}
    text = data.get("text", "")
    _add_log("Speech → Text", "Microphone input", text)
    return jsonify({"text": text})


@app.route('/transcribe', methods=['POST'])
def transcribe():
    print(">>> /transcribe called")
    audio_file = request.files.get('audio')
    if not audio_file:
        print(">>> No audio file in request")
        return jsonify({"error": "No audio file"}), 400

    print(f">>> Got audio: {audio_file.filename}, content_type={audio_file.content_type}")
    tmp_in = None
    tmp_wav = None
    try:
        # Save uploaded audio to temp file
        suffix = os.path.splitext(audio_file.filename or "")[1] or ".webm"
        tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        raw = audio_file.read()
        print(f">>> Audio size: {len(raw)} bytes")
        tmp_in.write(raw)
        tmp_in.close()

        # Convert any format → WAV using pydub (ffmpeg)
        audio_seg = AudioSegment.from_file(tmp_in.name)
        print(f">>> Converted: duration={len(audio_seg)}ms, channels={audio_seg.channels}, rate={audio_seg.frame_rate}")
        audio_seg = audio_seg.set_channels(1).set_frame_rate(16000)
        tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp_wav.close()
        audio_seg.export(tmp_wav.name, format="wav")

        # Transcribe with Google Speech API
        recognizer = sr.Recognizer()
        with sr.AudioFile(tmp_wav.name) as source:
            audio = recognizer.record(source)
        text = recognizer.recognize_google(audio)
        _add_log("Speech → Text", "Microphone input", text)
        return jsonify({"text": text})
    except sr.UnknownValueError:
        return jsonify({"text": ""})
    except sr.RequestError as e:
        return jsonify({"error": f"Google API error: {e}"}), 500
    except Exception as e:
        print(f"Transcribe error: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        for f in [tmp_in, tmp_wav]:
            if f:
                try: os.unlink(f.name)
                except OSError: pass


@app.route('/text-to-asl')
def text_to_asl():
    text = request.args.get("text", "").upper()
    asl_sequence = []
    for ch in text:
        if ch.isalpha():
            asl_sequence.append({
                "letter": ch,
                "image": f"/asl/{ch}.png",
            })
        elif ch == " ":
            asl_sequence.append({"letter": " ", "image": None})
    _add_log("Text → ASL", text, f"{len(asl_sequence)} signs")
    return jsonify({"input": text, "sequence": asl_sequence})


# ─── 3. Mode switching ───
@app.route('/mode', methods=['GET', 'POST'])
def system_mode():
    global mode
    if request.method == 'POST':
        data = request.json or {}
        mode = data.get("mode", mode)
    return jsonify({"mode": mode})


# ─── 4. Communication logs ───
@app.route('/logs')
def get_logs():
    search = request.args.get("search", "").lower()
    direction = request.args.get("direction", "")
    filtered = communication_logs
    if search:
        filtered = [l for l in filtered
                    if search in l["input"].lower() or search in l["output"].lower()]
    if direction:
        filtered = [l for l in filtered if l["direction"] == direction]
    return jsonify({"logs": filtered[:50], "total": len(filtered)})


@app.route('/logs/clear', methods=['POST'])
def clear_logs():
    communication_logs.clear()
    return jsonify({"status": "cleared"})


# ─── 5. System / hardware status ───
@app.route('/status')
def status():
    return jsonify({
        "mode": mode,
        "current_letter": current_letter,
        "current_text": current_text,
        "confidence": confidence,
        "hardware": hardware_status,
        "uptime": time.time(),
        "log_count": len(communication_logs),
    })


if __name__ == "__main__":
    app.run(debug=True)
