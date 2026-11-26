"""
4-Digit Number Repetition Counter (image upload version, no Tesseract needed)

Upload screenshots/photos containing 4-digit numbers.
The app reads the numbers with RapidOCR (pip-only), lets you review/fix them,
then counts repeats across ALL images.
"""
import os
import re
import threading
from collections import Counter

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB per request

TARGET_WIDTH = 1080
_engine = None
_lock = threading.Lock()


def get_engine():
    """Load the OCR model on first use (keeps startup fast and memory low)."""
    global _engine
    if _engine is None:
        from rapidocr_onnxruntime import RapidOCR
        _engine = RapidOCR()
    return _engine


def load_gray(img_bytes):
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Not a readable image")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    if w != TARGET_WIDTH:
        interp = cv2.INTER_AREA if w > TARGET_WIDTH else cv2.INTER_CUBIC
        gray = cv2.resize(gray, None, fx=TARGET_WIDTH / w, fy=TARGET_WIDTH / w, interpolation=interp)
    return gray


def run_ocr(gray):
    with _lock:  # one OCR at a time keeps memory low on small servers
        result, _ = get_engine()(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
    tokens = []
    for _box, text, _conf in (result or []):
        tokens += re.findall(r"\d+", text)
    return [t for t in tokens if len(t) == 4], [t for t in tokens if len(t) != 4]


def read_numbers(img_bytes):
    """Two OCR passes (plain + black/white). Returns (numbers, odd_tokens, confident)."""
    gray = load_gray(img_bytes)
    otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    if np.mean(otsu) < 127:
        otsu = cv2.bitwise_not(otsu)
    a_nums, a_odd = run_ocr(gray)
    b_nums, b_odd = run_ocr(otsu)
    if len(b_nums) > len(a_nums):
        best, odd = b_nums, b_odd
    else:
        best, odd = a_nums, a_odd
    confident = sorted(a_nums) == sorted(b_nums) and len(best) > 0
    return best, odd, confident


def analyze(numbers):
    """Your original counting logic."""
    frequency = Counter(numbers)
    repeated = sorted(
        ((n, c) for n, c in frequency.items() if c >= 2),
        key=lambda x: (-x[1], x[0]),
    )
    return {
        "repeated": [{"number": n, "count": c} for n, c in repeated],
        "total": len(numbers),
        "unique": len(frequency),
        "repeated_count": len(repeated),
        "once": sum(1 for c in frequency.values() if c == 1),
    }


# Optional password: set APP_PASSWORD (and optionally APP_USER) as environment
# variables on the server. If APP_PASSWORD is not set, the app is open.
@app.before_request
def require_login():
    password = os.environ.get("APP_PASSWORD")
    if not password:
        return None
    auth = request.authorization
    user = os.environ.get("APP_USER", "admin")
    if auth and auth.username == user and auth.password == password:
        return None
    return Response("Login required", 401, {"WWW-Authenticate": 'Basic realm="Counter"'})


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/extract", methods=["POST"])
def extract():
    results = []
    for f in request.files.getlist("images"):
        try:
            nums, odd, confident = read_numbers(f.read())
            results.append({"name": f.filename, "numbers": nums, "odd": odd, "confident": confident})
        except Exception as e:
            results.append({"name": f.filename, "numbers": [], "odd": [], "confident": False, "error": str(e)})
    return jsonify(results)


@app.route("/analyze", methods=["POST"])
def analyze_route():
    data = request.get_json(force=True)
    # Accept "per image" groups so we can show where each repeat came from
    groups = data.get("groups", [])
    all_numbers, where = [], {}
    for g in groups:
        for n in re.findall(r"\b\d{4}\b", " ".join(g.get("numbers", []))):
            all_numbers.append(n)
            where.setdefault(n, []).append(g.get("name", "?"))
    result = analyze(all_numbers)
    for item in result["repeated"]:
        item["images"] = where[item["number"]]
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
