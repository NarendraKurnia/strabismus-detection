import cv2
import numpy as np
import base64
from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
from pathlib import Path

app = Flask(__name__)

STATUS_NOT_DETECTED = "TIDAK TERDETEKSI"

# ============================================================
# 1. Memuat Model AI
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_DETEKSI_PATH = BASE_DIR / "models" / "model_mata.pt"
MODEL_KLASIFIKASI_PATH = BASE_DIR / "models" / "model_juling.pt"

detector = None
classifier = None


def load_models():
    """Memuat model YOLO untuk deteksi mata dan klasifikasi strabismus."""
    global detector, classifier
    try:
        if not MODEL_DETEKSI_PATH.exists() or not MODEL_KLASIFIKASI_PATH.exists():
            print("❌ Error: File model tidak ditemukan!")
            return False
        detector = YOLO(str(MODEL_DETEKSI_PATH))
        classifier = YOLO(str(MODEL_KLASIFIKASI_PATH))
        print("✅ Model berhasil dimuat.")
        return True
    except Exception as e:
        print(f"❌ Error memuat model: {e}")
        return False


# ============================================================
# 2. Deteksi Mata & Klasifikasi Strabismus
# ============================================================
def detect_and_classify(image):
    """Deteksi area mata dengan YOLO, lalu klasifikasi strabismus."""
    hasil_deteksi = []
    status = "NORMAL"

    # Deteksi Area Mata dengan YOLO
    results_det = detector.predict(image, conf=0.15, verbose=False)
    regions = []

    for r in results_det:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            regions.append([x1, y1, x2, y2])

    # Fallback ke Haar Cascade jika YOLO tidak menemukan mata
    if not regions:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )
        eyes = eye_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        for x, y, w, h in eyes:
            regions.append([x, y, x + w, y + h])

    if not regions:
        return hasil_deteksi, STATUS_NOT_DETECTED

    # Ambil koordinat pembatas terkecil dan terbesar
    min_x = min(r[0] for r in regions)
    min_y = min(r[1] for r in regions)
    max_x = max(r[2] for r in regions)
    max_y = max(r[3] for r in regions)

    pad_x, pad_y = 20, 20
    strip_x1 = max(0, min_x - pad_x)
    strip_y1 = max(0, min_y - pad_y)
    strip_x2 = min(image.shape[1], max_x + pad_x)
    strip_y2 = min(image.shape[0], max_y + pad_y)

    strip_img = image[strip_y1:strip_y2, strip_x1:strip_x2]

    # Jalankan Klasifikasi Strabismus
    results_cls = classifier.predict(strip_img, conf=0.3, verbose=False)

    if len(results_cls[0].boxes) > 0:
        for box in results_cls[0].boxes:
            bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy().astype(int)
            class_id = int(box.cls[0])
            label_name = classifier.names[class_id].upper()
            conf_score = float(box.conf[0])

            abs_x1, abs_y1 = bx1 + strip_x1, by1 + strip_y1
            abs_x2, abs_y2 = bx2 + strip_x1, by2 + strip_y1

            hasil_deteksi.append(
                {
                    "box": [int(abs_x1), int(abs_y1), int(abs_x2), int(abs_y2)],
                    "label": label_name,
                    "conf": round(conf_score, 4),
                }
            )

            if "JULING" in label_name or "STRABISMUS" in label_name:
                status = "STRABISMUS (JULING)"
    else:
        for x1, y1, x2, y2 in regions:
            hasil_deteksi.append(
                {
                    "box": [int(x1), int(y1), int(x2), int(y2)],
                    "label": "NORMAL",
                    "conf": 0.0,
                }
            )

    return hasil_deteksi, status


# ============================================================
# 3. Menggambar Box Deteksi secara Estetik
# ============================================================
def draw_boxes(image, detections):
    """Menggambar bounding box deteksi pada gambar."""
    img_copy = image.copy()
    for d in detections:
        x1, y1, x2, y2 = d["box"]
        label = d["label"]
        conf = d["conf"]

        warna = (
            (0, 0, 255)
            if "JULING" in label or "STRABISMUS" in label
            else (0, 255, 0)
        )
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), warna, 3)
        text_label = f"{label} {conf:.1%}" if conf > 0 else label

        (tw, th), _ = cv2.getTextSize(
            text_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        cv2.rectangle(img_copy, (x1, y1 - 30), (x1 + tw, y1), warna, -1)
        cv2.putText(
            img_copy,
            text_label,
            (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return img_copy


# ============================================================
# 4. Helper: encode/decode gambar base64
# ============================================================
def decode_base64_image(data_uri):
    """Decode base64 data URI menjadi numpy array (BGR)."""
    # Hapus header "data:image/...;base64," jika ada
    if "," in data_uri:
        data_uri = data_uri.split(",", 1)[1]
    img_bytes = base64.b64decode(data_uri)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


def encode_image_base64(image, ext=".jpg"):
    """Encode numpy array (BGR) menjadi base64 data URI."""
    _, buffer = cv2.imencode(ext, image)
    b64 = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


# ============================================================
# 5. Routes
# ============================================================
@app.route("/")
def index():
    """Halaman utama."""
    models_loaded = detector is not None and classifier is not None
    return render_template("index.html", models_loaded=models_loaded)


@app.route("/detect", methods=["POST"])
def detect():
    """
    Endpoint deteksi strabismus.
    Mode Camera : { image: base64, roi: {x, y, w, h}, mode: "camera" }
    Mode Upload : { image: base64, mode: "upload" }
    """
    if detector is None or classifier is None:
        return jsonify({"error": "Model belum dimuat"}), 500

    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "Data gambar tidak ditemukan"}), 400

    # Decode gambar
    cv_image = decode_base64_image(data["image"])
    if cv_image is None:
        return jsonify({"error": "Gagal decode gambar"}), 400

    mode = data.get("mode", "upload")

    # ===========================================================
    # MODE: CAMERA (mirror + crop ROI)
    # ===========================================================
    if mode == "camera":
        # Mirror mode (horizontal flip) — konsisten seperti cermin
        cv_image_mirrored = cv2.flip(cv_image, 1)
        h, w = cv_image_mirrored.shape[:2]

        # ROI dari request, atau default 50% x 70% centered
        roi = data.get("roi", None)
        if roi:
            face_roi_x = int(roi["x"])
            face_roi_y = int(roi["y"])
            face_roi_w = int(roi["w"])
            face_roi_h = int(roi["h"])
        else:
            face_roi_w, face_roi_h = int(w * 0.5), int(h * 0.7)
            face_roi_x = (w - face_roi_w) // 2
            face_roi_y = (h - face_roi_h) // 2

        # Crop ROI
        roi_img = cv_image_mirrored[
            face_roi_y : face_roi_y + face_roi_h,
            face_roi_x : face_roi_x + face_roi_w,
        ]

        if roi_img.size == 0:
            return jsonify(
                {
                    "status": STATUS_NOT_DETECTED,
                    "detections": [],
                    "result_image": encode_image_base64(cv_image_mirrored),
                    "preview_image": encode_image_base64(cv_image_mirrored),
                }
            )

        # Deteksi & klasifikasi pada area ROI saja
        detections, status = detect_and_classify(roi_img)

        # Konversi koordinat ROI ke koordinat gambar penuh
        for d in detections:
            d["box"][0] += face_roi_x
            d["box"][1] += face_roi_y
            d["box"][2] += face_roi_x
            d["box"][3] += face_roi_y

        # Gambar hasil deteksi
        result_img = draw_boxes(cv_image_mirrored, detections)
        # Tambahkan bounding box area deteksi pada gambar hasil
        cv2.rectangle(
            result_img,
            (face_roi_x, face_roi_y),
            (face_roi_x + face_roi_w, face_roi_y + face_roi_h),
            (255, 255, 0),
            2,
        )
        cv2.putText(
            result_img,
            "AREA DETEKSI",
            (face_roi_x + 10, face_roi_y + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )

        # Gambar preview (dengan overlay gelap di luar ROI)
        preview_img = cv_image_mirrored.copy()
        cv2.rectangle(
            preview_img,
            (face_roi_x, face_roi_y),
            (face_roi_x + face_roi_w, face_roi_y + face_roi_h),
            (255, 255, 0),
            3,
        )
        cv2.putText(
            preview_img,
            "AREA WAJAH",
            (face_roi_x + 10, face_roi_y + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )
        # Overlay gelap di luar bounding box
        overlay = preview_img.copy()
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[
            face_roi_y : face_roi_y + face_roi_h,
            face_roi_x : face_roi_x + face_roi_w,
        ] = 255
        overlay[mask == 0] = (overlay[mask == 0] * 0.4).astype(np.uint8)

        return jsonify(
            {
                "status": status,
                "detections": detections,
                "result_image": encode_image_base64(result_img),
                "preview_image": encode_image_base64(overlay),
            }
        )

    # ===========================================================
    # MODE: UPLOAD (tanpa mirror, full image)
    # ===========================================================
    else:
        detections, status = detect_and_classify(cv_image)
        result_img = draw_boxes(cv_image, detections)

        return jsonify(
            {
                "status": status,
                "detections": detections,
                "result_image": encode_image_base64(result_img),
            }
        )


# ============================================================
# 6. Startup
# ============================================================
if __name__ == "__main__":
    if load_models():
        app.run(debug=True, host="0.0.0.0", port=5000)
    else:
        print("❌ Gagal memuat model. Pastikan file model berada di folder 'models/'.")
