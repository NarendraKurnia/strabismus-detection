# app.py
import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile
import os
from pathlib import Path

# Konfigurasi halaman
st.set_page_config(
    page_title="Deteksi Mata Juling",
    page_icon="👁️",
    layout="wide"
)

# Load models
@st.cache_resource
def load_models():
    BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    MODEL_DETEKSI_PATH = BASE_DIR / "models" / "model_mata.pt"
    MODEL_KLASIFIKASI_PATH = BASE_DIR / "models" / "model_juling.pt"
    
    try:
        detector = YOLO(str(MODEL_DETEKSI_PATH))
        classifier = YOLO(str(MODEL_KLASIFIKASI_PATH))
        return detector, classifier
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None, None

# Deteksi mata dan klasifikasi
def detect_and_classify(image, detector, classifier):
    hasil_deteksi = []
    status = "NORMAL"
    
    # Deteksi area mata
    results_det = detector.predict(image, conf=0.15, verbose=False)
    regions = []
    
    for r in results_det:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            regions.append([x1, y1, x2, y2])
    
    if not regions:
        # Fallback ke Haar Cascade
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in eyes:
            regions.append([x, y, x + w, y + h])
    
    if not regions:
        return hasil_deteksi, "TIDAK TERDETEKSI"
    
    # Ambil area strip mata
    min_x = min([r[0] for r in regions])
    min_y = min([r[1] for r in regions])
    max_x = max([r[2] for r in regions])
    max_y = max([r[3] for r in regions])
    
    pad_x, pad_y = 20, 20
    strip_x1 = max(0, min_x - pad_x)
    strip_y1 = max(0, min_y - pad_y)
    strip_x2 = min(image.shape[1], max_x + pad_x)
    strip_y2 = min(image.shape[0], max_y + pad_y)
    
    strip_img = image[strip_y1:strip_y2, strip_x1:strip_x2]
    
    # Klasifikasi
    results_cls = classifier.predict(strip_img, conf=0.3, verbose=False)
    
    if len(results_cls[0].boxes) > 0:
        for box in results_cls[0].boxes:
            bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy().astype(int)
            class_id = int(box.cls[0])
            label_name = classifier.names[class_id].upper()
            conf_score = float(box.conf[0])
            
            abs_x1 = bx1 + strip_x1
            abs_y1 = by1 + strip_y1
            abs_x2 = bx2 + strip_x1
            abs_y2 = by2 + strip_y1
            
            hasil_deteksi.append({
                'box': [abs_x1, abs_y1, abs_x2, abs_y2],
                'label': label_name,
                'conf': conf_score
            })
            
            if "JULING" in label_name or "STRABISMUS" in label_name:
                status = "STRABISMUS (JULING)"
    else:
        for (x1, y1, x2, y2) in regions:
            hasil_deteksi.append({
                'box': [x1, y1, x2, y2],
                'label': "NORMAL",
                'conf': 0.0
            })
    
    return hasil_deteksi, status

# Gambar bounding box
def draw_boxes(image, detections):
    img_copy = image.copy()
    for d in detections:
        x1, y1, x2, y2 = d['box']
        label = d['label']
        conf = d['conf']
        
        warna = (0, 0, 255) if "JULING" in label or "STRABISMUS" in label else (0, 255, 0)
        
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), warna, 2)
        text_label = f"{label} {conf:.1%}" if conf > 0 else label
        
        (tw, th), _ = cv2.getTextSize(text_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img_copy, (x1, y1 - 25), (x1 + tw, y1), warna, -1)
        cv2.putText(img_copy, text_label, (x1, y1 - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    
    return img_copy

# Halaman Live Camera dengan ROI
def live_camera_page(detector, classifier):
    st.subheader("📸 Live Camera Detection")
    
    # Area ROI (mata)
    roi_width = 400
    roi_height = 200
    
    # Inisialisasi kamera
    camera = cv2.VideoCapture(0)
    
    # Placeholder untuk video
    frame_placeholder = st.empty()
    warning_placeholder = st.empty()
    status_placeholder = st.empty()
    
    capture_button = st.button("📸 Ambil Foto", disabled=True, key="capture_btn")
    captured_image = st.session_state.get('captured_image', None)
    
    # Stream video
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Preview Kamera")
        video_placeholder = st.empty()
    
    with col2:
        st.markdown("### Hasil Deteksi")
        result_placeholder = st.empty()
    
    stop_btn = st.button("⏹️ Stop Kamera")
    
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    while not stop_btn and camera.isOpened():
        ret, frame = camera.read()
        if not ret:
            st.error("Gagal mengakses kamera")
            break
        
        frame = cv2.flip(frame, 1)
        height, width = frame.shape[:2]
        
        # Posisi ROI di tengah
        roi_x = (width - roi_width) // 2
        roi_y = (height - roi_height) // 2
        
        # Deteksi wajah
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
        
        # Gambar ROI
        overlay = frame.copy()
        cv2.rectangle(overlay, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), (0, 255, 0), 3)
        cv2.putText(overlay, "AREA MATA", (roi_x + 10, roi_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Cek posisi wajah
        face_in_roi = False
        for (fx, fy, fw, fh) in faces:
            face_center_x = fx + fw // 2
            face_center_y = fy + fh // 2
            roi_center_x = roi_x + roi_width // 2
            roi_center_y = roi_y + roi_height // 2
            
            # Deteksi area mata dalam ROI
            eye_roi = gray[fy:fy+fh, fx:fx+fw]
            eyes = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml').detectMultiScale(eye_roi, 1.1, 5)
            
            for (ex, ey, ew, eh) in eyes:
                eye_abs_x = fx + ex
                eye_abs_y = fy + ey
                # Cek apakah mata berada dalam ROI
                if (roi_x < eye_abs_x + ew and roi_x + roi_width > eye_abs_x and
                    roi_y < eye_abs_y + eh and roi_y + roi_height > eye_abs_y):
                    face_in_roi = True
                    cv2.rectangle(overlay, (fx, fy), (fx+fw, fy+fh), (255, 0, 0), 2)
                    break
        
        if face_in_roi:
            warning_placeholder.success("✅ Posisi wajah sudah pas! Silakan klik 'Ambil Foto'")
            capture_button.disabled = False
        else:
            warning_placeholder.warning("⚠️ Posisikan wajah Anda di AREA MATA yang tersedia!")
            capture_button.disabled = True
        
        video_placeholder.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), channels="RGB")
        
        # Tombol capture menggunakan session state
        if st.button("📸 Ambil Foto", key="capture_btn_live"):
            st.session_state.captured_image = frame.copy()
            captured_image = frame.copy()
            break
        
        # Update tombol stop setiap iterasi
        if st.button("⏹️ Stop Kamera", key="stop_btn_live"):
            break
    
    camera.release()
    
    # Proses hasil capture
    if captured_image is not None:
        st.success("Gambar berhasil diambil! Memproses...")
        
        # Deteksi dan klasifikasi
        detections, status = detect_and_classify(captured_image, detector, classifier)
        
        # Gambar hasil
        result_img = draw_boxes(captured_image, detections)
        
        result_placeholder.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), channels="RGB")
        
        # Tampilkan status
        if status == "STRABISMUS (JULING)":
            status_placeholder.error(f"🟡 HASIL: {status}")
        elif status == "TIDAK TERDETEKSI":
            status_placeholder.warning(f"⚠️ HASIL: {status} - Tidak dapat mendeteksi mata")
        else:
            status_placeholder.success(f"✅ HASIL: {status}")

# Halaman Upload Foto
def upload_page(detector, classifier):
    st.subheader("📤 Upload Foto Wajah")
    
    uploaded_file = st.file_uploader(
        "Pilih foto wajah Anda",
        type=["jpg", "jpeg", "png", "bmp"],
        help="Upload foto dengan posisi wajah menghadap kamera"
    )
    
    if uploaded_file is not None:
        # Baca gambar
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Foto Asli", use_container_width=True)
        
        with col2:
            # Proses deteksi
            with st.spinner("Memproses deteksi mata..."):
                detections, status = detect_and_classify(image, detector, classifier)
                result_img = draw_boxes(image, detections)
                st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), caption="Hasil Deteksi", use_container_width=True)
            
            # Tampilkan hasil
            st.markdown("---")
            st.markdown("### 📊 Hasil Screening")
            
            if status == "STRABISMUS (JULING)":
                st.error(f"🟡 DIAGNOSA: {status}")
                st.warning("⚠️ Disarankan untuk berkonsultasi dengan dokter mata")
            elif status == "TIDAK TERDETEKSI":
                st.warning(f"⚠️ {status} - Pastikan mata terlihat jelas dalam foto")
            else:
                st.success(f"✅ DIAGNOSA: {status}")
                st.info("👁️ Mata Anda terdeteksi normal")

# Main App
def main():
    st.title("👁️ YUK DETEKSI DINI MATA KAMU")
    st.markdown("Upload foto wajah kamu untuk cek kesehatan mata strabismus (juling)")
    
    # Load models
    detector, classifier = load_models()
    
    if detector is None or classifier is None:
        st.error("Model tidak dapat dimuat. Pastikan file model tersedia di folder 'models'")
        st.info("Folder struktur yang diharapkan:\n- models/model_mata.pt\n- models/model_juling.pt")
        return
    
    # Pilihan input
    input_method = st.radio(
        "Pilih metode input:",
        ["📤 Upload Foto", "📸 Live Camera"],
        horizontal=True
    )
    
    if input_method == "📤 Upload Foto":
        upload_page(detector, classifier)
    else:
        live_camera_page(detector, classifier)
    
    # Footer
    st.markdown("---")
    st.markdown("### ℹ️ Informasi")
    st.markdown("""
    - **Deteksi Strabismus (Mata Juling)** menggunakan teknologi AI
    - Hasil deteksi bersifat screening awal, bukan diagnosis medis
    - Untuk diagnosis akurat, konsultasikan dengan dokter mata
    """)

if __name__ == "__main__":
    main()