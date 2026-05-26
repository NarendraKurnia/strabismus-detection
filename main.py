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

# Inisialisasi session state
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False
if 'captured_image' not in st.session_state:
    st.session_state.captured_image = None
if 'processing_done' not in st.session_state:
    st.session_state.processing_done = False

# Load models
@st.cache_resource
def load_models():
    BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    MODEL_DETEKSI_PATH = BASE_DIR / "models" / "model_mata.pt"
    MODEL_KLASIFIKASI_PATH = BASE_DIR / "models" / "model_juling.pt"
    
    try:
        # Cek apakah file model ada
        if not MODEL_DETEKSI_PATH.exists():
            st.warning(f"Model deteksi tidak ditemukan di {MODEL_DETEKSI_PATH}")
            return None, None
        if not MODEL_KLASIFIKASI_PATH.exists():
            st.warning(f"Model klasifikasi tidak ditemukan di {MODEL_KLASIFIKASI_PATH}")
            return None, None
            
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

# Fungsi untuk membuka kamera dan mengambil frame
def get_camera_frame():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return None
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None

# Komponen Live Camera dengan JavaScript
def live_camera_component():
    st.markdown("""
    <style>
    .camera-container {
        position: relative;
        width: 100%;
        max-width: 640px;
        margin: 0 auto;
    }
    video {
        width: 100%;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .roi-overlay {
        position: absolute;
        border: 3px solid #00ff00;
        border-radius: 10px;
        box-shadow: 0 0 0 9999px rgba(0,0,0,0.5);
        pointer-events: none;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Menggunakan HTML5 video stream
    video_html = """
    <div class="camera-container">
        <video id="video" autoplay playsinline style="width:100%; max-width:640px; border-radius:10px;"></video>
        <canvas id="canvas" style="display:none;"></canvas>
        <div id="roi" style="position:absolute; top:50%; left:50%; transform:translate(-50%,-50%); 
             width:400px; height:200px; border:3px solid #00ff00; border-radius:10px;
             box-shadow:0 0 0 9999px rgba(0,0,0,0.5); pointer-events:none;"></div>
    </div>
    <div style="text-align:center; margin-top:10px;">
        <p id="warning" style="color:orange; font-weight:bold;">⚠️ Posisikan wajah Anda di AREA MATA yang tersedia!</p>
    </div>
    <script>
        const video = document.getElementById('video');
        const canvas = document.getElementById('canvas');
        const context = canvas.getContext('2d');
        const roi = document.getElementById('roi');
        const warning = document.getElementById('warning');
        
        // Akses kamera
        navigator.mediaDevices.getUserMedia({ video: true })
            .then(stream => {
                video.srcObject = stream;
                video.play();
            })
            .catch(err => {
                console.error("Error accessing camera: ", err);
                warning.innerText = "❌ Gagal mengakses kamera! Pastikan izin kamera diberikan.";
                warning.style.color = "red";
            });
        
        // Fungsi untuk mengambil foto
        window.capturePhoto = function() {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            context.drawImage(video, 0, 0, canvas.width, canvas.height);
            const imageData = canvas.toDataURL('image/png');
            return imageData;
        };
    </script>
    """
    
    st.components.v1.html(video_html, height=500)
    
    # Tombol ambil foto
    if st.button("📸 Ambil Foto", key="capture_btn_live"):
        # Ambil frame dari kamera menggunakan OpenCV
        frame = get_camera_frame()
        if frame is not None:
            st.session_state.captured_image = frame
            st.session_state.processing_done = False
            st.rerun()
        else:
            st.error("Gagal mengambil foto. Pastikan kamera terhubung.")

# Halaman Live Camera dengan OpenCV (Alternatif)
def live_camera_page_alternative(detector, classifier):
    st.subheader("📸 Live Camera Detection")
    
    # Informasi ROI
    st.info("👁️ Area hijau adalah target area mata. Posisikan mata Anda di dalam area tersebut.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Preview Kamera")
        # Placeholder untuk video frame
        frame_placeholder = st.empty()
        
        # Tombol kontrol
        if st.button("🎥 Mulai Kamera", key="start_cam"):
            st.session_state.camera_active = True
        
        if st.button("⏹️ Stop Kamera", key="stop_cam"):
            st.session_state.camera_active = False
            st.session_state.captured_image = None
        
        ambil_foto = st.button("📸 Ambil Foto", key="capture_btn")
        
        # ROI dimensions
        roi_width = 400
        roi_height = 200
        
        # Face detector
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Proses kamera
        if st.session_state.camera_active:
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            if not cap.isOpened():
                st.error("❌ Tidak dapat mengakses kamera! Pastikan kamera terhubung dan tidak digunakan aplikasi lain.")
                st.session_state.camera_active = False
            else:
                ret, frame = cap.read()
                if ret:
                    frame = cv2.flip(frame, 1)
                    height, width = frame.shape[:2]
                    
                    # Posisi ROI
                    roi_x = (width - roi_width) // 2
                    roi_y = (height - roi_height) // 2
                    
                    # Deteksi wajah dan mata
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
                    
                    # Cek posisi mata di ROI
                    face_in_roi = False
                    for (x, y, w, h) in faces:
                        face_roi = gray[y:y+h, x:x+w]
                        eyes = eye_cascade.detectMultiScale(face_roi, 1.1, 5)
                        
                        for (ex, ey, ew, eh) in eyes:
                            eye_x = x + ex
                            eye_y = y + ey
                            
                            # Cek apakah mata dalam ROI
                            if (roi_x < eye_x + ew and roi_x + roi_width > eye_x and
                                roi_y < eye_y + eh and roi_y + roi_height > eye_y):
                                face_in_roi = True
                                # Gambar bounding box wajah
                                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                                break
                    
                    # Gambar ROI
                    cv2.rectangle(frame, (roi_x, roi_y), (roi_x+roi_width, roi_y+roi_height), (0, 255, 0), 3)
                    cv2.putText(frame, "AREA MATA", (roi_x+10, roi_y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Tampilkan peringatan
                    warning_placeholder = st.empty()
                    if face_in_roi:
                        warning_placeholder.success("✅ Posisi wajah sudah pas! Silakan klik 'Ambil Foto'")
                        if ambil_foto:
                            st.session_state.captured_image = frame.copy()
                            st.session_state.camera_active = False
                            st.session_state.processing_done = False
                            st.rerun()
                    else:
                        warning_placeholder.warning("⚠️ Posisikan wajah Anda di AREA MATA yang tersedia!")
                    
                    # Tampilkan frame
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                
                cap.release()
    
    with col2:
        st.markdown("### Hasil Deteksi")
        if st.session_state.captured_image is not None and not st.session_state.processing_done:
            with st.spinner("Memproses deteksi mata..."):
                detections, status = detect_and_classify(st.session_state.captured_image, detector, classifier)
                result_img = draw_boxes(st.session_state.captured_image, detections)
                st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), caption="Hasil Deteksi", use_container_width=True)
                
                # Tampilkan status
                if status == "STRABISMUS (JULING)":
                    st.error(f"🟡 DIAGNOSA: {status}")
                    st.warning("⚠️ Disarankan untuk berkonsultasi dengan dokter mata")
                elif status == "TIDAK TERDETEKSI":
                    st.warning(f"⚠️ {status} - Pastikan mata terlihat jelas dalam foto")
                else:
                    st.success(f"✅ DIAGNOSA: {status}")
                    st.info("👁️ Mata Anda terdeteksi normal")
                
                st.session_state.processing_done = True
        elif st.session_state.captured_image is not None and st.session_state.processing_done:
            # Tampilkan hasil yang sudah ada
            detections, status = detect_and_classify(st.session_state.captured_image, detector, classifier)
            result_img = draw_boxes(st.session_state.captured_image, detections)
            st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), caption="Hasil Deteksi", use_container_width=True)
            
            if status == "STRABISMUS (JULING)":
                st.error(f"🟡 DIAGNOSA: {status}")
            elif status == "TIDAK TERDETEKSI":
                st.warning(f"⚠️ {status}")
            else:
                st.success(f"✅ DIAGNOSA: {status}")

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
        
        # Demo mode tanpa model
        if st.button("Lanjutkan dengan Mode Demo"):
            st.session_state.demo_mode = True
            st.rerun()
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
        live_camera_page_alternative(detector, classifier)
    
    # Footer
    st.markdown("---")
    st.markdown("### ℹ️ Informasi")
    st.markdown("""
    - **Deteksi Strabismus (Mata Juling)** menggunakan teknologi AI
    - Hasil deteksi bersifat screening awal, bukan diagnosis medis
    - Untuk diagnosis akurat, konsultasikan dengan dokter mata
    - Pastikan cahaya cukup saat menggunakan kamera
    """)

if __name__ == "__main__":
    main()