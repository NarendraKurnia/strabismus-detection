import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
from pathlib import Path

# 1. Konfigurasi Halaman & Gaya Kustom (Elegan & Modern)
st.set_page_config(
    page_title="Deteksi Mata Juling - AI Screening",
    page_icon="👁️",
    layout="wide"
)

# Custom CSS untuk menyamai nuansa warna hangat & profesional (seperti gambar referensi)
st.markdown("""
    <style>
    /* Mengubah font global dan background card */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
    
    html, body, [data-testid="stSidebarUserContent"] {
        font-family: 'Poppins', sans-serif;
    }
    
    /* Banner Header bergaya pastel cream seperti gambar */
    .hero-banner {
        background-color: #FCE6A4;
        padding: 30px;
        border-radius: 15px;
        margin-bottom: 25px;
        border-left: 8px solid #D4A373;
    }
    .hero-title {
        color: #4A3E3D;
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 5px;
    }
    .hero-subtitle {
        color: #6B5B52;
        font-size: 16px;
    }
    
    /* Desain Card Berwarna Putih Bersih */
    .content-card {
        background-color: #FFFFFF;
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        margin-bottom: 20px;
        border: 1px solid #EAEAEA;
    }
    
    /* Mempercantik tampilan Radio Button Input */
    div[data-testid="stRadio"] > label {
        font-weight: bold;
        color: #4A3E3D;
    }
    </style>
""", unsafe_unsafe_html=True)

# 2. Load Models dengan Cache
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

# 3. Core Engine: Deteksi mata dan klasifikasi
def detect_and_classify(image, detector, classifier):
    hasil_deteksi = []
    status = "NORMAL"
    
    results_det = detector.predict(image, conf=0.15, verbose=False)
    regions = []
    
    for r in results_det:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            regions.append([x1, y1, x2, y2])
    
    if not regions:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in eyes:
            regions.append([x, y, x + w, y + h])
    
    if not regions:
        return hasil_deteksi, "TIDAK TERDETEKSI"
    
    min_x, min_y = min([r[0] for r in regions]), min([r[1] for r in regions])
    max_x, max_y = max([r[2] for r in regions]), max([r[3] for r in regions])
    
    pad_x, pad_y = 20, 20
    strip_x1 = max(0, min_x - pad_x)
    strip_y1 = max(0, min_y - pad_y)
    strip_x2 = min(image.shape[1], max_x + pad_x)
    strip_y2 = min(image.shape[0], max_y + pad_y)
    
    strip_img = image[strip_y1:strip_y2, strip_x1:strip_x2]
    
    results_cls = classifier.predict(strip_img, conf=0.3, verbose=False)
    
    if len(results_cls[0].boxes) > 0:
        for box in results_cls[0].boxes:
            bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy().astype(int)
            class_id = int(box.cls[0])
            label_name = classifier.names[class_id].upper()
            conf_score = float(box.conf[0])
            
            abs_x1, abs_y1 = bx1 + strip_x1, by1 + strip_y1
            abs_x2, abs_y2 = bx2 + strip_x1, by2 + strip_y1
            
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

def draw_boxes(image, detections):
    img_copy = image.copy()
    for d in detections:
        x1, y1, x2, y2 = d['box']
        label = d['label']
        conf = d['conf']
        
        warna = (0, 0, 255) if "JULING" in label or "STRABISMUS" in label else (0, 255, 0)
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), warna, 3)
        text_label = f"{label} {conf:.1%}" if conf > 0 else label
        
        (tw, th), _ = cv2.getTextSize(text_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img_copy, (x1, y1 - 30), (x1 + tw, y1), warna, -1)
        cv2.putText(img_copy, text_label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return img_copy

# 4. Halaman Live Camera Menggunakan Kamera Browser Native (Stabil & Mendukung Cloud)
def live_camera_page(detector, classifier):
    st.markdown('<div class="content-card"><h3>📸 Ambil Foto via Live Camera</h3>'
                '<p style="color: #666;">Izinkan browser mengakses kamera Anda. Pastikan wajah berada tepat di tengah frame.</p></div>', unsafe_html=True)
    
    img_file = st.camera_input("Arahkan pandangan mata Anda lurus ke kamera")
    
    if img_file is not None:
        bytes_data = img_file.getvalue()
        cv_image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB), caption="Foto Masuk", use_container_width=True)
            
        with col2:
            with st.spinner("Menganalisis posisi mata..."):
                detections, status = detect_and_classify(cv_image, detector, classifier)
                result_img = draw_boxes(cv_image, detections)
                st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), caption="Hasil Analisis AI", use_container_width=True)
                
            tampilkan_hasil_diagnosa(status)

# 5. Halaman Upload Foto
def upload_page(detector, classifier):
    st.markdown('<div class="content-card"><h3>📤 Unggah File Citra Wajah</h3>'
                '<p style="color: #666;">Gunakan foto berkualitas tinggi dengan pencahayaan terang dari arah depan.</p></div>', unsafe_html=True)
    
    uploaded_file = st.file_uploader(
        "Pilih file gambar Anda",
        type=["jpg", "jpeg", "png", "bmp"],
        help="Mendukung format JPG, PNG, atau BMP"
    )
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Foto Asli", use_container_width=True)
        
        with col2:
            with st.spinner("Memproses deteksi mata..."):
                detections, status = detect_and_classify(image, detector, classifier)
                result_img = draw_boxes(image, detections)
                st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), caption="Hasil Deteksi", use_container_width=True)
            
            tampilkan_hasil_diagnosa(status)

# Helper untuk mempercantik box hasil diagnosis
def tampilkan_hasil_diagnosa(status):
    st.markdown("---")
    st.markdown("### 📊 Hasil Resmi Screening")
    if status == "STRABISMUS (JULING)":
        st.error(f"🚨 **DIAGNOSA AWAL: {status}**")
        st.warning("⚠️ **Rekomendasi:** Hasil ini mendeteksi adanya indikasi deviasi mata. Sangat disarankan untuk melakukan pemeriksaan komprehensif ke Dokter Spesialis Mata (Oftalmolog).")
    elif status == "TIDAK TERDETEKSI":
        st.warning(f"🔍 **STATUS: {status}**")
        st.info("Pastikan wajah menghadap lurus ke depan, area mata tidak tertutup kacamata/rambut, dan pencahayaan mencukupi.")
    else:
        st.success(f"✅ **DIAGNOSA AWAL: {status} (Kondisi Normal)**")
        st.info("👁️ Sistem mendeteksi arah pandang bola mata Anda sejajar dan simetris.")

# 6. Main App Structure
def main():
    # Mengaplikasikan Banner Atas Mirip Gambar Desain UI
    st.markdown("""
        <div class="hero-banner">
            <div class="hero-title">👁️ YUK DETEKSI DINI MATA KAMU</div>
            <div class="hero-subtitle">Unggah foto wajah atau gunakan kamera langsung untuk mengecek indikasi kesehatan mata strabismus (juling).</div>
        </div>
    """, unsafe_html=True)
    
    detector, classifier = load_models()
    
    if detector is None or classifier is None:
        st.error("Sistem gagal memuat file bobot model `.pt`. Pastikan direktori `/models` sudah benar.")
        return
    
    # Switcher Metode Input Bergaya Modern
    input_method = st.radio(
        "Pilih Metode Pengambilan Data:",
        ["📤 Upload Foto Wajah", "📸 Gunakan Live Camera (Browser)"],
        horizontal=True
    )
    
    st.markdown('<div style="margin-top: 15px;"></div>', unsafe_html=True)
    
    if "Upload Foto" in input_method:
        upload_page(detector, classifier)
    else:
        live_camera_page(detector, classifier)
    
    # Footer Informasi Klinis
    st.markdown("---")
    st.markdown(
        '<div style="text-align: center; color: #777; font-size: 13px;">'
        '<strong>Disclaimer:</strong> Aplikasi ini merupakan alat skrining berbasis kecerdasan buatan (AI) '
        'dan tidak menggantikan diagnosis medis resmi profesional. © 2026 Deteksi Strabismus AI.'
        '</div>', 
        unsafe_html=True
    )

if __name__ == "__main__":
    main()