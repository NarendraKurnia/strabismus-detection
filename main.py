import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
from pathlib import Path

# 1. Konfigurasi Halaman & Tema Visual Kustom
st.set_page_config(
    page_title="Deteksi Mata Juling - AI Mirror Screening",
    page_icon="👁️",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
    
    html, body, [data-testid="stSidebarUserContent"] {
        font-family: 'Poppins', sans-serif;
    }
    
    /* Banner Header Pastel Cream */
    .hero-banner {
        background-color: #FCE6A4;
        padding: 35px;
        border-radius: 16px;
        margin-bottom: 25px;
        border-left: 8px solid #D4A373;
    }
    .hero-title {
        color: #4A3E3D;
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 8px;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        color: #6B5B52;
        font-size: 16px;
        line-height: 1.5;
    }
    
    /* Container Box / Card Putih Elegan */
    .content-card {
        background-color: #FFFFFF;
        padding: 25px;
        border-radius: 14px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.04);
        margin-bottom: 20px;
        border: 1px solid #EFEFEF;
    }
    
    /* Mempercantik Tampilan Radio Button */
    div[data-testid="stRadio"] > label {
        font-weight: 600;
        color: #4A3E3D;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# 2. Inisialisasi Session State
if 'captured_image' not in st.session_state:
    st.session_state.captured_image = None

# 3. Memuat Model AI Pintar
@st.cache_resource
def load_models():
    BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    MODEL_DETEKSI_PATH = BASE_DIR / "models" / "model_mata.pt"
    MODEL_KLASIFIKASI_PATH = BASE_DIR / "models" / "model_juling.pt"
    
    try:
        if not MODEL_DETEKSI_PATH.exists() or not MODEL_KLASIFIKASI_PATH.exists():
            return None, None
            
        detector = YOLO(str(MODEL_DETEKSI_PATH))
        classifier = YOLO(str(MODEL_KLASIFIKASI_PATH))
        return detector, classifier
    except Exception as e:
        return None, None

# 4. Deteksi Mata & Klasifikasi Strabismus
def detect_and_classify(image, detector, classifier):
    hasil_deteksi = []
    status = "NORMAL"
    
    # Deteksi Area Mata dengan YOLO
    results_det = detector.predict(image, conf=0.15, verbose=False)
    regions = []
    
    for r in results_det:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            regions.append([x1, y1, x2, y2])
    
    # Fallback ke Haar Cascade jika YOLO gagal mendeteksi
    if not regions:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in eyes:
            regions.append([x, y, x + w, y + h])
    
    if not regions:
        return hasil_deteksi, "TIDAK TERDETEKSI"
    
    # Ambil koordinat pembatas terkecil dan terbesar
    min_x, min_y = min([r[0] for r in regions]), min([r[1] for r in regions])
    max_x, max_y = max([r[2] for r in regions]), max([r[3] for r in regions])
    
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

# 5. Menggambar Box Deteksi secara Estetik
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

# 6. Komponen Tampilan Hasil Diagnosa yang Rapi
def tampilkan_hasil_diagnosa(status):
    st.markdown('<div style="margin-top:20px;"></div>', unsafe_allow_html=True)
    st.markdown("### 📊 Hasil Analisis Cermin AI")
    if status == "STRABISMUS (JULING)":
        st.error(f"🚨 **DIAGNOSA AWAL: {status}**")
        st.warning("⚠️ **Rekomendasi Ahli:** Deteksi cermin AI mendeteksi adanya indikasi sudut juling pada mata. Disarankan untuk menjadwalkan pemeriksaan menyeluruh bersama Dokter Spesialis Mata.")
    elif status == "TIDAK TERDETEKSI":
        st.warning(f"🔍 **STATUS: {status}**")
        st.info("Sistem kesulitan memetakan posisi mata Anda dengan jelas. Pastikan wajah tegak lurus menghadap kamera tanpa terhalang rambut atau kacamata reflektif.")
    else:
        st.success(f"✅ **DIAGNOSA AWAL: {status} (Kondisi Normal)**")
        st.info("👁️ Cermin AI mendeteksi sumbu bola mata kanan dan kiri Anda sejajar serta simetris.")

# 7. Halaman Live Camera (Kompatibel Penuh Cloud, Mirror Mode, dan Area Wajah)
def live_camera_page(detector, classifier):
    st.markdown('<div class="content-card"><h3>📸 Pemindaian via Live Cermin AI</h3>'
                '<p style="color: #666; margin: 0;">Berikan izin akses kamera. Posisikan wajah Anda tepat di dalam <strong>Area Wajah (Kotak Teal)</strong> dan mata di dalam Kotak Putih.</p></div>', unsafe_allow_html=True)
    
    # Input Kamera Browser Native
    img_file = st.camera_input("Arahkan pandangan mata Anda lurus ke 'Cermin' Cerdas ini")
    
    if img_file is not None:
        # 1. Konversi ke Format OpenCV
        bytes_data = img_file.getvalue()
        cv_image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        
        # 2. PERUBAHAN UTAMA: MIRROR MODE (Balik Horizontal)
        # Menjadikan Live Cam terasa natural seperti cermin
        cv_image_mirrored = cv2.flip(cv_image, 1)
        
        # 3. PERUBAHAN UTAMA: DETEKSI AREA WAJAH & MATA PADA PREVIEW
        # Mendapatkan dimensi gambar untuk membuat overlay
        h, w = cv_image_mirrored.shape[:2]
        cv_image_with_areas = cv_image_mirrored.copy()
        
        # Definisi Kotak Target Area Wajah (Teal) - Target posisi kepala
        face_roi_w, face_roi_h = int(w * 0.5), int(h * 0.7) # Kotak wajah 50% lebar, 70% tinggi
        face_roi_x, face_roi_y = (w - face_roi_w) // 2, (h - face_roi_h) // 2 # Centered
        
        # Definisi Kotak Target Area Mata (Putih) - Lebih spesifik
        eye_roi_w, eye_roi_h = int(w * 0.35), int(h * 0.22) # Kotak mata 35% lebar, 22% tinggi
        eye_roi_x, eye_roi_y = (w - eye_roi_w) // 2, int(h * 0.28) # Posisi mata lebih ke atas
        
        # Menggambar Kotak Area Wajah (Teal)
        cv2.rectangle(cv_image_with_areas, (face_roi_x, face_roi_y), 
                      (face_roi_x + face_roi_w, face_roi_y + face_roi_h), (255, 255, 0), 4) # Teal/Cyan
        cv2.putText(cv_image_with_areas, "AREA WAJAH", (face_roi_x + 10, face_roi_y + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
        
        # Menggambar Kotak Area Mata (Putih)
        cv2.rectangle(cv_image_with_areas, (eye_roi_x, eye_roi_y), 
                      (eye_roi_x + eye_roi_w, eye_roi_y + eye_roi_h), (255, 255, 255), 3) # Putih
        cv2.putText(cv_image_with_areas, "AREA MATA", (eye_roi_x + 10, eye_roi_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Layout Kolom Preview dan Hasil
        col1, col2 = st.columns(2)
        with col1:
            # Menampilkan Preview berkotak (dan sudah di-mirror)
            st.image(cv2.cvtColor(cv_image_with_areas, cv2.COLOR_BGR2RGB), caption="Preview 'Cermin' Cerdas", use_container_width=True)
            
        with col2:
            with st.spinner("Cermin AI sedang memetakan geometri simetris mata Anda..."):
                # Menjalankan Diagnosa pada gambar ber-mirror (tanpa kotak overlay)
                detections, status = detect_and_classify(cv_image_mirrored, detector, classifier)
                
                # Menggambar hasil bounding box AI (juga pada gambar ber-mirror)
                result_img = draw_boxes(cv_image_mirrored, detections)
                st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), caption="Hasil Analisis Geometri AI", use_container_width=True)
                
            tampilkan_hasil_diagnosa(status)

# 8. Halaman Unggah File Foto
def upload_page(detector, classifier):
    st.markdown('<div class="content-card"><h3>📤 Unggah Dokumen Foto</h3>'
                '<p style="color: #666; margin: 0;">Gunakan foto beresolusi tajam dengan pencahayaan yang cukup cerah dari depan wajah.</p></div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Pilih dokumen gambar",
        type=["jpg", "jpeg", "png", "bmp"],
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Diagnosa pada foto (tanpa mirror, karena ini adalah file foto asli)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Foto Asli", use_container_width=True)
        
        with col2:
            with st.spinner("Memproses deteksi area mata..."):
                detections, status = detect_and_classify(image, detector, classifier)
                result_img = draw_boxes(image, detections)
                st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), caption="Hasil Analisis Geometri", use_container_width=True)
            
            tampilkan_hasil_diagnosa(status)

# 9. Main Orchestrator
def main():
    # Render Banner Hero Elegan
    st.markdown("""
        <div class="hero-banner">
            <div class="hero-title">👁️ YUK DETEKSI DINI MATA KAMU</div>
            <div class="hero-subtitle">Gunakan teknologi skrining cermin cerdas berbasis kecerdasan buatan (AI) untuk menganalisis simetris arah bola mata dan indikasi strabismus secara instan.</div>
        </div>
    """, unsafe_allow_html=True)
    
    detector, classifier = load_models()
    
    if detector is None or classifier is None:
        st.error("Gagal memuat sistem deteksi pintar. Pastikan file model 'model_mata.pt' dan 'model_juling.pt' berada di dalam direktori folder 'models'.")
        return
    
    # Radio Menu Selector Horizontal Elegan
    input_method = st.radio(
        "Pilih Metode Skrining:",
        ["📤 Unggah File Foto Wajah", "📸 Gunakan Cermin Live AI"],
        horizontal=True
    )
    
    st.markdown('<div style="margin-top: 15px;"></div>', unsafe_allow_html=True)
    
    if "Unggah File" in input_method:
        upload_page(detector, classifier)
    else:
        live_camera_page(detector, classifier)
    
    # Catatan Disclaimer Kaki
    st.markdown("---")
    st.markdown(
        '<div style="text-align: center; color: #888; font-size: 13px;">'
        '<strong>Pemberitahuan:</strong> Aplikasi skrining cermin AI ini merupakan instrumen mandiri awal. '
        'Hasil pengujian tidak dapat dijadikan basis mutlak pengganti diagnosis klinis kedokteran mata resmi.'
        '</div>', 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()