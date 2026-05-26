import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox
import os
import sys

os.environ['YOLO_VERBOSE'] = 'False'


BASE_DIR = Path(__file__).resolve().parent
MODEL_DETEKSI_PATH = BASE_DIR / "models" / "model_mata.pt"
MODEL_KLASIFIKASI_PATH = BASE_DIR / "models" / "model_juling.pt"


if not MODEL_DETEKSI_PATH.exists() or not MODEL_KLASIFIKASI_PATH.exists():
    print("\n❌ ERROR: Salah satu model tidak ditemukan!")
    sys.exit(1)


try:
    detector = YOLO(str(MODEL_DETEKSI_PATH))
    classifier = YOLO(str(MODEL_KLASIFIKASI_PATH))
    print("\n✅ Model Deteksi & Klasifikasi Berhasil Dimuat.")
except Exception as e:
    print(f"❌ Gagal Memuat Model: {e}")
    sys.exit(1)

def process_screening():
    root = tk.Tk()
    root.withdraw()
    img_path = filedialog.askopenfilename(
        title="Pilih Foto Mata",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
    )
    root.destroy()

    if not img_path:
        return

    img = cv2.imread(img_path)
    if img is None:
        messagebox.showerror("Error", f"Gagal membaca gambar:\n{img_path}")
        return

    img_display = img.copy()

    results_det = detector.predict(img, conf=0.15, verbose=False, device='cpu')
    regions = []

    for r in results_det:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            regions.append([x1, y1, x2, y2])

    if not regions:
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in eyes:
            regions.append([x, y, x + w, y + h])

    if not regions:
        regions.append([0, 0, img.shape[1], img.shape[0]])

    status_pasien = "NORMAL"
    list_mata_terdeteksi = []

    min_x = min([r[0] for r in regions])
    min_y = min([r[1] for r in regions])
    max_x = max([r[2] for r in regions])
    max_y = max([r[3] for r in regions])
    
    pad_x, pad_y = 20, 20
    strip_x1 = max(0, min_x - pad_x)
    strip_y1 = max(0, min_y - pad_y)
    strip_x2 = min(img.shape[1], max_x + pad_x)
    strip_y2 = min(img.shape[0], max_y + pad_y)
    
    strip_img = img[strip_y1:strip_y2, strip_x1:strip_x2]
    
    results_cls = classifier.predict(strip_img, conf=0.3, verbose=False, device='cpu')
    
    boxes = results_cls[0].boxes
    if len(boxes) > 0:
        for box in boxes:
            bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy().astype(int)
            class_id = int(box.cls[0])
            label_name = classifier.names[class_id].upper()
            conf_score = float(box.conf[0])
            
            abs_x1 = bx1 + strip_x1
            abs_y1 = by1 + strip_y1
            abs_x2 = bx2 + strip_x1
            abs_y2 = by2 + strip_y1
            
            list_mata_terdeteksi.append({
                'box': [abs_x1, abs_y1, abs_x2, abs_y2],
                'label': label_name,
                'conf': conf_score
            })
            
            if "JULING" in label_name or "STRABISMUS" in label_name:
                status_pasien = "STRABISMUS (JULING)"
    else:
        for (x1, y1, x2, y2) in regions:
            list_mata_terdeteksi.append({
                'box': [x1, y1, x2, y2],
                'label': "NORMAL",
                'conf': 0.0 # Tidak ada skor dari klasifikasi
            })

    for d in list_mata_terdeteksi:
        x1, y1, x2, y2 = d['box']
        label = d['label']
        conf = d['conf']
        
        warna = (0, 0, 255) if "JULING" in label or "STRABISMUS" in label else (0, 255, 0)

        cv2.rectangle(img_display, (x1, y1), (x2, y2), warna, 2)

        text_label = f"{label} {conf:.1%}"
        
        (tw, th), _ = cv2.getTextSize(text_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img_display, (x1, y1 - 25), (x1 + tw, y1), warna, -1) # Box penuh warna
        cv2.putText(img_display, text_label, (x1, y1 - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow("Hasil Deteksi & Confidence Score", img_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    while True:
        process_screening()
        lagi = input("\nCoba lagi? (y/n): ").strip().lower()
        if lagi != 'y':
            break