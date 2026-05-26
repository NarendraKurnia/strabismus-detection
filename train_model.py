import os
import tensorflow as tf
from ultralytics import YOLO
from tensorflow.keras import layers, models
from pathlib import Path

# --- CONFIG PATH ---
BASE_DIR = Path(__file__).parent
DATASET_YOLO = str(BASE_DIR / "dataset/MATA/data.yaml")
DATASET_CNN = str(BASE_DIR / "dataset/STRABISMUS")
SAVE_DIR = str(BASE_DIR / "models")

# Membuat folder models jika belum ada
os.makedirs(SAVE_DIR, exist_ok=True)

def train_yolo():
    print("\n--- Memulai Training YOLO (Mode: CPU) ---")
    # Pastikan file config ada
    if not os.path.exists(DATASET_YOLO):
        print(f"❌ Error: File {DATASET_YOLO} tidak ditemukan!")
        return

    model = YOLO("yolov8n.pt")
    
    # Paksa menggunakan device='cpu'
    # project='runs' akan membuat folder di dalam folder project kamu saat ini
    model.train(
        data=DATASET_YOLO, 
        epochs=7,        # Dikurangi ke 10 agar tidak terlalu lama di CPU
        imgsz=640, 
        device='cpu', 
        project=str(BASE_DIR / 'runs'), 
        name='train_mata'
    )
    
    # Cari hasil terbaik
    best_path = BASE_DIR / "runs/train_mata/weights/best.pt"
    if best_path.exists():
        import shutil
        shutil.copy(str(best_path), os.path.join(SAVE_DIR, "best.pt"))
        print(f"✅ YOLO Selesai. Model disimpan di: {SAVE_DIR}/best.pt")

def train_cnn():
    print("\n--- Memulai Training CNN (Mode: CPU) ---")
    img_height, img_width = 112, 224
    
    if not os.path.exists(DATASET_CNN):
        print(f"❌ Error: Folder dataset CNN {DATASET_CNN} tidak ditemukan!")
        return

    # Load Dataset
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_CNN, 
        image_size=(img_height, img_width), 
        validation_split=0.2, 
        subset="training", 
        seed=123
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_CNN, 
        image_size=(img_height, img_width), 
        validation_split=0.2, 
        subset="validation", 
        seed=123
    )

    # Memaksa TensorFlow menggunakan CPU
    with tf.device('/CPU:0'):
        base = tf.keras.applications.MobileNetV2(
            input_shape=(img_height, img_width, 3), 
            include_top=False, 
            weights='imagenet'
        )
        base.trainable = False

        model = models.Sequential([
            layers.Input(shape=(img_height, img_width, 3)),
            layers.Rescaling(1./127.5, offset=-1),
            base,
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid') 
        ])

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # Epochs dikurangi agar lebih cepat di CPU
        model.fit(train_ds, validation_data=val_ds, epochs=10)
        
        # Simpan model
        model.save(os.path.join(SAVE_DIR, "cnn_biner_strabismus.keras"))
        print(f"✅ CNN Selesai. Model disimpan di: {SAVE_DIR}/cnn_biner_strabismus.keras")

if __name__ == "__main__":
    # Matikan peringatan optimasi CPU (oneDNN) agar log bersih
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    train_yolo()
    train_cnn()