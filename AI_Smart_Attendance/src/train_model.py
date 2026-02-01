import cv2
import os
import numpy as np
from PIL import Image

# ===============================
# PATH
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "trained_model.yml")

# ===============================
# INIT
# ===============================
recognizer = cv2.face.LBPHFaceRecognizer_create()
faces = []
labels = []
label_map = {}
current_label = 0

print("[INFO] Membaca dataset...")

# ===============================
# LOAD DATASET
# ===============================
for person_name in os.listdir(DATASET_PATH):
    person_path = os.path.join(DATASET_PATH, person_name)

    if not os.path.isdir(person_path):
        continue

    label_map[current_label] = person_name

    for image_name in os.listdir(person_path):
        image_path = os.path.join(person_path, image_name)

        try:
            image = Image.open(image_path).convert("L")
            image = image.resize((200, 200))
            image_np = np.array(image, "uint8")

            faces.append(image_np)
            labels.append(current_label)
        except:
            print(f"[WARNING] Gagal membaca {image_path}")

    current_label += 1

print(f"[INFO] Total wajah   : {len(faces)}")
print(f"[INFO] Total orang  : {len(label_map)}")

# ===============================
# TRAIN
# ===============================
print("[INFO] Training model...")
recognizer.train(faces, np.array(labels))

# ===============================
# SAVE MODEL
# ===============================
os.makedirs(MODEL_DIR, exist_ok=True)
recognizer.save(MODEL_PATH)

print("[SUCCESS] Model disimpan di:", MODEL_PATH)
