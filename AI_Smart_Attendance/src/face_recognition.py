import cv2
import os

print("FACE RECOGNITION STARTED")

# ==============================
# PATH AMAN (ANTI ERROR WINDOWS)
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "..", "model", "trained_model.yml")
cascade_path = os.path.join(BASE_DIR, "..", "haarcascade", "haarcascade_frontalface_default.xml")
dataset_path = os.path.join(BASE_DIR, "..", "dataset", "train")

print("Model exists   :", os.path.exists(model_path))
print("Cascade exists :", os.path.exists(cascade_path))
print("Dataset exists :", os.path.exists(dataset_path))

# ==============================
# LOAD MODEL & CASCADE
# ==============================
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(model_path)

face_cascade = cv2.CascadeClassifier(cascade_path)

# ==============================
# LABEL MAP (PASTI KONSISTEN)
# ==============================
label_map = {}
for idx, folder in enumerate(sorted(os.listdir(dataset_path))):
    label_map[idx] = folder

print("[INFO] Label Map:", label_map)

# ==============================
# BUKA KAMERA
# ==============================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ KAMERA TIDAK TERDETEKSI")
    exit()

print("✅ KAMERA AKTIF")

# ==============================
# LOOP KAMERA
# ==============================
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Gagal membaca frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(100, 100)
    )

    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (200, 200))

        label, confidence = recognizer.predict(face_img)

        name = label_map.get(label, "Unknown")

        # Semakin kecil confidence = semakin mirip
        if confidence < 70:
            text = f"{name} ({round(confidence, 2)})"
            color = (0, 255, 0)
        else:
            text = "Unknown"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(
            frame,
            text,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

    cv2.imshow("AI Smart Attendance - Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Keluar...")
        break

# ==============================
# CLEAN UP
# ==============================
cap.release()
cv2.destroyAllWindows()
