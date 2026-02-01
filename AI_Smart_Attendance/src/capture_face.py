import cv2
import os

name = "Raihan"
save_path = f"dataset/{name}"
os.makedirs(save_path, exist_ok=True)

cam = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

count = 0

while True:
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        count += 1
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200))
        cv2.imwrite(f"{save_path}/{count}.jpg", face)

        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    cv2.imshow("Capture Face", frame)

    if cv2.waitKey(1) & 0xFF == 27 or count >= 100:
        break

cam.release()
cv2.destroyAllWindows()

