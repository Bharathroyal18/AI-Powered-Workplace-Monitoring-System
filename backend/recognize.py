import cv2
import json
import os
import time

# Load employee names
with open("employees.json", "r") as file:
    employees = json.load(file)

# Load face detector
face_detector = cv2.CascadeClassifier(
    "haarcascade/haarcascade_frontalface_default.xml"
)

# Load blur-robust LBPH model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer/blur_trainer.yml")

camera = cv2.VideoCapture(0)

print()
print("Starting face recognition...")
print("Press ESC to stop.")

last_name = "No face detected"

while True:

    ret, frame = camera.read()

    if not ret:
        print("Camera not found")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:

        face = gray[y:y+h, x:x+w]

        employee_id, confidence = recognizer.predict(face)

        if confidence < 70:

            name = employees.get(
                str(employee_id),
                "Unknown"
            )

            last_name = name

        else:

            name = "Unknown"
            last_name = name

        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (255, 0, 0),
            2
        )

        cv2.putText(
            frame,
            name,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2
        )

    cv2.putText(
        frame,
        "Recognized: " + last_name,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )

    cv2.imshow(
        "Face Recognition",
        frame
    )

    key = cv2.waitKey(1) & 0xff

    if key == 27:
        break

camera.release()
cv2.destroyAllWindows()

print()
print("Recognition stopped.")
print("Last recognized employee:", last_name)