import cv2
import os

employee_id = input("Enter Employee ID: ")

folder = "test_dataset/" + employee_id

if not os.path.exists(folder):
    print("Employee folder not found!")
    print("Use 101, 102, or 103.")
    exit()

face_detector = cv2.CascadeClassifier(
    "haarcascade/haarcascade_frontalface_default.xml"
)

camera = cv2.VideoCapture(0)

count = 0

print()
print("Starting test image capture...")
print("Look at the camera.")
print("Move your face slightly left and right.")
print("Press ESC to stop.")

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

        count += 1

        face = gray[y:y+h, x:x+w]

        file_name = folder + "/test_" + str(count) + ".jpg"

        cv2.imwrite(file_name, face)

        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (255, 0, 0),
            2
        )

    cv2.imshow("Test Image Capture", frame)

    key = cv2.waitKey(100) & 0xff

    if key == 27:
        break

    if count >= 20:
        break

camera.release()
cv2.destroyAllWindows()

print()
print("Test image capture completed!")
print("Employee ID:", employee_id)
print("Images captured:", count)