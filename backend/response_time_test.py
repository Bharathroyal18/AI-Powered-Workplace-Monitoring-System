import cv2
import time

# Load LBPH model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer/trainer.yml")

# Load Haar Cascade
face_detector = cv2.CascadeClassifier(
    "haarcascade/haarcascade_frontalface_default.xml"
)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera not found")
    exit()

print()
print("Response Time Test")
print("------------------")
print("Look at the camera.")
print("Press ESC to stop.")

response_times = []

while True:

    ret, frame = camera.read()

    if not ret:
        print("Camera not found")
        break

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.equalizeHist(gray)

    # Start timing
    start_time = time.perf_counter()

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:

        recognizer.predict(
            gray[y:y+h, x:x+w]
        )

    # End timing
    end_time = time.perf_counter()

    if len(faces) > 0:

        response_time = (
            end_time - start_time
        ) * 1000

        response_times.append(
            response_time
        )

        cv2.putText(
            frame,
            "Response: "
            + str(round(response_time, 2))
            + " ms",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )

    cv2.imshow(
        "Response Time Test",
        frame
    )

    key = cv2.waitKey(1) & 0xff

    if key == 27:
        break


camera.release()
cv2.destroyAllWindows()

print()

if len(response_times) > 0:

    average_time = (
        sum(response_times)
        / len(response_times)
    )

    minimum_time = min(response_times)
    maximum_time = max(response_times)

    print("Response Time Results")
    print("---------------------")
    print(
        "Samples:",
        len(response_times)
    )
    print(
        "Average Response Time:",
        round(average_time, 2),
        "ms"
    )
    print(
        "Minimum Response Time:",
        round(minimum_time, 2),
        "ms"
    )
    print(
        "Maximum Response Time:",
        round(maximum_time, 2),
        "ms"
    )

else:

    print("No face detected.")