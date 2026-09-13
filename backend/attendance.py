import cv2
import json
import time
import sys


# Load employees
with open("employees.json", "r") as file:
    employees = json.load(file)


# Load face detector
face_detector = cv2.CascadeClassifier(
    "haarcascade/haarcascade_frontalface_default.xml"
)


# Load trained model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer/blur_trainer.yml")


# Get attendance settings from dashboard
if len(sys.argv) >= 3:

    session_minutes = float(sys.argv[1])

    required_percentage = float(sys.argv[2])

else:

    session_minutes = float(
        input("Enter session duration (minutes): ")
    )

    required_percentage = float(
        input("Enter required attendance percentage: ")
    )


session_seconds = session_minutes * 60

required_seconds = (
    session_seconds * required_percentage / 100
)


# Store presence time
presence_time = {}

for employee_id in employees:

    presence_time[employee_id] = 0


# Open camera
camera = cv2.VideoCapture(0)


print()
print("Attendance System Started")
print("-------------------------")
print("Session duration:", session_minutes, "minutes")
print("Required attendance:", required_percentage, "%")
print("Required presence:", required_seconds, "seconds")
print()
print("Press ESC to stop early.")


start_time = time.time()
previous_time = start_time


while True:

    current_time = time.time()

    elapsed_time = current_time - start_time

    if elapsed_time >= session_seconds:
        break


    ret, frame = camera.read()

    if not ret:

        print("Camera not found")
        break


    # Actual time between frames
    frame_time = current_time - previous_time

    previous_time = current_time


    # Convert to grayscale
    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    # Improve contrast
    gray = cv2.equalizeHist(gray)


    # Detect faces
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )


    recognized_ids = set()


    # Recognize each detected face
    for (x, y, w, h) in faces:

        face = gray[y:y+h, x:x+w]


        employee_id, confidence = recognizer.predict(face)


        if confidence < 70:

            employee_id = str(employee_id)


            if employee_id in employees:

                recognized_ids.add(employee_id)

                name = employees[employee_id]

            else:

                name = "Unknown"

        else:

            name = "Unknown"


        # Draw face rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (255, 0, 0),
            2
        )


        # Display employee name
        cv2.putText(
            frame,
            name,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2
        )


    # Add actual presence time
    for employee_id in recognized_ids:

        presence_time[employee_id] += frame_time


    # Remaining session time
    remaining = session_seconds - elapsed_time


    cv2.putText(
        frame,
        "Remaining: " + str(round(remaining, 1)) + " sec",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )


    cv2.imshow(
        "Attendance",
        frame
    )


    key = cv2.waitKey(1) & 0xff


    if key == 27:
        break


camera.release()

cv2.destroyAllWindows()


# Show attendance results
print()
print("Attendance Results")
print("------------------")


for employee_id in employees:

    name = employees[employee_id]

    seconds = presence_time[employee_id]


    attendance = (
        seconds / session_seconds
    ) * 100


    if attendance >= required_percentage:

        status = "PRESENT"

    else:

        status = "ABSENT"


    print()

    print("Employee:", name)

    print("ID:", employee_id)

    print(
        "Presence:",
        round(seconds / 60, 2),
        "minutes"
    )

    print(
        "Presence:",
        round(seconds, 2),
        "seconds"
    )

    print(
        "Attendance:",
        round(attendance, 2),
        "%"
    )

    print("Status:", status)