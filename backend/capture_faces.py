import cv2
import os
import json
import sys

# Get employee ID and name
if len(sys.argv) >= 3:
    employee_id = sys.argv[1]
    employee_name = sys.argv[2]
else:
    employee_id = input("Enter Employee ID: ")
    employee_name = input("Enter Employee Name: ")

# Load employees
if os.path.exists("employees.json"):
    with open("employees.json", "r") as file:
        employees = json.load(file)
else:
    employees = {}

# Add employee
employees[employee_id] = employee_name

with open("employees.json", "w") as file:
    json.dump(employees, file, indent=4)

# Create employee folder
folder = "dataset/" + employee_id
os.makedirs(folder, exist_ok=True)

# Load Haar Cascade
face_detector = cv2.CascadeClassifier(
    "haarcascade/haarcascade_frontalface_default.xml"
)

camera = cv2.VideoCapture(0)

count = 0

print()
print("Starting face capture...")
print("Employee:", employee_name)
print("ID:", employee_id)
print("Look at the camera.")
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

        file_name = (
            folder + "/" +
            employee_name + "_" +
            str(count) + ".jpg"
        )

        cv2.imwrite(file_name, face)

        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (255, 0, 0),
            2
        )

        cv2.putText(
            frame,
            employee_name,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2
        )

    cv2.imshow("Face Capture", frame)

    key = cv2.waitKey(100) & 0xff

    if key == 27:
        break

    if count >= 30:
        break

camera.release()
cv2.destroyAllWindows()

print()
print("Face capture completed!")
print("Employee ID:", employee_id)
print("Employee Name:", employee_name)
print("Images captured:", count)