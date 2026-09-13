import cv2
import os
import numpy as np
import json

dataset_path = "dataset_blur"

# Read all registered employees automatically
with open("employees.json", "r") as file:
    employees = json.load(file)

faces = []
ids = []

print()
print("Training Blur-Robust Model")
print("--------------------------")

for employee_id in employees:

    employee_folder = os.path.join(
        dataset_path,
        employee_id
    )

    if not os.path.isdir(employee_folder):
        continue

    for image_name in os.listdir(employee_folder):

        image_path = os.path.join(
            employee_folder,
            image_name
        )

        image = cv2.imread(
            image_path,
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            continue

        faces.append(image)
        ids.append(int(employee_id))


print("Total training images:", len(faces))
print("Employee IDs:", set(ids))

if len(faces) == 0:
    print("No training images found.")
    exit()

recognizer = cv2.face.LBPHFaceRecognizer_create()

recognizer.train(
    faces,
    np.array(ids)
)

os.makedirs("trainer", exist_ok=True)

recognizer.write(
    "trainer/blur_trainer.yml"
)

print()
print("Blur-robust model trained successfully!")
print("Model includes Employee IDs:", sorted(set(ids)))
print("Model saved at: trainer/blur_trainer.yml")