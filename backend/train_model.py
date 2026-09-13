import cv2
import os
import numpy as np

dataset_path = "dataset"

# Only these employees should be trained
employees = ["101", "103"]

faces = []
ids = []

print()
print("Training LBPH Model")
print("-------------------")

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

recognizer = cv2.face.LBPHFaceRecognizer_create()

recognizer.train(
    faces,
    np.array(ids)
)

os.makedirs("trainer", exist_ok=True)

recognizer.write(
    "trainer/trainer.yml"
)

print()
print("LBPH model trained successfully!")
print("Model includes only Employee IDs:", employees)
print("Model saved at: trainer/trainer.yml")