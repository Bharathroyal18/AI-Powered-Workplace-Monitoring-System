import cv2
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer/blur_trainer.yml")

employees = ["101", "103"]

print()
print("Normal Test - Blur-Robust Model")
print("-------------------------------")

for employee_id in employees:

    folder = "test_dataset/" + employee_id

    total = 0
    correct = 0

    for image_name in os.listdir(folder):

        image_path = os.path.join(
            folder,
            image_name
        )

        image = cv2.imread(
            image_path,
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            continue

        total += 1

        predicted_id, confidence = recognizer.predict(image)

        if str(predicted_id) == employee_id:
            correct += 1

    accuracy = (correct / total) * 100

    print()
    print("Employee ID:", employee_id)
    print("Test Images:", total)
    print("Correct Predictions:", correct)
    print("Normal Accuracy:", round(accuracy, 2), "%")