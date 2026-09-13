import cv2
import os

# Load trained LBPH model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer/trainer.yml")

employees = ["101", "103"]

print()
print("Occlusion Robustness Test")
print("-------------------------")

for employee_id in employees:

    folder = "test_dataset/" + employee_id

    total = 0
    correct = 0

    for image_name in os.listdir(folder):

        image_path = os.path.join(folder, image_name)

        image = cv2.imread(
            image_path,
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            continue

        # Cover the lower part of the face
        height, width = image.shape

        cv2.rectangle(
            image,
            (0, int(height * 0.60)),
            (width, height),
            0,
            -1
        )

        total += 1

        predicted_id, confidence = recognizer.predict(image)

        if str(predicted_id) == employee_id:
            correct += 1

    if total > 0:
        accuracy = (correct / total) * 100
    else:
        accuracy = 0

    print()
    print("Employee ID:", employee_id)
    print("Test Images:", total)
    print("Correct Predictions:", correct)
    print("Occlusion Accuracy:", round(accuracy, 2), "%")
