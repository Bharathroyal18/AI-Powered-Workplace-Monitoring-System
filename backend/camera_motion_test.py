import cv2
import os

# Load trained LBPH model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer/trainer.yml")

employees = ["101", "103"]

print()
print("Camera Motion / Background Test")
print("--------------------------------")

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

        # Simulate slight camera movement
        shifted_image = cv2.warpAffine(
            image,
            cv2.getRotationMatrix2D(
                (image.shape[1] // 2, image.shape[0] // 2),
                2,
                1
            ),
            (image.shape[1], image.shape[0])
        )

        total += 1

        predicted_id, confidence = recognizer.predict(
            shifted_image
        )

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
    print(
        "Camera Motion Accuracy:",
        round(accuracy, 2),
        "%"
    )
