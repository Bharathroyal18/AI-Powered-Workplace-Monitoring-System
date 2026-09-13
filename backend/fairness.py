import cv2
import os

# Load trained LBPH model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer/trainer.yml")

employees = ["101", "103"]

accuracy_list = []

print()
print("Fairness / Per-User Evaluation")
print("------------------------------")

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

        total += 1

        predicted_id, confidence = recognizer.predict(image)

        if str(predicted_id) == employee_id:
            correct += 1

    if total > 0:
        accuracy = (correct / total) * 100
    else:
        accuracy = 0

    accuracy_list.append(accuracy)

    print()
    print("Employee ID:", employee_id)
    print("Total Test Images:", total)
    print("Correct Predictions:", correct)
    print("Accuracy:", round(accuracy, 2), "%")


# Calculate best and worst accuracy
if len(accuracy_list) > 0:

    best_accuracy = max(accuracy_list)
    worst_accuracy = min(accuracy_list)

    consistency_gap = best_accuracy - worst_accuracy

    print()
    print("Fairness Summary")
    print("----------------")
    print("Best User Accuracy:", round(best_accuracy, 2), "%")
    print("Worst User Accuracy:", round(worst_accuracy, 2), "%")
    print("Consistency Gap:", round(consistency_gap, 2), "percentage points")
