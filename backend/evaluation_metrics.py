import cv2
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer/trainer.yml")

employees = ["101", "103"]

# Store all actual and predicted IDs
actual_ids = []
predicted_ids = []

print()
print("Precision, Recall and F1-Score")
print("-----------------------------")

for employee_id in employees:

    folder = "test_dataset/" + employee_id

    for image_name in os.listdir(folder):

        image_path = os.path.join(folder, image_name)

        image = cv2.imread(
            image_path,
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            continue

        predicted_id, confidence = recognizer.predict(image)

        actual_ids.append(employee_id)
        predicted_ids.append(str(predicted_id))


# Calculate metrics for each employee
for employee_id in employees:

    true_positive = 0
    false_positive = 0
    false_negative = 0

    for actual, predicted in zip(
        actual_ids,
        predicted_ids
    ):

        if actual == employee_id and predicted == employee_id:
            true_positive += 1

        elif actual != employee_id and predicted == employee_id:
            false_positive += 1

        elif actual == employee_id and predicted != employee_id:
            false_negative += 1

    # Precision
    if true_positive + false_positive > 0:
        precision = (
            true_positive
            / (true_positive + false_positive)
        )
    else:
        precision = 0

    # Recall
    if true_positive + false_negative > 0:
        recall = (
            true_positive
            / (true_positive + false_negative)
        )
    else:
        recall = 0

    # F1 Score
    if precision + recall > 0:
        f1_score = (
            2 * precision * recall
            / (precision + recall)
        )
    else:
        f1_score = 0

    print()
    print("Employee ID:", employee_id)
    print(
        "Precision:",
        round(precision * 100, 2),
        "%"
    )
    print(
        "Recall:",
        round(recall * 100, 2),
        "%"
    )
    print(
        "F1-Score:",
        round(f1_score * 100, 2),
        "%"
    )