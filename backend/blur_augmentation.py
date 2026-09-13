import cv2
import os
import json

dataset_path = "dataset"
output_path = "dataset_blur"

# Read all employees automatically
with open("employees.json", "r") as file:
    employees = json.load(file)

print()
print("Creating Blur-Augmented Dataset")
print("--------------------------------")

for employee_id in employees:

    employee_folder = os.path.join(
        dataset_path,
        employee_id
    )

    output_folder = os.path.join(
        output_path,
        employee_id
    )

    if not os.path.isdir(employee_folder):
        continue

    os.makedirs(output_folder, exist_ok=True)

    count = 0

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

        # Save original image
        original_name = "original_" + str(count) + ".jpg"

        cv2.imwrite(
            os.path.join(output_folder, original_name),
            image
        )

        # Create blurred image
        blurred = cv2.GaussianBlur(
            image,
            (7, 7),
            0
        )

        blur_name = "blur_" + str(count) + ".jpg"

        cv2.imwrite(
            os.path.join(output_folder, blur_name),
            blurred
        )

        count += 1

    print(
        "Employee",
        employee_id,
        "- images created:",
        count * 2
    )

print()
print("Blur augmentation completed!")