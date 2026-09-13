import cv2
import os

# Employee that needs improvement
employee_id = "103"

source_folder = "dataset/" + employee_id
output_folder = "dataset_augmented/" + employee_id

os.makedirs(output_folder, exist_ok=True)

print()
print("Fairness Improvement")
print("--------------------")
print("Employee ID:", employee_id)
print("Creating augmented training images...")

count = 0

for image_name in os.listdir(source_folder):

    image_path = os.path.join(
        source_folder,
        image_name
    )

    image = cv2.imread(
        image_path,
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        continue

    # Original image
    output_path = os.path.join(
        output_folder,
        "original_" + str(count) + ".jpg"
    )

    cv2.imwrite(output_path, image)
    count += 1

    # Slightly brighter image
    bright_image = cv2.convertScaleAbs(
        image,
        alpha=1.15,
        beta=15
    )

    output_path = os.path.join(
        output_folder,
        "bright_" + str(count) + ".jpg"
    )

    cv2.imwrite(output_path, bright_image)
    count += 1

    # Slightly darker image
    dark_image = cv2.convertScaleAbs(
        image,
        alpha=0.85,
        beta=-10
    )

    output_path = os.path.join(
        output_folder,
        "dark_" + str(count) + ".jpg"
    )

    cv2.imwrite(output_path, dark_image)
    count += 1

    # Small rotation
    height, width = image.shape

    center = (
        width // 2,
        height // 2
    )

    matrix = cv2.getRotationMatrix2D(
        center,
        3,
        1.0
    )

    rotated_image = cv2.warpAffine(
        image,
        matrix,
        (width, height)
    )

    output_path = os.path.join(
        output_folder,
        "rotated_" + str(count) + ".jpg"
    )

    cv2.imwrite(output_path, rotated_image)
    count += 1

print()
print("Fairness augmentation completed!")
print("Employee ID:", employee_id)
print("Augmented images created:", count)
print("Saved in:", output_folder)