import os
import argparse
import cv2

integers_to_classes = {
    0: '1',
    1: '2.1/3',
    2: '2.2',
    3: '2.3_6.1',
    4: '4.1',
    5: '4.2',
    6: '4.3',
    7: '5.1',
    8: '5.2',
    9: '6.2',
    10: '7',
    11: '7E',
    12: '8',
    13: '9',
    14: 'LQ',
    15: 'MP',
    16: 'BG',
}

def crop_images(images_folder, labels_folder, output_folder):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get the list of image files
    image_files = [f for f in os.listdir(images_folder) if f.endswith('.jpg')]

    # Process each image file
    for image_file in image_files:
        image_path = os.path.join(images_folder, image_file)

        # Find the corresponding label file
        label_file = image_file.replace('.jpg', '.txt')
        label_path = os.path.join(labels_folder, label_file)

        # Read the image
        image = cv2.imread(image_path)

        # Read the label file
        with open(label_path, 'r') as file:
            labels = file.readlines()

        # Process each bounding box label
        for i, label in enumerate(labels):
            class_id, x_center, y_center, width, height = label.split()[:5]

            

            # Calculate coordinates of the bounding box
            x_min = int(float(x_center) * image.shape[1] - float(width) * image.shape[1] / 2)
            y_min = int(float(y_center) * image.shape[0] - float(height) * image.shape[0] / 2)
            x_max = int(float(x_center) * image.shape[1] + float(width) * image.shape[1] / 2)
            y_max = int(float(y_center) * image.shape[0] + float(height) * image.shape[0] / 2)

            # Crop the bounding box region
            cropped_image = image[y_min:y_max, x_min:x_max]

            # Check if the cropped image is empty
            if cropped_image.size == 0:
                continue

            print(image_file)

            class_label = integers_to_classes[int(class_id)]

            # Create class folder if it doesn't exist
            class_folder = os.path.join(output_folder, class_label)
            os.makedirs(class_folder, exist_ok=True)

            # Save the cropped image
            output_path = os.path.join(class_folder, f'{image_file}_{i}_{class_id}.jpg')
            cv2.imwrite(output_path, cropped_image)

        print(f'Cropped images saved in {output_folder}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Crop images based on YOLOv5 labels')
    parser.add_argument('--images', '-i', type=str, help='Path to the folder containing the images')
    parser.add_argument('--labels', '-l', type=str, help='Path to the folder containing the label files')
    parser.add_argument('--output', '-o', type=str, help='Path to the output folder')

    args = parser.parse_args()

    images_folder = args.images
    labels_folder = args.labels
    output_folder = args.output

    crop_images(images_folder, labels_folder, output_folder)
