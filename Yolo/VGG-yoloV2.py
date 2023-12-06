import json
import os
import argparse
import cv2

classes_to_integers = {
    '1': 1,
    '0': 0
#     'HT': 16,
#     'TWU': 17,
}

def vgg_to_yolo(json_path, images_folder, output_folder):
    with open(json_path) as f:
        data = json.load(f)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for key, value in data.items():
        image_name = value['filename']
        regions = value['regions']
        img = cv2.imread(os.path.join(images_folder, image_name))
        if img is None:
            print("skipped ", image_name)
            continue
        im_height, im_width, channels = img.shape

        with open(os.path.join(output_folder, image_name.replace('.jpg', '.txt')), 'w') as f:
            for region in regions:
                print(os.path.join(output_folder, image_name))
                shape_attributes = region['shape_attributes']
                class_label = "0" #region['region_attributes'].get('class', 0)

                class_label_int = "0" #classes_to_integers.get(class_label, )  # Default to 0 if class not found

                if class_label_int == -1:
                    print(f"Warning: Class label '{class_label}' not found in conversion dictionary.")
                    print(image_name)

                x = shape_attributes['x']
                y = shape_attributes['y']
                width = shape_attributes['width']
                height = shape_attributes['height']

                x_center = (x + (width / 2)) / im_width
                y_center = (y + (height / 2)) / im_height
                width = width / im_width
                height = height / im_height

                f.write(f"{class_label_int} {x_center} {y_center} {width} {height}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('json_path', help='Path to VGG JSON file')
    parser.add_argument('images_folder', type=str, help='Path to input images folder')
    parser.add_argument('output_folder', type=str, help='Path to output folder for YOLO txt annotations')
    args = parser.parse_args()

    vgg_to_yolo(args.json_path, args.images_folder, args.output_folder)


