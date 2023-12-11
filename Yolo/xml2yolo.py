import os
import xml.etree.ElementTree as ET
import argparse
from pathlib import Path

def convert_coordinates(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x_center = (box[0] + box[2]) / 2.0
    y_center = (box[1] + box[3]) / 2.0
    width = box[2] - box[0]
    height = box[3] - box[1]

    x_center = x_center * dw
    width = width * dw
    y_center = y_center * dh
    height = height * dh

    return x_center, y_center, width, height

def xml_to_yolo(xml_path, output_folder):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    image_filename = root.find('filename').text
    image_name, _ = os.path.splitext(image_filename)

    size = (
        float(root.find('size/width').text),
        float(root.find('size/height').text)
    )

    yolo_lines = []
    for obj in root.findall('object'):
        # Fixed class_id of 0
        class_id = 0
        box = [
            float(obj.find('bndbox/xmin').text),
            float(obj.find('bndbox/ymin').text),
            float(obj.find('bndbox/xmax').text),
            float(obj.find('bndbox/ymax').text)
        ]
        yolo_coords = convert_coordinates(size, box)
        yolo_line = f"{class_id} {' '.join(map(str, yolo_coords))}"
        yolo_lines.append(yolo_line)


    output_path = os.path.join(output_folder, f"{image_name}.txt")
    with open(output_path, 'w') as f:
        f.write('\n'.join(yolo_lines))

#Main function
def Main():

    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="directory containing the xml annotations")
    parser.add_argument("output_dir", help="directory that will contain the output txt")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    xml_files = [f for f in os.listdir(input_dir) if f.endswith('.xml')]

    for xml_file in xml_files:
        xml_path = os.path.join(input_dir, xml_file)
        xml_to_yolo(xml_path, output_dir)

if __name__ == "__main__":
    Main()
