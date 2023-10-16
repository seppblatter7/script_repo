import argparse
from email.mime import image
import os
import cv2 as cv
import torch
from PIL import Image
import matplotlib.pyplot as plt
import tkinter as tk
from PIL import ImageTk
import time
import numpy as np
import torch.nn.functional as F
import random
import string

#Generates a random colour based on each different class name 
def generate_color(class_name):
    random.seed(class_name)
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return (r, g, b)

#Load yoloV5 model using "torch.hub"
def load_model(model_name):
    # Load the YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_name)
    model.eval()
    print(model.names)
    type(model)
    return model

def detect_objects(model, input_path, output_folder, save_txt, conf = 0.5):
    def process_image(frame, frame_pil, model, save_txt, frame_count, image_file = None):
        # Perform object detection
        results = model(frame_pil)

        predictions = []

        orig_frame = np.copy(frame)

        # Parse the results and draw bounding boxes
        for result in results.pred:
            for i, det in enumerate(result):
                class_index = int(det[5])

                obj_bbox_name = model.names[class_index]

                bbox = det[:4].tolist()
                
                if det[4].item() < conf:
                    continue
                
                confidence = "{:.2f}".format(det[4].item())
                
                # Extract bounding box coordinates
                x_center = (bbox[0] + bbox[2]) / 2 
                y_center = (bbox[1] + bbox[3]) / 2
                width = (bbox[2] - bbox[0])  
                height = (bbox[3] - bbox[1]) 


                x1, y1, x2, y2 = map(int, bbox)

                label_txt = f"{class_index} {x_center/frame.shape[1]} {y_center/frame.shape[0]} {width/frame.shape[1]} {height/frame.shape[0]}"

                label = obj_bbox_name + f" ({confidence})"

                predictions.append(label_txt)

                # Draw bounding box and confidence with the label
                text_size, _ = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                text_width, text_height = text_size[0], text_size[1]
                cv.rectangle(frame, (x1, y1 - text_height - 11), (x1 + text_width, y1 - 8), (255, 255, 255), cv.FILLED)
                cv.rectangle(frame, (x1, y1), (x2, y2), generate_color(obj_bbox_name), 2)
                cv.putText(frame, label, (x1, y1 - 10),
                           cv.FONT_HERSHEY_SIMPLEX, 0.4, generate_color(obj_bbox_name), 1)

        if save_txt:
            current_timestamp = int(time.time())
            # Create the labels folder
            labels_folder = os.path.join(output_folder, 'labels')
            os.makedirs(labels_folder, exist_ok=True)
            images_folder = os.path.join(output_folder, 'images')
            os.makedirs(images_folder, exist_ok=True)

            output_image_file = os.path.join(images_folder, f'{frame_count}_{current_timestamp}.jpg')

            if image_file == None:
                output_labels_file = os.path.join(labels_folder, f'{frame_count}_{current_timestamp}.txt')
            else:
                output_labels_file = os.path.join(labels_folder ,(image_file + ".txt"))

            print(output_labels_file)
            with open(output_labels_file, 'w') as file:
                for prediction in predictions:
                    file.write(f'{prediction}\n')
            cv.imwrite(output_image_file, frame)

        return frame
    
    #image_file = input_path
    
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Check if the input is a directory
    if os.path.isdir(input_path):
        print("Image")
        # Get a list of image files in the directory
        image_files = [os.path.join(input_path, file) for file in os.listdir(input_path) if file.endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

        # Sort the image files
        image_files.sort()

        # Get the total number of images
        total_images = len(image_files)

        # Iterate over the image files
        for i, image_file in enumerate(image_files):
            print(f'Processing image {i + 1}/{total_images}: {image_file}')

            # Read the image
            frame = cv.imread(image_file)

            # Convert the frame to PIL Image
            frame_pil = Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))

            # Process the frame
            frame = process_image(frame, frame_pil, model, save_txt, 0, os.path.splitext(os.path.basename(image_file))[0])

            output_path = os.path.join(output_folder, f'pred_{image_file}.jpg')
            cv.imwrite(output_path, frame)
    else:
            print(f'Processing image {i + 1}/{total_images}: {image_file}')

            # Read the image
            frame = cv.imread(image_file)

            # Convert the frame to PIL Image
            frame_pil = Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))

            # Process the frame
            frame = process_image(frame, frame_pil, model, save_txt, 0, os.path.splitext(os.path.basename(image_file))[0])

            output_path = os.path.join(output_folder, f'pred_{image_file}.jpg')
            cv.imwrite(output_path, frame)

def writeFile (total_images, image_file, frame, frame_pil, model, save_txt, output_folder):

    print(f'Processing image {i + 1}/{total_images}: {image_file}')

    # Read the image
    frame = cv.imread(image_file)

    # Convert the frame to PIL Image
    frame_pil = Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))

    # Process the frame
    frame = process_image(frame, frame_pil, model, save_txt, 0, os.path.splitext(os.path.basename(image_file))[0])

    output_path = os.path.join(output_folder, f'pred_{image_file}.jpg')
    cv.imwrite(output_path, frame)




# Main function
def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='YOLOv5 Object Detection')
    parser.add_argument('--input', '-i', type=str, help='Path to the input image-file/images-folder')
    parser.add_argument('--model', '-m', type=str, default='yolov5s.pt', help='Path to the model to use (yolov5s.pt, yolov5m.pt, yolov5l.pt)')
    parser.add_argument('--output_dir', '-od', type=str, default='output', help='Output folder path')
    parser.add_argument('--save_txt', '-st', action='store_true', help='Save txt for detected objects')

    args = parser.parse_args()

    input_path = args.input
    model_name = args.model
    output_folder = args.output_dir
    save_txt = args.save_txt

     # Load the YOLOv5 model
    model = load_model(model_name)

    # Apply object detection
    detect_objects(model, input_path, output_folder, save_txt)

if __name__ == '__main__':
    main()