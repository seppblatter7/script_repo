import argparse
import os
import cv2 as cv
import torch
from torchvision import transforms
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

def detect_objects(model, input_path, output_folder, save_txt, frame_ratio, conf = 0.5):

    def process_image(frame, frame_pil, model, save_txt, frame_count, frame_ratio, image_file = None):
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
            if frame_count % frame_ratio == 0:
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
                cv.imwrite(output_image_file, orig_frame)
                

        return frame


    def process_video(cap, total_frames, video_writer):
        frame_count = 0
        prev_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Calculate the processing time for the previous frame
            curr_time = time.time()
            processing_time = curr_time - prev_time
            prev_time = curr_time

            print(f"Processing ({frame_count}/{total_frames}) fps:", int(1 / processing_time))

            # Convert the frame to PIL Image
            frame_pil = Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))

            # Process the frame
            frame = process_image(frame, frame_pil, model, save_txt, frame_count, frame_ratio)
            
            #frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)  # Convert from RGB to BGR
            frame = cv.resize(frame, (1280, 720))
            video_writer.write(frame)

            frame_count += 1


    cap = cv.VideoCapture(input_path)

    # Get total number of frames in the video
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    # Create the output folder if it doesn't 
    os.makedirs(output_folder, exist_ok=True)   

    # Prepare the output video writer if the output format is mp4
    video_writer = None
    output_fps = cap.get(cv.CAP_PROP_FPS)
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    outputVideopath = os.path.join(output_folder, 'editedVideo')
    os.makedirs(outputVideopath, exist_ok=True)
    output_path = os.path.join(outputVideopath, "out_" + os.path.basename(input_path))
    video_writer = cv.VideoWriter(output_path, fourcc, output_fps, (1280, 720))
    video_writer.set(cv.VIDEOWRITER_PROP_QUALITY, 50)

    try:
        # Process the video frames
        process_video(cap, total_frames, video_writer)

    except KeyboardInterrupt:
        # Handle keyboard interrupt (Ctrl+C)
        print('Keyboard interrupt detected. Saving video output...')

    finally:
        # Release the capture, close windows, and release the video writer
        cap.release()
        if video_writer is not None:
            video_writer.release()
            print("Video saved!")
        #cv.destroyAllWindows()


  


# Main function
def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='YOLOv5 Object Detection')
    parser.add_argument('--input', '-i', type=str, help='Path to the input .mp4 file')
    parser.add_argument('--model', '-m', type=str, default='yolov5s.pt', help='Path to the model to use (yolov5s.pt, yolov5m.pt, yolov5l.pt)')
    parser.add_argument('--output_dir', '-od', type=str, default='output', help='Output folder path')
    parser.add_argument('--save_txt', '-st', action='store_true', help='Save txt and realtive images for detected objects')
    parser.add_argument('--frame_ratio', '-fr', type=int, default=1, help='Indicates that it saves a frame every "parameter_entered" (-frame_ratio)')


    args = parser.parse_args()

    input_path = args.input
    model_name = args.model
    output_folder = args.output_dir
    save_txt = args.save_txt
    frame_ratio = args.frame_ratio

     # Load the YOLOv5 model
    model = load_model(model_name)

    # Apply object detection
    detect_objects(model, input_path, output_folder, save_txt, frame_ratio)

if __name__ == '__main__':
    main()