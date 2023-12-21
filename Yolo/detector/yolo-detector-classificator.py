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


def generate_color(class_name):
    random.seed(class_name)
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return (r, g, b)

def display_image(frame, interval = 0):

    # Create a Tkinter window
    window = tk.Tk()

    # Create a PIL image from the frame
    image = Image.fromarray(frame)

    # Convert the PIL image to Tkinter-compatible format
    image_tk = ImageTk.PhotoImage(image)

    # Create a Tkinter label and display the image
    label = tk.Label(window)
    label.pack()
    label.imgtk = image_tk  # Store the image to keep it accessible

    # Update the image in the label
    label.config(image=image_tk)

    # Function to close the window
    def close_window():
        window.destroy()

    # Update the Tkinter window
    window.update()

    # Set the interval for automatic updates
    if interval > 0:
        window.after(interval, close_window)

    # Start the Tkinter event loop
    window.mainloop()

# +

def load_model(model_name, device, conf):
    # Load the YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_name)
    model.eval()
    model.to(device)
    model.conf = conf
    print("Detection model classes: ")
    print(model.names)
    return model


# +

def load_cls_model(classification_model_name, device):
    # Load the YOLOv5 or classification model from ultralytics/yolov5 repository
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=classification_model_name)  # local model
    model.eval()
    model.to(device)
    print("Classification model classes: ")
    print(model.names)
    return model


# -

def load_labels(labels_path):
    with open(labels_path, 'r') as file:
        labels = file.read().splitlines()
    return labels

def ResizeAndPad(img, size = (1280, 1280), padColor=0):
    w, h = img.size 
    #h, w = img.shape[:2]
    sh, sw = size
    
    # interpolation method
    if h > sh or w > sw: # shrinking image
        interp = cv.INTER_AREA
    else: # stretching image
        interp = cv.INTER_CUBIC
    # aspect ratio of image
    aspect = w/h  # if on Python 2, you might need to cast as a float: float(w)/h

    # compute scaling and pad sizing
    if aspect > 1: # horizontal image
        new_w = sw
        new_h = np.round(new_w/aspect).astype(int)
        pad_vert = (sh-new_h)/2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < 1: # vertical image
        new_h = sh
        new_w = np.round(new_h*aspect).astype(int)
        pad_horz = (sw-new_w)/2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else: # square image
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

    if len(img.size) == 3 and not isinstance(padColor, (list, tuple, np.ndarray)): # color image but only one color provided
        padColor = [padColor]*3

    # Scale the image
    scaled_img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Create a new image with padded size and fill it with the padding color
    img_padded = Image.new("RGB", (sw, sh), color=padColor)

    # Paste the scaled image onto the center of the padded image
    img_padded.paste(scaled_img, (pad_left, pad_top))

    return scaled_img

def detect_objects(model, classification_model, input_path, apply_classification, output_format, output_folder, save_txt, square_crop, save_crop, class_indices, device, conf = 0.5, letterbox = False):
    def process_image(frame, frame_pil, model, classification_model, apply_classification, save_txt, frame_count, square_crop, save_crop, image_file = None):
        # Perform object detection
        results = model(frame_pil)

        predictions = []

        orig_frame = frame

        # Parse the results and draw bounding boxes
        for result in results.pred:
            for i, det in enumerate(result):
                class_index = int(det[5])

                if class_indices is not None and class_index not in class_indices:
                    continue  # Skip this object if class not in class_indices

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

                # Convert bounding box to square if requested
                if square_crop:
                    size = max(width, height)
                    x_center = (bbox[0] + bbox[2]) / 2
                    y_center = (bbox[1] + bbox[3]) / 2
                    bbox = [
                        x_center - size / 2,
                        y_center - size / 2,
                        x_center + size / 2,
                        y_center + size / 2,
                    ]
                    # Ensure the square crop stays within image boundaries
                    bbox = [
                        max(0, bbox[0]),
                        max(0, bbox[1]),
                        min(frame.shape[1], bbox[2]),
                        min(frame.shape[0], bbox[3]),
                    ]

                # Extract bounding box coordinates
                x1, y1, x2, y2 = map(int, bbox)

                obj_cls_name = ""

                label_txt = f"{class_index} {x_center/frame.shape[1]} {y_center/frame.shape[0]} {width/frame.shape[1]} {height/frame.shape[0]}"

                label = obj_bbox_name + f" ({confidence}) - " + obj_cls_name 

                predictions.append(label_txt)

                # Draw bounding box and confidence with the label
                text_size, _ = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                text_width, text_height = text_size[0], text_size[1]
                cv.rectangle(frame, (x1, y1 - text_height - 11), (x1 + text_width, y1 - 8), (255, 255, 255), cv.FILLED)
                cv.rectangle(frame, (x1, y1), (x2, y2), generate_color(obj_bbox_name + obj_cls_name), 2)
                cv.putText(frame, label, (x1, y1 - 10),
                           cv.FONT_HERSHEY_SIMPLEX, 0.4, generate_color(obj_bbox_name + obj_cls_name), 1)

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
            cv.imwrite(output_image_file, orig_frame)

        return frame

    def process_video(cap, total_frames, output_format, output_folder, video_writer):
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
            frame = process_image(frame, frame_pil, model, classification_model, apply_classification, save_txt, frame_count ,square_crop, save_crop)

            # Display the frame or save it as an image or video
            if output_format == 'display':
                display_image(frame, interval=1)
            elif output_format == 'jpg':
                output_path = os.path.join(output_folder, f'output{frame_count}.jpg')
                #cv.imwrite(output_path, frame)
            elif output_format == 'mp4':
                #frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)  # Convert from RGB to BGR
                frame = cv.resize(frame, (1280, 720))
                video_writer.write(frame)

            frame_count += 1

    # Check if the input is a directory
    if os.path.isdir(input_path):
        print("Image")
        # Get a list of image files in the directory
        image_files = [os.path.join(input_path, file) for file in os.listdir(input_path) if file.endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

        # Sort the image files
        image_files.sort()

        # Get the total number of images
        total_images = len(image_files)

        # Create the output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        # Iterate over the image files
        for i, image_file in enumerate(image_files):
            print(f'Processing image {i + 1}/{total_images}: {image_file}')

            # Read the image
            frame = cv.imread(image_file)

            # Convert the frame to PIL Image
            frame_pil = Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))

            # Process the frame
            frame = process_image(frame, frame_pil, model, classification_model, apply_classification, save_txt, 0, square_crop, save_crop, os.path.splitext(os.path.basename(image_file))[0])

            # Save the frame as an image
            if output_format == 'jpg':
                output_path = os.path.join(output_folder, f'output{i}.jpg')
                #cv.imwrite(output_path, frame)

    else:
        print("Video")
        # Load the input video
        if input_path.startswith('rtsp://'):
            cap = cv.VideoCapture(input_path)
        else:
            cap = cv.VideoCapture(input_path)

        # Get total number of frames in the video
        total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

        # Create the output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        # Prepare the output video writer if the output format is mp4
        video_writer = None
        if output_format == 'mp4':
            output_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
            output_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
            if input_path.startswith('rtsp://'):
                output_fps = 14
            else:
                output_fps = cap.get(cv.CAP_PROP_FPS)
            fourcc = cv.VideoWriter_fourcc(*'mp4v')
            output_path = os.path.join(output_folder, "out_" + os.path.basename(input_path))
            video_writer = cv.VideoWriter(output_path, fourcc, output_fps, (1280, 720))
            video_writer.set(cv.VIDEOWRITER_PROP_QUALITY, 50)

        try:
            # Process the video frames
            process_video(cap, total_frames, output_format, output_folder, video_writer)

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
    parser.add_argument('--input', '-i', type=str, help='Path to the input file (image, video, or rtsp stream)')
    parser.add_argument('--model', '-m', type=str, default='yolov5s.pt', help='Path to the model to use (yolov5s.pt, yolov5m.pt, yolov5l.pt)')
    parser.add_argument('--output_format', '-of', type=str, default=None, help='Output format (display, jpg, mp4)')
    parser.add_argument('--output_dir', '-od', type=str, default='output', help='Output folder path')
    parser.add_argument('--device', '-d', type=str, default='cpu', help='Device to use (cpu, cuda)')
    parser.add_argument('--cls_model', '-cm', type=str, default='yolov5n-cls.pt', help='Path to the model to use (yolov5s-cls.pt, yolov5m-cls.pt, yolov5l-cls.pt)')
    parser.add_argument('--apply_cls', '-ac', action='store_true', default= False, help='Apply classification model to the detected objects')
    parser.add_argument('--save_txt', '-st', action='store_true', help='Save txt for detected objects')
    parser.add_argument('--square_crop', '-sqc', action='store_true', default=False, help='Convert bounding boxes to square boxes and apply classification')
    parser.add_argument('--save_crop', '-scrop', action='store_true', help='Save cropped images after classification')
    parser.add_argument('--classes', '-c', type=int, nargs='+', default=None, help='List of class indices to filter detected objects')
    parser.add_argument('--confidence', '-conf', type=float, default=0.5, help='Confidence')
    parser.add_argument('--letterbox', '-lb', action='store_true', default=False, help='Letterbox')

    args = parser.parse_args()

    input_path = args.input
    model_name = args.model
    output_format = args.output_format
    output_folder = args.output_dir
    device = args.device
    classification_model_name = args.cls_model
    apply_classification = args.apply_cls
    save_txt = args.save_txt
    square_crop = args.square_crop
    save_crop = args.save_crop
    print("save_crop ", save_crop)
    class_indices = args.classes  # Get the provided class indices
    letterbox = args.letterbox
    conf = float(args.confidence)
    
    # Check if CUDA is available and the device is set to GPU
    if device == 'cuda' and not torch.cuda.is_available():
        print('CUDA is not available. Switching to CPU.')
        device = 'cpu'

    # Set the device
    device = torch.device(device)

    # Load the YOLOv5 model
    model = load_model(model_name, device, conf)


    # Load the classification model if apply_classification is True
    classification_model = None
    if apply_classification:
        classification_model = load_cls_model(classification_model_name, device)

    # Apply object detection
    detect_objects(model, classification_model, input_path, apply_classification, output_format, output_folder, save_txt, square_crop, save_crop, class_indices, device, conf, letterbox)

if __name__ == '__main__':
    main()
