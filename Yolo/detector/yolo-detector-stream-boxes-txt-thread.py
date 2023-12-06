import sys
import time
from tkinter import Image
import cv2
import numpy as np
import torch
import argparse
import os
from PIL import Image
import random
from threaded_videocapture import ThreadedVideoCapture

def poll_rateF(input):
    if (input.startswith("rtsp://")):
        print("rtsp")
        cap = cv2.VideoCapture(input)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_queue_size = 10
        pr = fps
    elif(input.endswith(".mp4") and (not (input.startswith("rtsp://")))):
        print("mp4")
        cap = cv2.VideoCapture(input, cv2.CAP_FFMPEG)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_queue_size = 3000
        pr = fps + 10
    else:
        print("Formato di input non valido")
        sys.exit(1)
    cap.release()
    return fps, frame_queue_size, pr

#Generates a random colour based on each different class name 
def generate_color(class_name):
    random.seed(class_name)
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return (r, g, b)

def load_model(model_name):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_name)
    model.eval()
    return model

def detect_objects(model, input, fqs, pr, save_txt, output_path, conf = 0.5):
    print("aaaaaapr" + str(pr))
    # Crea il lettore RTSP
    with ThreadedVideoCapture(input, frame_queue_size=fqs, timeout=10) as tvc:
        frame_count = 0
        # Crea una finestra per la visualizzazione
        cv2.namedWindow('Stream con Bounding Boxes', cv2.WINDOW_NORMAL)

        while True:
            ret, frame = tvc.read() 
            if not ret:
                print("ThreadedVideoCapture has timed out.")
                break
            
            frame_count+=1
            print(frame_count, tvc.fps, tvc.actual_poll_rate, fqs)

            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            results = model(frame_pil)

            predictions = []

            orig_frame = np.copy(frame)

            # Disegna le bounding boxes sul frame
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

                    obj_cls_name = ""

                    label_txt = f"{class_index} {x_center/frame.shape[1]} {y_center/frame.shape[0]} {width/frame.shape[1]} {height/frame.shape[0]}"

                    label = obj_bbox_name + f" ({confidence}) - " + obj_cls_name

                    predictions.append(label_txt)

                    # Draw bounding box and confidence with the label
                    text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                    text_width, text_height = text_size[0], text_size[1]
                    cv2.rectangle(frame, (x1, y1 - text_height - 11), (x1 + text_width, y1 - 8), (255, 255, 255), cv2.FILLED)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), generate_color(obj_bbox_name + obj_cls_name), 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, generate_color(obj_bbox_name + obj_cls_name), 1)

           
            if save_txt:
                #retrives the current timestamp in order to generate unique fileames
                current_timestamp = int(time.time())

                # Create the labels folder
                labels_folder = os.path.join(output_path, 'labels')
                os.makedirs(labels_folder, exist_ok=True)
                images_folder = os.path.join(output_path, 'images')
                os.makedirs(images_folder, exist_ok=True)

                output_image_file = os.path.join(images_folder, f'{frame_count}_{current_timestamp}.jpg')
                output_labels_file = os.path.join(labels_folder, f'{frame_count}_{current_timestamp}.txt')

                print(output_labels_file)
                with open(output_labels_file, 'w') as file:
                    for prediction in predictions:
                        file.write(f'{prediction}\n')
                cv2.imwrite(output_image_file, orig_frame)


            # Visualizza il frame con le bounding boxes
            cv2.imshow('Stream con Bounding Boxes', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("ciao")
                break

        # Rilascia le risorse
        tvc.release()
        cv2.destroyAllWindows() 

def main():
    # Parsa gli argomenti da linea di comando
    parser = argparse.ArgumentParser(description='YOLOv5 Object Detection')
    parser.add_argument('--input', '-i', type=str, help='Streamed video URL RTSP')
    parser.add_argument('--model', '-m', type=str, default='yolov5s.pt', help='Path to the model you want use (yolov5s.pt, yolov5m.pt, yolov5l.pt)')
    parser.add_argument('--save_txt', '-txt', type=bool, default=0, help='1: It saves frames and associated labels, 0/(Not inserted): visualize just the video with the bounding boxes')
    parser.add_argument('--output', '-out', type=str, help='output path for labels and frames directories')

    args = parser.parse_args()

    input = args.input
    model_name = args.model
    save_txt = args.save_txt
    output_path= args.output

    # Carica il modello YOLOv5
    model = load_model(model_name)

    fps, fqsz, pr = poll_rateF(input)

    # Applica il rilevamento degli oggetti
    detect_objects(model, input, fqsz, pr, save_txt, output_path)

if __name__ == '__main__':
    main()