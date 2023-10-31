from PIL import Image
from PIL import ImageTk
import cv2
import numpy as np
import torch
import argparse
import os

def load_model(model_name):
    """
    Carica il modello YOLOv5 specificato.

    Args:
        model_name (str): Nome del file del modello da caricare.

    Returns:
        model: Modello YOLOv5 caricato.
    """
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_name)
    model.eval()
    return model


def detect_objects(model, rtsp_url, frame_ratio, output_folder):
    # Funzione detect_objects() con la condizione per il salvataggio delle etichette

    # Verifica se è stato specificato il percorso della cartella di output
    save_labels_flag = False
    if output_folder:
        save_labels_flag = True
        # Crea la cartella di output per i frame se non esiste già
        frames_output_folder = os.path.join(output_folder, 'frames')
        os.makedirs(frames_output_folder, exist_ok=True)
        # Crea la cartella di output per i file di testo delle etichette se non esiste già
        labels_output_folder = os.path.join(output_folder, 'labels')
        os.makedirs(labels_output_folder, exist_ok=True)

    cap = cv2.VideoCapture(rtsp_url)
    frame_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


        if frame_count % frame_ratio == 0:
            # Esegui il rilevamento degli oggetti sul frame
            pred = model(frame_pil)
            
            if save_labels_flag:
                frame_name = f'frame_{frame_count}.jpg'
                frame_path = os.path.join(frames_output_folder, frame_name)
                cv2.imwrite(frame_path, frame)

                # Salva le etichette delle bounding box in un file di testo
                save_labels(labels_output_folder, frame_name, pred)

        frame_count += 1

    cap.release()

def save_labels(output_folder, frame_name, pred):
    """
    Salva le etichette delle bounding box associate al frame in un file di testo.

    Args:
        output_folder (str): Cartella di output per salvare il file di testo.
        frame_name (str): Nome del frame.
        pred (Tensor): Tensor contenente le informazioni sulle bounding box.

    Returns:
        None
    """
    file_name = os.path.splitext(frame_name)[0] + '.txt'
    file_path = os.path.join(output_folder, file_name)

    with open(file_path, 'w') as f:
        for det in pred.pred:
            print (det)
            x1, y1, x2, y2 = det[:4]
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            label = f'{det[6]}, Confidence: 1.0'  # Modifica qui per aggiungere la classe e la confidenza corrette
            line = f'{label} {x1} {y1} {x2} {y2}\n'
            f.write(line)

def main():
    # Parsa gli argomenti da linea di comando
    parser = argparse.ArgumentParser(description='YOLOv5 Object Detection')
    parser.add_argument('--rtsp_url', '-ru', type=str, help='URL RTSP dello stream video')
    parser.add_argument('--model', '-m', type=str, default='yolov5s.pt', help='Percorso del modello da utilizzare (yolov5s.pt, yolov5m.pt, yolov5l.pt)')
    parser.add_argument('--frame_ratio', '-fr', type=int, default=1, help='Indica la frequenza di salvataggio dei frame')
    parser.add_argument('--output_folder', '-o', type=str, default='', help='Cartella di output per i file di testo delle etichette')

    args = parser.parse_args()

    rtsp_url = args.rtsp_url
    model_name = args.model
    frame_ratio = args.frame_ratio
    output_folder = args.output_folder

    # Carica il modello YOLOv5
    model = load_model(model_name)

    # Applica il rilevamento degli oggetti
    detect_objects(model, rtsp_url, frame_ratio, output_folder)

if __name__ == '__main__':
    main()