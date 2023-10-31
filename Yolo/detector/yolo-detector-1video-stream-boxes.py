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

def detect_objects(model, rtsp_url, frame_ratio):
    """
    Applica il rilevamento degli oggetti al flusso video in input utilizzando il modello specificato.

    Args:
        model: Modello YOLOv5 utilizzato per il rilevamento degli oggetti.
        rtsp_url (str): URL RTSP del flusso video.
        frame_ratio (int): Frequenza di salvataggio dei frame.

    Returns:
        None
    """
    # Crea il lettore RTSP
    cap = cv2.VideoCapture(rtsp_url)

    # Crea una finestra per la visualizzazione
    cv2.namedWindow('Stream con Bounding Boxes', cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Effettua la rilevazione degli oggetti
        results = model(frame)

        # Estrai le informazioni sulle bounding boxes
        pred = results.pred[0]

        # Disegna le bounding boxes sul frame
        for det in pred:
            x1, y1, x2, y2, conf, cls = det[:6]
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            label = f'Class {int(cls)} - Confidence: {conf:.2f}'
            color = (0, 255, 0)  # Colore verde
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Visualizza il frame con le bounding boxes
        cv2.imshow('Stream con Bounding Boxes', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Rilascia le risorse
    cap.release()
    cv2.destroyAllWindows()

def main():
    # Parsa gli argomenti da linea di comando
    parser = argparse.ArgumentParser(description='YOLOv5 Object Detection')
    parser.add_argument('--rtsp_url', '-ru', type=str, help='URL RTSP dello stream video')
    parser.add_argument('--model', '-m', type=str, default='yolov5s.pt', help='Percorso del modello da utilizzare (yolov5s.pt, yolov5m.pt, yolov5l.pt)')
    parser.add_argument('--frame_ratio', '-fr', type=int, default=1, help='Indica la frequenza di salvataggio dei frame')

    args = parser.parse_args()

    rtsp_url = args.rtsp_url
    model_name = args.model
    frame_ratio = args.frame_ratio

    # Carica il modello YOLOv5
    model = load_model(model_name)

    # Applica il rilevamento degli oggetti
    detect_objects(model, rtsp_url, frame_ratio)

if __name__ == '__main__':
    main()