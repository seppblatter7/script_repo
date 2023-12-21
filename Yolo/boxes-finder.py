import os
import argparse

def cerca(path, range_val):
    min_height_list = []

    # Cicla attraverso i file nella cartella specificata
    for filename in os.listdir(path):
        if filename.endswith('.txt'):
            file_path = os.path.join(path, filename)

            # Apri il file e cicla attraverso ogni linea
            with open(file_path, 'r') as file:
                for line in file:
                    # Estrai l'ultimo elemento di ogni linea come altezza
                    height = float(line.strip().split()[-1])

                    # Aggiungi l'altezza alla lista
                    min_height_list.append(height)

    # Ordina la lista delle altezze e prendi i primi "range_val" valori
    min_height_list.sort()
    min_height_values = min_height_list[:range_val]

    # Stampa i valori minimi
    print(f"I {range_val} valori minimi di altezza sono:")
    for value in min_height_values:
        print(value)

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='YOLOv5 Object Detection')
    parser.add_argument('--input', '-i', type=str, help='Path to the input folder')
    parser.add_argument('--range', '-r', type=int, help='Number of values visualized')

    args = parser.parse_args()

    input_path = args.input
    range_val = args.range

    cerca(input_path, range_val)

if __name__ == '__main__':
    main()
