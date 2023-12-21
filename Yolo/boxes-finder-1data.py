import os
import argparse
import sys 

def cerca(path, range_val, threshold):
    min_height_info_list = []
    el_height_info_list = []
    eliminated_files = []  # Lista per tenere traccia dei file eliminati

    # Cicla attraverso le cartelle "train" e "valid" in "dataset"
    for subset in ["train", "valid"]:
        subset_path = os.path.join(path, subset)
        labels_path = os.path.join(subset_path, "labels")

        # Controlla se la cartella "labels" esiste in "train" o "valid"
        if os.path.exists(labels_path):
            # Cicla attraverso i file nella cartella "labels"
            for filename in os.listdir(labels_path):
                if not filename.endswith('.txt'):
                    sys.exit()
                file_path = os.path.join(labels_path, filename)
                   
                # Apri il file e cicla attraverso ogni linea
                with open(file_path, 'r') as file:
                    lines = file.readlines()

                new_lines = []

                for line_num, line in enumerate(lines, start=1):
                    # Estrai l'ultimo elemento di ogni linea come altezza
                    raw_height = float(line.strip().split()[-1])
                    height = round(raw_height*640, 5)

                    if height > threshold:
                        new_lines.append(line)
                        # Aggiungi alle informazioni dei file eliminati
                        min_height_info_list.append((os.path.relpath(file_path, path), line_num, height))

                    else:
                        eliminated_files.append((os.path.relpath(file_path, path), line_num, height, filename))
                        # Aggiungi le informazioni alla lista
                        el_height_info_list.append((os.path.relpath(file_path, path), line_num, height))

                if threshold != 0:
                    with open(file_path, 'w') as file:
                        file.writelines(new_lines)

    # Ordina la lista delle informazioni basandoti sui valori minimi di altezza
    min_height_info_list.sort(key=lambda x: x[2])

    # Prendi i primi "range_val" valori minimi di altezza con le relative informazioni
    min_height_info_values = min_height_info_list[:range_val]

    # Stampa i valori minimi di altezza arrotondati a 3 cifre decimali con il nome del file e il numero della riga
    # Stampa i valori minimi di altezza arrotondati a 3 cifre decimali con il nome del file e il numero della riga
    print(f"I {range_val} valori minimi di altezza sono:")
    for i, info in enumerate(min_height_info_values):
        filename, line_num, height = info
        print(f"{round(height, 3)}| nome file: {filename}| n°riga: {line_num} ")
        if i + 1 >= range_val:  # Smetti di stampare dopo il numero specificato di valori
            break

    if threshold == 0:
        print("\nFile eliminati sotto il valore di threshold: 0")


    # Stampa i file eliminati in ordine di grandezza
    else:
        print("\nFile eliminati sotto il valore di threshold:")
        eliminated_files.sort(key=lambda x: x[2])  # Ordina per altezza eliminata
        for eliminated_info in eliminated_files:
            filename, line_num, height, original_filename = eliminated_info
            print(f"{round(height, 3)}| nome file: {filename}| n°riga: {line_num}")



def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='YOLOv5 Object Detection')
    parser.add_argument('--input', '-i', type=str, help='Path to the input folder')
    parser.add_argument('--range', '-r', type=int, help='Number of values visualized')
    parser.add_argument('--threshold', '-th', type=float, default=0, help='Threshold height above which we want the detection present on the line to be eliminated.')

    args = parser.parse_args()

    input_path = args.input
    range_val = args.range
    threshold = args.threshold

    cerca(input_path, range_val, threshold)

if __name__ == '__main__':
    main()
