import os
import argparse

def cerca(path, range_val, threshold):
    min_height_info_list = []
    eliminated_files = []  # Lista per tenere traccia dei file eliminati

    # Cicla attraverso i file nella cartella specificata
    for filename in os.listdir(path):
        if filename.endswith('.txt'):
            file_path = os.path.join(path, filename)

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
                else:
                    eliminated_files.append((filename, line_num, height))  # Aggiungi alle informazioni dei file eliminati

                # Aggiungi le informazioni alla lista
                min_height_info_list.append((raw_height, filename, line_num, height))

            if threshold != 0:
                with open(file_path, 'w') as file:
                    file.writelines(new_lines)

    # Ordina la lista delle informazioni basandoti sui valori minimi di altezza
    min_height_info_list.sort(key=lambda x: x[0])
    
    # Prendi i primi "range_val" valori minimi di altezza con le relative informazioni
    min_height_info_values = min_height_info_list[:range_val]

    # Stampa i valori minimi di altezza arrotondati a 3 cifre decimali con il nome del file e il numero della riga
    print(f"I {range_val} valori minimi di altezza sono:")
    for raw_height, filename, line_num, height in min_height_info_values:
        print(f"{round(height, 3)}| nome file: {filename}| n°riga: {line_num} ")

    # Stampa i file eliminati
    # Stampa i file eliminati in ordine di grandezza
    print("\nFile eliminati sotto il valore di threshold:")
    eliminated_files.sort(key=lambda x: x[2])  # Ordina per altezza eliminata
    for eliminated_file, eliminated_line_num, eliminated_height in eliminated_files:
        print(f"{round(eliminated_height, 3)}| nome file: {eliminated_file}| n°riga: {eliminated_line_num}")


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
