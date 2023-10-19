import argparse
import os

def remove_matching_lines(folder_path, excluded_indices):
    # Cicla attraverso i file .txt nella cartella
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            
            # Legge le righe dal file
            with open(file_path, 'r') as file:
                lines = file.readlines()
            
            # Rimuove le righe con corrispondenza di indici classe
            updated_lines = [line for line in lines if not any(line.startswith(f"{excluded_index} ") for excluded_index in excluded_indices)]
            
            # Sovrascrive il file .txt con le righe aggiornate
            with open(file_path, 'w') as file:
                file.writelines(updated_lines)

# Crea il parser degli argomenti
parser = argparse.ArgumentParser()
parser.add_argument('--input', '-i', type=str, help='Path to the input folder')
parser.add_argument('--excluded_indices', '-ex_i', type=int, nargs='+', help='Indices of classes to exclude')

# Parsa gli argomenti dalla linea di comando
args = parser.parse_args()

# Esempio di utilizzo
input_folder = args.input  # Ottieni il percorso della cartella di input
excluded_indices = args.excluded_indices  # Ottieni la lista degli indici classe da escludere

remove_matching_lines(input_folder, excluded_indices)

