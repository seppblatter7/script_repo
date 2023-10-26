import argparse
import os

def create_association_matrix(mother_indices, excluded_indices):
    association_matrix = []

    for mother_index in mother_indices:
        class_associations = [mother_index]
        num_classes = int(input(f"Inserisci il numero di classi da eliminare in favore della classe madre {mother_index}: "))

        for i in range(num_classes):
            class_to_exclude = input(f"Inserisci la {i+1} classe da eliminare in favore della classe madre {mother_index}: ")
            if class_to_exclude in excluded_indices or class_to_exclude in mother_indices or mother_indices in excluded_indices:
                print(f"L'indice {class_to_exclude} è presente in excluded_indices o in mother_indices. Il programma verrà interrotto.")
                raise SystemExit  # Interrompe il programma
            elif any(element == class_to_exclude for element in excluded_indices) or any(element == class_to_exclude for element in mother_indices) or any(element == excluded_indices for element in mother_indices):
                print(f"Almeno un elemento in class_to_exclude è uguale a un elemento in excluded_indices o in mother_indices. Il programma verrà interrotto.")
                raise SystemExit  # Interrompe il programma
            else:
                class_associations.append(class_to_exclude)
                break


        association_matrix.append(class_associations)

    return association_matrix

def process_files(input_folder, excluded_indices, association_matrix):
    txt_files = [f for f in os.listdir(input_folder) if f.endswith(".txt")]

    for file_name in txt_files:
        file_path = os.path.join(input_folder, file_name)

        with open(file_path, "r") as file:
            lines = file.readlines()

        modified_lines = []
        for line in lines:
            class_name = line.split()[0]

            if excluded_indices and not any(index is None for index in excluded_indices):
                if class_name in excluded_indices and excluded_indices is not None:
                    continue

            for row in association_matrix:
                if class_name in row[1:]:
                    mother_class = row[0]
                    modified_line = line.replace(class_name, str(mother_class))
                    line = modified_line
                    break

            modified_lines.append(line)

        with open(file_path, "w") as file:
            file.writelines(modified_lines)

def Main():
    # Crea il parser degli argomenti
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, help='Path to the input folder')
    parser.add_argument('--excluded_indices', '-ex_i', type=str, nargs='+', help='Indices of classes to exclude')
    parser.add_argument('--mother_indices', '-mot_i', type=int, nargs='+', help='Indices of mother classes')

    # Parsa gli argomenti dalla linea di comando
    args = parser.parse_args()

    # Esempio di utilizzo
    input_folder = args.input  # Ottieni il percorso della cartella di input
    excluded_indices = args.excluded_indices
    mother_indices = args.mother_indices  # Ottieni la lista degli indici classe "madre"

    association_matrix = []
    if mother_indices:
        association_matrix = create_association_matrix(mother_indices, excluded_indices)

    process_files(input_folder, excluded_indices, association_matrix)

if __name__ == '__main__':
    Main()