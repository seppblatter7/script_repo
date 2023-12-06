import argparse
import os

def create_association_matrix(mother_indices, excluded_indices):
    association_matrix = []

    for i, mother_index in enumerate(mother_indices):
        class_associations = [mother_index]
        

        # Inserisci il numero di classi da eliminare in favore della classe madre
        num_classes = int(input(f"Inserisci il numero di classi da eliminare in favore della classe madre '{mother_index}':\nEnter the number of classes to be eliminated in favor of the parent class '{mother_index}':\n"))

        for j in range(num_classes):
            if excluded_indices is None:
                excluded_indices = []

            # Inserisci la j-esima classe da eliminare in favore della classe madre
            class_to_exclude = int(input(f"Inserisci la {j+1} classe da eliminare in favore della classe madre '{mother_index}':\nEnter the {j+1} class to be eliminated in favor of the parent class '{mother_index}':\n"))
            if any(element == class_to_exclude for element in excluded_indices) or any(element == class_to_exclude for element in mother_indices) or any(element == excluded_indices for element in mother_indices):
                print(f"At least one element in class_to_exclude is equal to an element in excluded_indices or mother_indices. The program will be terminated.")
            class_associations.append(class_to_exclude)

        association_matrix.append(class_associations)
        print(association_matrix)

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

            modified_line = None

            if excluded_indices and class_name in excluded_indices:
                continue

            for row in association_matrix:
                if class_name in map(str, row[1:]):
                    # Sostituisci solo il primo elemento della riga con l'elemento della classe madre
                    modified_line = "{} {}".format(row[0], line.split(' ', 1)[1])
                    break

            if modified_line is None:
                modified_line = line

            modified_lines.append(modified_line)

        with open(file_path, "w") as file:
            file.writelines(modified_lines)

def Main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, help='Path to the input folder')
    parser.add_argument('--excluded_indices', '-ex_i', type=str, nargs='+', help='Indices of classes that will be erased with the entire line associated')
    parser.add_argument('--mother_indices', '-mot_i', type=int, nargs='+', help='Indices of mother classes that will replace the classes we do not want to remove but just transform. The program will ask you for every mother index entered how many class index you want to remove then based on this let you insert the indices')

    args = parser.parse_args()

    input_folder = args.input
    excluded_indices = args.excluded_indices
    mother_indices = args.mother_indices

    association_matrix = []
    if mother_indices:
        association_matrix = create_association_matrix(mother_indices, excluded_indices)

    process_files(input_folder, excluded_indices, association_matrix)

if __name__ == '__main__':
    Main()
