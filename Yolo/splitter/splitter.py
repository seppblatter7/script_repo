import os
import argparse
import random
import shutil
import argparse
import glob

def change_extension (file_name, new_extension):

    #divide file name by "."
    base_name, extension = os.path.splitext(file_name)
    #create new file name
    new_file_name = base_name + new_extension

    return new_file_name

def split_data_folder(input_folder):

    images_folder = os.path.join(input_folder, "images")
    labels_folder = os.path.join(input_folder, "labels")

    # Ottieni la lista dei file presenti nelle cartelle "images"
    image_files = os.listdir(images_folder)

    num_images = len(glob.glob(os.path.join(images_folder, "*.jpg")))
    num_labels = len(glob.glob(os.path.join(labels_folder, "*.txt")))
    
    # Check that images e labels contains same numbers of file 
    if num_images != num_labels:
        print(f"Errore: il numero di file nelle cartelle 'images' e 'labels' non corrisponde \n labels:{num_labels} \n images:{num_images}")
        return

    nomiCartelle = ["train", "test", "val"]

    for i in nomiCartelle:
        
        os.makedirs(os.path.join(input_folder, i, "images"), exist_ok=True)
        os.makedirs(os.path.join(input_folder, i, "labels"), exist_ok=True)
        
    for i in image_files:

        soglia = random.randint(1, 10)

        if soglia >= 1 and soglia <= 8:
            
            s = change_extension(i, ".txt")

            #sposta in train immagine e relativa label
            shutil.move(os.path.join(images_folder, i), os.path.join(input_folder, "train", "images", i))
            shutil.move(os.path.join(labels_folder, s), os.path.join(input_folder, "train", "labels", s))

        elif soglia == 9:

            s = change_extension(i, ".txt")

            #sposta in test immagine e relativa label
            shutil.move(os.path.join(images_folder, i), os.path.join(input_folder, "test", "images", i))
            shutil.move(os.path.join(labels_folder, s), os.path.join(input_folder, "test", "labels", s))

        elif soglia == 10: 

            s = change_extension(i, ".txt")

            #sposta in val immagine e relativa label
            shutil.move(os.path.join(images_folder, i), os.path.join(input_folder, "val", "images", i))
            shutil.move(os.path.join(labels_folder, s), os.path.join(input_folder, "val", "labels", s))

# Main function
def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='splitter into train, test e val')
    parser.add_argument('--input', '-i', type=str, help='Path to the input folder')

    args = parser.parse_args()

    input_path = args.input

    split_data_folder(input_path)

if __name__ == '__main__':
    main()
