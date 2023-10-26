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

def split_data_folder(input_folder):
    """ 
    train_path = os.path.join(input_folder, "train")
    test_path = os.path.join(input_folder, "test")
    val_path = os.path.join(input_folder, "val") """



    for class_folder in os.listdir(input_folder):

        nomiCartelle = ["train", "test", "val"] 

        for i in nomiCartelle:
            os.makedirs(os.path.join(input_folder,i), exist_ok=True)
            
        classFolder_path = os.path.join (input_folder, class_folder)

        if os.path.isdir(classFolder_path):

            for i in os.listdir(classFolder_path):
                print (i)

                cropped_images = os.path.join(classFolder_path, i)

                soglia = random.randint(1, 10)

                if soglia >= 1 and soglia <= 8:       

                    os.makedirs(os.path.join(os.path.join(input_folder, "train"), class_folder), exist_ok=True)         

                    #copia in train immagine croppata
                    shutil.copy(cropped_images, os.path.join(input_folder, "train", class_folder))

                elif soglia == 9:

                    os.makedirs(os.path.join(os.path.join(input_folder, "test"), class_folder), exist_ok=True)         

                    #copia in train immagine croppata
                    shutil.copy(cropped_images, os.path.join(input_folder, "test", class_folder))
                

                elif soglia == 10: 

                    os.makedirs(os.path.join(os.path.join(input_folder, "val"), class_folder), exist_ok=True)         

                    #copia in train immagine croppata
                    shutil.copy(cropped_images, os.path.join(input_folder, "val", class_folder))
        



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
