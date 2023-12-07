import os
import argparse

def elimina_file(path, divisore):
    """
    Elimina i file contenuti nella cartella specificata, alternando i file da eliminare e quelli da non eliminare.

    Args:
    path: Il percorso della cartella.

    Returns:
    Il numero di file eliminati.
    """
   
    images = os.listdir(os.path.join(path, "images"))
    labels = os.listdir(os.path.join(path, "labels"))

    assert len(images) == len(labels)

    file_si = []
    file_no = []

    print(f"numero file in images: {len(images)}")

    for index, file in enumerate(images):
        if index % divisore == 0:
            file_si.append(file)
        else:
            file_no.append(file)

    numero_file_eliminati = len(file_no)
    print (f"file eliminati da images: {numero_file_eliminati}")
    print (file_no)


    for file in file_no:
        os.remove(os.path.join(os.path.join(path, "images"), file))
        os.remove(os.path.join(os.path.join(path, "labels"), file[:-4] + ".txt"))

    print(f"Eliminati {numero_file_eliminati} file")

def main():
  # Parse command-line arguments
  parser = argparse.ArgumentParser(description='YOLOv5 Object Detection')
  parser.add_argument('--input', '-i', type=str, help='Path to the input folder')
  parser.add_argument('--divisore', '-div', type=int, default=2, help='Specify which divisor will be used on the total amount of file in the folder')

  args = parser.parse_args()

  input_path = args.input
  div = args.divisore

  elimina_file(input_path, div)

if __name__ == '__main__':
  main()