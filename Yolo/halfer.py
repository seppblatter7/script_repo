import os
import argparse

def elimina_meta_file(path):
  """
  Elimina i file contenuti nella cartella specificata, alternando i file da eliminare e quelli da non eliminare.

  Args:
    path: Il percorso della cartella.

  Returns:
    Il numero di file eliminati.
  """

  for directory in ["test", "train", "valid"]:
    images_path = os.path.join(path, directory, "images")
    labels_path = os.path.join(path, directory, "labels")

    images = os.listdir(images_path)
    labels = os.listdir(labels_path)

    assert len(images) == len(labels)

    file_si = []
    file_no = []

    for index, file in enumerate(images):
      if index % 2 == 0:
        file_si.append(file)
      else:
        file_no.append(file)

    numero_file_eliminati = len(file_no)

    for file in file_no:
      os.remove(os.path.join(images_path, file))
      os.remove(os.path.join(labels_path, file[:-4] + ".txt"))

    print(f"Eliminati {numero_file_eliminati} file in {directory}")

def main():
  # Parse command-line arguments
  parser = argparse.ArgumentParser(description='YOLOv5 Object Detection')
  parser.add_argument('--input', '-i', type=str, help='Path to the input .mp4 file')

  args = parser.parse_args()

  input_path = args.input

  elimina_meta_file(input_path)

if __name__ == '__main__':
  main()