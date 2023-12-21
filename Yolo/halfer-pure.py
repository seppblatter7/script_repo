import os
import argparse

def elimina_file(path, divisore):

  images_path = os.path.join (path, "images")
  labels_path = os.path.join (path, "labels")
  
  file_in_cartella = os.listdir(images_path)

  print(len(file_in_cartella))

  file_si = []
  file_no = []

  for index, file in enumerate(file_in_cartella):  

    if index % divisore == 0:
      file_si.append(file)
    else:
      file_no.append(file)

  numero_file_eliminati = len(file_no)
  print(f"Numero di file che verranno eliminati: {numero_file_eliminati}") 

  for file in file_no:
    os.remove(os.path.join(images_path, file))
    os.remove(os.path.join(labels_path, file.replace("jpg", "txt")))

  print(f"Eliminati {numero_file_eliminati} file in {path}")

def main():
  # Parse command-line arguments
  parser = argparse.ArgumentParser(description='YOLOv5 Object Detection')
  parser.add_argument('--input', '-i', type=str, help='Path to the input folder ********* IMPORTANT: make sure your valid folder is named valid and not val')
  parser.add_argument('--divisore', '-div', type=int, default=2, help='Specify which divisor will be used on the total amount of file in the folder')

  args = parser.parse_args()

  input_path = args.input
  div = args.divisore

  elimina_file(input_path, div)

if __name__ == '__main__':
  main()