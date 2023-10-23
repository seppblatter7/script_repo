This is an helping guide to use python scripts in YoloV5 training operations.

- 'crop.py' 
    operates recognizing only truck plates, aquiring in input both images and labels path and giving out square-cropped images of the single plate-detections, categorized by class.



- 'splitter.py' 
    takes in input the folder which contains both images and labels and create in the same directory 3 new folders named "train", "test", "val" which will in turn contain subfolders 'images' and 'labels' that will be filled with files (maintaining the images-labels association) according to the formula:
        -train 80%
        -test 10%
        -val 10%
    of the total amount of the files.



- 'VGG-yoloV2.py' 
    accepts in input .Json file and images path, returning for each image a .txt file where we want to.



- 'yolo2vgg.py'
    taken in .txt and .jpg file directories returns both 'annotations.Json' and 'via_project.Json' where we choose to.



- 'remover-changer-class.py'
    takes .txt label files and cycling over each one, it locates the lines in which the index of the class we wish to remove/change is present and operates according to our instructions removing the entire class detected or just changing its index.



- '