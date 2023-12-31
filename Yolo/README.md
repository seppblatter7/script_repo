# Description of some scripts commonly used during Ai training with Yolo 


## Overview

This is an helpful guide that introduce you to some python scripts in Yolo training operations. Below you can read an accurate description of what each script does and what it requires to do it. 

Generally, these are programmes that sort files into folders according to the scheme we want, and operate on the files themselves to derive output such as 'cropped-images' or .Json files that can be uploaded to VGG to correct any classification errors of the neural network.


## Scripts description 

### - 'cropper.py' 
    It operates recognizing only truck plates, aquiring as input both images and labels path and giving out square-cropped images of the single plate-detections, categorized by class.

    example:
    python3 crop.py -i "imagesfolder_path" -l "labelsfolde_path" -o "outputfolder_path"


### - 'splitter.py' 
    It takes as input the folder which contains both images and labels and create in the same directory 3 new folders named "train", "test", "val" which will in turn contain subfolders, 'images' and 'labels', that will be filled with files (maintaining the images-labels association) according to the formula:
        -train 80%
        -test 10%
        -val 10%
    of the total amount of the files.

    example:
    python3 splitter.py -i "inputfolder_path"


### - 'cropped-splitter.py'
    You can use it when you have the class folders (splitted by cropped.py for example), and you want to divide following the "splitter.py" logic (train, test, val) creating in the main folder (on the level of the original class sub-folders) train, test, val folders which will in turn contain each all the class sub-folders.        


### - 'VGG-yoloV2.py' 
    It accepts as input .Json file and relative images, returning for each image a .txt file where we specificate in the parameters that contains the correct coordinates of every class detected.

    example:
    python3 VGG-yolov2.py ".Jsonfile_path" "imagesfolder_path" "outputfolder_path"


### - 'yolo2vgg.py'
    It works in reverse of "VGG-yoloV2".
    It Requires the .txt-folder and .jpg-file directories path and returns 2 .json file: 'annotations.Json' and 'via_project.Json' where we have previously specified. There is also an option to overwrite the .Json if they already exists, in the case we haven't put the -o flag scripts will not output anything.

    example:
    python3 yolo2vgg.py -o "imagesfolder_path" "labelsfolder_path" "output_path"


### - 'remover-changer-class.py'
    It takes as input .txt label files and cycling over each one, it locates the lines in which the class index we wish to remove/change is present and operates according to our instructions removing the entire class detected or just changing its index.

    example:
    python3 remover-changer-class.py  


### detector
    This section contains scripts that have the function of detecting classes desired by the user, differentiating by input types and output options. 

#### 'yolo-detector-classificator'
    It can receive as input one single video, one single image, a folder containing images or rstp stream. We have to specify the model we want to use, and with multiple option can carry out some interesting tasks. I choose to copy the code itself because I think it could be confusionary to describe in our language every options and how to use it meanwhile the "help" description is pretty clear.

    ('--input', '-i', type=str,                                          help='Path to the input file (image, video, or rtsp stream)')
    ('--model', '-m', type=str, default='yolov5s.pt',                    help='Path to the model to use (yolov5s.pt, yolov5m.pt, yolov5l.pt)')
    ('--output_format', '-of', type=str, default=None,                   help='Output format (display, jpg, mp4)')
    ('--output_dir', '-od', type=str, default='output',                  help='Output folder path')
    ('--device', '-d', type=str, default='cpu',                          help='Device to use (cpu, cuda)')
    ('--cls_model', '-cm', type=str, default='yolov5n-cls.pt',           help='Path to the model to use (yolov5s-cls.pt, yolov5m-cls.pt, yolov5l-cls.pt)')
    ('--apply_cls', '-ac', action='store_true', default= False,          help='Apply classification model to the detected objects')
    ('--save_txt', '-st', action='store_true',                           help='Save txt for detected objects')
    ('--square_crop', '-sqc', action='store_true', default=False,        help='Convert bounding boxes to square boxes and apply classification')
    ('--save_crop', '-scrop', action='store_true',                       help='Save cropped images after classification')
    ('--classes', '-c', type=int, nargs='+', default=None,               help='List of class indices to filter detected objects')
    ('--confidence', '-conf', type=float, default=0.5,                   help='Confidence')
    ('--letterbox', '-lb', action='store_true', default=False,           help='Letterbox')

#### 'yolo-detector-images.py'
    It can receive as input a single image as a folder containing many. It gives out both images and labels for every frame, and if flagged, also the cropped images for every class detected (squared if you desire)

    example:
    python3 yolo-detector-images.py -i "inputh_path" -m "model_path" -od "output_path" [-st] [-sqc] [-scrop]

#### 'yolo-detector-1video.py'
    It receive as input one single .mp4 file and model we want to use, giving out folder frames, (".jpg",".txt") with the possibility of ticking the frame ratio flag to choose to save one frame every 'argument_entered' or the save txt flag to save also annotations.

    example (with one frame saved every 15 and save txt selected):
    python3 yolo-detector-1video,py -i "single_video_path" -m "model_path" -od "outputfolder_path" -st -fr 15


#### 'yolo-detector-stream-boxes-txt.py'
    It takes as input an RSTP stream and visualize a window receives as input an RSTP stream generating as output the same stream but with the bounding boxes drawn around the detected classes, in case the value "1" is passed to the -txt flag it will also save the related images/labels for each frame of the video in the path passed via args -out.

    example:
    python3 yolo-detector-stream-boxes-txt.py -ru "rtsp_URL" -M "model_path" [-txt 1 (in order to save labels and images)] [-out "output_path"] 


#### 'yolo-detector-stream-boxes-txt-thread.py'
    Using the thread technolgy