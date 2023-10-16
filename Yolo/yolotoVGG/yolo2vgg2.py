import argparse
from pathlib import Path
import json
import cv2
try:
    import tqdm
except:
    print('TQDM not installed, progress bars will not be printed')


classes_to_integers = {
    '1': 0,
    '2.1/3': 1,
    '2.2': 2,
    '2.3/6.1': 3,
    '4.1': 4,
    '4.2': 5,
    '4.3': 6,
    '5.1': 7,
    '5.2': 8,
    '6.2': 9,
    '7': 10,
    '7E': 11,
    '8': 12,
    '9': 13,
    'LQ': 14,
    'MP': 15,
#     'HT': 16,
#     'TWU': 17,
}


class_names = list(classes_to_integers.keys())
class_names.sort(key=lambda x: classes_to_integers[x])

empty_project = r'{"_via_settings":{"ui":{"annotation_editor_height":35,"annotation_editor_fontsize":0.8,"leftsidebar_width":20,"image_grid":{"img_height":80,"rshape_fill":"none","rshape_fill_opacity":0.3,"rshape_stroke":"yellow","rshape_stroke_width":2,"show_region_shape":true,"show_image_policy":"all"},"image":{"region_label":"class","region_color":"class","region_label_font":"10px Sans","on_image_annotation_editor_placement":"NEAR_REGION"}},"core":{"buffer_size":"18","filepath":{},"default_filepath":".\\Immagini"},"project":{"name":"VIA project"}},"_via_img_metadata":{},"_via_attributes":{"region":{"class":{"type":"text","description":"IMDG class or placard name","default_value":""},"booleans":{"type":"checkbox","description":"Boolean information","options":{"other_truck":"Not on the main truck","truncated":"Only partially visible","side":"On the side of a truck"},"default_options":{}}},"file":{}},"_via_data_format_version":"2.0.10","_via_image_id_list":[]}'

def yolo2vgg(yolo_labels, im_w, im_h):
    if len(yolo_labels) == 5:
        cl_num, xc, yc, w, h = yolo_labels
    if len(yolo_labels) == 6:
        cl_num, xc, yc, w, h, conf = yolo_labels
    w_vgg = int(w * im_w)
    h_vgg = int(h * im_h)
    x_vgg = int((xc - w/2) * im_w)
    y_vgg = int((yc - h/2) * im_h)
    return {
        'shape_attributes': {
            'name': 'rect',
            'x': x_vgg,
            'y': y_vgg,
            'width': w_vgg,
            'height': h_vgg,
        },
        'region_attributes': {
            'class': class_names[cl_num],
            'booleans': {}
        },
    }

parser = argparse.ArgumentParser()
parser.add_argument("-o", help="overwrite save file", action='store_true')
parser.add_argument("-s", help="saved files suffix", default='')
parser.add_argument("images_dir", help="directory containing the images")
parser.add_argument("labels_dir", help="directory containing the YOLO labels")
parser.add_argument("save_dir", help="directory where the VGG annotations will be saved")
args = parser.parse_args()

images_dir = Path(args.images_dir)
labels_dir = Path(args.labels_dir)
save_dir = Path(args.save_dir)
annotations_file = save_dir/f'annotations{args.s}.json'
if not args.o and annotations_file.exists():
    print('Annotations file already exists, use "-o" to overwrite')
    exit(0)
project_file = save_dir/f'via_project{args.s}.json'
if not args.o and project_file.exists():
    print('Project file already exists, use "-o" to overwrite')
    exit(0)

annotations = {}
image_files = list(images_dir.glob('*.jpg'))
if not image_files:
    print(f'No images found in {images_dir.absolute()}')
    exit(0)
print(f"{len(list(labels_dir.rglob('*.txt')))} label files found.")
try:
    image_files = tqdm.tqdm(image_files)
except:
    pass
for image_file in image_files:
    regions = []
    label_file = labels_dir/(image_file.stem + '.txt')
    try:
        h, w, _ = cv2.imread(str(image_file)).shape

        with label_file.open() as label_data:
            lines = label_data.read().splitlines()
        
        for l in lines:
            region = list(map(float, l.split(' ')))
            region[0] = int(region[0])
            region = yolo2vgg(region, w, h)
            regions.append(region)
    except:
        pass
    
    filename = image_file.name
    size = image_file.stat().st_size
    
    annotations[filename + str(size)] = {
        'filename': filename,
        'size': size,
        'regions': regions,
        'file_attributes': {},
    }

with annotations_file.open(mode='w') as out:
    json.dump(annotations, out)
    print(f'Annotations saved to {annotations_file}')

project = json.loads(empty_project)
project['_via_img_metadata'] = annotations
project['_via_image_id_list'] = list(annotations.keys())


