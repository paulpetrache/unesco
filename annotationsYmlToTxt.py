import xml.etree.ElementTree as ET
import os

# Paths to your image and annotation directories
image_dir = 'images'
annotation_dir = 'annotations'
yolo_annotation_dir = 'yolo_annotations'

if not os.path.exists(yolo_annotation_dir):
    os.makedirs(yolo_annotation_dir)

def convert_to_yolo_format(img_size, box):
    dw = 1. / img_size[0]
    dh = 1. / img_size[1]
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

for file in os.listdir(annotation_dir):
    if file.endswith('.xml'):
        annotation_path = os.path.join(annotation_dir, file)
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        
        out_file = open(os.path.join(yolo_annotation_dir, file.replace('.xml', '.txt')), 'w')
        
        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            if int(difficult) == 1:
                continue
            cls = obj.find('name').text
            if cls != "licence":
                continue
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
            yolo_box = convert_to_yolo_format((w, h), b)
            out_file.write(f"0 {yolo_box[0]} {yolo_box[1]} {yolo_box[2]} {yolo_box[3]}\n")
        out_file.close()
