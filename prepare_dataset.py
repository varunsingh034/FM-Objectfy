import os
import cv2
import xml.etree.ElementTree as ET

ANNOTATION_PATH = "annotations"
IMAGE_PATH = "images"
OUTPUT_PATH = "dataset"

labels = ["with_mask", "without_mask", "mask_weared_incorrect"]
for label in labels:
    os.makedirs(os.path.join(OUTPUT_PATH, label), exist_ok=True)

for xml_file in os.listdir(ANNOTATION_PATH):
    tree = ET.parse(os.path.join(ANNOTATION_PATH, xml_file))
    root = tree.getroot()
    filename = root.find("filename").text
    img_path = os.path.join(IMAGE_PATH, filename)
    image = cv2.imread(img_path)
    if image is None:
        continue
    for obj in root.findall("object"):
        label = obj.find("name").text
        if label == "mask_weared_incorrect":
            label = "without_mask"  # Optionally merge classes
        if label not in labels:
            continue
        bbox = obj.find("bndbox")
        x1 = int(bbox.find("xmin").text)
        y1 = int(bbox.find("ymin").text)
        x2 = int(bbox.find("xmax").text)
        y2 = int(bbox.find("ymax").text)
        face = image[y1:y2, x1:x2]
        face = cv2.resize(face, (224, 224))
        save_path = os.path.join(OUTPUT_PATH, label, f"{filename}_{x1}.jpg")
        cv2.imwrite(save_path, face)
