import os

IMAGES_DIR = "dataset/images/images_300"
CAPTIONS_FILE = "dataset/Text/Flickr8k_text/filtered_text.txt"

image_names = set(
    os.path.splitext(img)[0]
    for img in os.listdir(IMAGES_DIR)
    if img.endswith(".jpg")
)

caption_names = set()

with open(CAPTIONS_FILE, "r", encoding="utf-8") as f:
    for line in f:
        img, _ = line.strip().split("\t", 1)
        caption_names.add(os.path.splitext(img)[0])

print("Images without captions:", image_names - caption_names)
print("Captions without images:", caption_names - image_names)


import os
from PIL import Image
import matplotlib.pyplot as plt

IMAGES_DIR = "dataset/images/images_300"
CAPTIONS_FILE = "dataset/Text/Flickr8k_text/filtered_text.txt"

with open(CAPTIONS_FILE, "r", encoding="utf-8") as f:
    lines = f.readlines()

# show first 5 image-caption pairs
for line in lines[:2]:
    img_name, caption = line.strip().split("\t")
    img_path = os.path.join(IMAGES_DIR, img_name)

    image = Image.open(img_path)
    plt.imshow(image)
    plt.axis("off")
    plt.title(caption)
    plt.show()
