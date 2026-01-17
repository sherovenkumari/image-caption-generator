import os
import shutil

SRC = "dataset/images/Flicker8k_Dataset"
DST = "dataset/images/images_300"

os.makedirs(DST, exist_ok=True)

images = os.listdir(SRC)[:300]

for img in images:
    shutil.copy(os.path.join(SRC, img), os.path.join(DST, img))

print("300 images copied successfully")
