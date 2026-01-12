import os
import random
import shutil

SOURCE_DIR = "dataset/images"
TARGET_DIR = "dataset/images_small"

os.makedirs(TARGET_DIR, exist_ok=True)

images = os.listdir(SOURCE_DIR)
selected_images = random.sample(images, 300)

for img in selected_images:
    shutil.copy(
        os.path.join(SOURCE_DIR, img),
        os.path.join(TARGET_DIR, img)
    )

print("300 random images copied successfully.")
