# features.py
import os
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

IMAGES_DIR = "dataset/images/images_300"
OUTPUT_FILE = "image_features.npy"

model = VGG16(weights="imagenet", include_top=False)
features = {}

for img in os.listdir(IMAGES_DIR):
    if img.endswith(".jpg"):
        path = os.path.join(IMAGES_DIR, img)
        image = load_img(path, target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape((1, *image.shape))
        image = preprocess_input(image)

        feature = model.predict(image, verbose=0)
        features[img] = feature.flatten()

np.save(OUTPUT_FILE, features)
print("Image features extracted and saved")
