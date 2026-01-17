# inference.py

import numpy as np
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# -----------------------------
# Load model
# -----------------------------
model = load_model("caption_model.keras")

# -----------------------------
# Load features
# -----------------------------
features = np.load("image_features.npy", allow_pickle=True).item()

# -----------------------------
# Load tokenizer
# -----------------------------
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

max_length = 34

# -----------------------------
# Temperature sampling
# -----------------------------
def sample_with_temperature(preds, temperature=0.8):
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    return np.random.choice(len(preds), p=preds)

# -----------------------------
# Caption generation
# -----------------------------
def generate_caption(photo):
    text = "<start>"
    last_word = None

    for _ in range(max_length):
        seq = tokenizer.texts_to_sequences([text])[0]
        seq = pad_sequences([seq], maxlen=max_length)

        preds = model.predict([photo, seq], verbose=0)[0]
        yhat = sample_with_temperature(preds)

        word = tokenizer.index_word.get(yhat)
        if word is None or word == "<end>" or word == last_word:
            break

        text += " " + word
        last_word = word

    return text.replace("<start>", "").strip()

# -----------------------------
# PICK AN IMAGE
# -----------------------------
img_name = list(features.keys())[0]  # first image
photo = features[img_name].reshape(1, -1)

# -----------------------------
# Generate caption
# -----------------------------
caption = generate_caption(photo)

# -----------------------------
# Display image + caption
# -----------------------------
img_path = f"dataset/images/images_300/{img_name}"

image = load_img(img_path, target_size=(224, 224))
plt.imshow(image)
plt.axis("off")
plt.title(caption)
plt.show()

print("Generated Caption:", caption)


