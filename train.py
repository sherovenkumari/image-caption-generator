# train.py
import numpy as np
import pickle
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from model import build_model

features = np.load("image_features.npy", allow_pickle=True).item()

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

vocab_size = len(tokenizer.word_index) + 1
max_length = 34

X1, X2, y = [], [], []

with open("dataset/Text/Flickr8k_text/filtered_text.txt") as f:
    for line in f:
        img, cap = line.strip().split("\t")
        cap = "<start> " + cap + " <end>"
        seq = tokenizer.texts_to_sequences([cap])[0]

        for i in range(1, len(seq)):
            in_seq = pad_sequences([seq[:i]], maxlen=max_length)[0]
            out_seq = to_categorical(seq[i], vocab_size)

            X1.append(features[img])
            X2.append(in_seq)
            y.append(out_seq)

X1, X2, y = np.array(X1), np.array(X2), np.array(y)

model = build_model(vocab_size, max_length)
model.fit([X1, X2], y, epochs=8, batch_size=32)

model.save("caption_model.keras")

print("Model trained and saved")
