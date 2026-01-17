# preprocess.py
import string
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

CAPTIONS_FILE = "dataset/Text/Flickr8k_text/filtered_text.txt"

captions = []
with open(CAPTIONS_FILE, "r", encoding="utf-8") as f:
    for line in f:
        _, cap = line.strip().split("\t")
        cap = cap.lower().translate(str.maketrans("", "", string.punctuation))
        captions.append(f"<start> {cap} <end>")

tokenizer = Tokenizer()
tokenizer.fit_on_texts(captions)

sequences = tokenizer.texts_to_sequences(captions)
max_len = max(len(seq) for seq in sequences)

with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("Captions preprocessed")
print("Vocabulary size:", len(tokenizer.word_index) + 1)
print("Max caption length:", max_len)
