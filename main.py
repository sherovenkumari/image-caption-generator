import os
import string
from tensorflow.keras.models import Model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical


# Paths
IMAGE_DIR = "dataset/images"
CAPTION_FILE = "dataset/captions.txt"

# Read captions
def load_captions(filename):
    captions = {}
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if len(line) < 1:
                continue

            image_id, caption = line.split("\t")
            image_id = image_id.split("#")[0]

            if image_id not in captions:
                captions[image_id] = []
            captions[image_id].append(caption)
    return captions


# Clean text
def clean_captions(captions):
    table = str.maketrans("", "", string.punctuation)

    for key, caps in captions.items():
        for i in range(len(caps)):
            caption = caps[i]
            caption = caption.lower()
            caption = caption.translate(table)
            caption = caption.replace("  ", " ")
            caption = "startseq " + caption + " endseq"
            caps[i] = caption


# Load and clean
captions_dict = load_captions(CAPTION_FILE)
clean_captions(captions_dict)

# Check output
first_image = list(captions_dict.keys())[0]
print("Sample image:", first_image)
print("Sample captions:")
for c in captions_dict[first_image][:3]:
    print(c)


from tensorflow.keras.preprocessing.text import Tokenizer

# Collect all captions into one list
all_captions = []
for key in captions_dict:
    for cap in captions_dict[key]:
        all_captions.append(cap)

print("Total captions:", len(all_captions))

# Create tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)

# Vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print("Vocabulary Size:", vocab_size)

# Check word to index
print("Word mapping example:")
for word in list(tokenizer.word_index.keys())[:10]:
    print(word, "->", tokenizer.word_index[word])


from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Load VGG16 model (exclude top layer)
model = VGG16(weights="imagenet")
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)  # feature vector layer

print("VGG16 model loaded successfully.")

# Function to extract features from a single image
def extract_features(image_path):
    img = load_img(image_path, target_size=(224,224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    feature = model.predict(img, verbose=0)
    return feature

# Test feature extraction on first image
first_image_path = os.path.join(IMAGE_DIR, list(os.listdir(IMAGE_DIR))[0])
feature_vector = extract_features(first_image_path)
print("Feature vector shape:", feature_vector.shape)

# Find maximum caption length
def max_caption_length(captions):
    max_len = 0
    for caps in captions.values():
        for cap in caps:
            max_len = max(max_len, len(cap.split()))
    return max_len

max_len = max_caption_length(captions_dict)
print("Max caption length:", max_len)

# Caption generator model
def define_model(vocab_size, max_len):
    # Image feature extractor input
    inputs1 = Input(shape=(4096,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)

    # Caption sequence input
    inputs2 = Input(shape=(max_len,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)

    # Merge both models
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model


# Create model
caption_model = define_model(vocab_size, max_len)

# Show model summary
caption_model.summary()

def data_generator(captions, features, tokenizer, max_len, vocab_size):
    while True:
        for key, caps in captions.items():
            if key not in features:
                continue

            feature = features[key][0]
            for cap in caps:
                seq = tokenizer.texts_to_sequences([cap])[0]
                for i in range(1, len(seq)):
                    in_seq = seq[:i]
                    out_seq = seq[i]

                    in_seq = pad_sequences([in_seq], maxlen=max_len)[0]
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]

                    yield ([feature, in_seq], out_seq)


# Extract features for all images
print("Extracting features for all images...")

features = {}
for img_name in os.listdir(IMAGE_DIR):
    img_path = os.path.join(IMAGE_DIR, img_name)
    features[img_name] = extract_features(img_path)

print("Feature extraction completed.")


# ---------------- TRAINING ----------------

steps = sum(len(caps) for caps in captions_dict.values())
epochs = 2   # beginner ke liye 5 enough hai

generator = data_generator(
    captions_dict,
    features,
    tokenizer,
    max_len,
    vocab_size
)

print("Training started...")

caption_model.fit(
    generator,
    epochs=epochs,
    steps_per_epoch=steps,
    verbose=1
)

print("Training completed.")



