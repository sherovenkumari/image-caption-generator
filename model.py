# model.py
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add

def build_model(vocab_size, max_length):
    image_input = Input(shape=(25088,))
    image_dense = Dense(256, activation="relu")(image_input)

    caption_input = Input(shape=(max_length,))
    caption_embed = Embedding(vocab_size, 256, mask_zero=True)(caption_input)
    caption_lstm = LSTM(256)(caption_embed)

    decoder = add([image_dense, caption_lstm])
    decoder = Dense(256, activation="relu")(decoder)
    output = Dense(vocab_size, activation="softmax")(decoder)

    model = Model(inputs=[image_input, caption_input], outputs=output)
    model.compile(loss="categorical_crossentropy", optimizer="adam")

    return model
