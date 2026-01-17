🖼️ Image Caption Generator using CNN and LSTM
📌 Project Overview

    This project implements an Image Caption Generator that automatically generates natural language descriptions for images using Deep Learning.
    It combines CNN-based image feature extraction with a sequence-based language model to predict meaningful captions.

    The project is designed to be simple to moderate level, suitable for training purposes, and runnable on a CPU-only system.

🎯 Objectives

    To understand how images can be converted into meaningful textual descriptions

    To learn the integration of computer vision and natural language processing

    To build and evaluate a basic end-to-end deep learning pipeline

    To generate captions for unseen images using a trained model

🧠 Model Architecture

    The project follows a two-branch architecture:

        1️⃣ Image Feature Extractor (CNN)

            Uses a pretrained CNN model (e.g., VGG16 / similar)

            Extracts a fixed-length feature vector from images

            Features are saved to avoid recomputation

        2️⃣ Caption Generator (Sequence Model)

            Uses Embedding + LSTM / Dense layers

            Takes:

            Image features

            Partial caption sequence

            Predicts the next word in the caption

📂 Project Structure

image-caption-generator/
│
├── dataset/
│   ├── images/
│   │   ├── Flicker8k_Dataset/
│   │   └── images_300/
│   └── Text/
│       └── Flickr8k_text/
│
├── preprocess.py          # Text preprocessing
├── features.py            # Image feature extraction
├── train.py               # Model training
├── model.py               # CNN + Transformer model
├── inference.py           # Caption generation (prediction)
├── image_filter.py        # Dataset filtering utilities
├── dataset_checking.py    # Dataset validation
│
├── image_features.npy     # Pre-extracted image features
├── caption_model.keras    # Trained model
│
└── README.md

📊 Dataset

    Based on Flickr8k dataset

    Custom reduced dataset of 300 images

    Each image is associated with multiple captions

    Dataset is included inside the repository



⚙️ Requirements

    Install dependencies using:

    pip install tensorflow numpy matplotlib pillow nltk


    ⚠️ Make sure you are using Python 3.8+

🚀 How to Run the Project

    1️⃣ Preprocess Text Data
        python preprocess.py

    2️⃣ Extract Image Features (Run once)
        python features.py
        This generates:

        image_features.npy

    3️⃣ Train the Model
        python train.py

        This saves:
        caption_model.keras

    4️⃣ Generate Caption for an Image
        python inference.py

🧪 Sample Output

    Input Image: dog_running.jpg
    Generated Caption: "a dog is running through the grass"


⚙️ Technologies & Libraries Used

    Python 3.x

    TensorFlow / Keras

    NumPy

    Matplotlib

    Pickle

    Pillow (PIL)

🏋️ Training Details

    Loss Function: Categorical Crossentropy

    Optimizer: Adam

    Metrics: Accuracy (token-level)

    Epochs: Configurable (CPU-friendly)

    Validation Split: 20%

    Metrics Observed

    Training Loss ↓

    Validation Loss ↓

    Training Accuracy ↑

    Validation Accuracy ↑

    Note: Accuracy is token-level due to sequence prediction nature.

📈 Model Evaluation

    Loss Curve: Shows learning convergence

    Accuracy Curve: Indicates word prediction improvement

    Qualitative Evaluation: Visual inspection of generated captions

🧪 Inference Example

    For a given input image, the model:

    Extracts image features

    Generates caption word-by-word

    Displays the image with the predicted caption

Sample Output:

Generated Caption: a brown and white dog is playing with a toy

💻 System Requirements

    CPU-based system (no GPU required)

    Minimum 8 GB RAM

    Windows / Linux / macOS

📈 Future Improvements

    Add BLEU score evaluation

    Support larger datasets (Flickr30k / MS-COCO)

    Integrate attention visualization

    Web or GUI interface


⚠️ Limitations

    Small dataset (300 images)

    CPU-only training

    Limited vocabulary

    Captions may repeat words or lack grammatical perfection

    These limitations are acceptable for training-level implementation.

👩‍💻 Author

    Sheroven Kumari
    Deep Learning Project – Image Caption Generator

📜 License

    This project is for educational and research purposes only.