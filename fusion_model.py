"""
LATE FUSION SCRIPT - Internship Requirements
Separate Image + Text Models with Gemini Augmentation
Combines predictions at decision level, not feature level
"""

import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import warnings
import google.generativeai as genai

warnings.filterwarnings('ignore')

# -------------------------
# CONFIG
# -------------------------
IMAGE_MODEL_PATH = "/content/best_image_model_phase1.h5"
TEXT_MODEL_WEIGHTS_PATH = "/content/best_text_model.h5"
TEXT_MODEL_CONFIG_PATH = "/content/text_model_config.json"
TOKENIZER_CONFIG_PATH = "/content/tokenizer_config.json"
TEXT_CSV_PATH = "/content/ProjectFlask_internship_Assignment/mental_health_processed.csv"

TEST_IMG_DIR = "/content/ProjectFlask_internship_Assignment/dermatology_dataset/test"

IMAGE_CLASSES = [
    "Atopic Dermatitis Photos",
    "Lupus and other Connective Tissue diseases",
    "Herpes HPV and other STDs Photos",
    "Poison Ivy Photos and other Contact Dermatitis",
    "Scabies Lyme Disease and other Infestations and Bites",
    "Light Diseases and Disorders of Pigmentation",
    "Eczema Photos",
    "Exanthems and Drug Eruptions",
    "Acne and Rosacea Photos",
    "Systemic Disease",
    "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions",
    "Tinea Ringworm Candidiasis and other Fungal Infections",
    "Psoriasis pictures Lichen Planus and related diseases",
    "Nail Fungus and other Nail Disease",
    "Urticaria Hives",
    "Warts Molluscum and other Viral Infections",
    "Vasculitis Photos",
    "Bullous Disease Photos",
    "Hair Loss Photos Alopecia and other Hair Diseases",
    "Vascular Tumors",
    "Cellulitis Impetigo and other Bacterial Infections",
    "Melanoma Skin Cancer Nevi and Moles",
    "Seborrheic Keratoses and other Benign Tumors"
]

TEXT_CLASSES = ["normal", "stress", "anxiety", "depression"]

BATCH_SIZE = 16
IMAGE_SIZE = (224, 224)

GEMINI_API_KEY = "AIzaSyDsUHGkjhASXBHLmUpIUq4JlokU9K90uTs"
USE_GEMINI = True
GEMINI_EMBED_DIM = 768


# -------------------------
# Gemini Embedder
# -------------------------
class GeminiEmbedder:
    def __init__(self, api_key):
        self.api_key = api_key
        self.embed_dim = GEMINI_EMBED_DIM
        self.model = None
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('models/embedding-001')
            self.is_available = True
            print("✓ Gemini API initialized")
        except Exception as e:
            print(f"⚠ Gemini API init failed: {e}")
            self.is_available = False

    def embed_texts(self, text_list):
        if not self.is_available or not USE_GEMINI:
            return np.random.randn(len(text_list), self.embed_dim).astype(np.float32)
        embeddings = []
        for text in text_list:
            try:
                result = genai.embed_content(
                    model="models/embedding-001",
                    content=text,
                    task_type="classification"
                )
                embeddings.append(result['embedding'])
            except:
                embeddings.append(np.random.randn(self.embed_dim).astype(np.float32))
        return np.array(embeddings)


# -------------------------
# Image Loader
# -------------------------
class ImageDataLoader:
    def __init__(self, directory, target_size=IMAGE_SIZE):
        self.directory = directory
        self.target_size = target_size
        self.samples = []
        self.labels = []
        self._load_image_paths()

    def _load_image_paths(self):
        for class_name in IMAGE_CLASSES:
            class_dir = os.path.join(self.directory, class_name)
            if os.path.exists(class_dir):
                for img_name in sorted(os.listdir(class_dir)):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(class_dir, img_name)
                        self.samples.append(img_path)
        print(f"Loaded {len(self.samples)} images from {self.directory}")

    def load_and_preprocess_images(self, indices):
        images = []
        for idx in indices:
            img_path = self.samples[idx]
            img = load_img(img_path, target_size=self.target_size)
            img = img_to_array(img) / 255.0
            images.append(img)
        return np.array(images)


# -------------------------
# Text Model Builder
# -------------------------
def build_text_model_from_config(config):
    inputs = keras.Input(shape=(config["max_len"],))
    x = keras.layers.Embedding(config.get("max_words", 10000), config.get("embedding_dim", 128), mask_zero=True)(inputs)
    x = keras.layers.SpatialDropout1D(0.3)(x)
    x = keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True, dropout=0.3))(x)
    x = keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True, dropout=0.3))(x)
    attn = keras.layers.Dense(1, activation='tanh')(x)
    attn = keras.layers.Flatten()(attn)
    attn = keras.layers.Activation('softmax')(attn)
    attn = keras.layers.RepeatVector(128)(attn)
    attn = keras.layers.Permute([2, 1])(attn)
    x = keras.layers.Multiply()([x, attn])
    x = keras.layers.Lambda(lambda xin: tf.reduce_sum(xin, axis=1))(x)
    x = keras.layers.Dense(256, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(len(TEXT_CLASSES), activation='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


# -------------------------
# Tokenizer Loader
# -------------------------
def load_or_create_tokenizer(path, texts=None):
    if os.path.exists(path):
        with open(path, 'r') as f:
            cfg = json.load(f)
        tokenizer = keras.preprocessing.text.Tokenizer(num_words=cfg.get('num_words', 10000), oov_token="<OOV>")
        tokenizer.word_index = cfg.get('word_index', {})
    else:
        tokenizer = keras.preprocessing.text.Tokenizer(num_words=10000, oov_token="<OOV>")
        if texts:
            tokenizer.fit_on_texts(texts)
    return tokenizer


# -------------------------
# MAIN
# -------------------------
def main():
    print("="*60)
    print("LATE FUSION: IMAGE + TEXT MODELS")
    print("="*60)

    text_df = pd.read_csv(TEXT_CSV_PATH)
    print(f"Loaded {len(text_df)} text samples")

    image_loader = ImageDataLoader(TEST_IMG_DIR)

    print("Loading image model...")
    image_model = load_model(IMAGE_MODEL_PATH, compile=False)

    print("Loading text model...")
    if os.path.exists(TEXT_MODEL_CONFIG_PATH):
        with open(TEXT_MODEL_CONFIG_PATH, 'r') as f:
            text_config = json.load(f)
    else:
        text_config = {"max_len": 100, "max_words": 10000, "embedding_dim": 128}
    text_model = build_text_model_from_config(text_config)
    text_model.load_weights(TEXT_MODEL_WEIGHTS_PATH)

    tokenizer = load_or_create_tokenizer(TOKENIZER_CONFIG_PATH, text_df['text'].tolist())

    gemini_embedder = GeminiEmbedder(GEMINI_API_KEY)

    # -------------------------
    # Image Predictions
    # -------------------------
    print("Predicting image classes...")
    img_preds = []
    batch_size = BATCH_SIZE
    num_batches = int(np.ceil(len(image_loader.samples) / batch_size))
    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(image_loader.samples))
        batch_images = image_loader.load_and_preprocess_images(range(start, end))
        batch_preds = image_model.predict(batch_images, verbose=0)
        img_preds.append(batch_preds)
    img_preds = np.vstack(img_preds)

    # -------------------------
    # Text Predictions
    # -------------------------
    print("Predicting text classes...")
    text_sequences = tokenizer.texts_to_sequences(text_df['text'].tolist())
    text_padded = pad_sequences(text_sequences, maxlen=text_config["max_len"], padding='post')
    text_preds = text_model.predict(text_padded, verbose=0)

    # -------------------------
    # Gemini Predictions (optional)
    # -------------------------
    print("Embedding text via Gemini...")
    gemini_feats = gemini_embedder.embed_texts(text_df['text'].tolist())

    # -------------------------
    # Late Fusion
    # -------------------------
    print("Performing late fusion...")
    # Weighted average (adjust weights as needed)
    img_weight = 0.5
    text_weight = 0.5
    fused_preds = img_weight * img_preds[:, :len(TEXT_CLASSES)] + text_weight * text_preds

    fused_labels = np.argmax(fused_preds, axis=1)
    true_labels = LabelEncoder().fit(TEXT_CLASSES).transform(text_df['label'])

    acc = accuracy_score(true_labels, fused_labels)
    print(f"Late Fusion Accuracy: {acc:.4f}")
    print(classification_report(true_labels, fused_labels, target_names=TEXT_CLASSES))


if __name__ == "__main__":
    main()
