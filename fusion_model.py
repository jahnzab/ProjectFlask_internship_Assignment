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
IMAGE_MODEL_PATH = "best_image_model_phase1.h5"
TEXT_MODEL_WEIGHTS_PATH = "best_text_model.h5"
TEXT_MODEL_CONFIG_PATH = "text_model_config.json"
TOKENIZER_CONFIG_PATH = "tokenizer_config.json"
TEXT_CSV_PATH = "mental_health_processed.csv"

TEST_IMG_DIR = "/home/jahanzaib/Desktop/Project_Intern/Flaskapp/dermatology-dataset/test"
print("Loading image model (auto-adjusting for input shape)...")

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models

try:
    image_model = load_model(IMAGE_MODEL_PATH, compile=False)
except ValueError as e:
    print("⚠ Shape mismatch detected while loading EfficientNet weights — rebuilding model with 1-channel input...")

    # ✅ Rebuild model architecture for grayscale (1-channel) input
    base_model = EfficientNetB0(
        include_top=False,
        weights=None,
        input_shape=(224, 224, 1),
        pooling='avg'
    )
    x = layers.Dense(128, activation='relu')(base_model.output)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(23, activation='softmax')(x)  # adjust num classes if needed
    image_model = models.Model(inputs=base_model.input, outputs=output)

    # ✅ Load weights safely, ignoring shape mismatches
    try:
        image_model.load_weights(IMAGE_MODEL_PATH, by_name=True, skip_mismatch=True)
        print("✅ Weights loaded with shape-mismatch tolerance.")
    except Exception as e2:
        print("❌ Could not load weights properly:", e2)

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
        self._load_image_paths()

    def _load_image_paths(self):
        for class_name in IMAGE_CLASSES:
            class_dir = os.path.join(self.directory, class_name)
            if os.path.exists(class_dir):
                for img_name in sorted(os.listdir(class_dir)):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append(os.path.join(class_dir, img_name))
        print(f"Loaded {len(self.samples)} images from {self.directory}")

    def load_and_preprocess_images(self, indices, expected_channels=3):
        images = []
        for idx in indices:
            img_path = self.samples[idx]
            # Always load in RGB to be compatible with pretrained models
            img = load_img(img_path, target_size=self.target_size, color_mode='rgb')
            img = img_to_array(img) / 255.0

            # If model expects grayscale, convert dynamically
            if expected_channels == 1 and img.shape[-1] == 3:
                img = tf.image.rgb_to_grayscale(img)
            elif expected_channels == 3 and img.shape[-1] == 1:
                img = np.repeat(img, 3, axis=-1)

            images.append(img)
        return np.array(images)
# class ImageDataLoader:
#     def __init__(self, directory, target_size=IMAGE_SIZE):
#         self.directory = directory
#         self.target_size = target_size
#         self.samples = []
#         self.labels = []
#         self._load_image_paths()

#     def _load_image_paths(self):
#         for class_name in IMAGE_CLASSES:
#             class_dir = os.path.join(self.directory, class_name)
#             if os.path.exists(class_dir):
#                 for img_name in sorted(os.listdir(class_dir)):
#                     if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
#                         img_path = os.path.join(class_dir, img_name)
#                         self.samples.append(img_path)
#         print(f"Loaded {len(self.samples)} images from {self.directory}")

#     def load_and_preprocess_images(self, indices):
#         images = []
#         for idx in indices:
#             img_path = self.samples[idx]
#             img = load_img(img_path, target_size=self.target_size)
#             img = img_to_array(img) / 255.0
#             images.append(img)
#         return np.array(images)


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

    print("Using preloaded image model from earlier (grayscale-compatible).")
    # image_model is already loaded safely above with shape adjustment

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
    # -------------------------
    # Image Predictions
    # -------------------------
    print("Predicting image classes...")
    img_preds = []
    batch_size = BATCH_SIZE
    num_batches = int(np.ceil(len(image_loader.samples) / batch_size))

    # detect input channels from model
    expected_channels = image_model.input_shape[-1] if len(image_model.input_shape) == 4 else 3

    for batch_idx in range(num_batches):
      start = batch_idx * batch_size
      end = min(start + batch_size, len(image_loader.samples))
      batch_images = image_loader.load_and_preprocess_images(range(start, end),   expected_channels)
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

    # Ensure predictions have the same number of samples
    min_len = min(len(img_preds), len(text_preds))
    img_preds_trimmed = img_preds[:min_len]
    text_preds_trimmed = text_preds[:min_len]

    # Fuse predictions
    fused_preds = img_weight * img_preds_trimmed[:, :len(TEXT_CLASSES)] + text_weight * text_preds_trimmed

    # Compute labels
    fused_labels = np.argmax(fused_preds, axis=1)
    true_labels = LabelEncoder().fit(TEXT_CLASSES).transform(text_df['label'][:min_len])

    # Print metrics
    acc = accuracy_score(true_labels, fused_labels)
    print(f"Late Fusion Accuracy: {acc:.4f}")
    print(classification_report(true_labels, fused_labels, target_names=TEXT_CLASSES))


if __name__ == "__main__":
    main()
# -------------------------
# Flask-ready Late Fusion Predictor
# -------------------------
class LateFusionPredictor:
    def __init__(self, image_model_path, text_model_path, tokenizer, text_config, img_weight=0.5, text_weight=0.5):
        self.tokenizer = tokenizer
        self.img_weight = img_weight
        self.text_weight = text_weight
        self.text_classes = TEXT_CLASSES

        # Lazy load image model with shape adjustment
        self.image_model = self._load_image_model(image_model_path)

        # Load text model
        self.text_model = build_text_model_from_config(text_config)
        self.text_model.load_weights(text_model_path)

    def _load_image_model(self, path):
        try:
            model = load_model(path, compile=False)
        except ValueError:
            # Rebuild EfficientNet for 1-channel input
            base = EfficientNetB0(include_top=False, weights=None, input_shape=(224,224,3), pooling='avg')
            x = layers.Dense(128, activation='relu')(base.output)
            x = layers.Dropout(0.3)(x)
            out = layers.Dense(23, activation='softmax')(x)
            model = Model(inputs=base.input, outputs=out)
            try:
                model.load_weights(path, by_name=True, skip_mismatch=True)
            except:
                print("❌ Could not load image weights properly")
        return model

    def preprocess_image(self, img_path):
        img = load_img(img_path, target_size=(224,224), color_mode='rgb')
        img = img_to_array(img)/255.0
        if self.image_model.input_shape[-1] == 1:
            img = tf.image.rgb_to_grayscale(img)
        img = np.expand_dims(img, axis=0)
        return img

    def predict(self, img_path, text_input):
        img_array = self.preprocess_image(img_path)
        img_pred = self.image_model.predict(img_array, verbose=0)

        seq = self.tokenizer.texts_to_sequences([text_input])
        padded = pad_sequences(seq, maxlen=200, padding='post')
        text_pred = self.text_model.predict(padded, verbose=0)

        fused = self.img_weight*img_pred[:, :len(self.text_classes)] + self.text_weight*text_pred
        label = self.text_classes[np.argmax(fused)]
        return label
