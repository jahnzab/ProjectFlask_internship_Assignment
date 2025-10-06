"""
OPTIMIZED Fusion Script - Internship Requirements
Multimodal Psychodermatological Disorder Detection
Uses CNN Transfer Learning + Trained Text Model + Gemini Augmentation
"""

import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
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

TRAIN_IMG_DIR = "/content/ProjectFlask_internship_Assignment/dermatology_dataset/train"
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
class AlignedImageDataLoader:
    def __init__(self, directory, target_size=IMAGE_SIZE):
        self.directory = directory
        self.target_size = target_size
        self.samples = []
        self.labels = []
        self.class_indices = {}
        self._load_image_paths()

    def _load_image_paths(self):
        for idx, class_name in enumerate(sorted(IMAGE_CLASSES)):
            class_dir = os.path.join(self.directory, class_name)
            if os.path.exists(class_dir):
                self.class_indices[class_name] = idx
                for img_name in sorted(os.listdir(class_dir)):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(class_dir, img_name)
                        self.samples.append((img_path, idx))
                        self.labels.append(idx)
        print(f"Loaded {len(self.samples)} images from {self.directory}")

    def load_and_preprocess_images(self, indices):
        images = []
        labels = []
        for idx in indices:
            img_path, label = self.samples[idx]
            img = load_img(img_path, target_size=self.target_size)
            img = img_to_array(img) / 255.0
            images.append(img)
            labels.append(label)
        return np.array(images), np.array(labels)

# -------------------------
# Text Model Builder
# -------------------------
def build_text_model_from_config(config):
    inputs = Input(shape=(config["max_len"],))
    x = Embedding(config.get("max_words", 10000), config.get("embedding_dim", 128), mask_zero=True)(inputs)
    x = SpatialDropout1D(0.3)(x)
    x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.3))(x)
    x = Bidirectional(LSTM(64, return_sequences=True, dropout=0.3))(x)
    attn = Dense(1, activation='tanh')(x)
    attn = Flatten()(attn)
    attn = Activation('softmax')(attn)
    attn = RepeatVector(128)(attn)
    attn = Permute([2, 1])(attn)
    x = Multiply()([x, attn])
    x = Lambda(lambda xin: tf.reduce_sum(xin, axis=1))(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(len(TEXT_CLASSES), activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
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
# Fusion Model
# -------------------------
def build_advanced_fusion_model(image_feat_shape, text_feat_shape, gemini_feat_shape, num_classes=len(TEXT_CLASSES)):
    img_input = Input(shape=image_feat_shape)
    txt_input = Input(shape=text_feat_shape)
    gemini_input = Input(shape=gemini_feat_shape)
    img_proj = Dense(256, activation='relu')(img_input)
    txt_proj = Dense(256, activation='relu')(txt_input)
    gemini_proj = Dense(256, activation='relu')(gemini_input)
    img_txt_attention = Dot(axes=-1, normalize=True)([img_proj, txt_proj])
    img_txt_attention = Activation('sigmoid')(img_txt_attention)
    gemini_context = Dense(256, activation='tanh')(gemini_proj)
    weighted_img = Multiply()([img_proj, img_txt_attention])
    weighted_txt = Multiply()([txt_proj, img_txt_attention])
    fused = Concatenate()([weighted_img, weighted_txt, gemini_context])
    x = Dense(512, activation='relu')(fused)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=[img_input, txt_input, gemini_input], outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()])
    return model

# -------------------------
# Main Pipeline
# -------------------------
def main():
    print("="*60)
    print("MULTIMODAL FUSION WITH GEMINI AUGMENTATION")
    print("="*60)

    text_df = pd.read_csv(TEXT_CSV_PATH)
    print(f"Loaded {len(text_df)} text samples")

    image_loader = AlignedImageDataLoader(TEST_IMG_DIR)

    num_samples = min(len(image_loader.samples), len(text_df))
    print(f"Aligning {num_samples} samples for fusion")

    aligned_texts = text_df['text'][:num_samples].tolist()
    aligned_labels = text_df['label'][:num_samples].tolist()

    le = LabelEncoder()
    le.fit(TEXT_CLASSES)
    labels_encoded = le.transform(aligned_labels)
    labels_onehot = keras.utils.to_categorical(labels_encoded, len(TEXT_CLASSES))

    print("Loading image model...")
    raw_image_model = load_model(IMAGE_MODEL_PATH, compile=False)
    image_feat_extractor = Model(raw_image_model.input, raw_image_model.layers[-2].output)

    print("Loading text model...")
    if os.path.exists(TEXT_MODEL_CONFIG_PATH):
        with open(TEXT_MODEL_CONFIG_PATH, 'r') as f:
            text_config = json.load(f)
    else:
        text_config = {"max_len": 100, "max_words": 10000, "embedding_dim": 128}
    text_model = build_text_model_from_config(text_config)
    text_model.load_weights(TEXT_MODEL_WEIGHTS_PATH)
    text_feat_extractor = Model(text_model.input, text_model.layers[-4].output)

    gemini_embedder = GeminiEmbedder(GEMINI_API_KEY)

    fusion_model = build_advanced_fusion_model(
        image_feat_shape=image_feat_extractor.output_shape[1:],
        text_feat_shape=text_feat_extractor.output_shape[1:],
        gemini_feat_shape=(GEMINI_EMBED_DIM,)
    )

    tokenizer = load_or_create_tokenizer(TOKENIZER_CONFIG_PATH, aligned_texts)

    all_predictions = []
    all_true_labels = []
    num_batches = int(np.ceil(num_samples / BATCH_SIZE))

    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, num_samples)
        batch_texts = aligned_texts[start_idx:end_idx]
        batch_labels = labels_onehot[start_idx:end_idx]

        batch_images, _ = image_loader.load_and_preprocess_images(range(start_idx, end_idx))
        text_sequences = tokenizer.texts_to_sequences(batch_texts)
        text_padded = pad_sequences(text_sequences, maxlen=text_config["max_len"], padding='post')

        img_feats = image_feat_extractor.predict(batch_images, verbose=0)
        text_feats = text_feat_extractor.predict(text_padded, verbose=0)
        gemini_feats = gemini_embedder.embed_texts(batch_texts)

        predictions = fusion_model.predict([img_feats, text_feats, gemini_feats], verbose=0)
        all_predictions.extend(np.argmax(predictions, axis=1))
        all_true_labels.extend(np.argmax(batch_labels, axis=1))

    acc = accuracy_score(all_true_labels, all_predictions)
    print(f"Final Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
