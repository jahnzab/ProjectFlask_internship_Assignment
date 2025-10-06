"""
OPTIMIZED Fusion Script - Keeps Trained Text Model + Gemini 2.0 Flash Option
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
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import requests

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

CLASS_NAMES = ["normal", "anxiety", "depression", "stress"]
NUM_CLASSES = len(CLASS_NAMES)
BATCH_SIZE = 32

# Gemini 2.0 Flash (optional)
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
GEMINI_API_URL = "https://generativeai.googleapis.com/v1beta2/models/gemini-2.0-flash:embedText"


# -------------------------
# FAST Image Data Generator
# -------------------------
class FastImageGenerator:
    def __init__(self, directory, target_size=(224, 224), batch_size=32):
        self.directory = directory
        self.target_size = target_size
        self.batch_size = batch_size
        self.datagen = ImageDataGenerator(rescale=1./255)

        self.generator = self.datagen.flow_from_directory(
            directory,
            target_size=target_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )

        self.num_samples = self.generator.n
        self.num_classes = self.generator.num_classes
        self.class_indices = self.generator.class_indices
        self.class_names = list(self.class_indices.keys())

    def get_all_batches(self):
        self.generator.reset()
        num_batches = int(np.ceil(self.num_samples / self.batch_size))
        for _ in range(num_batches):
            images, labels = next(self.generator)
            yield images, labels


# -------------------------
# Load Text Model
# -------------------------
def build_text_model(config):
    inputs = Input(shape=(config["max_len"],), name="text_input")
    x = Embedding(config["max_words"], config["embedding_dim"], mask_zero=True)(inputs)
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

    outputs = Dense(config["num_classes"], activation='softmax')(x)
    return Model(inputs=inputs, outputs=outputs, name="text_model")


def load_tokenizer(path):
    with open(path, 'r') as f:
        cfg = json.load(f)
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=cfg.get('num_words', 10000), oov_token="<OOV>")
    tokenizer.word_index = cfg['word_index']
    return tokenizer


# -------------------------
# Gemini Embedder (optional)
# -------------------------
def embed_text_with_gemini(text_list):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AIzaSyARUo4nMnsQp0c9PiDI_ueAS0Xa9ZgL3Is}"
    }
    data = {
        "instances": [{"content": text} for text in text_list]
    }
    response = requests.post(GEMINI_API_URL, headers=headers, json=data)
    if response.status_code != 200:
        print("Gemini API Error:", response.text)
        return None
    result = response.json()
    return np.array([inst["embedding"] for inst in result.get("predictions", [])])


# -------------------------
# Fusion Model Builder
# -------------------------
def build_fusion_model(image_feat_shape, text_feat_shape, num_classes=4, fusion_dim=256):
    img_input = Input(shape=image_feat_shape, name="image_input")
    txt_input = Input(shape=text_feat_shape, name="text_input")

    img_proj = Dense(fusion_dim, activation='relu')(img_input)
    txt_proj = Dense(fusion_dim, activation='relu')(txt_input)

    # Cross-modal attention with Lambda layers
    sim = Lambda(lambda x: tf.matmul(x[0], x[1], transpose_b=True), name="sim_matmul")([img_proj, txt_proj])
    img_att = Lambda(lambda x: tf.nn.softmax(x, axis=-1), name="img_attention")(sim)
    img_enh = Lambda(lambda x: tf.matmul(x[0], x[1]), name="img_enhancement")([img_att, txt_proj])

    txt_att = Lambda(lambda x: tf.nn.softmax(tf.transpose(x), axis=-1), name="txt_attention")(sim)
    txt_enh = Lambda(lambda x: tf.matmul(x[0], x[1]), name="txt_enhancement")([txt_att, img_proj])

    fused = Concatenate(name="fusion_concat")([
        img_proj, img_enh,
        txt_proj, txt_enh,
        Multiply()([img_proj, txt_proj])
    ])

    x = Dense(512, activation='relu')(fused)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=[img_input, txt_input], outputs=outputs, name="fusion_model")
    model.compile(optimizer=keras.optimizers.Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# -------------------------
# MAIN
# -------------------------
def main():
    print("="*70)
    print("OPTIMIZED MULTIMODAL FUSION - WITH TRAINED TEXT MODEL")
    print("="*70)

    print("\nLoading text data...")
    df = pd.read_csv(TEXT_CSV_PATH)
    texts = df['text'].tolist()
    labels = df['label'].tolist()

    le = LabelEncoder()
    le.fit(CLASS_NAMES)
    labels_encoded = le.transform(labels)
    labels_onehot = keras.utils.to_categorical(labels_encoded, num_classes=NUM_CLASSES)

    print(f"✓ Loaded {len(texts)} text samples")

    print("\nCreating image generators...")
    test_gen = FastImageGenerator(TEST_IMG_DIR, batch_size=BATCH_SIZE)

    print("\nLoading text model...")
    with open(TEXT_MODEL_CONFIG_PATH, 'r') as f:
        text_config = json.load(f)
    text_model = build_text_model(text_config)
    text_model.load_weights(TEXT_MODEL_WEIGHTS_PATH)
    text_feat_extractor = Model(inputs=text_model.input, outputs=text_model.layers[-2].output)
    print(f"✓ Text feature shape: {text_feat_extractor.output_shape}")

    print("\nLoading image model...")
    raw_image_model = load_model(IMAGE_MODEL_PATH, compile=False)
    image_feat_extractor = Model(inputs=raw_image_model.input, outputs=raw_image_model.layers[-2].output)
    print(f"✓ Image feature shape: {image_feat_extractor.output_shape}")

    print("\nBuilding fusion model...")
    fusion_model = build_fusion_model(
        image_feat_shape=image_feat_extractor.output_shape[1:],
        text_feat_shape=text_feat_extractor.output_shape[1:],
        num_classes=NUM_CLASSES
    )
    fusion_model.summary()

    print("\nLoading tokenizer...")
    tokenizer = load_tokenizer(TOKENIZER_CONFIG_PATH)

    print("\nEvaluating...")
    all_predictions = []
    all_true_labels = []
    batch_idx = 0
    for img_batch, img_labels_batch in test_gen.get_all_batches():
      start_idx = batch_idx * BATCH_SIZE
      end_idx = start_idx + len(img_batch)

      text_batch = texts[start_idx:end_idx]

      # Fixed tokenization
      text_seq = pad_sequences(
        tokenizer.texts_to_sequences(text_batch),
        maxlen=text_config["max_len"],
        padding='post'
      )

      text_feats = text_feat_extractor.predict(text_seq)
      img_feats = image_feat_extractor.predict(img_batch)

      preds = fusion_model.predict([img_feats, text_feats], verbose=0)

      all_predictions.extend(np.argmax(preds, axis=1))
      all_true_labels.extend(np.argmax(labels_onehot[start_idx:end_idx], axis=1))
      batch_idx += 1

    acc = accuracy_score(all_true_labels, all_predictions)
    print(f"\n✓ Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_true_labels, all_predictions, target_names=CLASS_NAMES))

    cm = confusion_matrix(all_true_labels, all_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f'Fusion Model Confusion Matrix (Acc: {acc:.4f})')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('fusion_confusion_matrix.png', dpi=200)
    plt.close()

    print("\nSaving fusion model...")
    fusion_model.save('fusion_model_with_text.h5')
    print("\n✓ DONE!")


if __name__ == "__main__":
    main()
