import os
import json
import math
import numpy as np
import pandas as pd
import requests
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Layer, Dense, Concatenate, Dropout, Input, Flatten, Activation, RepeatVector, Permute, Multiply, Lambda, BatchNormalization
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ========================
# CONFIG
# ========================

IMAGE_MODEL_PATH = "/content/best_image_model_phase1.h5"
TEXT_MODEL_WEIGHTS_PATH = "/content/best_text_model.h5"
TEXT_MODEL_CONFIG_PATH = "/content/text_model_config.json"
TOKENIZER_CONFIG_PATH = "/content/tokenizer_config.json"
TEXT_CSV_PATH = "/content/mental_health_processed.csv"

TRAIN_IMG_DIR = "/content/ProjectFlask_internship_Assignment/dermatology_dataset/train"
TEST_IMG_DIR = "/content/ProjectFlask_internship_Assignment/dermatology_dataset/test"

CLASS_NAMES = ["normal", "anxiety", "depression", "stress"]
NUM_CLASSES = len(CLASS_NAMES)
MAX_SEQ_LEN = 200
MAX_WORDS = 10000
EMBEDDING_DIM = 128

# Gemini API config
GEMINI_API_URL = "https://generativeai.googleapis.com/v1beta2/models/text-bison-001:generate"
GEMINI_API_KEY = "AIzaSyARUo4nMnsQp0c9PiDI_ueAS0Xa9ZgL3Is"  # Must be OAuth token from Google Cloud Console

# ========================
# HELPER FUNCTIONS
# ========================

def load_images_from_folder(folder_path, target_size=(224,224)):
    X_imgs = []
    sample_ids = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            sample_id = os.path.splitext(filename)[0]
            img_path = os.path.join(folder_path, filename)
            img = load_img(img_path, target_size=target_size)
            img_array = img_to_array(img) / 255.0
            X_imgs.append(img_array)
            sample_ids.append(sample_id)
    return np.array(X_imgs, dtype=np.float32), sample_ids

def encode_labels(labels):
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    return keras.utils.to_categorical(labels_encoded, num_classes=NUM_CLASSES), le

# ========================
# TEXT MODEL BUILDER
# ========================

def build_text_model(num_classes, max_words, max_len, embedding_dim):
    inputs = Input(shape=(max_len,))
    x = tf.keras.layers.Embedding(max_words, embedding_dim, mask_zero=True)(inputs)
    x = tf.keras.layers.SpatialDropout1D(0.3)(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3))(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.3))(x)

    attn = Dense(1, activation='tanh', name="attention_dense")(x)
    attn = Flatten(name="attention_flatten")(attn)
    attn = Activation('softmax', name="attention_activation")(attn)
    attn = RepeatVector(128, name="attention_repeat")(attn)
    attn = Permute([2, 1], name="attention_permute")(attn)
    x = Multiply(name="attention_multiply")([x, attn])
    x = Lambda(lambda xin: tf.reduce_sum(xin, axis=1), name="attention_lambda")(x)

    x = Dense(256, activation='relu', name="text_dense_256")(x)
    x = BatchNormalization(name="text_bn_256")(x)
    x = Dropout(0.5, name="text_dropout_256")(x)
    x = Dense(128, activation='relu', name="text_dense_128")(x)
    x = BatchNormalization(name="text_bn_128")(x)
    x = Dropout(0.4, name="text_dropout_128")(x)
    x = Dense(64, activation='relu', name="text_dense_64")(x)
    x = Dropout(0.3, name="text_dropout_64")(x)

    outputs = Dense(num_classes, activation='softmax', name="text_output")(x)
    return Model(inputs, outputs)

# ========================
# MULTIMODAL FUSION CLASS
# ========================

class MultimodalFusionTrainer:
    def __init__(self, image_model_path, text_model_weights_path, text_model_config_path, tokenizer_config_path, gemini_api_key):
        print("🔹 Loading image model...")
        self.image_model = load_model(image_model_path, compile=False)

        print("🔹 Rebuilding text model architecture...")
        with open(text_model_config_path) as f:
            config = json.load(f)

        self.text_model = build_text_model(config["num_classes"], config["max_words"], config["max_len"], config["embedding_dim"])
        self.text_model.load_weights(text_model_weights_path)
        print("✅ Text model rebuilt and weights loaded")

        with open(tokenizer_config_path) as f:
            tokenizer_config = json.load(f)
        self.tokenizer = Tokenizer(num_words=tokenizer_config["num_words"], oov_token="<OOV>")
        self.tokenizer.word_index = tokenizer_config["word_index"]

        self.fusion_model = None
        self.gemini_api_key = gemini_api_key

    def call_gemini(self, prompt: str):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.gemini_api_key}"
        }
        data = {
            "prompt": {"text": prompt},
            "temperature": 0.7
        }
        try:
            resp = requests.post(GEMINI_API_URL, headers=headers, json=data)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print("❌ Gemini API error:", e)
            return None

    def build_fusion_model(self, num_classes, fusion_type="gemini"):
        print(f"🔹 Building fusion model with fusion_type='{fusion_type}'")
        self.image_model.trainable = False
        self.text_model.trainable = False

        image_input = self.image_model.input
        image_features = self.image_model.output
        text_input = self.text_model.input
        text_features = self.text_model.output

        if fusion_type.lower() == "concat":
            combined = Concatenate(name="concat_features")([image_features, text_features])
        elif fusion_type.lower() == "gemini":
            img_aug = Dense(128, activation='relu', name="gemini_img_dense")(image_features)
            txt_aug = Dense(128, activation='relu', name="gemini_txt_dense")(text_features)
            combined = Concatenate(name="gemini_concat")([img_aug, txt_aug])
            combined = Dropout(0.3, name="gemini_dropout")(combined)
        else:
            combined = Concatenate(name="default_concat")([image_features, text_features])

        output = Dense(num_classes, activation='softmax', name="fusion_output")(combined)
        self.fusion_model = Model(inputs=[image_input, text_input], outputs=output)
        self.fusion_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print("✅ Fusion model built successfully")

    def evaluate(self, X_img, X_text, y_true, class_names):
        if self.fusion_model is None:
            raise ValueError("Fusion model is not built yet.")

        if len(X_img) == 0 or len(X_text) == 0 or len(y_true) == 0:
            raise ValueError("❌ Evaluation dataset is empty. Check dataset paths and filtering logic.")

        self.tokenizer.fit_on_texts(X_text)
        X_seq = self.tokenizer.texts_to_sequences(X_text)
        X_pad = pad_sequences(X_seq, maxlen=MAX_SEQ_LEN, padding='post')

        preds = self.fusion_model.predict([X_img, X_pad])
        y_pred = np.argmax(preds, axis=1)
        y_true_labels = np.argmax(y_true, axis=1)

        acc = accuracy_score(y_true_labels, y_pred)
        report = classification_report(y_true_labels, y_pred, target_names=class_names)
        cm = confusion_matrix(y_true_labels, y_pred)

        return {"accuracy": acc, "classification_report": report, "confusion_matrix": cm}

    def save_model(self, output_dir="fusion_model_deploy"):
        if self.fusion_model is None:
            raise ValueError("Fusion model not built.")
        os.makedirs(output_dir, exist_ok=True)
        self.fusion_model.save(os.path.join(output_dir, "fusion_model.h5"))
        print(f"✅ Fusion model saved to {output_dir}")

# ========================
# MAIN
# ========================

def main():
    print("🔹 Loading train and test datasets...")
    X_train_img, train_ids = load_images_from_folder(TRAIN_IMG_DIR)
    X_test_img, test_ids = load_images_from_folder(TEST_IMG_DIR)

    df_text = pd.read_csv(TEXT_CSV_PATH)
    if 'sample_id' not in df_text.columns:
        df_text['sample_id'] = df_text.index.astype(str)

    df_train_text = df_text[df_text['sample_id'].isin(train_ids)].reset_index(drop=True)
    df_test_text = df_text[df_text['sample_id'].isin(test_ids)].reset_index(drop=True)

    X_train_text = df_train_text['text'].values
    X_test_text = df_test_text['text'].values
    y_train, le = encode_labels(df_train_text['label'])
    y_test, _ = encode_labels(df_test_text['label'])

    print(f"✅ Loaded {len(X_train_img)} train and {len(X_test_img)} test samples.")

    trainer = MultimodalFusionTrainer(
        IMAGE_MODEL_PATH,
        TEXT_MODEL_WEIGHTS_PATH,
        TEXT_MODEL_CONFIG_PATH,
        TOKENIZER_CONFIG_PATH,
        GEMINI_API_KEY
    )

    trainer.build_fusion_model(NUM_CLASSES, fusion_type="gemini")

    # Gemini API test call
    gemini_response = trainer.call_gemini("Describe the fusion model architecture in detail.")
    print("\n💡 Gemini Response:", gemini_response)

    results = trainer.evaluate(X_test_img, X_test_text, y_test, CLASS_NAMES)
    print("\n🔹 Test Accuracy:", results['accuracy'])
    print("🔹 Classification Report:\n", results['classification_report'])
    print("🔹 Confusion Matrix:\n", results['confusion_matrix'])

    trainer.save_model(output_dir="fusion_model_deploy")


if __name__ == "__main__":
    main()
