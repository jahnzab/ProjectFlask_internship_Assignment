"""
Multimodal Psychodermatological Disorder Detection
Part 2: Text Model Training (Mental Health Data)
Author: Internship Project
Date: October 2025
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except:
    pass

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

class TextModelTrainer:
    def __init__(self, max_words=10000, max_len=200, embedding_dim=128):
        self.max_words = max_words
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.tokenizer = None
        self.model = None
        self.history = None
        self.label_mapping = {}
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def load_data(self, data_path):
        """
        Load multiple CSVs with survey questions and combine them.
        Automatically detects label column and combines all question columns into a single text column.
        """
        csv_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.csv')]
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {data_path}")

        dfs = []
        for f in csv_files:
            df = pd.read_csv(f)

            # Detect label column
            label_col_candidates = [col for col in df.columns if 'Label' in col or 'label' in col]
            if not label_col_candidates:
                raise ValueError(f"No label column found in {f}")
            label_col = label_col_candidates[0]

            # Exclude demographic columns (common columns) to identify question columns
            demographic_cols = ['Age','Gender','University','Department','Academic Year','Current CGPA','Did you receive a waiver or scholarship at your university?']
            question_cols = [col for col in df.columns if col not in demographic_cols + [label_col]]

            # Combine question columns into single text
            df['text'] = df[question_cols].astype(str).agg(' '.join, axis=1)
            df['label'] = df[label_col]

            dfs.append(df[['text','label']])

        df = pd.concat(dfs, ignore_index=True)
        print(f"Loaded {len(df)} samples from {len(csv_files)} CSV files")
        print(f"Class distribution:\n{df['label'].value_counts()}")

        return df

    def preprocess_text(self, text):
        # Lowercase
        text = text.lower()
        # Remove URLs, emails, mentions, hashtags
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'@\w+|#\w+', '', text)
        # Remove special characters/numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        # Tokenize and lemmatize
        words = [self.lemmatizer.lemmatize(w) for w in text.split() if w not in self.stop_words and len(w) > 2]
        return ' '.join(words)

    def prepare_data(self, df):
        print("\nPreprocessing text data...")
        df['processed_text'] = df['text'].apply(self.preprocess_text)
        df = df[df['processed_text'].str.len() > 0]

        # Encode labels
        unique_labels = sorted(df['label'].unique())
        self.label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        df['encoded_label'] = df['label'].map(self.label_mapping)

        print(f"Label mapping: {self.label_mapping}")

        # Tokenize and pad
        self.tokenizer = Tokenizer(num_words=self.max_words, oov_token='<OOV>')
        self.tokenizer.fit_on_texts(df['processed_text'])
        sequences = self.tokenizer.texts_to_sequences(df['processed_text'])
        X = pad_sequences(sequences, maxlen=self.max_len, padding='post', truncating='post')
        y = keras.utils.to_categorical(df['encoded_label'], num_classes=len(self.label_mapping))

        print(f"\nPreprocessing complete! Vocabulary size: {len(self.tokenizer.word_index)}")
        print(f"Sequence shape: {X.shape}, Labels shape: {y.shape}")
        return X, y

    def build_model(self, num_classes):
        print("\nBuilding text model...")
        inputs = keras.Input(shape=(self.max_len,))
        x = layers.Embedding(self.max_words, self.embedding_dim, mask_zero=True)(inputs)
        x = layers.SpatialDropout1D(0.3)(x)
        x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3))(x)
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.3))(x)

        # Attention
        attn = layers.Dense(1, activation='tanh')(x)
        attn = layers.Flatten()(attn)
        attn = layers.Activation('softmax')(attn)
        attn = layers.RepeatVector(128)(attn)
        attn = layers.Permute([2,1])(attn)
        x = layers.Multiply()([x, attn])
        x = layers.Lambda(lambda xin: tf.reduce_sum(xin, axis=1))(x)

        # Dense layers
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.3)(x)

        outputs = layers.Dense(num_classes, activation='softmax')(x)
        self.model = keras.Model(inputs, outputs)
        self.model.compile(optimizer=keras.optimizers.Adam(1e-3),
                           loss='categorical_crossentropy',
                           metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()])
        print(f"Model built successfully! Total parameters: {self.model.count_params():,}")
        return self.model

    def train(self, X_train, y_train, X_val, y_val, epochs=20):
        print("\nTraining text model...")
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
            ModelCheckpoint('best_text_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1)
        ]
        self.history = self.model.fit(
            X_train, y_train, validation_data=(X_val, y_val),
            epochs=epochs, batch_size=64, callbacks=callbacks, verbose=1
        )
        return self.history

    def evaluate(self, X_test, y_test):
        print("\nEvaluating model...")
        y_pred_probs = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = np.argmax(y_test, axis=1)
        class_names = [k for k,v in sorted(self.label_mapping.items(), key=lambda x: x[1])]
        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        cm = confusion_matrix(y_true, y_pred)

        print(f"Test Accuracy: {accuracy:.4f}")
        print(classification_report(y_true, y_pred, target_names=class_names))

        plt.figure(figsize=(10,8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix - Text Model')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('text_model_confusion_matrix.png', dpi=300)
        plt.close()
        return {'accuracy': accuracy, 'report': report, 'confusion_matrix': cm.tolist()}

    def plot_training_history(self):
        fig, axes = plt.subplots(2,2, figsize=(15,10))
        axes[0,0].plot(self.history.history['accuracy'], label='Train Acc')
        axes[0,0].plot(self.history.history['val_accuracy'], label='Val Acc')
        axes[0,0].set_title('Accuracy'); axes[0,0].legend(); axes[0,0].grid(True)

        axes[0,1].plot(self.history.history['loss'], label='Train Loss')
        axes[0,1].plot(self.history.history['val_loss'], label='Val Loss')
        axes[0,1].set_title('Loss'); axes[0,1].legend(); axes[0,1].grid(True)

        axes[1,0].plot(self.history.history['precision'], label='Train Precision'); axes[1,0].set_title('Precision'); axes[1,0].legend();

    
    def save_model_and_config(self, output_dir='models'):
        """
        Save model, tokenizer, and configuration
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model
        self.model.save(os.path.join(output_dir, 'text_model.h5'))
        
        # Save tokenizer
        tokenizer_config = {
            'word_index': self.tokenizer.word_index,
            'index_word': {v: k for k, v in self.tokenizer.word_index.items()},
            'num_words': self.max_words
        }
        
        with open(os.path.join(output_dir, 'tokenizer_config.json'), 'w') as f:
            json.dump(tokenizer_config, f, indent=4)
        
        # Save configuration
        config = {
            'label_mapping': self.label_mapping,
            'max_words': self.max_words,
            'max_len': self.max_len,
            'embedding_dim': self.embedding_dim,
            'num_classes': len(self.label_mapping)
        }
        
        with open(os.path.join(output_dir, 'text_model_config.json'), 'w') as f:
            json.dump(config, f, indent=4)
        
        print(f"Model saved to {output_dir}/")


def main():
    """
    Main training pipeline for text model
    """
    print("="*70)
    print("MULTIMODAL PSYCHODERMATOLOGICAL DISORDER DETECTION")
    print("Text Model Training Pipeline (Mental Health)")
    print("="*70)
    
    # Configuration
    CSV_PATH = '/content/ProjectFlask_internship_Assignment/mental_health_dataset'
    # Update with your dataset path
    MAX_WORDS = 10000
    MAX_LEN = 200
    EMBEDDING_DIM = 128
    EPOCHS = 20
    
    # Check GPU availability
    print(f"\nGPU Available: {tf.config.list_physical_devices('GPU')}")
    
    # Initialize trainer
    trainer = TextModelTrainer(
        max_words=MAX_WORDS,
        max_len=MAX_LEN,
        embedding_dim=EMBEDDING_DIM
    )
    
    # Load data
   
    
    df = trainer.load_data(CSV_PATH)
    
    # Prepare data
    X, y = trainer.prepare_data(df)
    
    # Split data: 70% train, 15% validation, 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"\nDataset split:")
    print(f"Train: {len(X_train)} samples")
    print(f"Validation: {len(X_val)} samples")
    print(f"Test: {len(X_test)} samples")
    
    # Build model
    trainer.build_model(num_classes=len(trainer.label_mapping))
    
    # Train model
    trainer.train(X_train, y_train, X_val, y_val, epochs=EPOCHS)
    
    # Plot training history
    trainer.plot_training_history()
    
    # Evaluate model
    metrics = trainer.evaluate(X_test, y_test)
    
    # Save model
    trainer.save_model_and_config()
    
    print("\n" + "="*70)
    print("Training completed successfully!")
    print("="*70)


if __name__ == "__main__":
    main()
