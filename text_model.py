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
        """
        Initialize Text Model Trainer
        
        Args:
            max_words: Maximum vocabulary size
            max_len: Maximum sequence length
            embedding_dim: Dimension of embedding layer
        """
        self.max_words = max_words
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.tokenizer = None
        self.model = None
        self.history = None
        self.label_mapping = {}
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def load_data(self, csv_path):
        """
        Load mental health dataset
        
        Expected CSV format:
        - 'text' column: contains the text data
        - 'label' column: contains the class label (e.g., 'stress', 'anxiety', 'depression', 'normal')
        """
        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        # Check required columns
        if 'text' not in df.columns or 'label' not in df.columns:
            raise ValueError("CSV must contain 'text' and 'label' columns")
        
        print(f"Loaded {len(df)} samples")
        print(f"Class distribution:\n{df['label'].value_counts()}")
        
        return df
    
    def preprocess_text(self, text):
        """
        Advanced text preprocessing
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize
        words = text.split()
        
        # Remove stopwords and lemmatize
        words = [
            self.lemmatizer.lemmatize(word) 
            for word in words 
            if word not in self.stop_words and len(word) > 2
        ]
        
        return ' '.join(words)
    
    def prepare_data(self, df):
        """
        Prepare text data for training
        """
        print("\nPreprocessing text data...")
        
        # Preprocess all texts
        df['processed_text'] = df['text'].apply(self.preprocess_text)
        
        # Remove empty texts
        df = df[df['processed_text'].str.len() > 0]
        
        # Encode labels
        unique_labels = sorted(df['label'].unique())
        self.label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        df['encoded_label'] = df['label'].map(self.label_mapping)
        
        print(f"Label mapping: {self.label_mapping}")
        
        # Tokenize texts
        self.tokenizer = Tokenizer(num_words=self.max_words, oov_token='<OOV>')
        self.tokenizer.fit_on_texts(df['processed_text'])
        
        # Convert to sequences
        sequences = self.tokenizer.texts_to_sequences(df['processed_text'])
        
        # Pad sequences
        X = pad_sequences(sequences, maxlen=self.max_len, padding='post', truncating='post')
        
        # One-hot encode labels
        y = keras.utils.to_categorical(df['encoded_label'], num_classes=len(self.label_mapping))
        
        print(f"\nPreprocessing complete!")
        print(f"Vocabulary size: {len(self.tokenizer.word_index)}")
        print(f"Sequence shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
        
        return X, y
    
    def build_model(self, num_classes):
        """
        Build text classification model with LSTM and Attention
        """
        print("\nBuilding text model...")
        
        # Input layer
        inputs = keras.Input(shape=(self.max_len,), name='text_input')
        
        # Embedding layer
        x = layers.Embedding(
            input_dim=self.max_words,
            output_dim=self.embedding_dim,
            input_length=self.max_len,
            mask_zero=True
        )(inputs)
        
        # Spatial dropout for regularization
        x = layers.SpatialDropout1D(0.3)(x)
        
        # Bidirectional LSTM layers
        x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3))(x)
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.3))(x)
        
        # Attention mechanism
        attention = layers.Dense(1, activation='tanh')(x)
        attention = layers.Flatten()(attention)
        attention = layers.Activation('softmax')(attention)
        attention = layers.RepeatVector(128)(attention)
        attention = layers.Permute([2, 1])(attention)
        
        # Apply attention
        x = layers.Multiply()([x, attention])
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
        
        # Output layer
        outputs = layers.Dense(num_classes, activation='softmax', name='text_output')(x)
        
        # Create model
        self.model = keras.Model(inputs, outputs, name='text_model_lstm_attention')
        
        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
        
        print(f"Model built successfully!")
        print(f"Total parameters: {self.model.count_params():,}")
        
        return self.model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=20):
        """
        Train the text model
        """
        print("\n" + "="*50)
        print("Training Text Model")
        print("="*50)
        
        # Callbacks
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
        
        checkpoint = ModelCheckpoint(
            'best_text_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=64,
            callbacks=[early_stop, checkpoint, reduce_lr],
            verbose=1
        )
        
        return self.history
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model and generate metrics
        """
        print("\nEvaluating model...")
        
        # Predictions
        y_pred_probs = self.model.predict(X_test, verbose=1)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        # Get class names
        class_names = [k for k, v in sorted(self.label_mapping.items(), key=lambda x: x[1])]
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        cm = confusion_matrix(y_true, y_pred)
        
        print(f"\nTest Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=class_names))
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                    xticklabels=class_names,
                    yticklabels=class_names)
        plt.title('Confusion Matrix - Text Model')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('text_model_confusion_matrix.png', dpi=300)
        plt.close()
        
        return {
            'accuracy': accuracy,
            'report': report,
            'confusion_matrix': cm.tolist()
        }
    
    def plot_training_history(self):
        """
        Plot training curves
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Train Accuracy')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Val Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Train Loss')
        axes[0, 1].plot(self.history.history['val_loss'], label='Val Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision
        axes[1, 0].plot(self.history.history['precision'], label='Train Precision')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Recall
        axes[1, 1].plot(self.history.history['recall'], label='Train Recall')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('text_model_training_history.png', dpi=300)
        plt.close()
    
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
    CSV_PATH = 'mental_health_dataset'
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
