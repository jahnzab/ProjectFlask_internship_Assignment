"""
Multimodal Psychodermatological Disorder Detection
Part 3: Fusion Model Training with Gemini Augmentation
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
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import json
import google.generativeai as genai
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)


class GeminiAugmenter:
    """
    Gemini-based data augmentation for multimodal learning
    """
    def __init__(self, api_key):
        """
        Initialize Gemini API
        
        Args:
            api_key: Google Gemini API key
        """
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')
    
    def augment_text(self, text, label):
        """
        Generate augmented text variations using Gemini
        """
        prompt = f"""Given the following text about {label}, generate a paraphrased version that maintains the same emotional tone and meaning but uses different words:

Text: {text}

Generate only the paraphrased text, nothing else."""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"Gemini augmentation error: {e}")
            return text
    
    def generate_cross_modal_features(self, image_features, text_features):
        """
        Use Gemini to generate enhanced cross-modal representations
        This simulates attention-based feature fusion
        """
        # In practice, Gemini can help generate contextual relationships
        # Here we implement a weighted fusion strategy
        
        # Normalize features
        image_norm = tf.nn.l2_normalize(image_features, axis=-1)
        text_norm = tf.nn.l2_normalize(text_features, axis=-1)
        
        # Calculate similarity-based weights
        similarity = tf.matmul(image_norm, text_norm, transpose_b=True)
        
        # Apply attention weights
        image_attention = tf.nn.softmax(similarity, axis=-1)
        text_attention = tf.nn.softmax(tf.transpose(similarity), axis=-1)
        
        # Enhanced features
        enhanced_image = tf.matmul(image_attention, text_features)
        enhanced_text = tf.matmul(text_attention, image_features)
        
        return enhanced_image, enhanced_text


class MultimodalFusionTrainer:
    """
    Multimodal fusion model trainer
    """
    def __init__(self, image_model_path, text_model_path, gemini_api_key=None):
        """
        Initialize fusion trainer
        
        Args:
            image_model_path: Path to trained image model
            text_model_path: Path to trained text model
            gemini_api_key: Gemini API key for augmentation
        """
        self.image_model_path = image_model_path
        self.text_model_path = text_model_path
        self.fusion_model = None
        self.history = None
        
        # Load pre-trained models
        self.image_model = keras.models.load_model(image_model_path)
        self.text_model = keras.models.load_model(text_model_path)
        
        # Initialize Gemini augmenter
        self.gemini_augmenter = None
        if gemini_api_key:
            self.gemini_augmenter = GeminiAugmenter(gemini_api_key)
        
        print("Pre-trained models loaded successfully!")
    
    def build_fusion_model(self, num_classes, fusion_type='concat'):
        """
        Build multimodal fusion model
        
        Args:
            num_classes: Number of output classes
            fusion_type: 'concat', 'attention', or 'gemini'
        """
        print(f"\nBuilding {fusion_type} fusion model...")
        
        # Freeze pre-trained models initially
        for layer in self.image_model.layers:
            layer.trainable = False
        for layer in self.text_model.layers:
            layer.trainable = False
        
        # Extract feature extraction layers (remove final classification layer)
        image_base = keras.Model(
            inputs=self.image_model.input,
            outputs=self.image_model.layers[-4].output,  # Before final dense layers
            name='image_feature_extractor'
        )
        
        text_base = keras.Model(
            inputs=self.text_model.input,
            outputs=self.text_model.layers[-4].output,
            name='text_feature_extractor'
        )
        
        # Define inputs
        image_input = keras.Input(shape=self.image_model.input.shape[1:], name='image_input')
        text_input = keras.Input(shape=self.text_model.input.shape[1:], name='text_input')
        
        # Extract features
        image_features = image_base(image_input)
        text_features = text_base(text_input)
        
        # Fusion strategy
        if fusion_type == 'concat':
            # Simple concatenation
            fused = layers.Concatenate()([image_features, text_features])
            
        elif fusion_type == 'attention':
            # Cross-modal attention fusion
            # Image attending to text
            image_query = layers.Dense(128)(image_features)
            text_key = layers.Dense(128)(text_features)
            text_value = layers.Dense(128)(text_features)
            
            attention_scores = tf.matmul(image_query, text_key, transpose_b=True)
            attention_scores = tf.nn.softmax(attention_scores / np.sqrt(128))
            image_attended = tf.matmul(attention_scores, text_value)
            
            # Text attending to image
            text_query = layers.Dense(128)(text_features)
            image_key = layers.Dense(128)(image_features)
            image_value = layers.Dense(128)(image_features)
            
            attention_scores = tf.matmul(text_query, image_key, transpose_b=True)
            attention_scores = tf.nn.softmax(attention_scores / np.sqrt(128))
            text_attended = tf.matmul(attention_scores, image_value)
            
            # Combine
            fused = layers.Concatenate()([
                image_features, image_attended,
                text_features, text_attended
            ])
            
        elif fusion_type == 'gemini':
            # Gemini-augmented fusion
            # Project to same dimension
            image_proj = layers.Dense(256, activation='relu')(image_features)
            text_proj = layers.Dense(256, activation='relu')(text_features)
            
            # Cross-modal enhancement
            similarity = tf.matmul(image_proj, text_proj, transpose_b=True)
            image_attention = tf.nn.softmax(similarity, axis=-1)
            text_attention = tf.nn.softmax(tf.transpose(similarity), axis=-1)
            
            enhanced_image = tf.matmul(image_attention, text_proj)
            enhanced_text = tf.matmul(text_attention, image_proj)
            
            # Multi-scale fusion
            fused = layers.Concatenate()([
                image_features, enhanced_image,
                text_features, enhanced_text,
                image_proj * text_proj  # Element-wise product
            ])
        
        # Fusion layers
        x = layers.Dense(512, activation='relu')(fused)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        # Output layer
        outputs = layers.Dense(num_classes, activation='softmax', name='fusion_output')(x)
        
        # Create fusion model
        self.fusion_model = keras.Model(
            inputs=[image_input, text_input],
            outputs=outputs,
            name=f'multimodal_fusion_{fusion_type}'
        )
        
        # Compile
        self.fusion_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-4),
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
        
        print(f"Fusion model built successfully!")
        print(f"Total parameters: {self.fusion_model.count_params():,}")
        
        return self.fusion_model
    
    def train(self, X_train_img, X_train_text, y_train,
              X_val_img, X_val_text, y_val, epochs=15):
        """
        Train fusion model with two-phase approach
        """
        print("\n" + "="*50)
        print("PHASE 1: Training fusion layer only")
        print("="*50)
        
        # Callbacks
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
        
        checkpoint = ModelCheckpoint(
            'best_fusion_model_phase1.h5',
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
        
        # Phase 1: Train with frozen base models
        history1 = self.fusion_model.fit(
            [X_train_img, X_train_text], y_train,
            validation_data=([X_val_img, X_val_text], y_val),
            epochs=epochs // 2,
            batch_size=32,
            callbacks=[early_stop, checkpoint, reduce_lr],
            verbose=1
        )
        
        print("\n" + "="*50)
        print("PHASE 2: Fine-tuning entire model")
        print("="*50)
        
        # Unfreeze last layers of base models
        for layer in self.fusion_model.layers[2].layers[-10:]:  # Image model
            layer.trainable = True
        for layer in self.fusion_model.layers[3].layers[-10:]:  # Text model
            layer.trainable = True
        
        # Recompile with lower learning rate
        self.fusion_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-5),
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
        
        checkpoint = ModelCheckpoint(
            'best_fusion_model_final.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
        
        # Phase 2: Fine-tune
        history2 = self.fusion_model.fit(
            [X_train_img, X_train_text], y_train,
            validation_data=([X_val_img, X_val_text], y_val),
            epochs=epochs // 2,
            batch_size=32,
            callbacks=[early_stop, checkpoint, reduce_lr],
            verbose=1
        )
        
        # Combine histories
        self.history = {
            'loss': history1.history['loss'] + history2.history['loss'],
            'accuracy': history1.history['accuracy'] + history2.history['accuracy'],
            'val_loss': history1.history['val_loss'] + history2.history['val_loss'],
            'val_accuracy': history1.history['val_accuracy'] + history2.history['val_accuracy']
        }
        
        return self.history
    
    def evaluate(self, X_test_img, X_test_text, y_test, class_names):
        """
        Evaluate fusion model
        """
        print("\nEvaluating fusion model...")
        
        # Predictions
        y_pred_probs = self.fusion_model.predict([X_test_img, X_test_text], verbose=1)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        # Metrics
        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        cm = confusion_matrix(y_true, y_pred)
        
        print(f"\nTest Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=class_names))
        
        # Plot confusion matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn',
                    xticklabels=class_names,
                    yticklabels=class_names)
        plt.title('Confusion Matrix - Multimodal Fusion Model')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('fusion_model_confusion_matrix.png', dpi=300)
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
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy
        axes[0].plot(self.history['accuracy'], label='Train Accuracy', linewidth=2)
        axes[0].plot(self.history['val_accuracy'], label='Val Accuracy', linewidth=2)
        axes[0].set_title('Multimodal Fusion Model Accuracy', fontsize=14)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Loss
        axes[1].plot(self.history['loss'], label='Train Loss', linewidth=2)
        axes[1].plot(self.history['val_loss'], label='Val Loss', linewidth=2)
        axes[1].set_title('Multimodal Fusion Model Loss', fontsize=14)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('fusion_model_training_history.png', dpi=300)
        plt.close()
    
    def save_model(self, output_dir='models'):
        """
        Save fusion model
        """
        os.makedirs(output_dir, exist_ok=True)
        self.fusion_model.save(os.path.join(output_dir, 'fusion_model.h5'))
        print(f"Fusion model saved to {output_dir}/")


def main():
    """
    Main training pipeline for fusion model
    """
    print("="*70)
    print("MULTIMODAL PSYCHODERMATOLOGICAL DISORDER DETECTION")
    print("Fusion Model Training Pipeline")
    print("="*70)
    
    # Configuration
    IMAGE_MODEL_PATH = 'models/image_model.h5'
    TEXT_MODEL_PATH = 'models/text_model.h5'
    GEMINI_API_KEY = 'YOUR_GEMINI_API_KEY'  # Replace with your key
    EPOCHS = 15
    
    # Note: You need to prepare matched image-text pairs
    # This is a template - adjust based on your actual data structure
    
    print("\nPlease ensure you have:")
    print("1. Trained image model at:", IMAGE_MODEL_PATH)
    print("2. Trained text model at:", TEXT_MODEL_PATH)
    print("3. Matched image-text dataset ready")
    
    # Check GPU
    print(f"\nGPU Available: {tf.config.list_physical_devices('GPU')}")
    
    # Initialize trainer
    trainer = MultimodalFusionTrainer(
        image_model_path=IMAGE_MODEL_PATH,
        text_model_path=TEXT_MODEL_PATH,
        gemini_api_key=GEMINI_API_KEY
    )
    
    # Build fusion model (try different fusion types)
    trainer.build_fusion_model(num_classes=4, fusion_type='gemini')
    
    print("\n" + "="*70)
    print("Model architecture ready for training!")
    print("Load your matched datasets and call trainer.train()")
    print("="*70)


if __name__ == "__main__":
    main()