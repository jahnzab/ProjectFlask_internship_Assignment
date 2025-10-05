"""
Multimodal Psychodermatological Disorder Detection
Part 1: Image Model Training with Transfer Learning
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
from tensorflow.keras.applications import EfficientNetB0, ResNet50
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from tqdm import tqdm
import json
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class ImageModelTrainer:
    def __init__(self, data_dir, img_size=(224, 224), batch_size=32):
        """
        Initialize the Image Model Trainer
        
        Args:
            data_dir: Path to image dataset directory
            img_size: Target image size (height, width)
            batch_size: Batch size for training
        """
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.model = None
        self.history = None
        self.class_names = []
        
    def load_and_preprocess_data(self):
        """
        Load images and labels with advanced preprocessing
        """
        print("Loading and preprocessing images...")
        
        images = []
        labels = []
        
        # Assuming directory structure: data_dir/class_name/images
        for class_idx, class_name in enumerate(sorted(os.listdir(self.data_dir))):
            class_path = os.path.join(self.data_dir, class_name)
            if not os.path.isdir(class_path):
                continue
                
            self.class_names.append(class_name)
            
            for img_name in tqdm(os.listdir(class_path), desc=f"Loading {class_name}"):
                img_path = os.path.join(class_path, img_name)
                
                try:
                    # Read image
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    
                    # Convert BGR to RGB
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Resize
                    img = cv2.resize(img, self.img_size)
                    
                    # Advanced preprocessing
                    # 1. Contrast Limited Adaptive Histogram Equalization (CLAHE)
                    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                    lab[:,:,0] = clahe.apply(lab[:,:,0])
                    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
                    
                    # 2. Denoise
                    img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
                    
                    # Normalize to [0, 1]
                    img = img.astype(np.float32) / 255.0
                    
                    images.append(img)
                    labels.append(class_idx)
                    
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
                    continue
        
        print(f"Loaded {len(images)} images from {len(self.class_names)} classes")
        
        # Convert to numpy arrays
        X = np.array(images)
        y = np.array(labels)
        
        # One-hot encode labels
        y = keras.utils.to_categorical(y, num_classes=len(self.class_names))
        
        return X, y
    
    def create_data_augmentation(self):
        """
        Create advanced data augmentation pipeline
        """
        return ImageDataGenerator(
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=False,
            fill_mode='reflect',
            brightness_range=[0.8, 1.2]
        )
    
    def build_model(self, base_model_name='EfficientNetB0', num_classes=None):
        """
        Build CNN model with transfer learning
        
        Args:
            base_model_name: 'EfficientNetB0' or 'ResNet50'
            num_classes: Number of output classes
        """
        if num_classes is None:
            num_classes = len(self.class_names)
        
        print(f"Building model with {base_model_name}...")
        
        # Load pre-trained base model
        if base_model_name == 'EfficientNetB0':
            base_model = EfficientNetB0(
                include_top=False,
                weights='imagenet',
                input_shape=(*self.img_size, 3)
            )
        elif base_model_name == 'ResNet50':
            base_model = ResNet50(
                include_top=False,
                weights='imagenet',
                input_shape=(*self.img_size, 3)
            )
        else:
            raise ValueError(f"Unknown base model: {base_model_name}")
        
        # Freeze base model initially
        base_model.trainable = False
        
        # Build custom head
        inputs = keras.Input(shape=(*self.img_size, 3))
        
        # Data augmentation layer (applies only during training)
        x = layers.RandomFlip("horizontal")(inputs)
        x = layers.RandomRotation(0.1)(x)
        x = layers.RandomZoom(0.1)(x)
        
        # Base model
        x = base_model(x, training=False)
        
        # Global pooling
        x = layers.GlobalAveragePooling2D()(x)
        
        # Dense layers with dropout
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        # Output layer
        outputs = layers.Dense(num_classes, activation='softmax', name='image_output')(x)
        
        # Create model
        self.model = keras.Model(inputs, outputs, name=f'image_model_{base_model_name}')
        
        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
        
        print(f"Model built successfully!")
        print(f"Total parameters: {self.model.count_params():,}")
        
        return self.model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=30):
        """
        Train the model with two-phase approach
        """
        print("\n" + "="*50)
        print("PHASE 1: Training with frozen base model")
        print("="*50)
        
        # Callbacks
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
        
        checkpoint = ModelCheckpoint(
            'best_image_model_phase1.h5',
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
        
        # Phase 1: Train with frozen base
        history1 = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs // 2,
            batch_size=self.batch_size,
            callbacks=[early_stop, checkpoint, reduce_lr],
            verbose=1
        )
        
        print("\n" + "="*50)
        print("PHASE 2: Fine-tuning with unfrozen base model")
        print("="*50)
        
        # Unfreeze base model for fine-tuning
        base_model = self.model.layers[3]  # Get base model layer
        base_model.trainable = True
        
        # Freeze first 80% of layers
        freeze_until = int(len(base_model.layers) * 0.8)
        for layer in base_model.layers[:freeze_until]:
            layer.trainable = False
        
        # Recompile with lower learning rate
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-5),
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
        
        # Update checkpoint path
        checkpoint = ModelCheckpoint(
            'best_image_model_final.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
        
        # Phase 2: Fine-tune
        history2 = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs // 2,
            batch_size=self.batch_size,
            callbacks=[early_stop, checkpoint, reduce_lr],
            verbose=1
        )
        
        # Combine histories
        self.history = {
            'loss': history1.history['loss'] + history2.history['loss'],
            'accuracy': history1.history['accuracy'] + history2.history['accuracy'],
            'val_loss': history1.history['val_loss'] + history2.history['val_loss'],
            'val_accuracy': history1.history['val_accuracy'] + history2.history['val_accuracy'],
            'precision': history1.history['precision'] + history2.history['precision'],
            'recall': history1.history['recall'] + history2.history['recall']
        }
        
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
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, target_names=self.class_names, output_dict=True)
        cm = confusion_matrix(y_true, y_pred)
        
        print(f"\nTest Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=self.class_names))
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names)
        plt.title('Confusion Matrix - Image Model')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('image_model_confusion_matrix.png', dpi=300)
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
        axes[0, 0].plot(self.history['accuracy'], label='Train Accuracy')
        axes[0, 0].plot(self.history['val_accuracy'], label='Val Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(self.history['loss'], label='Train Loss')
        axes[0, 1].plot(self.history['val_loss'], label='Val Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision
        axes[1, 0].plot(self.history['precision'], label='Train Precision')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Recall
        axes[1, 1].plot(self.history['recall'], label='Train Recall')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('image_model_training_history.png', dpi=300)
        plt.close()
    
    def save_model_and_config(self, output_dir='models'):
        """
        Save model and configuration
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model
        self.model.save(os.path.join(output_dir, 'image_model.h5'))
        
        # Save configuration
        config = {
            'class_names': self.class_names,
            'img_size': self.img_size,
            'num_classes': len(self.class_names)
        }
        
        with open(os.path.join(output_dir, 'image_model_config.json'), 'w') as f:
            json.dump(config, f, indent=4)
        
        print(f"Model saved to {output_dir}/")


def main():
    """
    Main training pipeline
    """
    print("="*70)
    print("MULTIMODAL PSYCHODERMATOLOGICAL DISORDER DETECTION")
    print("Image Model Training Pipeline")
    print("="*70)
    
    # Configuration
    DATA_DIR = 'dermatology_dataset'  # Update with your dataset path
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    EPOCHS = 30
    
    # Check GPU availability
    print(f"\nGPU Available: {tf.config.list_physical_devices('GPU')}")
    
    # Initialize trainer
    trainer = ImageModelTrainer(
        data_dir=DATA_DIR,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )
    
    # Load and preprocess data
    X, y = trainer.load_and_preprocess_data()
    
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
    trainer.build_model(base_model_name='EfficientNetB0')
    
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