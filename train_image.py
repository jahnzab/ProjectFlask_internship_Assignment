"""
Multimodal Psychodermatological Disorder Detection
Part 1: Image Model Training with Transfer Learning
Author: Internship Project
Date: October 2025
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0, ResNet50
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import json
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class ImageModelTrainer:
    def __init__(self, data_dir, img_size=(224, 224), batch_size=32):
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.model = None
        self.history = None
        self.class_names = []

    def load_and_preprocess_data(self):
        """
        Load train and validation datasets from folder structure
        """
        train_dir = os.path.join(self.data_dir, 'train')
        test_dir = os.path.join(self.data_dir, 'test')

        self.class_names = sorted(os.listdir(train_dir))

        # Load datasets
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            train_dir,
            labels='inferred',
            label_mode='categorical',
            image_size=self.img_size,
            batch_size=self.batch_size,
            shuffle=True
        )

        test_ds = tf.keras.preprocessing.image_dataset_from_directory(
            test_dir,
            labels='inferred',
            label_mode='categorical',
            image_size=self.img_size,
            batch_size=self.batch_size,
            shuffle=False
        )

        # Normalize images
        normalization_layer = layers.Rescaling(1./255)
        train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
        val_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

        return train_ds, val_ds

    def build_model(self, base_model_name='EfficientNetB0', num_classes=None):
        if num_classes is None:
            num_classes = len(self.class_names)

        print(f"Building model with {base_model_name}...")

        # Load pre-trained base model
        if base_model_name == 'EfficientNetB0':
            base_model = EfficientNetB0(include_top=False, weights='imagenet',
                                        input_shape=(*self.img_size, 3))
        elif base_model_name == 'ResNet50':
            base_model = ResNet50(include_top=False, weights='imagenet',
                                  input_shape=(*self.img_size, 3))
        else:
            raise ValueError(f"Unknown base model: {base_model_name}")

        base_model.trainable = False

        # Build custom head
        inputs = keras.Input(shape=(*self.img_size, 3))
        x = layers.RandomFlip("horizontal")(inputs)
        x = layers.RandomRotation(0.1)(x)
        x = layers.RandomZoom(0.1)(x)
        x = base_model(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)

        self.model = keras.Model(inputs, outputs, name=f'image_model_{base_model_name}')
        self.model.compile(optimizer=keras.optimizers.Adam(1e-3),
                           loss='categorical_crossentropy',
                           metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()])

        print(f"Model built successfully! Total parameters: {self.model.count_params():,}")
        return self.model

    def train(self, train_ds, val_ds, epochs=30):
        """
        Two-phase training: frozen base and fine-tuning
        """
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1)

        # Phase 1: Train frozen base
        checkpoint1 = ModelCheckpoint('best_image_model_phase1.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
        print("\n=== PHASE 1: Training with frozen base model ===")
        history1 = self.model.fit(train_ds, validation_data=val_ds,
                                  epochs=epochs//2, callbacks=[early_stop, checkpoint1, reduce_lr], verbose=1)

        # Phase 2: Fine-tuning
        base_model = self.model.layers[3]  # Assuming base model is the 4th layer
        base_model.trainable = True
        freeze_until = int(len(base_model.layers) * 0.8)
        for layer in base_model.layers[:freeze_until]:
            layer.trainable = False

        self.model.compile(optimizer=keras.optimizers.Adam(1e-5),
                           loss='categorical_crossentropy',
                           metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()])

        checkpoint2 = ModelCheckpoint('best_image_model_final.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
        print("\n=== PHASE 2: Fine-tuning ===")
        history2 = self.model.fit(train_ds, validation_data=val_ds,
                                  epochs=epochs//2, callbacks=[early_stop, checkpoint2, reduce_lr], verbose=1)

        # Combine histories
        self.history = {key: history1.history[key] + history2.history[key] for key in history1.history}
        return self.history

    def evaluate(self, val_ds):
        print("\nEvaluating model...")
        y_pred_probs = self.model.predict(val_ds, verbose=1)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = np.concatenate([y for x, y in val_ds], axis=0)
        y_true = np.argmax(y_true, axis=1)

        acc = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, target_names=self.class_names, output_dict=True)
        cm = confusion_matrix(y_true, y_pred)

        print(f"\nTest Accuracy: {acc:.4f}")
        print(classification_report(y_true, y_pred, target_names=self.class_names))

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Confusion Matrix - Image Model')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('image_model_confusion_matrix.png', dpi=300)
        plt.close()

        return {'accuracy': acc, 'report': report, 'confusion_matrix': cm.tolist()}

    def plot_training_history(self):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes[0, 0].plot(self.history['accuracy'], label='Train Accuracy')
        axes[0, 0].plot(self.history['val_accuracy'], label='Val Accuracy')
        axes[0, 0].set_title('Model Accuracy'); axes[0, 0].legend(); axes[0, 0].grid(True)
        axes[0, 1].plot(self.history['loss'], label='Train Loss')
        axes[0, 1].plot(self.history['val_loss'], label='Val Loss')
        axes[0, 1].set_title('Model Loss'); axes[0, 1].legend(); axes[0, 1].grid(True)
        axes[1, 0].plot(self.history['precision'], label='Train Precision')
        axes[1, 0].plot(self.history['val_precision'], label='Val Precision')
        axes[1, 0].set_title('Precision'); axes[1, 0].legend(); axes[1, 0].grid(True)
        axes[1, 1].plot(self.history['recall'], label='Train Recall')
        axes[1, 1].plot(self.history['val_recall'], label='Val Recall')
        axes[1, 1].set_title('Recall'); axes[1, 1].legend(); axes[1, 1].grid(True)
        plt.tight_layout()
        plt.savefig('image_model_training_history.png', dpi=300)
        plt.close()

    def save_model_and_config(self, output_dir='models'):
        os.makedirs(output_dir, exist_ok=True)
        self.model.save(os.path.join(output_dir, 'image_model.h5'))
        config = {'class_names': self.class_names, 'img_size': self.img_size, 'num_classes': len(self.class_names)}
       
def main():
    """
    Main training pipeline
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.applications import EfficientNetB0
    import os

    print("="*70)
    print("MULTIMODAL PSYCHODERMATOLOGICAL DISORDER DETECTION")
    print("Image Model Training Pipeline")
    print("="*70)
    
    # Configuration
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    EPOCHS = 30
    DATA_DIR = '/content/ProjectFlask_internship_Assignment/dermatology_dataset'

    # Load train dataset with validation split
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR + '/train',
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        shuffle=True,
        validation_split=0.15,
        subset='training',
        seed=42
    )

    val_dataset = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR + '/train',
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        shuffle=True,
        validation_split=0.15,
        subset='validation',
        seed=42
    )

    # Load test dataset
    test_dataset = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR + '/test',
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        shuffle=False
    )

    # Count samples
    train_samples = sum(1 for _ in train_dataset.unbatch())
    val_samples = sum(1 for _ in val_dataset.unbatch())
    test_samples = sum(1 for _ in test_dataset.unbatch())

    print(f"\nDataset split:")
    print(f"Train: {train_samples} samples")
    print(f"Validation: {val_samples} samples")
    print(f"Test: {test_samples} samples")

    # Initialize trainer (make sure you have ImageModelTrainer defined)
    trainer = ImageModelTrainer(
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    # Build model
    trainer.build_model(base_model_name='EfficientNetB0')

    # Train model using tf.data.Dataset objects
    trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=EPOCHS
    )

    # Plot training history
    trainer.plot_training_history()

    # Evaluate model
    metrics = trainer.evaluate(test_dataset=test_dataset)

    # Save model
    trainer.save_model_and_config()

    print("\n" + "="*70)
    print("Training completed successfully!")
    print("="*70)

if __name__ == "__main__":
    main()
