"""
Clasificator UNIVERSAL de Imagini folosind Transfer Learning cu VGG16

FUNCȚIONEAZĂ PENTRU ORICE TIP DE IMAGINI:
- Animale, vehicule, obiecte, fețe, produse, etc.
- Detectează automat clasele din folderele tale
- Se adaptează automat la numărul de clase

UTILIZARE:
python universal_classifier.py --data_dir "data/YOUR_PROJECT" --project_name "animals"
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import argparse
from pathlib import Path
import json


class UniversalImageClassifier:
    """
    Clasificator universal care se adaptează automat la orice dataset
    """
    
    def __init__(self, data_dir, project_name='my_project', img_size=224, batch_size=32):
        self.data_dir = Path(data_dir)
        self.project_name = project_name
        self.img_size = img_size
        self.batch_size = batch_size
        
        self.train_dir = self.data_dir / 'train'
        self.val_dir = self.data_dir / 'validation'
        self.test_dir = self.data_dir / 'test'
        
        self.model = None
        self.base_model = None
        self.class_names = None
        self.num_classes = None
        
    def detect_classes(self):
        """Detectează automat clasele din directoarele de date"""
        if not self.train_dir.exists():
            raise ValueError(f"❌ Directorul de antrenament nu există: {self.train_dir}")
        
        class_dirs = [d for d in self.train_dir.iterdir() if d.is_dir()]
        self.class_names = sorted([d.name for d in class_dirs])
        self.num_classes = len(self.class_names)
        
        print(f"\n{'='*60}")
        print(f"PROIECT: {self.project_name.upper()}")
        print(f"{'='*60}")
        print(f"📊 Clase detectate automat: {self.num_classes}")
        for i, name in enumerate(self.class_names, 1):
            print(f"   {i}. {name}")
        print(f"{'='*60}\n")
        
        return self.class_names
    
    def create_data_generators(self, augmentation=True):
        """Creează generatoare de date cu sau fără augmentare"""
        
        if augmentation:
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=30,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                brightness_range=[0.8, 1.2],
                fill_mode='nearest'
            )
        else:
            train_datagen = ImageDataGenerator(rescale=1./255)
        
        val_test_datagen = ImageDataGenerator(rescale=1./255)
        
        print("📁 Încărcare date...")
        
        train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=(self.img_size, self.img_size),
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True
        )
        
        val_generator = val_test_datagen.flow_from_directory(
            self.val_dir,
            target_size=(self.img_size, self.img_size),
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        test_generator = val_test_datagen.flow_from_directory(
            self.test_dir,
            target_size=(self.img_size, self.img_size),
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        print(f"✓ Train: {train_generator.samples} imagini")
        print(f"✓ Validation: {val_generator.samples} imagini")
        print(f"✓ Test: {test_generator.samples} imagini\n")
        
        return train_generator, val_generator, test_generator
    
    def build_model(self):
        """Construiește modelul VGG16 adaptat la numărul de clase"""
        
        print("🏗️  Construire model VGG16...")
        
        # Încarcă VGG16 pre-antrenat
        self.base_model = VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=(self.img_size, self.img_size, 3)
        )
        
        # Freeze straturile de bază
        for layer in self.base_model.layers:
            layer.trainable = False
        
        # Adaugă straturi personalizate
        x = self.base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu', name='fc1')(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu', name='fc2')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(self.num_classes, activation='softmax', name='predictions')(x)
        
        self.model = Model(inputs=self.base_model.input, outputs=predictions)
        
        print(f"✓ Model construit pentru {self.num_classes} clase")
        print(f"✓ Parametri VGG16: {self.base_model.count_params():,}")
        
        return self.model
    
    def compile_model(self, learning_rate=0.001):
        """Compilează modelul"""
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        print(f"✓ Model compilat (lr={learning_rate})\n")
    
    def train(self, train_gen, val_gen, epochs=10, callbacks=None):
        """Antrenează modelul"""
        print(f"🔥 Start antrenare ({epochs} epoci)...\n")
        
        history = self.model.fit(
            train_gen,
            epochs=epochs,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def unfreeze_and_finetune(self, train_gen, val_gen, epochs=10, unfreeze_from='block5_conv1'):
        """Fine-tuning prin dezghețarea straturilor superioare"""
        
        print(f"\n{'='*60}")
        print("FINE-TUNING: Dezghețare straturi")
        print(f"{'='*60}\n")
        
        # Dezgheață straturi
        set_trainable = False
        for layer in self.base_model.layers:
            if layer.name == unfreeze_from:
                set_trainable = True
            if set_trainable:
                layer.trainable = True
        
        # Recompilează cu learning rate mai mic
        self.compile_model(learning_rate=0.0001)
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3)
        ]
        
        history = self.train(train_gen, val_gen, epochs=epochs, callbacks=callbacks)
        
        return history
    
    def evaluate(self, test_gen):
        """Evaluare pe setul de test"""
        print(f"\n{'='*60}")
        print("EVALUARE FINALĂ")
        print(f"{'='*60}\n")
        
        test_loss, test_acc = self.model.evaluate(test_gen, verbose=0)
        print(f"✓ Test Loss: {test_loss:.4f}")
        print(f"✓ Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)\n")
        
        return test_loss, test_acc
    
    def predict_and_report(self, test_gen):
        """Generează raport de clasificare detaliat"""
        y_pred = self.model.predict(test_gen, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = test_gen.classes
        
        print(f"{'='*60}")
        print("RAPORT DE CLASIFICARE")
        print(f"{'='*60}\n")
        print(classification_report(y_true, y_pred_classes, 
                                    target_names=self.class_names, digits=4))
        
        return y_true, y_pred_classes
    
    def plot_history(self, history, finetune_history=None):
        """Vizualizează istoricul antrenării"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        fig.suptitle(f'Training History - {self.project_name}', fontsize=16, fontweight='bold')
        
        if finetune_history:
            epochs_initial = len(history.history['loss'])
            total_epochs = range(1, epochs_initial + len(finetune_history.history['loss']) + 1)
            train_loss = history.history['loss'] + finetune_history.history['loss']
            val_loss = history.history['val_loss'] + finetune_history.history['val_loss']
            train_acc = history.history['accuracy'] + finetune_history.history['accuracy']
            val_acc = history.history['val_accuracy'] + finetune_history.history['val_accuracy']
            finetune_start = epochs_initial
        else:
            total_epochs = range(1, len(history.history['loss']) + 1)
            train_loss = history.history['loss']
            val_loss = history.history['val_loss']
            train_acc = history.history['accuracy']
            val_acc = history.history['val_accuracy']
            finetune_start = None
        
        # Loss
        axes[0].plot(total_epochs, train_loss, 'b-', label='Train', linewidth=2)
        axes[0].plot(total_epochs, val_loss, 'r-', label='Validation', linewidth=2)
        if finetune_start:
            axes[0].axvline(x=finetune_start, color='green', linestyle='--', 
                           linewidth=2, label='Fine-tuning start')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Loss Curves')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[1].plot(total_epochs, train_acc, 'b-', label='Train', linewidth=2)
        axes[1].plot(total_epochs, val_acc, 'r-', label='Validation', linewidth=2)
        if finetune_start:
            axes[1].axvline(x=finetune_start, color='green', linestyle='--', 
                           linewidth=2, label='Fine-tuning start')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Accuracy Curves')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.project_name}_training_curves.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Grafic salvat: {self.project_name}_training_curves.png")
        plt.show()
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plotează matricea de confuzie"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(max(8, self.num_classes), max(6, self.num_classes)))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names)
        plt.title(f'Confusion Matrix - {self.project_name}', fontsize=16, fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'{self.project_name}_confusion_matrix.png', dpi=300, bbox_inches='tight')
        print(f"✓ Matrice salvată: {self.project_name}_confusion_matrix.png")
        plt.show()
    
    def save_model(self):
        """Salvează modelul și configurația"""
        model_path = f'{self.project_name}_model.h5'
        self.model.save(model_path)
        
        # Salvează și configurația
        config = {
            'project_name': self.project_name,
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'img_size': self.img_size
        }
        
        with open(f'{self.project_name}_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\n✓ Model salvat: {model_path}")
        print(f"✓ Config salvat: {self.project_name}_config.json")
    
    def predict_single_image(self, image_path):
        """Face predicție pe o imagine nouă"""
        from tensorflow.keras.preprocessing import image
        
        img = image.load_img(image_path, target_size=(self.img_size, self.img_size))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        
        predictions = self.model.predict(img_array, verbose=0)
        predicted_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_idx]
        
        print(f"\n🎯 Predicție pentru: {image_path}")
        print(f"   Clasă: {self.class_names[predicted_idx]}")
        print(f"   Încredere: {confidence*100:.2f}%")
        
        print("\n   Top 3 predicții:")
        top_3_idx = np.argsort(predictions[0])[-3:][::-1]
        for idx in top_3_idx:
            print(f"   - {self.class_names[idx]}: {predictions[0][idx]*100:.2f}%")
        
        return self.class_names[predicted_idx], confidence


def main():
    """Funcție principală cu argumente din linia de comandă"""
    
    parser = argparse.ArgumentParser(description='Clasificator Universal de Imagini cu VGG16')
    parser.add_argument('--data_dir', type=str, default='data/fruits',
                       help='Cale către directorul cu date')
    parser.add_argument('--project_name', type=str, default='image_classifier',
                       help='Numele proiectului')
    parser.add_argument('--img_size', type=int, default=224,
                       help='Dimensiunea imaginilor')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--initial_epochs', type=int, default=10,
                       help='Epoci pentru antrenarea inițială')
    parser.add_argument('--finetune_epochs', type=int, default=15,
                       help='Epoci pentru fine-tuning')
    parser.add_argument('--no_augmentation', action='store_true',
                       help='Dezactivează data augmentation')
    parser.add_argument('--predict', type=str, default=None,
                       help='Cale către imagine pentru predicție')
    
    args = parser.parse_args()
    
    # Inițializează clasificatorul
    classifier = UniversalImageClassifier(
        data_dir=args.data_dir,
        project_name=args.project_name,
        img_size=args.img_size,
        batch_size=args.batch_size
    )
    
    # Dacă se dorește doar predicție
    if args.predict:
        print(f"\n{'='*60}")
        print("MOD PREDICȚIE")
        print(f"{'='*60}")
        
        # Încarcă modelul salvat
        model_path = f'{args.project_name}_model.h5'
        if Path(model_path).exists():
            classifier.model = tf.keras.models.load_model(model_path)
            with open(f'{args.project_name}_config.json', 'r') as f:
                config = json.load(f)
            classifier.class_names = config['class_names']
            classifier.num_classes = config['num_classes']
            
            classifier.predict_single_image(args.predict)
        else:
            print(f"❌ Model nu a fost găsit: {model_path}")
        return
    
    # Detectează clasele
    classifier.detect_classes()
    
    # Creează generatoare de date
    train_gen, val_gen, test_gen = classifier.create_data_generators(
        augmentation=not args.no_augmentation
    )
    
    # Construiește și compilează modelul
    classifier.build_model()
    classifier.compile_model(learning_rate=0.001)
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1)
    ]
    
    # FAZA 1: Antrenare inițială
    print(f"\n{'='*60}")
    print("FAZA 1: ANTRENARE INIȚIALĂ (VGG16 FROZEN)")
    print(f"{'='*60}\n")
    
    history = classifier.train(train_gen, val_gen, 
                              epochs=args.initial_epochs, 
                              callbacks=callbacks)
    
    # Evaluare după prima fază
    loss_1, acc_1 = classifier.evaluate(val_gen)
    
    # FAZA 2: Fine-tuning
    print(f"\n{'='*60}")
    print("FAZA 2: FINE-TUNING")
    print(f"{'='*60}\n")
    
    finetune_history = classifier.unfreeze_and_finetune(
        train_gen, val_gen, 
        epochs=args.finetune_epochs
    )
    
    # Evaluare finală
    test_loss, test_acc = classifier.evaluate(test_gen)
    
    # Generează raport și predicții
    y_true, y_pred = classifier.predict_and_report(test_gen)
    
    # Vizualizări
    classifier.plot_history(history, finetune_history)
    classifier.plot_confusion_matrix(y_true, y_pred)
    
    # Salvează modelul
    classifier.save_model()
    
    # Rezumat final
    print(f"\n{'='*60}")
    print("REZUMAT FINAL")
    print(f"{'='*60}")
    print(f"✓ Proiect: {args.project_name}")
    print(f"✓ Clase: {classifier.num_classes}")
    print(f"✓ Accuracy inițial: {acc_1*100:.2f}%")
    print(f"✓ Accuracy final: {test_acc*100:.2f}%")
    print(f"✓ Îmbunătățire: +{(test_acc - acc_1)*100:.2f}%")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
