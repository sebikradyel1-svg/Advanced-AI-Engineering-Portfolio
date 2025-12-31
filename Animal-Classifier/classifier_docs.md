# Universal Image Classifier with VGG16

## Overview

A flexible, transfer learning-based image classification system that automatically adapts to any dataset. Built on VGG16 pre-trained weights, this classifier can handle any number of classes across diverse domains (animals, vehicles, objects, products, etc.).

## Features

- **Universal Architecture**: Works with any image dataset and number of classes
- **Automatic Class Detection**: Detects classes from your folder structure
- **Transfer Learning**: Leverages VGG16 pre-trained on ImageNet
- **Two-Phase Training**: Initial training + fine-tuning for optimal results
- **Data Augmentation**: Built-in image augmentation for better generalization
- **Comprehensive Reporting**: Detailed metrics, confusion matrices, and visualizations
- **Easy Prediction**: Simple interface for classifying new images

## Requirements

```bash
pip install tensorflow numpy matplotlib scikit-learn seaborn
```

**Required versions:**
- Python 3.7+
- TensorFlow 2.x
- NumPy
- Matplotlib
- Scikit-learn
- Seaborn

## Dataset Structure

Organize your data in the following structure:

```
data/
└── your_project/
    ├── train/
    │   ├── class1/
    │   │   ├── image1.jpg
    │   │   ├── image2.jpg
    │   │   └── ...
    │   ├── class2/
    │   │   └── ...
    │   └── classN/
    │       └── ...
    ├── validation/
    │   ├── class1/
    │   ├── class2/
    │   └── classN/
    └── test/
        ├── class1/
        ├── class2/
        └── classN/
```

**Notes:**
- Folder names become class labels
- Supports common image formats (jpg, jpeg, png)
- Recommended split: 70% train, 15% validation, 15% test

## Basic Usage

### Training a New Model

```bash
python universal_classifier.py \
    --data_dir "data/animals" \
    --project_name "animal_classifier" \
    --initial_epochs 10 \
    --finetune_epochs 15
```

### Making Predictions

```bash
python universal_classifier.py \
    --project_name "animal_classifier" \
    --predict "path/to/new_image.jpg"
```

## Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | `data/fruits` | Path to dataset directory |
| `--project_name` | `image_classifier` | Name for your project (used in output files) |
| `--img_size` | `224` | Image size (width and height in pixels) |
| `--batch_size` | `32` | Batch size for training |
| `--initial_epochs` | `10` | Epochs for initial training phase |
| `--finetune_epochs` | `15` | Epochs for fine-tuning phase |
| `--no_augmentation` | `False` | Disable data augmentation |
| `--predict` | `None` | Path to image for prediction |

## How It Works

### Training Pipeline

**Phase 1: Initial Training (Frozen VGG16)**
1. Loads VGG16 pre-trained on ImageNet
2. Freezes all convolutional layers
3. Adds custom classification head
4. Trains only the new layers

**Phase 2: Fine-Tuning**
1. Unfreezes top convolutional blocks
2. Reduces learning rate (0.001 → 0.0001)
3. Fine-tunes the entire network
4. Uses early stopping to prevent overfitting

### Model Architecture

```
VGG16 Base (frozen initially)
    ↓
GlobalAveragePooling2D
    ↓
Dense(512) + ReLU + Dropout(0.5)
    ↓
Dense(256) + ReLU + Dropout(0.5)
    ↓
Dense(num_classes) + Softmax
```

### Data Augmentation

When enabled, applies:
- Random rotation (±30°)
- Width/height shifts (±20%)
- Shear transformation (±20%)
- Zoom (±20%)
- Horizontal flipping
- Brightness adjustment (80-120%)

## Training Process Example

```python
from pathlib import Path
from universal_classifier import UniversalImageClassifier

# Initialize classifier
classifier = UniversalImageClassifier(
    data_dir="data/fruits",
    project_name="fruit_classifier",
    img_size=224,
    batch_size=32
)

# Detect classes automatically
classifier.detect_classes()

# Create data generators
train_gen, val_gen, test_gen = classifier.create_data_generators(
    augmentation=True
)

# Build and compile model
classifier.build_model()
classifier.compile_model(learning_rate=0.001)

# Phase 1: Initial training
history = classifier.train(
    train_gen, val_gen, 
    epochs=10
)

# Phase 2: Fine-tuning
finetune_history = classifier.unfreeze_and_finetune(
    train_gen, val_gen,
    epochs=15
)

# Evaluate
test_loss, test_acc = classifier.evaluate(test_gen)

# Generate reports
y_true, y_pred = classifier.predict_and_report(test_gen)

# Create visualizations
classifier.plot_history(history, finetune_history)
classifier.plot_confusion_matrix(y_true, y_pred)

# Save model
classifier.save_model()
```

## Prediction Example

```python
# Load and predict
classifier = UniversalImageClassifier(
    data_dir="data/fruits",
    project_name="fruit_classifier"
)

# Load saved model
classifier.model = tf.keras.models.load_model('fruit_classifier_model.h5')

# Predict on new image
predicted_class, confidence = classifier.predict_single_image(
    'path/to/apple.jpg'
)

print(f"Predicted: {predicted_class} ({confidence*100:.2f}%)")
```

## Output Files

After training, the following files are generated:

1. **`{project_name}_model.h5`** - Trained model weights
2. **`{project_name}_config.json`** - Model configuration and class names
3. **`{project_name}_training_curves.png`** - Loss and accuracy plots
4. **`{project_name}_confusion_matrix.png`** - Confusion matrix heatmap

### Sample config.json

```json
{
  "project_name": "animal_classifier",
  "num_classes": 5,
  "class_names": ["cat", "dog", "elephant", "horse", "lion"],
  "img_size": 224
}
```

## Performance Tips

### Getting Better Results

1. **More Data**: Aim for 100+ images per class
2. **Balanced Classes**: Equal number of images per class
3. **Data Quality**: Use clear, varied images
4. **Augmentation**: Keep enabled for small datasets
5. **Training Time**: Start with 10+15 epochs, increase if needed
6. **Early Stopping**: Patience prevents overfitting

### Common Issues

**Low Accuracy:**
- Increase training epochs
- Check data quality
- Ensure data augmentation is enabled
- Verify correct folder structure

**Overfitting:**
- Reduce epochs
- Add more dropout
- Get more training data
- Increase augmentation

**Out of Memory:**
- Reduce batch_size
- Reduce img_size to 128 or 160

## Advanced Usage

### Custom Callbacks

```python
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    ),
    ModelCheckpoint(
        'best_model.h5',
        monitor='val_accuracy',
        save_best_only=True
    )
]

history = classifier.train(
    train_gen, val_gen,
    epochs=20,
    callbacks=callbacks
)
```

### Adjusting Fine-Tuning

```python
# Unfreeze more layers for fine-tuning
finetune_history = classifier.unfreeze_and_finetune(
    train_gen, val_gen,
    epochs=20,
    unfreeze_from='block4_conv1'  # Unfreeze earlier
)
```

## Use Cases

This classifier works for:

- **Animal Recognition**: Cats, dogs, wildlife
- **Product Classification**: E-commerce items
- **Medical Imaging**: X-rays, scans (with proper data)
- **Quality Control**: Defect detection
- **Food Classification**: Dishes, ingredients
- **Vehicle Recognition**: Car models, types
- **Plant Identification**: Flowers, leaves
- **Document Classification**: Forms, receipts

## Citation

If you use this classifier in your research, please cite:

```
VGG16: Simonyan, K., & Zisserman, A. (2014). 
Very deep convolutional networks for large-scale image recognition.
arXiv preprint arXiv:1409.1556.
```

## License

This implementation is provided as-is for educational and research purposes.

## Support

For issues or questions:
- Check dataset structure matches requirements
- Verify all dependencies are installed
- Ensure sufficient disk space for model files
- Review error messages for specific issues

---

**Version**: 1.0  
**Last Updated**: 2025  
**Compatibility**: TensorFlow 2.x, Python 3.7+