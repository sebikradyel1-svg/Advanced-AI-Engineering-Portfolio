# Models Directory

Place your trained model files here:

1. `image_classifier_model.h5` - The trained TensorFlow/Keras model
2. `image_classifier_config.json` - Configuration with class names

## How to generate these files:

```bash
python universal_classifier.py \
    --data_dir data/your_dataset \
    --project_name image_classifier \
    --initial_epochs 10 \
    --finetune_epochs 15
```

Then copy the outputs:
```bash
mv image_classifier_model.h5 models/
mv image_classifier_config.json models/
```
