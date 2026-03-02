"""
Grad-CAM: Visual Explanations for VGG16 Image Classifier
Generates heatmaps showing which regions the model focuses on.
Reference: Selvaraju et al., ICCV 2017
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
import cv2
import logging

logger = logging.getLogger(__name__)


class GradCAM:
    """Grad-CAM for any Keras CNN. For VGG16, use layer block5_conv3."""

    def __init__(self, model, layer_name=None):
        self.model = model
        self.layer_name = layer_name or self._find_last_conv_layer()
        conv_layer = self.model.get_layer(self.layer_name)
        self.grad_model = Model(
            inputs=self.model.input,
            outputs=[conv_layer.output, self.model.output],
        )
        logger.info(f"GradCAM initialized: layer={self.layer_name}")

    def _find_last_conv_layer(self):
        for layer in reversed(self.model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                return layer.name
        raise ValueError("No Conv2D layer found")

    def generate(self, img_array, class_idx=None, img_size=None):
        if len(img_array.shape) == 3:
            img_array = np.expand_dims(img_array, axis=0)

        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(img_array)
            if class_idx is None:
                class_idx = tf.argmax(predictions[0])
            class_score = predictions[:, class_idx]

        grads = tape.gradient(class_score, conv_outputs)
        weights = tf.reduce_mean(grads, axis=(0, 1, 2))
        cam = tf.reduce_sum(conv_outputs[0] * weights, axis=-1).numpy()

        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam = cam / cam.max()

        if img_size is None:
            img_size = (img_array.shape[1], img_array.shape[2])
        heatmap = cv2.resize(cam, (img_size[1], img_size[0]))
        overlay = self._create_overlay(img_array[0], heatmap)
        return heatmap, overlay

    def _create_overlay(self, img, heatmap, alpha=0.4):
        img_uint8 = (img * 255).astype(np.uint8)
        hm_resized = cv2.resize(heatmap, (img_uint8.shape[1], img_uint8.shape[0]))
        hm_colored = cv2.applyColorMap(
            (hm_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        hm_colored = cv2.cvtColor(hm_colored, cv2.COLOR_BGR2RGB)
        return cv2.addWeighted(img_uint8, 1 - alpha, hm_colored, alpha, 0)

    def generate_for_top_k(self, img_array, predictions, class_names, k=3):
        top_k_idx = np.argsort(predictions[0])[-k:][::-1]
        results = []
        for idx in top_k_idx:
            heatmap, overlay = self.generate(img_array, class_idx=idx)
            results.append({
                "class_name": class_names[idx] if idx < len(class_names) else f"Class {idx}",
                "confidence": float(predictions[0][idx]),
                "heatmap": heatmap,
                "overlay": overlay,
            })
        return results
