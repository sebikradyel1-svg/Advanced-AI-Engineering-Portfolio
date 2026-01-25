"""
Universal Image Classifier - Streamlit Web Interface
Transfer Learning with VGG16 | Production-Ready Deployment

Author: Sebastian
Stack: TensorFlow, VGG16, Streamlit
"""

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import json
import os
from pathlib import Path
import time

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="🖼️ Universal Image Classifier",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS
# =============================================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .confidence-high { color: #00ff88; font-weight: bold; }
    .confidence-medium { color: #ffcc00; font-weight: bold; }
    .confidence-low { color: #ff6b6b; font-weight: bold; }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# MODEL LOADING (CACHED)
# =============================================================================
@st.cache_resource
def load_classifier():
    """Load the trained model and configuration"""
    model_path = os.getenv('MODEL_PATH', 'models/image_classifier_model.h5')
    config_path = os.getenv('CONFIG_PATH', 'models/image_classifier_config.json')
    
    # Check if model exists
    if not Path(model_path).exists():
        st.error(f"❌ Model not found at: {model_path}")
        st.info("Please ensure you have trained and saved the model.")
        return None, None
    
    # Load model
    model = load_model(model_path)
    
    # Load config
    if Path(config_path).exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = {
            'class_names': ['Unknown'],
            'num_classes': 1,
            'img_size': 224
        }
    
    return model, config

# =============================================================================
# PREDICTION FUNCTION
# =============================================================================
def predict_image(model, config, img):
    """Make prediction on uploaded image"""
    img_size = config.get('img_size', 224)
    
    # Preprocess image
    img_resized = img.resize((img_size, img_size))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    # Predict
    with st.spinner('🔮 Analyzing image...'):
        start_time = time.time()
        predictions = model.predict(img_array, verbose=0)
        inference_time = time.time() - start_time
    
    # Get results
    class_names = config.get('class_names', ['Unknown'])
    
    # Top 3 predictions
    top_3_idx = np.argsort(predictions[0])[-3:][::-1]
    results = []
    for idx in top_3_idx:
        results.append({
            'class': class_names[idx] if idx < len(class_names) else f'Class {idx}',
            'confidence': float(predictions[0][idx])
        })
    
    return results, inference_time

# =============================================================================
# SAMPLE IMAGES
# =============================================================================
def get_sample_images():
    """Get sample images from the sample_images directory"""
    sample_dir = Path('sample_images')
    if not sample_dir.exists():
        return []
    
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp']
    samples = []
    for ext in extensions:
        samples.extend(sample_dir.glob(ext))
    
    return sorted(samples)[:6]  # Max 6 samples

# =============================================================================
# MAIN APP
# =============================================================================
def main():
    # Header
    st.markdown('<h1 class="main-header">🖼️ Universal Image Classifier</h1>', unsafe_allow_html=True)
    st.markdown("""
    <p style="text-align: center; color: #666; font-size: 1.1rem;">
        Powered by VGG16 Transfer Learning | Real-time Classification
    </p>
    """, unsafe_allow_html=True)
    
    # Load model
    model, config = load_classifier()
    
    if model is None:
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown(f"""
        **Model:** VGG16 Transfer Learning  
        **Classes:** {config.get('num_classes', 'N/A')}  
        **Input Size:** {config.get('img_size', 224)}×{config.get('img_size', 224)}
        """)
        
        st.divider()
        
        st.header("📊 Supported Classes")
        class_names = config.get('class_names', [])
        for i, name in enumerate(class_names, 1):
            st.markdown(f"{i}. **{name.replace('_', ' ').title()}**")
        
        st.divider()
        
        st.header("🔧 Settings")
        show_details = st.checkbox("Show technical details", value=False)
        show_probabilities = st.checkbox("Show all probabilities", value=False)
    
    # Main content - Two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png', 'webp'],
            help="Supported formats: JPG, JPEG, PNG, WebP"
        )
        
        # Sample images gallery
        st.subheader("🖼️ Or try a sample image:")
        sample_images = get_sample_images()
        
        if sample_images:
            sample_cols = st.columns(3)
            selected_sample = None
            
            for idx, sample_path in enumerate(sample_images):
                with sample_cols[idx % 3]:
                    sample_img = Image.open(sample_path)
                    st.image(sample_img, caption=sample_path.stem, use_column_width=True)
                    if st.button(f"Use this", key=f"sample_{idx}"):
                        selected_sample = sample_path
            
            if selected_sample:
                uploaded_file = selected_sample
        else:
            st.info("No sample images found. Add images to `sample_images/` folder.")
    
    with col2:
        st.header("🎯 Prediction Results")
        
        if uploaded_file is not None:
            # Load and display image
            if isinstance(uploaded_file, Path):
                img = Image.open(uploaded_file)
            else:
                img = Image.open(uploaded_file)
            
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            st.image(img, caption="Uploaded Image", use_column_width=True)
            
            # Make prediction
            results, inference_time = predict_image(model, config, img)
            
            # Display main prediction
            main_result = results[0]
            confidence_pct = main_result['confidence'] * 100
            
            # Confidence color coding
            if confidence_pct >= 80:
                conf_class = "confidence-high"
                conf_emoji = "✅"
            elif confidence_pct >= 50:
                conf_class = "confidence-medium"
                conf_emoji = "⚠️"
            else:
                conf_class = "confidence-low"
                conf_emoji = "❓"
            
            st.markdown(f"""
            <div class="prediction-box">
                <h2>{conf_emoji} {main_result['class'].replace('_', ' ').title()}</h2>
                <h3 class="{conf_class}">{confidence_pct:.1f}% Confidence</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Top 3 predictions with progress bars
            st.subheader("📊 Top 3 Predictions")
            for i, result in enumerate(results, 1):
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.progress(result['confidence'], text=f"{result['class'].replace('_', ' ').title()}")
                with col_b:
                    st.write(f"**{result['confidence']*100:.1f}%**")
            
            # Technical details
            if show_details:
                st.divider()
                st.subheader("🔬 Technical Details")
                
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Inference Time", f"{inference_time*1000:.1f} ms")
                with col_b:
                    st.metric("Image Size", f"{img.size[0]}×{img.size[1]}")
                with col_c:
                    st.metric("Model Input", f"{config.get('img_size', 224)}×{config.get('img_size', 224)}")
            
            # All probabilities
            if show_probabilities and model is not None:
                st.divider()
                st.subheader("📈 All Class Probabilities")
                
                img_size = config.get('img_size', 224)
                img_resized = img.resize((img_size, img_size))
                img_array = image.img_to_array(img_resized)
                img_array = np.expand_dims(img_array, axis=0) / 255.0
                predictions = model.predict(img_array, verbose=0)
                
                class_names = config.get('class_names', [])
                for idx, prob in enumerate(predictions[0]):
                    class_name = class_names[idx] if idx < len(class_names) else f'Class {idx}'
                    st.progress(float(prob), text=f"{class_name}: {prob*100:.2f}%")
        
        else:
            st.info("👆 Upload an image or select a sample to get predictions")
    
    # Footer
    st.divider()
    st.markdown("""
    <p style="text-align: center; color: #888; font-size: 0.9rem;">
        Built with ❤️ using TensorFlow & Streamlit | 
        <a href="https://github.com/yourusername/image-classifier" target="_blank">View on GitHub</a>
    </p>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
