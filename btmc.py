import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import pickle
import logging
import tensorflow as tf
import cv2
import io
import os

# Streamlit app
st.set_page_config(page_title="BTMC (Late Fusion)", page_icon="🧠", layout="wide")

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Check GPU availability
gpus = tf.config.list_physical_devices('GPU')

# Model paths (preserving lowercase and typo)
model_paths = [
    "Models/efficientnetv2b0.keras",
    "Models/Convnexttiny.keras"
]

# Load models
@st.cache_resource
def load_models():
    return [load_model(path) for path in model_paths]

try:
    models = load_models()
except Exception as e:
    st.error(f"❌ Failed to load models: {e}")
    st.stop()

# Class labels
labels = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# Load test data
@st.cache_data
def load_test_data():
    try:
        with open("data/test_data.pkl", 'rb') as f:
            data = pickle.load(f)
        return data['X_test'], data['y_test'], data['filenames_test']
    except Exception as e:
        st.error(f"❌ Failed to load test data: {e}")
        raise

X_test, y_test, filenames_test = load_test_data()

# Night mode CSS (inspired by old app)
def get_css(night_mode):
    if night_mode:
        return """
        <style>
        .main {
            background: linear-gradient(135deg, #1a1a1a 0%, #2c2c2c 100%);
            padding: 20px;
            color: #e0e0e0;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }
        .sidebar .sidebar-content {
            background: #2c3e50;
            color: #e0e0e0;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        }
        h1, h2, h3 {
            color: #66b3ff;
            font-weight: 600;
        }
        .stButton>button {
            background: #007bff;
            color: #ffffff;
            border-radius: 10px;
            padding: 10px 20px;
            border: none;
            font-weight: 500;
            transition: all 0.3s ease;
            box-shadow: 0 2px 8px rgba(0, 123, 255, 0.3);
        }
        .stButton>button:hover {
            background: #0056b3;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 86, 179, 0.4);
        }
        .stSelectbox, .stTextInput, .stFileUploader, .stForm {
            background: #2c3e50;
            border-radius: 10px;
            padding: 10px;
            color: #e0e0e0;
            border: 1px solid #495057;
        }
        .stMetric {
            background: #343a40;
            color: #e0e0e0;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        }
        .tooltip {
            position: relative;
            display: inline-block;
            cursor: help;
            color: #66b3ff;
        }
        .tooltip .tooltiptext {
            visibility: hidden;
            max-width: 200px;
            background: #444;
            color: #e0e0e0;
            text-align: center;
            border-radius: 8px;
            padding: 8px;
            position: absolute;
            z-index: 1000;
            top: -5px;
            left: 50%;
            transform: translateX(-50%) translateY(-100%);
            opacity: 0;
            transition: opacity 0.3s;
        }
        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }
        </style>
        """
    else:
        return """
        <style>
        .main {
            background: linear-gradient(135deg, #f5f5f5 0%, #e0e0e0 100%);
            padding: 20px;
            color: #333333;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }
        .sidebar .sidebar-content {
            background: #e6f0fa;
            color: #333333;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        }
        h1, h2, h3 {
            color: #003366;
            font-weight: 600;
        }
        .stButton>button {
            background: #0066cc;
            color: #ffffff;
            border-radius: 10px;
            padding: 10px 20px;
            border: none;
            font-weight: 500;
            transition: all 0.3s ease;
            box-shadow: 0 2px 8px rgba(0, 102, 204, 0.2);
        }
        .stButton>button:hover {
            background: #0055aa;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 85, 170, 0.3);
        }
        .stSelectbox, .stTextInput, .stFileUploader, .stForm {
            background: #ffffff;
            border-radius: 10px;
            padding: 10px;
            color: #333333;
            border: 1px solid #ced4da;
        }
        .stMetric {
            background: #ffffff;
            color: #333333;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        }
        .tooltip {
            position: relative;
            display: inline-block;
            cursor: help;
            color: #0066cc;
        }
        .tooltip .tooltiptext {
            visibility: hidden;
            max-width: 200px;
            background: #555;
            color: #ffffff;
            text-align: center;
            border-radius: 8px;
            padding: 8px;
            position: absolute;
            z-index: 1000;
            top: -5px;
            left: 50%;
            transform: translateX(-50%) translateY(-100%);
            opacity: 0;
            transition: opacity 0.3s;
        }
        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }
        </style>
        """

# Visualization function
def visualize_inference(image, true_label_idx=None, filename="Image", pred1=None, pred2=None, fused_pred=None, night_mode=False):
    true_label = labels[true_label_idx] if true_label_idx is not None else "Unknown"
    pred1_class = np.argmax(pred1) if pred1 is not None else 0
    pred2_class = np.argmax(pred2) if pred2 is not None else 0
    fused_class = np.argmax(fused_pred) if fused_pred is not None else 0
    pred1_conf = pred1[pred1_class] if pred1 is not None else 0.0
    pred2_conf = pred2[pred2_class] if pred2 is not None else 0.0
    fused_conf = fused_pred[fused_class] if fused_pred is not None else 0.0
    pred_labels = [labels[pred1_class], labels[pred2_class], labels[fused_class]]
    correct = [pred1_class == true_label_idx, pred2_class == true_label_idx, fused_class == true_label_idx] if true_label_idx is not None else [False, False, False]

    # Colors (inspired by old app)
    bg_color = '#1a1a1a' if night_mode else '#f5f5f5'
    title_color = '#66b3ff' if night_mode else '#003366'
    text_color = '#e0e0e0' if night_mode else '#000000'
    accent_color = '#007bff' if night_mode else '#0066cc'  # Blue for uploads
    correct_color = '#008000'  # Green for correct
    incorrect_color = '#FF0000'  # Red for incorrect

    # Create figure
    fig = plt.figure(figsize=(8,5), facecolor=bg_color)
    gs = plt.GridSpec(3, 3, height_ratios=[0.1, 0.5, 0.4], hspace=0.3, wspace=0.2)

    # Title
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis('off')
    title_text = f"Image {filename.split('.')[0]} | True: {true_label} | Fusion: {pred_labels[2]} ({fused_conf:.2f})" if true_label_idx is not None else f"Image {filename.split('.')[0]} | Fusion: {pred_labels[2]} ({fused_conf:.2f})"   
    title_color_to_use = correct_color if true_label_idx is not None and correct[2] else incorrect_color if true_label_idx is not None else accent_color
    ax_title.text(
        0.5, 0.5, title_text, fontsize=10, fontweight='bold', ha='center', va='center',
        color=title_color_to_use,
        bbox=dict(facecolor='#343a40' if night_mode else 'white', edgecolor=accent_color, boxstyle='round,pad=0.3')
    )

    # Image panel
    ax_img = fig.add_subplot(gs[1, 1])
    ax_img.imshow(image, cmap='gray')
    ax_img.set_title("Brain MRI", fontsize=10, fontweight='bold', color=title_color, pad=5)
    if true_label_idx is not None:
        ax_img.text(
            10, 20, f"True: {true_label}",
            color='white', fontweight='bold', fontsize=6,
        bbox=dict(facecolor='black', edgecolor='white', linewidth=0.4, alpha=0.7, boxstyle='round,pad=0.2')
        )
    ax_img.text(
        10, image.shape[0] - 20, f"Fusion: {pred_labels[2]} ({fused_conf:.2f})",
        color='white', fontweight='bold', fontsize=6,
        bbox=dict(facecolor='black', edgecolor='white', linewidth=0.4, alpha=0.7, boxstyle='round,pad=0.2')
    )
    ax_img.axis('off')
    plt.colorbar(ax_img.imshow(image, cmap='gray'), ax=ax_img, fraction=0.046, pad=0.04, cmap='gray')

    # Model info boxes
    ax1 = fig.add_subplot(gs[2, 0])
    ax1.axis('off')
    ax1.text(0.5, 0.8, "EfficientnetV2B0", fontsize=8, fontweight='bold', ha='center', color=title_color)
    ax1.text(
        0.5, 0.65, f"Pred: {pred_labels[0]}",
        ha='center', fontsize=8,
        color=correct_color if true_label_idx is not None and correct[0] else incorrect_color if true_label_idx is not None else accent_color
    )
    ax1.text(0.5, 0.55, f"Conf: {pred1_conf:.2f}", ha='center', fontsize=8, color=text_color)
    ax1.text(
        0.5, 0.45, f"Prob: ({', '.join([f'{p:.2f}' for p in pred1])})",
        ha='center', fontsize=7, color=text_color
    )

    ax2 = fig.add_subplot(gs[2, 1])
    ax2.axis('off')
    ax2.text(0.5, 0.8, "ConvNeXtTiny", fontsize=8, fontweight='bold', ha='center', color=title_color)
    ax2.text(
        0.5, 0.65, f"Pred: {pred_labels[1]}",
        ha='center', fontsize=8,
        color=correct_color if true_label_idx is not None and correct[1] else incorrect_color if true_label_idx is not None else accent_color
    )
    ax2.text(0.5, 0.55, f"Conf: {pred2_conf:.2f}", ha='center', fontsize=8, color=text_color)
    ax2.text(
        0.5, 0.45, f"Prob: ({', '.join([f'{p:.2f}' for p in pred2])})",
        ha='center', fontsize=7, color=text_color
    )

    ax3 = fig.add_subplot(gs[2, 2])
    ax3.axis('off')
    ax3.text(0.5, 0.8, "Fusion", fontsize=9, fontweight='bold', ha='center', color=title_color)
    ax3.text(
        0.5, 0.65, f"Pred: {pred_labels[2]}",
        ha='center', fontsize=8,
        color=correct_color if true_label_idx is not None and correct[2] else incorrect_color if true_label_idx is not None else accent_color
    )
    ax3.text(0.5, 0.55, f"Conf: {fused_conf:.2f}", ha='center', fontsize=8, color=text_color)
    ax3.text(
        0.5, 0.45, f"Prob: ({', '.join([f'{p:.2f}' for p in fused_pred])})",
        ha='center', fontsize=7, color=text_color
    )

    # Manual layout adjustment instead of tight_layout
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.3, hspace=0.4)

    return fig

# Image preprocessing for uploads
def process_uploaded_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to load image")
    img = cv2.resize(img, (224, 224))
    img_array = np.array(img)
    return img_array

# Initialize session state
if 'night_mode' not in st.session_state:
    st.session_state.night_mode = False


# Apply CSS
st.markdown(get_css(st.session_state.night_mode), unsafe_allow_html=True)

# Define background color for figure saving
bg_color = '#1a1a1a' if st.session_state.night_mode else '#f5f5f5'

# Sidebar
with st.sidebar:
    st.title("⚙️ App Panel")
    night_mode = st.toggle(
        "🌙 Night Mode",
        value=st.session_state.night_mode,
        key="night_mode_toggle",
        help="Toggle between light and dark themes for better visibility."
    )
    if night_mode != st.session_state.night_mode:
        st.session_state.night_mode = night_mode
        st.rerun()
    
    st.success(f"{'GPU' if gpus else 'CPU'} in use")
    with st.expander("ℹ️ About the Project"):
        st.markdown("""
        **Objective**: Predict brain tumor types using MRI scans.
        - **Models**: EfficientnetV2B0, ConvNeXtTiny, and late fusion
        - **Classes**: Glioma, Meningioma, No Tumor, Pituitary
        - **Tabs**:
          - 📊 Explore Test Scans: Analyze preloaded test images
          - 📤 Predict New Image: Upload and predict on new MRI scans
        - **Colors**: 🟢 Correct, 🔴 Incorrect, 🔵 Upload predictions
        """)
    st.info("Select or upload an image to start.")

# Main content
st.header("🧠 Brain Tumor Classification Analysis")
st.markdown("Use preloaded MRI scans or upload your own to predict tumor types.")

# Tabs
tab1, tab2 = st.tabs(["📊 Explore Test Scans", "📤 Predict New Image"])

# Tab 1: Explore Test Scans
with tab1:
    st.subheader("Browse Test Images")
    with st.container():
        image_options = [f.split('.')[0] for f in filenames_test]
        selected_image = st.selectbox(
            "Select Image:",
            ['Select an image'] + image_options,
            key="test_image_select",
            help="Choose a test MRI image for prediction"
        )
        
        if selected_image != 'Select an image':
            image_index = image_options.index(selected_image)
            img = X_test[image_index]
            true_label_idx = np.argmax(y_test[image_index])
            filename = filenames_test[image_index]  # Use actual filename from filenames_test

            with st.form(key="test_predict_form"):
                submit_button = st.form_submit_button("🚀 Run Prediction")

            if submit_button:
                with st.spinner("Running prediction..."):
                    pred1 = models[0].predict(img.reshape(1, 224, 224, 3), verbose=0)[0]
                    pred2 = models[1].predict(img.reshape(1, 224, 224, 3), verbose=0)[0]
                    fused_pred = (pred1 + pred2) / 2

                st.markdown("### Prediction Results")
                with st.container():
                    col1, col2, col3 = st.columns(3)
                    col1.metric("True Label", labels[true_label_idx])
                    col2.metric("Predicted Label", labels[np.argmax(fused_pred)])
                    col3.metric("Fusion Confidence", f"{fused_pred[np.argmax(fused_pred)]:.2f}")

                fig = visualize_inference(img, true_label_idx, filename, pred1, pred2, fused_pred, night_mode=st.session_state.night_mode)
                st.pyplot(fig)

                # Download PNG
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=300, facecolor=bg_color, bbox_inches='tight')
                buf.seek(0)
                st.download_button(
                    label="📥 Download Visualization",
                    data=buf,
                    file_name=f"{filename.split('.')[0]}_prediction.png",
                    mime="image/png",
                    key="download_test"
                )
                plt.close(fig)

# Tab 2: Predict New Image
with tab2:
    st.subheader("Upload MRI for Prediction")
    with st.container():
        with st.form(key="upload_form"):
            col1, col2 = st.columns([2, 1])
            with col1:
                patient_id = st.text_input(
                    "Patient ID (Optional):",
                    help="Enter a unique identifier for the image"
                )
                uploaded_file = st.file_uploader(
                    "Upload MRI Image",
                    type=["png", "jpg", "jpeg"],
                    key="upload_image",
                    help="Upload a single MRI image (PNG/JPG/JPEG)"
                )
            with col2:
                with st.expander("📋 Instructions"):
                    st.markdown("""
                    - 🖼️ Upload one MRI image (PNG/JPG/JPEG)
                    - 📏 Image will be resized to 224x224
                    - 🔍 Prediction uses EfficientnetV2B0, ConvNeXtTiny, and fusion
                    """)
            submit_button = st.form_submit_button("🚀 Run Prediction")
        
        if uploaded_file and submit_button:
            try:
                img_array = process_uploaded_image(uploaded_file)
                with st.container():
                    st.markdown("### Image Preview")
                    st.image(img_array, caption="Uploaded Image", width=200, use_container_width=False)
                
                with st.spinner("Processing image..."):
                    pred1 = models[0].predict(img_array.reshape(1, 224, 224, 3), verbose=0)[0]
                    pred2 = models[1].predict(img_array.reshape(1, 224, 224, 3), verbose=0)[0]
                    fused_pred = (pred1 + pred2) / 2

                st.markdown("### Prediction Results")
                with st.container():
                    col1, col2 = st.columns(2)
                    col1.metric("Predicted Label", labels[np.argmax(fused_pred)])
                    col2.metric("Fusion Confidence", f"{fused_pred[np.argmax(fused_pred)]:.2f}")

                fig = visualize_inference(img_array, filename=patient_id or "Uploaded", pred1=pred1, pred2=pred2, fused_pred=fused_pred, night_mode=st.session_state.night_mode)
                st.pyplot(fig)

                # Download PNG
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=300, facecolor=bg_color, bbox_inches='tight')
                buf.seek(0)
                st.download_button(
                    label="📥 Download Visualization",
                    data=buf,
                    file_name=f"prediction_{patient_id or 'uploaded'}.png",
                    mime="image/png",
                    key="download_upload"
                )
                plt.close(fig)
            except Exception as e:
                st.error(f"❌ Error processing image: {e}")