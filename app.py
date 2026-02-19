import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from model import AudioCNN
from utils import AudioProcessor, process_uploaded_audio
import os
import io
import librosa

# Page Config
st.set_page_config(
    page_title="SonicVision: The AI Audio Brain",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Wow" factor
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: #FFFFFF;
    }
    
    /* Header Styling */
    h1 {
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #00d2ff, #3a7bd5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding-bottom: 20px;
    }
    
    h2, h3, h4, h5, h6, p, span, div, label, .stMarkdown, .stMetricLabel, .stMetricValue {
        color: #FFFFFF !important;
    }
    
    /* Force white text on all labels including widget labels */
    .stTextInput > label, .stSelectbox > label, .stFileUploader > label {
        color: #FFFFFF !important;
    }
    
    /* Fix file uploader button text color (Browse files) */
    [data-testid="stFileUploader"] button {
        color: #000000 !important;
    }
    [data-testid="stFileUploader"] button * {
        color: #000000 !important;
    }
    /* Fix file uploader instructions (Drag and drop, Limit 200MB) */
    [data-testid="stFileUploader"] section {
        color: #000000 !important;
    }
    [data-testid="stFileUploader"] section * {
        color: #000000 !important;
    }
    
    /* Card/Container Styling */
    .stCard {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(0, 0, 0, 0.2);
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 50px;
        font-weight: bold;
        transition: transform 0.2s;
    }
    .stButton>button:hover {
        transform: scale(1.05);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = 'best_model.pth'
    
    if not os.path.exists(model_path):
        st.error(f"Model file '{model_path}' not found. Please ensure it is in the project directory.")
        return None, None, None, None

    checkpoint = torch.load(model_path, map_location=device)
    classes = checkpoint['classes']
    model = AudioCNN(num_classes=len(classes))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    processor = AudioProcessor()
    
    return model, classes, device, processor

def app():
    st.title("🧠 SonicVision")
    st.markdown("<h3 style='text-align: center; margin-top: -20px; margin-bottom: 40px;'>The AI Audio Brain Visualizer</h3>", unsafe_allow_html=True)

    # Load Model
    model, classes, device, processor = load_model()
    
    if model is None:
        return

    # Sidebar
    st.sidebar.header("Input Audio")
    uploaded_file = st.sidebar.file_uploader("Upload a WAV or MP3 file", type=['wav', 'mp3'])
    
    # Example Audio Buttons
    st.sidebar.markdown("---")
    st.sidebar.subheader("Or try an example:")
    if st.sidebar.button("🐦 Chirping Birds"):
        example_path = "chirping_birds.wav"
        if os.path.exists(example_path):
            with open(example_path, "rb") as f:
                # Read into memory immediately to avoid closed file errors
                file_bytes = f.read()
                uploaded_file = io.BytesIO(file_bytes)
                uploaded_file.name = "chirping_birds.wav" # fake name for display if needed
        else:
            st.sidebar.error("Example file 'chirping_birds.wav' not found.")

    if uploaded_file:
        # Process Audio
        # Ensure we are at the start of the file
        if hasattr(uploaded_file, 'seek'):
            uploaded_file.seek(0)
            
        audio_bytes = uploaded_file.read()
        
        # Display Audio Player
        # Container for visual grouping
        st.markdown('<div class="stCard">', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("🎧 Input")
            st.audio(audio_bytes, format='audio/wav')
            
            # Process for waveforms/spectrograms
            # Reset pointer again for processing function if it reads it again (though we passed bytes)
            audio_data, sample_rate = process_uploaded_audio(audio_bytes)
            st.metric("Duration", f"{len(audio_data)/sample_rate:.2f}s")

        with col2:
            st.subheader("📊 Spectrogram")
            processed_spectrogram = processor.process_audio_chunk(audio_data)
            spectrogram_np = processed_spectrogram.squeeze().numpy()
            
            fig, ax = plt.subplots(figsize=(10, 3))
            # Dark background for plot
            fig.patch.set_facecolor('none')
            ax.set_facecolor('none')
            
            img = librosa.display.specshow(spectrogram_np, sr=44100, x_axis='time', y_axis='hz', ax=ax, cmap='magma')
            cbar = fig.colorbar(img, ax=ax, format='%+2.0f dB')
            
            # Colorbar text color
            cbar.ax.yaxis.set_tick_params(color='white')
            plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
            
            # Axis colors
            ax.spines['bottom'].set_color('white')
            ax.spines['top'].set_color('white') 
            ax.spines['left'].set_color('white')
            ax.spines['right'].set_color('white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.tick_params(axis='x', colors='white')
            ax.tick_params(axis='y', colors='white')
            ax.set_title("Mel Spectrogram (Input to AI)", color='white')
            
            st.pyplot(fig)
        
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # 2. AI Inference
        processed_spectrogram = processed_spectrogram.to(device)
        with torch.no_grad():
            outputs, feature_maps = model(processed_spectrogram, return_features_maps=True)
            probabilities = nn.functional.softmax(outputs, dim=1)
            top3_probs, top3_indices = torch.topk(probabilities[0], 3)

        # Display Predictions
        st.markdown('<div class="stCard">', unsafe_allow_html=True)
        st.subheader("🤖 AI Classification")
        pred_cols = st.columns(3)
        for i, (prob, idx) in enumerate(zip(top3_probs, top3_indices)):
            class_name = classes[idx]
            confidence = prob.item()
            with pred_cols[i]:
                st.metric(label=f"Rank #{i+1}", value=class_name)
                # Custom progress bar color logic can be added here if needed, 
                # but standard st.progress is used for now.
                st.progress(confidence)
                st.caption(f"Confidence: {confidence:.2%}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)

        # 3. Inside the Brain (Feature Maps)
        st.markdown('<div class="stCard">', unsafe_allow_html=True)
        st.header("🧠 Inside the AI's Brain: Layer Activations")
        st.write("Visualize what different layers of the Neural Network are 'seeing'. Early layers detect edges/textures; deep layers detect abstract concepts.")
        
        layer_names = list(feature_maps.keys())
        # Filter layers to show interesting ones
        shown_layers = ['conv1', 'layer1', 'layer2', 'layer3', 'layer4']
        shown_layers = [l for l in shown_layers if l in layer_names] 

        tabs = st.tabs([name.upper() for name in shown_layers])

        for i, layer_name in enumerate(shown_layers):
            with tabs[i]:
                features = feature_maps[layer_name]
                
                if features.dim() == 4:
                    activation_map = torch.mean(features, dim=1).squeeze().cpu().numpy()
                    
                    st.write(f"**Layer {layer_name} Summary Activation Map**")
                    
                    fig, ax = plt.subplots(figsize=(12, 4))
                    fig.patch.set_facecolor('none')
                    ax.set_facecolor('none')
                    
                    sns.heatmap(activation_map, cmap='viridis', ax=ax)
                    ax.axis('off')
                    st.pyplot(fig)
                    
                    st.markdown("#### Individual Channel Activations (First 4 Filters)")
                    if features.size(1) >= 4:
                        cols = st.columns(4)
                        for j in range(4):
                            filter_map = features[0, j, :, :].cpu().numpy()
                            with cols[j]:
                                fig, ax = plt.subplots(figsize=(3, 2))
                                fig.patch.set_facecolor('none')
                                ax.imshow(filter_map, cmap='inferno')
                                ax.axis('off')
                                ax.set_title(f"Filter {j}", color='white', fontsize=10)
                                st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    app()
