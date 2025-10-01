import streamlit as st
import numpy as np
import cv2
from PIL import Image
from src.model import MLP

class_name = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z',
    'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't'
]

@st.cache_resource
def load_model():
    try:
        layer_dims = [784, 512, 256, 47]
        mlp = MLP(layer_dims=layer_dims, init='he', lr=0.01)
        weights = np.load('models/mlp_weights.npz')
        for key in weights.keys():
            mlp.parameters[key] = weights[key]
        return mlp
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def preprocess_image(image):
    img_array = np.array(image)
    
    if len(img_array.shape) == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    img_resized = cv2.resize(img_array, (28, 28))
    
    img_inverted = 255 - img_resized
    
    img_normalized = img_inverted.astype(np.float32) / 255.0
    
    img_normalized = (img_normalized - 0.13) / 0.30
    
    img_flattened = img_normalized.flatten()
    
    img_input = img_flattened.reshape(-1, 1)
    
    return img_input, img_normalized

def predict_character(model, processed_image):
    try:
        pred, _ = model.forward(processed_image)
        
        pred_class = np.argmax(pred, axis=0)[0]
        confidence = np.max(pred, axis=0)[0]
        
        pred_name = class_name[pred_class]
        
        top3_idxs = np.argsort(pred[:, 0])[-3:][::-1]
        top3_probs = pred[top3_idxs, 0]
        top3_preds = [class_name[i] for i in top3_idxs]
        
        return pred_name, confidence, top3_preds, top3_probs
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, None, None, None

def main():
    st.set_page_config(
        page_title="Character Recognition",
        page_icon="🔤",
        layout="wide"
    )
    
    st.markdown("<h1 style='text-align: center;'>🔤 Character Recognition</h1>",
    unsafe_allow_html=True)

    st.markdown("<p style='text-align: center;'>Upload an image of a handwritten character and get predictions from my MLP model!</p>",
    unsafe_allow_html=True)
    
    model = load_model()
    if model is None:
        st.error("Failed to load model. Please check if 'models/mlp_weights.npz' exists.")
        return
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("<h2 style='text-align: center;'>📤 Upload Image</h2>", unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            help="Upload an image containing a single handwritten character"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Original Image", width='stretch')
            
            processed_img, normalized_img = preprocess_image(image)
    
    with col2:
        st.markdown("<h2 style='text-align: center;'>🤖 Prediction Results</h2>", unsafe_allow_html=True)
        
        if uploaded_file is not None:
            with st.spinner("Making prediction..."):
                pred_name, confidence, top3_preds, top3_probs = predict_character(model, processed_img)
            
            if pred_name is not None:
                st.markdown(f"<h1 style='text-align: center; color: #1f77b4; font-size: 72px;'>{pred_name}</h1>", 
                          unsafe_allow_html=True)
                st.markdown(f"<p style='text-align: center; font-size: 24px;'>Confidence: {confidence:.2%}</p>", 
                          unsafe_allow_html=True)
                
                st.subheader("📊 Top 3 Predictions")
                if top3_preds is not None and top3_probs is not None:
                    for i, (cls, prob) in enumerate(zip(top3_preds, top3_probs)):
                        col_rank, col_char, col_prob = st.columns([1, 2, 3])
                        
                        with col_rank:
                            st.write(f"#{i+1}")
                        with col_char:
                            st.markdown(f"**{cls}**")
                        with col_prob:
                            st.progress(prob)
                            st.write(f"{prob:.2%}")
                
                st.subheader("Preprocessed Image (28x28)")
                st.image(normalized_img, caption="Processed for Model", width=200, clamp=True)
        else:
            st.info("Upload an image to see predictions")

if __name__ == "__main__":
    main()
