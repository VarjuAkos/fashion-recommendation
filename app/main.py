import streamlit as st
import pickle
import os
import numpy as np
from PIL import Image
from app.object_detection import detect_objects
from app.feature_extraction import FeatureExtractor
from streamlit_cropper import st_cropper

MODEL_PATH = os.path.join('data', 'models', 'fashion_knn_model.pkl')
DATA_DIR = os.path.join('data', 'images')
MIN_DISPLAY_WIDTH = 600
MIN_DISPLAY_HEIGHT = 800

def preprocess_image(upload):
    try:
        image = Image.open(upload)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Resize image if it's too small - UI cant handle small images well
        if image.width < MIN_DISPLAY_WIDTH or image.height < MIN_DISPLAY_HEIGHT:
            aspect_ratio = image.width / image.height
            if aspect_ratio > MIN_DISPLAY_WIDTH / MIN_DISPLAY_HEIGHT:
                display_size = (MIN_DISPLAY_WIDTH, int(MIN_DISPLAY_WIDTH / aspect_ratio))
            else:
                display_size = (int(MIN_DISPLAY_HEIGHT * aspect_ratio), MIN_DISPLAY_HEIGHT)
            image = image.resize(display_size, Image.LANCZOS)
            
        image_array = np.array(image)
        # Ensure the image has 3 channels
        if image_array.shape[-1] != 3:
            raise ValueError("Image must have 3 color channels")
        processed_image = Image.fromarray(image_array)
        print("--Image processed--")
        
        return processed_image, None
    except Exception as e:
        return None, str(e)
    
# Load the pre-trained model and data
@st.cache_resource
def load_model_and_data():
    with open(MODEL_PATH, "rb") as f:
        knn, _, image_paths = pickle.load(f)
    
    image_paths = []
    for path in os.listdir(DATA_DIR):
        if path.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_paths.append(os.path.join(DATA_DIR, path))
    
    return knn, image_paths

@st.cache_resource
def get_feature_extractor():
    return FeatureExtractor()

def main():
    st.set_page_config(
        page_title="BePacek",
        page_icon="🔥"
    )

    st.markdown("<h1 style='text-align: center; color: white;'>💪 BePacek 👠</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: white;'>✨ We recommend similar fashion items based on your clothes. ✨</h3>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: white;'>📸 Take an image to get started with AI magic 🧠</h3>", unsafe_allow_html=True)

    knn, image_paths = load_model_and_data()
    
    feature_extractor = get_feature_extractor()

    uploaded_file = st.camera_input("Quickstart", label_visibility="hidden")
    # uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], accept_multiple_files=False, label_visibility="collapsed")

    if uploaded_file:
        print("--Cheese 😎--")
        image, error = preprocess_image(uploaded_file)
        if error:
            st.error(f"Error processing image: {error}")
            st.stop()

        with st.spinner('Detecting clothes 👀'):
            detected_items = detect_objects(image)

        if detected_items:
            item_options = [f"{item[1]} ({item[2]:.2f}%)" for item in detected_items]
            selected_item = st.selectbox("Detected items:", item_options)
        else:
            st.warning("No clothes were detected on the image.")

        # Image cropping
        st.write("Uploaded Image | Select area of interest:")
        aspect_ratio = (3, 4)
        
        cropped_img = st_cropper(image, aspect_ratio=aspect_ratio, box_color = 'red', return_type="image",stroke_width=2)
        print("--Image cropped--")

        if st.button("Get Recommendations"):
            # Extract features
            with st.spinner('Extracting features...'):
                features = feature_extractor.extract_features(cropped_img)
                print("--Features extracted--")

            # Get recommendations
            with st.spinner('Finding similar items...'):
                distances, indices = knn.kneighbors([features], n_neighbors=6)
            
            # Display recommendations
            st.subheader("Recommended Items:")
            cols = st.columns(5)
            for i, idx in enumerate(indices[0][1:]):  # Skip the first one as it's the input image
                with cols[i % 5]:
                    rec_img = Image.open(image_paths[idx])
                    st.image(rec_img, caption=f"Recommendation {i+1}", use_column_width=True)
            print("--Recommendations displayed--")
if __name__ == "__main__":
    main()