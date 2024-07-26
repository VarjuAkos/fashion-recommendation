import streamlit as st
import numpy as np
import pickle
from PIL import Image
from object_detection import detect_objects
from feature_extraction import FeatureExtractor
from streamlit_cropper import st_cropper
import tensorflow as tf

# Load the pre-trained model and data
@st.cache_resource
def load_model_and_data():
    with open("fashion_knn_model.pkl", "rb") as f:
        knn, feature_array, image_paths = pickle.load(f)
    return knn, feature_array, image_paths

# Create feature extractor outside of Streamlit cache
@st.cache_resource
def get_feature_extractor():
    return FeatureExtractor()

def main():
    st.title("BePacek👗👠\n - The one and only Fashion Recommendation System👒👜")

    # Load model and data
    knn, feature_array, image_paths = load_model_and_data()
    
    # Get feature extractor
    feature_extractor = get_feature_extractor()

    # Discpription of the app and how to use it
    st.write("""
    
            ✨ This app uses a 🧠 machine learning model to recommend similar fashion items based on an image you provide. ✨ \n
            📸 Upload an Image of a fashion item using the file uploader below, then click the 'Get Recommendations' button to see similar items. 👕\n
            ⬇️ Upload Your Fashion Item Here ⬇️""")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Perform object detection
        with st.spinner('Detecting items in the image...'):
            detected_items = detect_objects(image)

         # Create a combo box for detected items
        if detected_items:
            item_options = [f"{item[1]} ({item[2]:.2f}%)" for item in detected_items]
            selected_item = st.selectbox("Detected item:", item_options)
        else:
            st.warning("No items detected in the image.")


        # Image cropping
        st.write("Select area of interest:")
        aspect_ratio = (60, 80)
        cropped_img = st_cropper(image, aspect_ratio=aspect_ratio, return_type="image")

        # Display cropped image
        st.image(cropped_img, caption='Selected Area', use_column_width=True)

        if st.button("Get Recommendations"):
            # Extract features
            with st.spinner('Extracting features...'):
                features = feature_extractor.extract_features(cropped_img)

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

if __name__ == "__main__":
    main()