import streamlit as st
from PIL import Image
import numpy as np
from object_detection import detect_objects
from feature_extraction import FeatureExtractor

# Initialize the feature extractor
feature_extractor = FeatureExtractor()

def main():
    st.title("BePacek - Fashion Recommendation System")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Perform object detection
        with st.spinner('Detecting items in the image...'):
            detected_items = detect_objects(image)
        
        if detected_items:
            st.write("Detected Items:")
            st.write(", ".join(detected_items))
            
            # Extract features
            with st.spinner('Extracting features...'):
                features = feature_extractor.extract_features(image)
            st.write(f"Feature vector shape: {features.shape}")
            
            # Multi-select box for detected items
            selected_items = st.multiselect("Select items for recommendations:", detected_items)

            if st.button("Search for recommendations"):
                if selected_items:
                    for item in selected_items:
                        st.write(f"Recommendations for {item}:")
                        # Placeholder for recommendations
                        for i in range(6):
                            st.image(np.random.rand(100,100,3), caption=f"Recommended {item} {i+1}", width=100)
                else:
                    st.warning("Please select at least one item for recommendations.")
        else:
            st.warning("No clothing items detected in the image. Please try another image.")

if __name__ == "__main__":
    main()