import streamlit as st
from PIL import Image
import numpy as np
import os
from object_detection import detect_objects
from feature_extraction import FeatureExtractor
from recommendation import RecommendationSystem

# Initialize components
feature_extractor = FeatureExtractor()
recommender = RecommendationSystem()

# Load or create recommendation system
if os.path.exists("recommender.pkl"):
    recommender.load("recommender.pkl")
else:
    st.warning("Recommendation system not found. Please add items to the system.")

def main():
    st.title("Fashion Recommendation System")

    # Sidebar for adding new items
    st.sidebar.title("Add New Item")
    new_item = st.sidebar.file_uploader("Choose an image...", key="new_item")
    if new_item:
        new_image = Image.open(new_item)
        new_features = feature_extractor.extract_features(new_image)
        recommender.add_item(new_features, new_item.name)
        recommender.fit()
        recommender.save("recommender.pkl")
        st.sidebar.success(f"Added {new_item.name} to the system!")

    # Main content
    uploaded_file = st.file_uploader("Choose an image for recommendations...", key="recommendation")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        with st.spinner('Processing image...'):
            detected_items = detect_objects(image)
            features = feature_extractor.extract_features(image)

        if detected_items:
            st.write("Detected Items:", ", ".join(detected_items))
            
            selected_items = st.multiselect("Select items for recommendations:", detected_items)

            if st.button("Get Recommendations"):
                if selected_items:
                    recommendations = recommender.get_recommendations(features)
                    
                    for i, rec in enumerate(recommendations):
                        st.write(f"Recommendation {i+1}:")
                        rec_image = Image.open(rec)  # This assumes recommendations are file paths
                        st.image(rec_image, caption=f"Recommended Item {i+1}", width=200)
                else:
                    st.warning("Please select at least one item for recommendations.")
        else:
            st.warning("No clothing items detected in the image. Please try another image.")

if __name__ == "__main__":
    main()