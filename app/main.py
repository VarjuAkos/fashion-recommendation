import streamlit as st
import pickle
import os
from PIL import Image
from app.object_detection import detect_objects
from app.feature_extraction import FeatureExtractor
from streamlit_cropper import st_cropper

MODEL_PATH = os.path.join('data', 'models', 'fashion_knn_model.pkl')
DATA_DIR = os.path.join('data', 'images')

def image_to_base64(image):
    import base64
    from io import BytesIO

    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# Load the pre-trained model and data
@st.cache_resource
def load_model_and_data():
    with open(MODEL_PATH, "rb") as f:
        knn, feature_array, image_paths = pickle.load(f)
    
    image_paths = []
    for path in os.listdir(DATA_DIR):
        if path.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_paths.append(os.path.join(DATA_DIR, path))
    
    return knn, feature_array, image_paths

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
        #st.image(image, caption='Uploaded Image', use_column_width=True)

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
        st.write("Uploaded Image | Select area of interest:")
        aspect_ratio = (3, 4)
        
        cropped_img = st_cropper(image, aspect_ratio=aspect_ratio, box_color = 'red', return_type="image")

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