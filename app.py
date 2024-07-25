import streamlit as st
import numpy as np
import pickle
from PIL import Image, ImageDraw
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from object_detection import detect_objects_with_boxes

# Load the pre-trained model and data
@st.cache_resource
def load_model_and_data():
    with open("fashion_knn_model.pkl", "rb") as f:
        knn, feature_array, image_paths = pickle.load(f)
    return knn, feature_array, image_paths

# Feature extraction function
@st.cache_resource
def get_feature_extractor():
    base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(80, 60, 3))
    x = base_model.output
    x = tf.keras.layers.GlobalMaxPooling2D()(x)
    return tf.keras.Model(inputs=base_model.input, outputs=x)

# Preprocess and extract features from an image
def extract_features(img, feature_extractor):
    img = img.resize((60, 80))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = feature_extractor.predict(img_array)
    return features.flatten()

# Main function
def main():
    st.title("Fashion Recommendation System")

    # Load model and data
    knn, feature_array, image_paths = load_model_and_data()
    feature_extractor = get_feature_extractor()

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Perform object detection
        with st.spinner('Detecting items in the image...'):
            detected_items = detect_objects_with_boxes(image)

        if detected_items:
            # Display image with bounding boxes
            draw = ImageDraw.Draw(image)
            for item, box in detected_items:
                draw.rectangle(box, outline="red", width=2)
            st.image(image, caption='Detected Items', use_column_width=True)

            # Let user select an item
            item_names = [item for item, _ in detected_items]
            selected_item = st.selectbox("Select an item for recommendations:", item_names)

            if st.button("Get Recommendations"):
                # Get bounding box for selected item
                selected_box = next(box for item, box in detected_items if item == selected_item)

                # Crop and resize the selected item
                cropped_img = image.crop(selected_box)
                resized_img = Image.new('RGB', (60, 80), (255, 255, 255))
                cropped_img.thumbnail((60, 80), Image.LANCZOS)
                offset = ((60 - cropped_img.width) // 2, (80 - cropped_img.height) // 2)
                resized_img.paste(cropped_img, offset)

                # Extract features
                features = extract_features(resized_img, feature_extractor)

                # Get recommendations
                distances, indices = knn.kneighbors([features], n_neighbors=6)
                
                # Display recommendations
                st.subheader("Recommended Items:")
                cols = st.columns(5)
                for i, idx in enumerate(indices[0][1:]):  # Skip the first one as it's the input image
                    with cols[i % 5]:
                        rec_img = Image.open(image_paths[idx])
                        st.image(rec_img, caption=f"Recommendation {i+1}", use_column_width=True)

        else:
            st.warning("No clothing items detected in the image. Please try another image.")

if __name__ == "__main__":
    main()