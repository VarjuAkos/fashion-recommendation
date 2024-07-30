import os
import numpy as np
from PIL import Image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.models import Model
from sklearn.neighbors import NearestNeighbors
import pickle


DATA_DIR = os.path.join('data', 'images')
MODEL_OUTPUT_DIR = os.path.join('data', 'models')
MODEL_OUTPUT_PATH = os.path.join(MODEL_OUTPUT_DIR, 'fashion_knn_model.pkl')

# Feature Extractor
def create_feature_extractor():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(60, 80, 3)) # Change input shape (224, 224, 3)
    x = base_model.output
    x = GlobalMaxPooling2D()(x)
    return Model(inputs=base_model.input, outputs=x)

# Image Preprocessing
def preprocess_image(img_path):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((60, 80)) # Change image size (224, 224)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)      # function is designed for the ImageNet dataset and is intended to normalize the input images to match the expected format.

# Extract Features
def extract_features(image_path, feature_extractor):
    img_array = preprocess_image(image_path)
    features = feature_extractor.predict(img_array) # generate predictions for input images
    return features.flatten() 

# Main Preprocessing and Training
def preprocess_and_train(dataset_path, output_path):
    feature_extractor = create_feature_extractor()
    
    features = []
    image_paths = []

    for img_name in os.listdir(dataset_path):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(dataset_path, img_name)
            try:
                feature = extract_features(img_path, feature_extractor)
                features.append(feature)
                image_paths.append(img_path)
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")

    feature_array = np.array(features)

    # Train KNN model
    knn = NearestNeighbors(n_neighbors=6, metric='cosine')
    knn.fit(feature_array)

    # Save the model, features, and image paths
    with open(output_path, 'wb') as f:
        pickle.dump((knn, feature_array, image_paths), f)

    print(f"Preprocessing complete. Model and data saved to {output_path}")

if __name__ == "__main__":
    preprocess_and_train(DATA_DIR, MODEL_OUTPUT_PATH)