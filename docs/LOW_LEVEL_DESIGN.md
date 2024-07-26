# Fashion Recommendation System: Low-Level Design

## 1. Component Overview

The Fashion Recommendation System consists of the following main components:

1. User Interface (app.py)
2. Object Detection (object_detection.py)
3. Feature Extraction (feature_extraction.py)
4. Data Preprocessing and Model Training (preprocess_and_train.py)
5. Recommendation Engine (integrated in app.py)

## 2. Detailed Component Design

### 2.1 User Interface (app.py)

The user interface is built using Streamlit and serves as the main entry point for the application.

Key Functions:
- `load_model_and_data()`: Loads the pre-trained KNN model and associated data.
- `get_feature_extractor()`: Creates and caches the FeatureExtractor instance.
- `main()`: The main function that sets up the Streamlit interface and orchestrates the workflow.

Workflow:
1. User uploads an image
2. Image is displayed and passed to object detection
3. Detected items are presented in a dropdown
4. User crops the image using an interactive tool
5. Cropped image is passed to feature extraction
6. Extracted features are used to find recommendations
7. Recommendations are displayed to the user

### 2.2 Object Detection (object_detection.py)

This component uses a pre-trained MobileNetV2 model for detecting fashion items in uploaded images.

Key Functions:
- `preprocess_image(image)`: Resizes and preprocesses the image for the model.
- `detect_objects(image)`: Performs object detection and returns detected items with confidence scores.

Workflow:
1. Image is preprocessed to fit model input requirements
2. MobileNetV2 model processes the image
3. Predictions are decoded and filtered for fashion-related items
4. Detected items are returned as a list of tuples (item_name, confidence_score)

### 2.3 Feature Extraction (feature_extraction.py)

This component uses a pre-trained ResNet50 model to extract feature vectors from images.

Key Class: `FeatureExtractor`

Methods:
- `__init__()`: Initializes the ResNet50 model without top layers and adds a GlobalMaxPooling2D layer.
- `extract_features(img)`: Processes an image and returns its feature vector.

Workflow:
1. Image is resized to 224x224 pixels
2. Image is preprocessed for ResNet50
3. Features are extracted using the modified ResNet50 model
4. Feature vector is flattened and normalized

### 2.4 Data Preprocessing and Model Training (preprocess_and_train.py)

This script prepares the dataset and trains the KNN model used for recommendations.

Key Functions:
- `create_feature_extractor()`: Creates the ResNet50-based feature extractor.
- `preprocess_image(img_path)`: Loads and preprocesses images from the dataset.
- `extract_features(image_path, feature_extractor)`: Extracts features from a single image.
- `preprocess_and_train(dataset_path, output_path)`: Main function that processes the entire dataset and trains the KNN model.

Workflow:
1. Load and preprocess each image in the dataset
2. Extract features from each preprocessed image
3. Compile all features into a single array
4. Train a KNN model on the feature array
5. Save the trained model, feature array, and image paths to a file

### 2.5 Recommendation Engine (integrated in app.py)

The recommendation engine uses the trained KNN model to find similar fashion items.

Key Components:
- KNN model (loaded from the saved file)
- Feature array and image paths (loaded from the saved file)

Workflow:
1. Receive feature vector of the user-selected image area
2. Use KNN model to find k nearest neighbors in the feature space
3. Retrieve the corresponding image paths for the nearest neighbors
4. Return these as recommendations

## 3. Data Flow

1. User uploads an image → UI (app.py)
2. Image → Object Detection (object_detection.py) → Detected Items → UI
3. User selects area of interest → UI
4. Cropped Image → Feature Extraction (feature_extraction.py) → Feature Vector → UI
5. Feature Vector → Recommendation Engine → Similar Item Indices → UI
6. Similar Item Indices → Image Paths → UI displays recommendations

## 4. Key Algorithms and Models

1. Object Detection: MobileNetV2 (pre-trained on ImageNet)
   - Input: 224x224x3 image
   - Output: List of detected objects with confidence scores

2. Feature Extraction: ResNet50 (pre-trained on ImageNet, top layers removed)
   - Input: 224x224x3 image
   - Output: 2048-dimensional feature vector

3. Recommendation: K-Nearest Neighbors
   - Training Input: n x 2048 feature array (n = number of images in dataset)
   - Query Input: 2048-dimensional feature vector
   - Output: Indices of k most similar items

## 5. Error Handling and Edge Cases

- Image Upload: Check for valid image formats and handle upload errors
- Object Detection: Handle cases where no fashion items are detected
- Feature Extraction: Ensure proper error handling for image processing issues
- Recommendation: Handle cases where not enough similar items are found

## 6. Performance Considerations

- Use of `@st.cache_resource` for model loading and feature extractor creation to optimize performance
- Potential for batch processing in feature extraction during dataset preprocessing
- Consideration of dataset size and its impact on KNN performance

## 7. Security Considerations

- Ensure proper handling and storage of user-uploaded images
- Implement rate limiting to prevent abuse of the recommendation system
- Consider adding user authentication for personalized recommendations in future versions

## 8. Scalability

- Current design suitable for moderate-sized datasets
- For larger datasets, consider:
  - Implementing approximate nearest neighbor algorithms (e.g., Annoy, FAISS)
  - Distributed processing for dataset preprocessing and feature extraction
  - Caching frequently requested recommendations

This low-level design provides a comprehensive overview of the system's architecture, components, and their interactions. It serves as a guide for understanding the current implementation and as a foundation for future improvements and scaling considerations.