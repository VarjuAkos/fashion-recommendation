# Fashion Recommendation System

## 1. Project Synopsis

The Fashion Recommendation System is an interactive web application that allows users to upload images of fashion items and receive personalized recommendations based on selected areas of interest. The system utilizes advanced machine learning techniques, including object detection and feature extraction, to analyze fashion items and find similar products.

Key Features:
- Image upload functionality
- Object detection to identify fashion items in uploaded images
- Interactive image cropping to focus on specific areas of interest
- Feature extraction using ResNet50 for accurate similarity matching
- K-Nearest Neighbors algorithm for finding similar fashion items
- User-friendly interface built with Streamlit

## 2. Project High-Level Design

The project consists of several key components:

1. **User Interface**: Built using Streamlit, providing an intuitive way for users to interact with the system.
2. **Object Detection**: Utilizes MobileNetV2 to identify fashion items in uploaded images.
3. **Image Processing**: Includes an interactive cropping tool for users to select specific areas of interest.
4. **Feature Extraction**: Employs ResNet50 to extract meaningful features from the selected image area.
5. **Recommendation Engine**: Uses K-Nearest Neighbors algorithm to find similar fashion items based on extracted features.
6. **Data Management**: Handles the storage and retrieval of fashion item features and image paths.

## 3. Project Low-Level Design

The project is structured into several Python scripts:

- `main.py`: Main application script, handles the Streamlit UI and orchestrates the entire process.
- `object_detection.py`: Contains the MobileNetV2-based object detection functionality.
- `feature_extraction.py`: Implements the ResNet50-based feature extraction.
- `preprocess_and_train.py`: Preprocesses the dataset and trains the KNN model.

Key Functions and Classes:
- `FeatureExtractor`: Class for extracting features from images using ResNet50.
- `detect_objects`: Function for identifying fashion items in images.
- `st_cropper`: Streamlit component for interactive image cropping.
- `load_model_and_data`: Function to load the pre-trained KNN model and associated data.

## 4. Output

The system provides the following outputs:

1. Detected fashion items in the uploaded image, displayed in a dropdown menu.
2. An interactive cropping interface for the user to select a specific area of interest.
3. A display of the cropped area selected by the user.
4. A set of 5 recommended fashion items similar to the selected area, displayed as images.

## 5. Explainable AI (How the Model Works)

The Fashion Recommendation System works through several steps:

1. **Object Detection**: When an image is uploaded, MobileNetV2 is used to detect and classify fashion items in the image. This gives users an idea of what the system recognizes in their upload.

2. **Feature Extraction**: After the user selects an area of interest, the system uses ResNet50, a deep convolutional neural network, to extract a feature vector from this area. ResNet50 has been pre-trained on millions of images and can capture complex visual features.

3. **Similarity Matching**: The extracted feature vector is then compared to a database of pre-processed fashion item features using the K-Nearest Neighbors algorithm. This algorithm finds the most similar items based on the distance between feature vectors in high-dimensional space.

4. **Recommendation Generation**: The system retrieves the images corresponding to the most similar feature vectors, presenting these as recommendations to the user.

This approach allows the system to make recommendations based on visual similarity, which can capture nuanced aspects of fashion items that might be difficult to describe in words.

## Deployment Instructions

To deploy the Fashion Recommendation System:

1. Clone the repository:
   ```
   git clone [repository-url]
   cd fashion-recommendation-system
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Prepare the dataset and train the model:
   ```
   python preprocess_and_train.py
   ```

5. Run the Streamlit app:
   ```
   streamlit run run.py
   ```

6. Open a web browser and navigate to the URL provided by Streamlit (typically `http://localhost:8501`).

Note: Ensure you have a suitable dataset of fashion images for the system to use as a recommendation base. The `preprocess_and_train.py` script should be run on this dataset before starting the application.

## Requirements

- Python 3.7+
- TensorFlow 2.x
- Streamlit
- NumPy
- Pillow
- scikit-learn
- streamlit-cropper

For a complete list of dependencies, refer to the `requirements.txt` file.

## Future Improvements

- Implement more advanced object detection models for better accuracy in identifying fashion items.
- Expand the dataset to cover a wider range of fashion items and styles.
- Add user accounts and personalization based on user preferences and history.
- Implement a feedback system to continuously improve recommendations.