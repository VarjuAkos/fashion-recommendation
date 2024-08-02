# Fashion Recommendation System

## Project Synopsis

The Fashion Recommendation System is an interactive web application that allows users to upload images of fashion items and receive personalized recommendations based on selected areas of interest. The system utilizes advanced machine learning techniques, including object detection and feature extraction, to analyze fashion items and find similar products.

Key Features:
- Image upload functionality
- Object detection to identify fashion items in uploaded images
- Interactive image cropping to focus on specific areas of interest
- Feature extraction using ResNet50 for accurate similarity matching
- K-Nearest Neighbors algorithm for finding similar fashion items
- User-friendly interface built with Streamlit

![Demo](../demo.gif)


## Deployment Instructions

To deploy the Fashion Recommendation System, follow these steps:

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

4. Prepare the dataset:
   - Download the dataset and the model from [[dataset and model link](https://www.dropbox.com/scl/fo/loo7a6c58eh9vdnqm4nr3/AEkORjjAMvcQKU5z2QDY_mg?rlkey=n5w4bxgae4cmgywnrebr6addf&st=zaqcaav8&dl=0)].
   - Create a folder structure: `data/images` and `data/models`.
   - Place the downloaded images into the `data/images` folder.

5. Train the model (optional):
   - If you want to train the model yourself (this may take a few hours), run:
     ```
     python -m app.preprocess_and_train
     ```
   - This will create a `fashion_knn_model.pkl` file in the `data/models` folder.
   - If you already have a pre-trained model, place the `fashion_knn_model.pkl` file directly in the `data/models` folder.

6. Run the Streamlit app:
   ```
   streamlit run run.py
   ```

7. Open a web browser and navigate to the URL provided by Streamlit (typically `http://localhost:8501`).

Note: If you've placed a pre-trained model in the `data/models` folder, you can skip step 5 and run the application directly.

Note: Ensure you have a suitable dataset of fashion images for the system to use as a recommendation base. The `preprocess_and_train.py` script should be run on this dataset before starting the application.

## Requirements

- Python 3.7+
- TensorFlow 2.x
- Streamlit
- NumPy
- Pillow
- scikit-learn
- streamlit-cropper
- ...

For a complete list of dependencies, refer to the `requirements.txt` file.

## 2. Explainable AI (How the Model Works)

The Fashion Recommendation System works through several steps:

1. **Object Detection**: When an image is uploaded, MobileNetV2 is used to detect and classify fashion items in the image. This gives users an idea of what the system recognizes in their upload.

2. **Feature Extraction**: After the user selects an area of interest, the system uses ResNet50, a deep convolutional neural network, to extract a feature vector from this area. ResNet50 has been pre-trained on millions of images and can capture complex visual features.

3. **Similarity Matching**: The extracted feature vector is then compared to a database of pre-processed fashion item features using the K-Nearest Neighbors algorithm. This algorithm finds the most similar items based on the distance between feature vectors in high-dimensional space.

4. **Recommendation Generation**: The system retrieves the images corresponding to the most similar feature vectors, presenting these as recommendations to the user.

This approach allows the system to make recommendations based on visual similarity, which can capture nuanced aspects of fashion items that might be difficult to describe in words.

## Future Improvements

- Implement more advanced object detection models for better accuracy in identifying fashion items.
- Expand the dataset to cover a wider range of fashion items and styles.
- Add user accounts and personalization based on user preferences and history.
- Implement a feedback system to continuously improve recommendations.









