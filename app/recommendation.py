import numpy as np
from sklearn.neighbors import NearestNeighbors
import pickle

class RecommendationSystem:
    def __init__(self):
        self.feature_list = []
        self.image_paths = []
        self.knn = None

    def add_item(self, features, image_path):
        self.feature_list.append(features)
        self.image_paths.append(image_path)

    def fit(self):
        feature_array = np.array(self.feature_list)
        self.knn = NearestNeighbors(n_neighbors=6, metric='cosine')
        self.knn.fit(feature_array)

    def get_recommendations(self, input_features, n_recommendations=5):
        distances, indices = self.knn.kneighbors([input_features], n_neighbors=n_recommendations+1)
        
        # Exclude the first result as it's the input image itself
        recommended_paths = [self.image_paths[i] for i in indices[0][1:]]
        return recommended_paths

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump((self.feature_list, self.image_paths, self.knn), f)

    def load(self, filename):
        with open(filename, 'rb') as f:
            self.feature_list, self.image_paths, self.knn = pickle.load(f)

# Example usage
if __name__ == "__main__":
    recommender = RecommendationSystem()
    
    # Simulate adding items to the system
    for i in range(1000):
        fake_features = np.random.rand(2048)
        fake_image_path = f"path/to/image_{i}.jpg"
        recommender.add_item(fake_features, fake_image_path)
    
    recommender.fit()
    
    # Test recommendation
    test_features = np.random.rand(2048)
    recommendations = recommender.get_recommendations(test_features)
    print("Recommended image paths:", recommendations)
    
    # Save and load test
    recommender.save("recommender.pkl")
    new_recommender = RecommendationSystem()
    new_recommender.load("recommender.pkl")
    
    # Test loaded recommender
    new_recommendations = new_recommender.get_recommendations(test_features)
    print("Recommendations after loading:", new_recommendations)