# Import necessary libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import VGG16 # type: ignore
from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
dataset = pd.read_csv('styles.csv')

# Define the image processing model
image_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Define the recommendation model
class FashionRecommender:
    def __init__(self, dataset):
        self.dataset = dataset
        self.shirt_features = []
        self.pant_features = []
        self.shoe_features = []

    def extract_features(self, image_path):
        img = load_img(image_path, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        features = image_model.predict(img_array)
        return features

    def get_recommendations(self, input_image, outfit_type, color):
        input_features = self.extract_features(input_image)
        input_features = input_features.flatten()

        # Filter the dataset based on outfit type and color
        filtered_dataset = self.dataset[(self.dataset['outfit_type'] == outfit_type) & (self.dataset['color'] == color)]

        # Calculate the similarity between the input features and the filtered dataset
        similarities = []
        for index, row in filtered_dataset.iterrows():
            shirt_features = self.extract_features(row['shirt_image'])
            pant_features = self.extract_features(row['pant_image'])
            shoe_features = self.extract_features(row['shoe_image'])
            features = np.concatenate((shirt_features, pant_features, shoe_features), axis=1)
            similarity = cosine_similarity([input_features], [features])
            similarities.append(similarity)

        # Get the top recommendations
        top_indices = np.argsort(similarities)[-5:]
        recommendations = filtered_dataset.iloc[top_indices]

        return recommendations