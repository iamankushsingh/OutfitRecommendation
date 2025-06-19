# -*- coding: utf-8 -*-
"""RBL Project - Fashion Recommendation System"""

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.neighbors import NearestNeighbors

# Load the pre-trained ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# Function to preprocess image and extract features
def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array)
    return features.flatten()

# Directory paths for each category (Update these paths as needed)
tshirt_dir = r"C:\Users\nidhi\OneDrive\Desktop\FRC\T-Shirt"
watch_dir = r"C:\Users\nidhi\OneDrive\Desktop\FRC\Watch"
shoes_dir = r"C:\Users\nidhi\OneDrive\Desktop\FRC\shoes"
pants_dir = r"C:\Users\nidhi\OneDrive\Desktop\FRC\Jeans"

# Extract features for all t-shirt, watch, shoes, and pants images
tshirt_features = []
watch_features = []
shoes_features = []
pants_features = []

# Extract features for all images in each category
for img_name in os.listdir(tshirt_dir):
    img_path = os.path.join(tshirt_dir, img_name)
    tshirt_features.append(extract_features(img_path, model))

for img_name in os.listdir(watch_dir):
    img_path = os.path.join(watch_dir, img_name)
    watch_features.append(extract_features(img_path, model))

for img_name in os.listdir(shoes_dir):
    img_path = os.path.join(shoes_dir, img_name)
    shoes_features.append(extract_features(img_path, model))

for img_name in os.listdir(pants_dir):
    img_path = os.path.join(pants_dir, img_name)
    pants_features.append(extract_features(img_path, model))

# Convert lists to numpy arrays
tshirt_features = np.array(tshirt_features)
watch_features = np.array(watch_features)
shoes_features = np.array(shoes_features)
pants_features = np.array(pants_features)

# Fit KNN models for each category (watch, shoes, pants)
watch_knn = NearestNeighbors(n_neighbors=1, metric='euclidean')
watch_knn.fit(watch_features)

shoes_knn = NearestNeighbors(n_neighbors=1, metric='euclidean')
shoes_knn.fit(shoes_features)

pants_knn = NearestNeighbors(n_neighbors=1, metric='euclidean')
pants_knn.fit(pants_features)

# Function to recommend items for a given t-shirt
def recommend_items(tshirt_img_path):
    # Extract features for the input t-shirt image
    tshirt_feature = extract_features(tshirt_img_path, model).reshape(1, -1)

    # Find the nearest watch, shoes, and pants based on the t-shirt
    watch_index = watch_knn.kneighbors(tshirt_feature, return_distance=False)[0][0]
    shoes_index = shoes_knn.kneighbors(tshirt_feature, return_distance=False)[0][0]
    pants_index = pants_knn.kneighbors(tshirt_feature, return_distance=False)[0][0]

    # Return recommended watch, shoes, and pants
    recommended_watch = os.listdir(watch_dir)[watch_index]
    recommended_shoes = os.listdir(shoes_dir)[shoes_index]
    recommended_pants = os.listdir(pants_dir)[pants_index]

    return recommended_watch, recommended_shoes, recommended_pants

# Test the recommendation system
tshirt_img = r'C:\Users\nidhi\OneDrive\Desktop\FRC\T-Shirt\12582.jpg'  # Update this path
watch, shoes, pants = recommend_items(tshirt_img)
print(f"Recommended watch: {watch}, shoes: {shoes}, pants: {pants}")

# Function to display an image with a title
def display_image(img_path, title):
    img = mpimg.imread(img_path)
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')

# Full paths to the recommended images
watch_img_path = os.path.join(watch_dir, watch)
shoes_img_path = os.path.join(shoes_dir, shoes)
pants_img_path = os.path.join(pants_dir, pants)

# Plotting the recommended images
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
display_image(watch_img_path, 'Recommended Watch')

plt.subplot(1, 3, 2)
display_image(shoes_img_path, 'Recommended Shoes')

plt.subplot(1, 3, 3)
display_image(pants_img_path, 'Recommended Pants')

plt.show()

# ------------------------------
# Evaluation: Calculate average Euclidean distances
# ------------------------------

total_watch_dist = 0
total_shoes_dist = 0
total_pants_dist = 0

tshirt_images = os.listdir(tshirt_dir)

for img_name in tshirt_images:
    tshirt_path = os.path.join(tshirt_dir, img_name)
    tshirt_feat = extract_features(tshirt_path, model).reshape(1, -1)

    watch_name, shoes_name, pants_name = recommend_items(tshirt_path)

    watch_feat = extract_features(os.path.join(watch_dir, watch_name), model).reshape(1, -1)
    shoes_feat = extract_features(os.path.join(shoes_dir, shoes_name), model).reshape(1, -1)
    pants_feat = extract_features(os.path.join(pants_dir, pants_name), model).reshape(1, -1)

    total_watch_dist += euclidean_distances(tshirt_feat, watch_feat)[0][0]
    total_shoes_dist += euclidean_distances(tshirt_feat, shoes_feat)[0][0]
    total_pants_dist += euclidean_distances(tshirt_feat, pants_feat)[0][0]

n = len(tshirt_images)
avg_watch_dist = total_watch_dist / n
avg_shoes_dist = total_shoes_dist / n
avg_pants_dist = total_pants_dist / n

print("\n--- Evaluation Results ---")
print(f"Average Euclidean Distance (Watch): {avg_watch_dist:.4f}")
print(f"Average Euclidean Distance (Shoes): {avg_shoes_dist:.4f}")
print(f"Average Euclidean Distance (Pants): {avg_pants_dist:.4f}")