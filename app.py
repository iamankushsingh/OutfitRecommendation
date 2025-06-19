import streamlit as st
import os
import numpy as np
import cv2
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.neighbors import NearestNeighbors
from PIL import Image

# Load the pre-trained ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# Function to preprocess image and extract features
def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))  # Resizing only for model processing
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array)
    return features.flatten()

# Directories for each category
base_dir = r"C:\Users\nidhi\OneDrive\Desktop\FRC"
tshirt_dir = os.path.join(base_dir, "T-Shirt")
watch_dir = os.path.join(base_dir, "Watch")
shoes_dir = os.path.join(base_dir, "shoes")
pants_dir = os.path.join(base_dir, "Jeans")

# Initialize KNN models
watch_knn = NearestNeighbors(n_neighbors=1, metric='euclidean')
shoes_knn = NearestNeighbors(n_neighbors=1, metric='euclidean')
pants_knn = NearestNeighbors(n_neighbors=1, metric='euclidean')

# Load features into KNN models
def load_features():
    watch_features, shoes_features, pants_features = [], [], []
    for img_name in os.listdir(watch_dir):
        img_path = os.path.join(watch_dir, img_name)
        watch_features.append(extract_features(img_path, model))
    for img_name in os.listdir(shoes_dir):
        img_path = os.path.join(shoes_dir, img_name)
        shoes_features.append(extract_features(img_path, model))
    for img_name in os.listdir(pants_dir):
        img_path = os.path.join(pants_dir, img_name)
        pants_features.append(extract_features(img_path, model))
    return np.array(watch_features), np.array(shoes_features), np.array(pants_features)

watch_features, shoes_features, pants_features = load_features()
watch_knn.fit(watch_features)
shoes_knn.fit(shoes_features)
pants_knn.fit(pants_features)

# Function to recommend items based on T-shirt
def recommend_items(tshirt_img_path):
    tshirt_feature = extract_features(tshirt_img_path, model).reshape(1, -1)
    
    watch_index = watch_knn.kneighbors(tshirt_feature, return_distance=False)[0][0]
    shoes_index = shoes_knn.kneighbors(tshirt_feature, return_distance=False)[0][0]
    pants_index = pants_knn.kneighbors(tshirt_feature, return_distance=False)[0][0]

    recommended_watch = os.listdir(watch_dir)[watch_index]
    recommended_shoes = os.listdir(shoes_dir)[shoes_index]
    recommended_pants = os.listdir(pants_dir)[pants_index]

    return recommended_watch, recommended_shoes, recommended_pants, watch_index, shoes_index, pants_index

# Function to upscale image resolution
def upscale_image(img_path, scale=2):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    height, width = img.shape[:2]
    upscaled = cv2.resize(img, (width * scale, height * scale), interpolation=cv2.INTER_CUBIC)
    return Image.fromarray(cv2.cvtColor(upscaled, cv2.COLOR_BGR2RGB))

# Streamlit UI
st.title("Fashion Recommendation System")
uploaded_file = st.file_uploader("Upload T-Shirt Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    tshirt_img_path = os.path.join("tempDir", uploaded_file.name)
    with open(tshirt_img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Display input image with improved resolution
    st.image(upscale_image(tshirt_img_path, scale=2), caption="Input T-Shirt", use_column_width=False)

    # Get recommendations
    watch, shoes, pants, watch_dist, shoes_dist, pants_dist = recommend_items(tshirt_img_path)


# Add this code towards the end of your Streamlit code where you display the Euclidean distance visuals

    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

# # Reduce vectors to 2D using PCA
# def plot_2d_vector_space(tshirt_feature, watch_feature, shoes_feature, pants_feature):
#     # Combine the features into one array for PCA transformation
#     all_vectors = np.vstack([tshirt_feature, watch_features, shoes_features, pants_features])

#     # Apply PCA to reduce the dimensions to 2D
#     pca = PCA(n_components=2)
#     reduced_vectors = pca.fit_transform(all_vectors)

#     # Prepare the labels for each feature vector
#     labels = ['T-Shirt', 'Watch', 'Shoes', 'Pants']
#     colors = ['black', 'red', 'blue', 'green']

#     # Plot the reduced 2D vectors
#     fig, ax = plt.subplots(figsize=(8, 6))

#     # Plot T-shirt as the origin point
#     ax.scatter(reduced_vectors[0][0], reduced_vectors[0][1], color='black', label='T-Shirt', s=100)

#     # Plot the other items (Watch, Shoes, Pants) and draw vectors from T-shirt
#     for i in range(1, 4):
#         ax.quiver(reduced_vectors[0][0], reduced_vectors[0][1], reduced_vectors[i][0] - reduced_vectors[0][0], reduced_vectors[i][1] - reduced_vectors[0][1],
#                   angles='xy', scale_units='xy', scale=1, color=colors[i], label=labels[i])

#     # Label the points and add a title
#     for i in range(4):
#         ax.text(reduced_vectors[i][0], reduced_vectors[i][1], labels[i], color=colors[i], fontsize=12, ha='right')

#     ax.set_title('2D Euclidean Vectors from T-Shirt to Recommendations')
#     ax.set_xlabel('PCA Component 1')
#     ax.set_ylabel('PCA Component 2')
#     ax.legend(loc='upper right')
#     ax.grid(True)

#     # Display the plot in Streamlit
#     st.pyplot(fig)

# # Get the features for T-shirt, Watch, Shoes, and Pants
#     tshirt_feature = extract_features(tshirt_img_path, model).reshape(1,-1)
#     watch_feature = extract_features(os.path.join(watch_dir, watch), model).reshape(1,-1)
#     shoes_feature = extract_features(os.path.join(shoes_dir, shoes), model).reshape(1,-1)
#     pants_feature = extract_features(os.path.join(pants_dir, pants), model).reshape(1,-1)

#     # Plot the 2D vector space
#     plot_2d_vector_space(tshirt_feature, watch_feature, shoes_feature, pants_feature)



    # Display recommended images with improved resolution
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(upscale_image(os.path.join(watch_dir, watch), scale=2), caption="Recommended Watch", use_column_width=False)
    with col2:
        st.image(upscale_image(os.path.join(shoes_dir, shoes), scale=2), caption="Recommended Shoes", use_column_width=False)
    with col3:
        st.image(upscale_image(os.path.join(pants_dir, pants), scale=2), caption="Recommended Pants", use_column_width=False)

    # Add a catchy heading for the Euclidean distance metrics
    st.subheader("Here are the Euclidean Distances for the Recommendations:")
    
    # Display the distances
    st.write(f"📏 **Watch Distance:** {watch_dist:.2f}")
    st.write(f"📏 **Shoes Distance:** {shoes_dist:.2f}")
    st.write(f"📏 **Pants Distance:** {pants_dist:.2f}")

    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    # # Reduce vectors to 2D
    # pca = PCA(n_components=2)
    # all_vectors = np.vstack([tshirt_feature, watch_features[watch_index], shoes_features[shoes_index], pants_features[pants_index]])
    # reduced = pca.fit_transform(all_vectors)

    # labels = ['T-Shirt', 'Watch', 'Shoes', 'Pants']
    # colors = ['black', 'red', 'blue', 'green']

    # plt.figure(figsize=(8, 6))
    # origin = reduced[0]

    # # Plot vectors from T-shirt to recommended items
    # for i in range(1, 4):
    #     vec = reduced[i] - origin
    #     plt.quiver(origin[0], origin[1], vec[0], vec[1], angles='xy', scale_units='xy', scale=1, color=colors[i], label=labels[i])

    # # Plot T-shirt point
    # plt.scatter(origin[0], origin[1], color='black', label='T-Shirt', s=100)

    # plt.title('Euclidean Vectors from T-Shirt to Recommendations (PCA-reduced)')
    # plt.xlabel('PCA Component 1')
    # plt.ylabel('PCA Component 2')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    
    ##########
    
    # Data Visualization 
    import matplotlib.pyplot as plt

    categories = ['Watch', 'Shoes', 'Pants']
    distances = [watch_dist, shoes_dist, pants_dist]

    fig, ax = plt.subplots()
    ax.bar(categories, distances, color=['blue', 'green', 'orange'])
    ax.set_title('Euclidean Distances for Recommendations')
    ax.set_ylabel('Distance')
    ax.set_ylim(0, max(distances) + 10)

    st.pyplot(fig)

    # # 2nd Visulization
    # import matplotlib.pyplot as plt

    # # Assuming you already have the distances
    # labels = ['Watch', 'Shoes', 'Pants']
    # distances = [watch_dist, shoes_dist, pants_dist]

    # plt.figure(figsize=(6, 6))
    # plt.pie(distances, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff','#99ff99'])
    # plt.title('Proportion of Euclidean Distances Across Categories')
    # plt.show()

    # # 3rd
    # plt.figure(figsize=(8, 4))
    # plt.plot(labels, distances, marker='o', color='purple')
    # plt.title('Euclidean Distance Comparison')
    # plt.xlabel('Category')
    # plt.ylabel('Distance')
    # plt.grid(True)
    # plt.show()
    import matplotlib.pyplot as plt
    import streamlit as st

# # 2nd Visualization: PIE CHART
#     fig1, ax1 = plt.subplots(figsize=(6, 6))
#     ax1.pie([watch_dist, shoes_dist, pants_dist], labels=['Watch', 'Shoes', 'Pants'],
#         autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff','#99ff99'])
#     ax1.set_title('Proportion of Euclidean Distances Across Categories')
#     st.pyplot(fig1)

# 3rd Visualization: LINE CHART
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(['Watch', 'Shoes', 'Pants'], [watch_dist, shoes_dist, pants_dist],
         marker='o', color='purple')
    ax2.set_title('Euclidean Distance Comparison')
    ax2.set_xlabel('Category')
    ax2.set_ylabel('Distance')
    ax2.grid(True)
    st.pyplot(fig2)