import csv
import os
import cv2
import numpy as np

from PCA import PCA
def load_images_from_folder(root_folder):
    data = []
    labels = []
    for person_folder in sorted(os.listdir(root_folder)):
        person_path = os.path.join(root_folder, person_folder)
        if os.path.isdir(person_path):
            for filename in sorted(os.listdir(person_path)):
                if filename.endswith('.jpg')| filename.endswith('.png'):
                    img_path = os.path.join(person_path, filename)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img_flat = img.flatten()
                        data.append(img_flat)
                        labels.append(person_folder)  # Add folder name as label
    return np.array(data), labels

def save_mean_vector(X, output_file="mean_vector.csv"):
    # Mean of each pixel across all images (axis=0 means column-wise)
    mean_vector = np.mean(X, axis=0)

    # Save to CSV (one row with D columns)
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(mean_vector)


image_data, image_labels = load_images_from_folder("../data/ORL database")

# Save mean vector
save_mean_vector(image_data, output_file="../data/mean_vector.csv")

# Perform PCA and save features
pca = PCA(n_components=50, output_file="../data/pca_features.csv")
pca.fit(image_data, image_labels)


