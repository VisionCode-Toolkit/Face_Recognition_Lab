import os
import cv2
import numpy as np

from PCA import PCA
def load_images_from_folder(root_folder):
    data = []
    for person_folder in sorted(os.listdir(root_folder)): # loops over each person
        person_path = os.path.join(root_folder, person_folder) # add the path of person to original path
        if os.path.isdir(person_path): # checks if the path is actually a folder
            for filename in sorted(os.listdir(person_path)): # loops over each image for one person
                if filename.endswith('.jpg'):
                    img_path = os.path.join(person_path, filename)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # read as grayscale
                    if img is not None:
                        img_flat = img.flatten()  # convert 2D image to 1D array
                        data.append(img_flat)
    return np.array(data)

# Load image data
image_data = load_images_from_folder("../data/ORL database")

# Step 2: Fit PCA
pca = PCA(n_components=50)
pca.fit(image_data)

loaded = np.loadtxt("../data/pca_data.csv", delimiter=",")
print("Loaded shape:", loaded.shape)

