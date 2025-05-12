import cv2
import numpy as np
import csv

class PCA:
    def __init__(self, n_components=50, image_viewer = None,output_file="pca_features.csv"):
        self.n_components = n_components
        self.output_file = output_file
        self.image_viewer = image_viewer

    def fit(self, X, labels):
        # solve mean and center the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # covariance
        cov = np.cov(X_centered.T)
        # eigenvalues & eigen vectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        idxs = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idxs]
        self.components = eigenvectors[:, :self.n_components]  # (D, n_components)

        projected_data = np.dot(X_centered, self.components)  # (N, n_components)

        # save PCA features with labels
        with open(self.output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ["label"] + [f"feature_{i}" for i in range(self.n_components)]
            writer.writerow(header)

            for label, features in zip(labels, projected_data):
                writer.writerow([label] + list(features))

    def transform(self):
        self.mean = np.loadtxt("data/mean_vector.csv", delimiter=",")
        self.components = np.loadtxt("data/pca_data.csv", delimiter=",", skiprows=1)
        modified_image = self.image_viewer.current_image.modified_image

        if modified_image.ndim == 3 and modified_image.shape[2] == 3:
            modified_image = cv2.cvtColor(modified_image, cv2.COLOR_BGR2GRAY)

        # Resize to match training image size
        modified_image = cv2.resize(modified_image, (70, 80))
        modified_image = modified_image.flatten()
        centered = modified_image - self.mean
        pca_features = np.dot(centered, self.components.T)

        return pca_features

