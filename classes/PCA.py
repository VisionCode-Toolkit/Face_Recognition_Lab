import numpy as np

class PCA:
    def __init__(self, n_components, image_viewer = None, pca_file = "../data/pca_data.csv"):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.image_viewer = image_viewer
        self.pca_file = pca_file

    def fit(self, X):
        # Mean centering
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        # Covariance matrix
        cov = np.cov(X.T)

        # Eigenvalues, eigenvectors Calculations
        eigenvalues, eigenvectors = np.linalg.eigh(cov)


        # Sort eigenvalues and eigenvectors
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idxs]
        eigenvalues = eigenvalues[idxs]

        # Store top components
        self.components = eigenvectors[:, :self.n_components]

        # Save to CSV: stack mean and components
        data_to_save = np.vstack([self.mean, self.components.T])
        np.savetxt(self.pca_file, data_to_save, delimiter=",")




    def transform(self):
        # Load mean and components from file
        data_loaded = np.loadtxt(self.pca_file, delimiter=",")

        self.mean = data_loaded[0]  # First row is the mean
        self.components = data_loaded[1:].T  # Rest is components (transpose back)

        # Get current image
        modified_image = self.image_viewer.current_image.modified_image

        # Subtract mean and project
        centered = modified_image - self.mean
        return np.dot(centered, self.components)