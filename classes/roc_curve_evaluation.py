import numpy as np
import cv2
import os
import joblib
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from classes.PCA import PCA

# loading all data inside or outside
svm_model_path = "D:\SBE\Third Year\Second Term\Computer Vision\Tasks\Task5\Face_Recognition_Lab\classes\svm_pca_model.pkl"
in_db_folder = "D:\SBE\Third Year\Second Term\Computer Vision\Tasks\Task5\Face_Recognition_Lab\data\ORL database"
out_db_folder = "D:\SBE\Third Year\Second Term\Computer Vision\Tasks\Task5\Face_Recognition_Lab\data\outside_data"


svm_model = joblib.load(svm_model_path)

pca = PCA(n_components=50)

# load (inside >> pos)
X_in = []
y_in = []
for label in range(1, 42):
    folder = os.path.join(in_db_folder, str(label))
    if os.path.exists(folder):
        for filename in os.listdir(folder):
            if filename.endswith(('.jpg', '.pgm', '.png')):
                img_path = os.path.join(folder, filename)
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Failed to load {img_path}")
                    continue
                pca_features = pca.transform(img)
                X_in.append(pca_features)
                y_in.append(1)  # positive in data
X_in = np.array(X_in)
y_in = np.array(y_in)

# load (outside >> neg)
X_out = []
y_out = []
for filename in os.listdir(out_db_folder):
    if filename.endswith(('.jpg', '.png', '.jpeg')):
        img_path = os.path.join(out_db_folder, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load {img_path}")
            continue
        pca_features = pca.transform(img)
        X_out.append(pca_features)
        y_out.append(0)  # meaning it's neg not in data
X_out = np.array(X_out)
y_out = np.array(y_out)

# combine all to draw the curve
X_all = np.vstack((X_in, X_out))
y_all = np.hstack((y_in, y_out))

# apply the svm model
probs = svm_model.predict_proba(X_all)
max_probs = np.max(probs, axis=1)

# compute roc
fpr, tpr, thresholds = roc_curve(y_all, max_probs)
roc_auc = auc(fpr, tpr)

# plotting
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve: In-Database vs Not In-Database')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# try different thereshold
print("Sample Thresholds, TPR, FPR:")
for i in range(0, len(thresholds), len(thresholds)//10):
    print(f"Threshold: {thresholds[i]:.3f}, TPR: {tpr[i]:.3f}, FPR: {fpr[i]:.3f}")