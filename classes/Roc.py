import pandas as pd
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt

class Roc():
    def init(self):
        self.model = None
        self.data = None
        self.n_classes = 0

    def make_classifier(self):
        df = pd.read_csv("data\pca_features.csv")
        y = df.iloc[:, 0].values       # the label that represent the person
        X = df.iloc[:, 1:].values     # the others is features
        classes = np.unique(y)
        n_classes = len(classes)
        self.n_classes = n_classes
        y_bin = label_binarize(y, classes=classes)
        X_train, X_test, y_train, y_test = train_test_split(X, y_bin, test_size=0.3, random_state=42)
        self.data = [X_train, X_test, y_train, y_test]
        model = OneVsRestClassifier(SVC(kernel='linear', probability=True))
        model.fit(X_train, y_train)
        self.model = model
        y_scores = model.predict_proba(X_test)
        return y_scores

    def compute_roc(self, y_true, y_score, thresholds = np.linspace(0, 1, 1000)):
        tpr_list = []
        fpr_list = []
        P = np.sum(y_true == 1)
        N = np.sum(y_true == 0)
        for thresh in thresholds:
            y_pred = (y_score >= thresh).astype(int)
            TP = np.sum((y_pred == 1) & (y_true == 1))
            FP = np.sum((y_pred == 1) & (y_true == 0))
            tpr = TP / P if P else 0
            fpr = FP / N if N else 0
            tpr_list.append(tpr)
            fpr_list.append(fpr)
        fpr_array = np.array(fpr_list)
        tpr_array = np.array(tpr_list)
        sorted_indices = np.argsort(fpr_array)
        fpr_sorted = fpr_array[sorted_indices]
        tpr_sorted = tpr_array[sorted_indices]
        return fpr_sorted, tpr_sorted

    def make_roc(self, classes_per_plot=10, save=False):
        y_scores = self.make_classifier()
        thresholds = np.linspace(0, 1, 1000)


        num_figs = (self.n_classes + classes_per_plot - 1) // classes_per_plot

        for fig_idx in range(num_figs):
            start = fig_idx * classes_per_plot
            end = min((fig_idx + 1) * classes_per_plot, self.n_classes)

            plt.figure(figsize=(10, 6))
            for i in range(start, end):
                fpr, tpr = self.compute_roc(self.data[3][:, i], y_scores[:, i], thresholds)
                auc = np.trapz(tpr, fpr)
                plt.plot(fpr, tpr, label=f"Class {i} (AUC = {auc:.2f})")

            plt.plot([0, 1], [0, 1], 'k--', lw=1)
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC Curves for Classes {start} to {end - 1}")
            plt.legend(loc='lower right', fontsize='small', ncol=2)
            plt.grid(True)
            plt.tight_layout()

            if save:
                plt.savefig(f"rocclasses{start}to{end - 1}.png", dpi=300)

        plt.show()

# Roc().make_roc(classes_per_plot=3, save=True)