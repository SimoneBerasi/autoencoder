import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score

def compute_segmentation_roc(anomaly_maps, ground_truth_maps):
    anomaly_maps = np.array(anomaly_maps)
    ground_truth_labels = np.array(ground_truth_maps)

    anomaly_maps = np.concatenate(anomaly_maps, axis=0)
    ground_truth_labels = np.concatenate(ground_truth_labels, axis=0)

    anomaly_maps = np.concatenate(anomaly_maps, axis=0)
    ground_truth_labels = np.concatenate(ground_truth_labels / 255, axis=0)

    roc_score = roc_auc_score(ground_truth_labels, anomaly_maps, max_fpr=0.3)
    # fpr, tpr, thresholds = roc_curve(ground_truth_labels, anomaly_maps)

    return roc_score  # , #fpr, tpr

if __name__ == "__main__":
    pass