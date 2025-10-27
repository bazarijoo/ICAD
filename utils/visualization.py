import matplotlib.pyplot as plt
import numpy as np
import torch
import os

def plot_validation_scores(all_scores, file_name, file_path, mode=""):
    """
    Plot distribution of anomaly scores on validation (normal) data.
    """
    all_scores = all_scores.numpy() if isinstance(all_scores, torch.Tensor) else all_scores

    plt.figure(figsize=(10,6))
    plt.hist(all_scores, bins=20, alpha=0.7, color='green', density=True, range=(0, 100))
    plt.xlabel('Anomaly Score (- Log-Likelihood)')
    plt.ylabel('Density')
    plt.title(f'Anomaly Score Distribution on {mode} Data (Normal Visits Only)')
    plt.grid(True)
    plt.savefig(f"{file_path}/{file_name}.png")

def plot_anomaly_score_distribution(all_scores, all_labels, file_name, file_path):
    """
    Plots the distribution of anomaly scores for normal vs anomalous points.
    """
    
    os.makedirs(file_path, exist_ok=True)
    
    all_scores = all_scores.numpy() if isinstance(all_scores, torch.Tensor) else all_scores
    all_labels = all_labels.numpy() if isinstance(all_labels, torch.Tensor) else all_labels

    all_scores = np.array(all_scores)  # Ensure all_scores is a NumPy array
    all_labels = np.array(all_labels)  # Ensure all_labels is a NumPy array

    plt.figure(figsize=(10, 6))
    for label in np.unique(all_labels):
        label_scores = all_scores[all_labels == label]
        plt.hist(label_scores, bins=20, alpha=0.7, label=f"Label {label}", density=True, range=(0, 100))
    
    plt.xlabel('Anomaly Score (- Log-Likelihood)')
    plt.ylabel('Density')
    plt.title('Anomaly Score Distribution by Label')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{file_path}/{file_name}.png")