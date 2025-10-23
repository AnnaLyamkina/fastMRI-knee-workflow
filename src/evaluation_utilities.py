import torch
import torch.nn.functional as F
import torchmetrics
import torchmetrics.functional as tm
from tqdm import tqdm
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import roc_curve, roc_auc_score
from src.models import denorm_image

def calculate_metrics_on_sample(prediction, target_data, mean, std):
    target_denorm = denorm_image(target_data, mean = mean, std = std)
    slice_max = target_denorm.max()
    # 1. Root Mean Squared Error (RMSE)
    mse = F.mse_loss(prediction, target_denorm)
    rmse = torch.sqrt(mse).item()
    
    # 2. Peak Signal-to-Noise Ratio (PSNR)
    psnr = tm.peak_signal_noise_ratio(
        prediction, 
        target_denorm, 
        data_range=slice_max
    ).item()
    
    # 3. Structural Similarity Index Measure (SSIM)

    ssim = tm.structural_similarity_index_measure(
        prediction, 
        target_denorm, 
        data_range=slice_max,
        reduction='elementwise_mean' 
    ).item()
    
    return rmse, psnr, ssim
    
def calculate_metrics_model(data_loader, model, nsamples = None, model_type = "Unet_only"):
    """
    Calculates the average RMSE, PSNR, and SSIM across all batches 
    in a data loader.
    
    NOTE: Assumes images are normalized to the range [0.0, 1.0].
    
    Args:
        data_loader (torch.utils.data.DataLoader): Data loader containing (input, target) pairs.
        model (torch.nn.Module): The trained reconstruction model.
        device (torch.device): The device (e.g., 'cpu' or 'cuda') to run calculations on.
        
    Returns:
        tuple: (avg_rmse, avg_psnr, avg_ssim)
    """
    rmse_scores = []
    psnr_scores = []
    ssim_scores = []
    samples_processed = 0

    model.eval() 
    
    with torch.no_grad():
        if model_type == "Unet_only":
            for input_data, target_data, mean, std, max_value in data_loader:

                if nsamples:
                    if samples_processed >= nsamples:
                        break

                prediction = model(input_data, mean, std)

                rmse, psnr, ssim = calculate_metrics_on_sample(prediction, target_data, mean = mean, std = std)

                rmse_scores.append(rmse)
                psnr_scores.append(psnr)
                ssim_scores.append(ssim)

                samples_processed += 1
    
        elif model_type == "Unet_DataConsistency":
            for kspace_masked, mask, masked_image, target_data, mean, std, max_value in data_loader:

                if nsamples:
                    if samples_processed >= nsamples:
                        break

                prediction = model(kspace_masked, mask, masked_image, mean, std)
            
                rmse, psnr, ssim = calculate_metrics_on_sample(prediction, target_data, mean = mean, std = std)
                rmse_scores.append(rmse)
                psnr_scores.append(psnr)
                ssim_scores.append(ssim)

                samples_processed += 1

    print(f"Samples processed: {samples_processed}")
    
    return rmse_scores, psnr_scores, ssim_scores

def visualize_metrics(rmse_scores, psnr_scores, ssim_scores, model_name = ""):
 # Group scores and labels for iteration
    df_metrics = pd.DataFrame({
        'RMSE': rmse_scores,
        'PSNR': psnr_scores,
        'SSIM': ssim_scores,
    })
    metadata = [
        {"color":"red", "label": "Lower is better"},
        {"color":"green", "label": "Higher is better"},
        {"color":"blue", "label": "Higher is better"}
    ]
    # 1. --- Visualization: Individual Box Plots for Distribution ---
    # We use separate plots since the score scales (Y-axes) are very different.
    
    # Set up a 1x3 grid of subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    plt.suptitle(f'Distribution of Reconstruction Metrics for {model_name}', fontsize=20, y=1.02)

    for i, metric in enumerate(df_metrics.columns):
        # Convert list to DataFrame for simple Seaborn input (wide format)
        color = metadata[i]["color"]
        # Plot Box Plot
        sns.boxplot(
            data=df_metrics[[metric]], 
            y=metric, 
            ax=axes[i], 
            color=color, 
            linewidth=1.5,
            width=0.4,
            fliersize=5
        )
        
        # Add mean line for context
        mean_score = df_metrics[metric].mean()
        if metric == "RMSE":
            axes[i].axhline(
            mean_score, 
            color='red', 
            linestyle='--', 
            linewidth=2, 
            label=f'Mean: {mean_score:.2e}'
        )
        
        else:
            axes[i].axhline(
                mean_score, 
                color='red', 
                linestyle='--', 
                linewidth=2, 
                label=f'Mean: {mean_score:.3f}'
            )
        
        # Styling
        axes[i].set_title(metric, fontsize=16, color=color)
        axes[i].set_ylabel(metadata[i]["label"], fontsize=12)
        axes[i].set_xlabel('') # Clear the default X label
        axes[i].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False) # Remove X ticks
        axes[i].legend()
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

    # 2. --- Visualization: Detailed Distribution Plot (Focus on SSIM) ---
    # Shows the smoothness and frequency of scores, highlighting stability.
    plt.figure(figsize=(12, 5))
    
    sns.histplot(
        ssim_scores, 
        kde=True, 
        bins=20, 
        color="#2E86C1", # Using SSIM color for consistency
        line_kws={'linewidth': 3, 'alpha': 0.8}
    )
    
    # Add mean line
    mean_ssim = np.mean(ssim_scores)
    plt.axvline(
        mean_ssim, 
        color='red', 
        linestyle='--', 
        linewidth=2, 
        label=f'Mean SSIM: {mean_ssim:.2f}'
    )
    
    plt.title(f'Distribution of SSIM for {model_name}', fontsize=16)
    plt.xlabel('SSIM Score', fontsize=12)
    plt.ylabel('Frequency (Samples)', fontsize=12)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

###########################
## Classifier evaluation ##
###########################

def calculate_predictions(model, dataset):
    """
    Calculates predictions for a classifier
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device);
    preds = []
    labels = []

    with torch.no_grad():
        for data, label, _, _ in tqdm(dataset):
            data = data.to(device).float().unsqueeze(0)
            pred = torch.sigmoid(model(data)[0].cpu())
            preds.append(pred)
            labels.append(label)
    preds = torch.tensor(preds)
    labels = torch.tensor(labels).int()

    return preds, labels

def calculate_classifier_metrics(preds, labels):
    threshold = []
    accuracy= []
    precision = []
    f1 = []
    recall = []
    for t in np.linspace(0, 1, 20, endpoint=False):
        threshold.append(t)
        accuracy.append(torchmetrics.Accuracy(task='binary', threshold=t)(preds, labels).numpy().item())
        precision.append(torchmetrics.Precision(task='binary', threshold=t)(preds, labels).numpy().item())
        recall.append(torchmetrics.Recall(task='binary', threshold=t)(preds, labels).numpy().item())
        f1.append(torchmetrics.F1Score(task='binary', threshold=t)(preds, labels).numpy().item())
    df_metrics = pd.DataFrame({"threshold": threshold, "accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1})
    threshold_f1 = df_metrics.loc[df_metrics["f1_score"].idxmax()]["threshold"]
    
    return df_metrics, threshold_f1

def show_confusion_matrix(preds, labels, threshold, model_name= "Classifier"):
    cm = torchmetrics.ConfusionMatrix(task='binary', threshold=threshold)(preds, labels).numpy()
    
    plt.figure(figsize=(7, 6))
    annotation_font_settings = {'fontsize': 18, 'fontweight': 'bold'}
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d',          
        cmap='RdBu_r',    
        linewidths=.5,    
        cbar=True,        
        linecolor='black',
        annot_kws=annotation_font_settings
    )
    
    # plt.xticks(ticks=np.arange(len(class_names)) + 0.5, labels=class_names, fontsize=12)
    # plt.yticks(ticks=np.arange(len(class_names)) + 0.5, labels=class_names, rotation=0, fontsize=12)
    
    plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
    plt.ylabel('True Label (Actual)', fontsize=14, fontweight='bold')
    plt.title(f'Confusion Matrix for {model_name} at Threshold = {threshold:.2f}', fontsize=16)
    
    # Optional: Add cell labels (TN, FP, FN, TP)
    labels_pos = [
        (0.25, 0.85, 'True Negatives (TN)'), 
        (0.25, 0.35, 'False Positives (FP)'), 
        (0.75, 0.85, 'False Negatives (FN)'), 
        (0.75, 0.35, 'True Positives (TP)')
    ]
    
    for x_offset, y_offset, text in labels_pos:
        plt.text(
            x_offset, y_offset, text, 
            horizontalalignment='center', 
            verticalalignment='center', 
            color='black', 
            fontsize=16, 
            transform=plt.gca().transAxes
        )
        
    plt.tight_layout()
    plt.show()

def plot_classifier_metrics_vs_threshold(df_metrics, optimal_threshold = None, model_name = "Classifier"):
    """
    Creates an interactive plot of classifier metrics plotted against the prediction threshold.
    Outputs threshold and corresponding metrics for maximized F1 Score.

    Args:
        df_metrics(pd.DataFrame): DataFrame containing 'threshold', 'accuracy',
                                        'recall', 'precision', and 'f1_score' columns.
    """
    
    if optimal_threshold:
        optimal_row = df_metrics.loc[df_metrics["threshold"] == optimal_threshold].iloc[0]
    else:
        optimal_row = df_metrics.loc[df_metrics['f1_score'].idxmax()]
        optimal_threshold = optimal_row['threshold']
    

    print(f"--- Optimal Metrics")
    print(f"Threshold: {optimal_threshold:.3f}")
    print(f"Precision: {optimal_row['precision']:.3f}")
    print(f"Recall:    {optimal_row['recall']:.3f}")
    print("----------------------------------")
    
    fig = px.line(
        df_metrics,
        x='threshold',
        y=["accuracy","precision", "recall", "f1_score"],
        title=f'Classifier Performance Metrics vs. Prediction Threshold for {model_name}',
        markers=True,
        color_discrete_map={
            'precision': '#2196F3', # Blue
            'recall': '#FF9800',    # Orange
            'accuracy': '#4CAF50',   # Green,
            'f1_score': 'red'
        },
        height=800,
        width = 800
    )

    fig.add_vline(
        x= optimal_row['threshold'], 
        line_dash="dash", 
        line_color="red", 
        line_width=3, 
        annotation_text=f"Optimal F1 Threshold: {optimal_row['threshold']:.2f}",
        annotation_position="top right"
    )

    fig.update_layout(
        # xaxis_range=[0, 1.05],
        # yaxis_range=[0, 1.05],
        xaxis_title='Prediction Threshold',
        yaxis_title='Metric Value',
        hovermode='x unified',
        template='simple_white',
        legend_title_text='Metric',
        title_x=0.5
    )

    fig.show()

def plot_roc_curve_and_auc(scores, labels, model_name="Classifier Model"):
    """
    Calculates and plots the Receiver Operating Characteristic (ROC) curve 
    and determines the Area Under the Curve (AUC).

    Args:
        scores (list or numpy.array): The predicted probability scores
                                              for the positive class.
        labels (list or numpy.array): The true binary labels (0 or 1).
        model_name (str): The name of the model for the plot title.
    """
    
    fpr, tpr, _ = roc_curve(labels, scores)
    
    roc_auc = roc_auc_score(labels, scores)
    
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='#4f46e5', lw=3, 
             label=f'ROC curve ({model_name}) (AUC = {roc_auc:.4f})')
    
    plt.plot([0, 1], [0, 1], color='#f87171', lw=2, linestyle='--', 
             label='Random Chance (AUC = 0.5)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    
    plt.xlabel('False Positive Rate (FPR)', fontsize=14)
    plt.ylabel('True Positive Rate (TPR) / Recall', fontsize=14)
    plt.title(f'Receiver Operating Characteristic (ROC) Curve for {model_name}', fontsize=16)
    
    plt.grid(color='gray', linestyle=':', linewidth=0.5)
    
    plt.legend(loc="lower right", fontsize=12, frameon=True, shadow=True)

    plt.tight_layout()
    plt.show()
    
    return roc_auc