import torch
import torch.nn.functional as F
import torchmetrics.functional as tm
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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
