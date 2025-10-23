import torch
import torch.nn as nn
import torch.optim
import torchvision
import torchmetrics
import pytorch_lightning as pl
import torch.nn.functional as F
import os
import glob
from pathlib import Path
from typing import Tuple
from fastmri.models import Unet
from fastmri import ifft2c, fft2c, complex_abs
from fastmri.data.transforms import center_crop

from src.datasets import to_fastmri_complex_format

def find_latest_checkpoint(directory_path):
    search_pattern = os.path.join(directory_path, "*.ckpt")
    list_of_files = glob.glob(search_pattern)
    
    if not list_of_files:
        print(f"Error: No .ckpt files found in directory: {directory_path}")
        return None
    latest_file = max(list_of_files, key=os.path.getmtime)
    return latest_file

def load_fastmri_unet(weights_path: Path) -> Unet:
    """
    Loads the U-Net model architecture and weights.
    """
    print(f"Loading U-Net model from: {weights_path.name}...")
    
    model = Unet(in_chans=1, out_chans=1, chans=256, num_pool_layers=4, drop_prob=0.0)
    checkpoint = torch.load(weights_path, map_location=torch.device('cpu'))
    print("Checkpoint loaded")
    model.load_state_dict(checkpoint)
    print("State loaded")
    model.eval()
    return model

def denorm_image(image, mean, std):
    return image*std + mean

def norm_image(image, mean, std):
    return (image - mean)/std

def pad_image(image_tensor: torch.Tensor, target_shape: Tuple[int, int]) -> torch.Tensor:
    """
    Pads an tensor in FastMRI float format ([B, 1, H_in, W_in]) 
    to the target (H_target, W_target) dimensions by zero-padding the H and W dimensions.
    
    The output tensor maintains the FastMRI float format ([B, 1, H_target, W_target]).
    """
    
    H_in, W_in = image_tensor.shape[2:]
    H_target, W_target = target_shape

    # If shapes already match, return early
    if H_in == H_target and W_in == W_target:
        return image_tensor
        
    # Calculate padding amounts for height (dim 1)
    pad_h_needed = H_target - H_in
    pad_h_before = pad_h_needed // 2
    pad_h_after = pad_h_needed - pad_h_before
    
    # Calculate padding amounts for width (dim 2)
    pad_w_needed = W_target - W_in
    pad_w_before = pad_w_needed // 2
    pad_w_after = pad_w_needed - pad_w_before
    
    # F.pad takes padding dimensions in the order: (W_start, W_end, H_start, H_end, ...)
    # The last two dimensions of the tensor (W and H) are padded/cropped
    padding_dims = (pad_w_before, pad_w_after, pad_h_before, pad_h_after)

    # Apply padding to the H and W dimensions
    padded_tensor = F.pad(image_tensor, padding_dims, "constant", 0)
    
    return padded_tensor

class UNetDenormalizationWrapper(nn.Module):
    """
    Wraps a fixed, pre-trained UNet model:  
    Runs UNet, and denormalizes the output.
    """
    def __init__(self, weights_path: Path):
        super().__init__()
        
        # Load the pre-trained UNet
        self.unet = load_fastmri_unet(weights_path)
        
        # Freeze all UNet parameters
        for param in self.unet.parameters():
            param.requires_grad = False
        
        print("UNet Output Denormalization Wrapper initialized.")

    def forward(self, X: torch.Tensor, mean_tensor: torch.Tensor, std_tensor: torch.Tensor, 
                ) -> torch.Tensor:
        """
        Args:
            X: The initial zero-filled image. Its magnitude 
                       is assumed to be the normalized input for the UNet.
            mean_tensor, std_tensor: Normalization factors 
                                                       used to reverse the scaling.

        Returns:
            The final reconstructed magnitude image, denormalized to the original scale.
        """
        
        # The UNet is fixed, so we don't need gradients
        with torch.no_grad():
            
            # 1. UNet Inference: Pass the normalized image through the UNet
            X_unet_output = self.unet(X)
            
            # 2. Denormalization: Scale the output back to the original pixel range
            X_denormalized = denorm_image(X_unet_output, std = std_tensor, mean = mean_tensor)
            
        return X_denormalized
    
class DataConsistencyWrapper(nn.Module):
    """
    Unrolled Network that uses the fixed U-Net loaded with weights.
    The regularization weight (lambda) is now a fixed hyperparameter.
    """
    def __init__(self, weights_path: Path, num_iterations: int = 5, lambda_value: float = 1):
        super().__init__()
        self.num_iterations = num_iterations
        
        # Load the pre-trained UNet
        self.unet = load_fastmri_unet(weights_path)
        # Freeze all UNet parameters
        for param in self.unet.parameters():
            param.requires_grad = False
        
        self.lambda_value = lambda_value 

        print(f"Data Consistency Wrapper initialized with fixed lambda: {self.lambda_value}")
        
    def forward(self, kspace_masked: torch.Tensor, mask: torch.Tensor, 
                X: torch.Tensor, mean_tensor: torch.Tensor, std_tensor: torch.Tensor) -> torch.Tensor:
        """
        Performs the unrolled iterative reconstruction.

        Args:
            kspace_masked: The zero-filled k-space data (under-sampled).
            mask: The binary sampling mask.
            X: Initial zero-filled image (for the first iteration).
            Y: The k-space version of X (for the first iteration).
            mean_tensor, std_tensor, max_value_tensor: Normalization factors.

        Returns:
            The final reconstructed image (X_final) in its original value range.
        """

        with torch.no_grad():
            
            # Unrolled Iterative Loop (T steps)
            for i in range(self.num_iterations):

                X_current = self.unet(X)
                # should denorm before padding with 0
                X_denorm = denorm_image(X_current, mean=mean_tensor, std = std_tensor)
                X_padded = pad_image(X_denorm, (kspace_masked.shape[1:3]))   
                X_complex = to_fastmri_complex_format(X_padded)
                K_current = fft2c(X_complex)

                # Apply Data Consistency Projection (P_D(S_k))
                # P_D(S_k) = (1 - mask) * S_k + mask * (S_k + lambda * kspace_masked) / (1 + lambda)

                # Ensure the mask is also compatible for broadcasting against [B, H, W, 2]
                if mask.dim() == 3:
                    mask = mask.unsqueeze(-1) # Convert [B, H, W] to [B, H, W, 1]
                # Calculate the numerator term: S_k + lambda * kspace_masked
                numerator = K_current + self.lambda_value * kspace_masked
                # Calculate the full DC projection
                kspace_projected = (1 - mask) * K_current + mask * (numerator / (1 + self.lambda_value))
                # Transform back to image space (X_projected)
                X_projected = ifft2c(kspace_projected)
                X_projected_magnitude = complex_abs(X_projected).unsqueeze(1)
                # Crop to ensure dimensions match the input of the next UNet step
                # This handles complex-to-real-to-complex transformations in PyTorch
                X = center_crop(X_projected_magnitude, X.shape[-2:])
                # normalize again for the next iteration
                X = norm_image(X, mean=mean_tensor, std = std_tensor)

        X = denorm_image(X, mean = mean_tensor, std = std_tensor)

        return X
    
class MeniscusTearModel(pl.LightningModule):
    """
    A  binary classification of single-channel MRI image using a pre-trained ResNet-18 backbone.
    
    The model loads ResNet-18 pre-trained on ImageNet, then modifies the first 
    convolutional layer for 1 input channel and the final fully connected layer 
    for 1 output feature (for BCEWithLogitsLoss).
    """
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        # We used a pretrained Resnet18 NN
        self.model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        
        # freeze the parameters:
        for name, param in self.model.named_parameters():
            param.requires_grad = False

        # The original conv1 takes 3 input channels. We replace it with a new layer
        # that takes 1 input channel (for grayscale/reconstructed images).

        self.model.conv1 = torch.nn.Conv2d(
            in_channels=1, 
            out_channels=64, 
            kernel_size=(7, 7), 
            stride=(2, 2), 
            padding=(3, 3), 
            bias=False
        )
        # Binary classification 
        self.model.fc = torch.nn.Linear(in_features=512, out_features=1)
        # unfreeze the fc layer
        for param in self.model.fc.parameters():
            param.requires_grad = True

        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = torch.optim.Adam(trainable_params, lr=1e-4)
        # Use BCEWithLogitsLoss which is robust and handles positive weighting for imbalanced datasets
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        
        # Initialize metrics
        self.train_acc = torchmetrics.Accuracy(task="binary")
        self.val_acc = torchmetrics.Accuracy(task="binary")

    def forward(self, data):
        """Passes input data through the ResNet backbone."""
        pred = self.model(data)
        return pred
    
    def training_step(self, batch, batch_idx):
        """Logic for a single training step."""
        image, label, _, _ = batch
        # Ensure label is float for loss computation
        label = label.float() 
        # Prediction: We take the first element [:, 0] to match the label shape
        pred_logits = self(image)[:, 0] 
        loss = self.loss_fn(pred_logits, label)
        
        # Log loss and batch accuracy
        self.log("Train Loss", loss, on_step=True, on_epoch=True)
        # Use sigmoid to get probabilities before computing binary accuracy
        self.train_acc.update(torch.sigmoid(pred_logits), label.int())
        self.log("Step Train Acc", self.train_acc, on_step=True, on_epoch=False)
        return loss
    
    def on_training_epoch_end(self):
        # After one epoch compute the whole train_data accuracy
        self.log("Train Acc", self.train_acc.compute())
        self.train_acc.reset()
        
    def validation_step(self, batch, batch_idx):
        # Same steps as in the training_step
        image, label, _, _  = batch
        label = label.float()
        pred = self(image)[:,0]  # make sure prediction and label have same shape

        loss = self.loss_fn(pred, label)
        
        # Log validation metrics
        self.log("Val Loss", loss)
        self.log("Step Val Acc", self.val_acc(torch.sigmoid(pred), label.int()))
        return loss
    
    def on_validation_epoch_end(self):
        self.log("Val Acc", self.val_acc.compute(), prog_bar=True)
        self.val_acc.reset()
    
    def configure_optimizers(self):
        #Caution! You always need to return a list here (just pack your optimizer into one :))
        return [self.optimizer]