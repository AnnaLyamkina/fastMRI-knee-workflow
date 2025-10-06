import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import os
from pathlib import Path
from typing import List, Tuple, Any

# Import crucial fastMRI transformation utilities
from fastmri.data.transforms import center_crop #, center_crop_to_smallest
from fastmri import ifft2c, complex_abs

# --- Mask Generation Helper Function ---

def generate_undersampling_mask(shape: Tuple, AF: int, center_fraction: float, power: float) -> torch.Tensor:
    """
    Generates a 2D Cartesian undersampling mask with Variable Density.
    The mask is 1D in the frequency-encoding direction (W) and tiled 
    across the phase-encoding direction (H).
    
    Args:
        shape: Shape of the k-space slice, typically [C, H, W].
        AF (int): Acceleration factor.
        center_fraction (float): fraction of low frequencies fully sampled.
        power: power in density equation (for Variable Density profile).

    Returns:
        torch.Tensor: A 3D mask tensor of shape [1, H, W] ready for broadcasting.
    """
    # Determine H and W from the shape (assuming [C, H, W] or [H, W] input)
    if len(shape) == 3:
        H, W = shape[1], shape[2]
    elif len(shape) == 2:
        H, W = shape
    else:
        raise ValueError("Shape must be 2D or 3D.")

    # 1. Determine center lines (ACR width) based on W
    center_lines_abs = int(W * center_fraction)
    # Ensure center lines count is an even number
    center_lines = center_lines_abs if center_lines_abs % 2 == 0 else center_lines_abs + 1

    # 2. Create 1D probability density function (PDF)
    x = np.linspace(-1, 1, W)
    # The probability density is inversely proportional to the AF and position
    probs = 1.0 / (1.0 + (np.abs(x) ** power) * AF)

    # 3. Stochastic sampling
    mask_1d = np.random.rand(W) < probs

    # 4. Fully sample the center lines (Autocalibration Region - ACR)
    center_start = (W - center_lines) // 2
    mask_1d[center_start:center_start + center_lines] = True
        
    # 5. Create the 2D mask by tiling the 1D mask across the phase-encoding dimension (H)
    # The result is a 2D NumPy array of shape [H, W]
    mask_2d = np.tile(mask_1d, (H, 1))

    # 6. Convert to PyTorch tensor, change to float, and add explicit Coil dimension [1, H, W]
    # This prepares the mask for multiplication with k-space tensor [C, H, W] via broadcasting
    mask_tensor = torch.from_numpy(mask_2d).float()
    
    return mask_tensor

# --- 1. Base Dataset for Shared K-space Loading Logic ---

class FastMRIBaseDataset(torch.utils.data.Dataset):
    """
    Base class containing shared logic for loading k-space and its normalization attributes 
    from a pre-split CSV file.
    
    NOTE: This class now uses the local `generate_undersampling_mask` function.
    """
    def __init__(self, 
                 data_root: Path, 
                 manifest_path: Path, 
                 acceleration_factor: int, 
                 center_fraction: float = 0.05, 
                 power: float = 4.0, 
                 target_resolution: Tuple[int, int] = (320, 320)):
        
        self.data_root = data_root
        self.target_resolution = target_resolution
        self.acceleration_factor = acceleration_factor
        
        # Mask parameters
        self.center_fraction = center_fraction
        self.power = power 
        
        # 1. Load the pre-split manifest (e.g., train.csv)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found at {manifest_path}")
            
        self.manifest = pd.read_csv(manifest_path)

        # 2. Ensure slice and label are integers and use the original column names
        self.manifest['slice'] = self.manifest['slice'].astype(int)
        self.manifest['label'] = self.manifest['label'].astype(int)

        # 3. Create the final list of samples (tuple: file_id, slice_idx, label)
        # We use the original 'file' and 'label' column names here.
        self.samples = self.manifest[['file', 'slice', 'label']].values.tolist()

        if self.manifest.empty:
            raise RuntimeError("Manifest is empty after loading. Check paths and files.")

    def __len__(self):
        return len(self.samples)

    def _load_kspace_and_attributes(self, file_id: str, slice_idx: int) -> Tuple[torch.Tensor, float, float, Tuple[int, int]]:
        """
        Loads a single k-space slice and both normalization attributes.

        Args:
            file_id: The file name (without .h5 extension) from the 'file' column.
            slice_idx: The slice index.

        Returns: (kspace_tensor, global_image_max, kspace_l2_norm, shape)
        """
        # The file_id is used to construct the path
        file_path = self.data_root / f"{file_id}.h5"
        
        if not file_path.exists():
             raise FileNotFoundError(f"HDF5 file not found for {file_id} at {file_path}")

        with h5py.File(file_path, 'r') as hf:
            # 1. Load k-space and select the correct slice
            kspace_full = hf['kspace'][slice_idx]
            
            # 2. Get the normalization attributes
            # 'max' is for IMAGES (used in A and C)
            global_image_max = hf.attrs.get('max', 1.0) 
            # 'norm' is for K-SPACE (used in B)
            kspace_l2_norm = hf.attrs.get('norm', 1.0)
            
            # 3. Get the original full shape of the k-space (H, W)
            shape = kspace_full.shape
            
        # Convert NumPy array to complex PyTorch tensor
        kspace_tensor = torch.from_numpy(kspace_full)
        
        return kspace_tensor, global_image_max, kspace_l2_norm, shape

    def _get_masked_kspace(self, kspace_full: torch.Tensor) -> torch.Tensor:
        """Generates the mask using class parameters and applies it to the k-space."""
        # kspace_full is [Coil, H, W]
        shape = kspace_full.shape 
        
        # Generate the mask [1, H, W]
        mask_tensor = generate_undersampling_mask( # Renamed to mask_tensor for clarity
            shape=shape, # Pass the [C, H, W] shape for dimension inference
            AF=self.acceleration_factor, 
            center_fraction=self.center_fraction, 
            power=self.power 
        )
        
        # Apply the mask: Kspace [C, H, W] * Mask [1, H, W] (broadcasting handles the C dimension)
        kspace_masked = kspace_full * mask_tensor
        
        return kspace_masked


# --- 2. Pipeline A: Real-Space Reconstruction Dataset (RealSpaceReconDataset) ---

class RealSpaceReconDataset(FastMRIBaseDataset):
    """
    Dataset for Pipeline A: Image-to-Image Reconstruction.
    Returns: (Zero-filled magnitude image, Fully-sampled magnitude image)
    """
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        file_id, slice_idx, _ = self.samples[idx]

        # We only need the image max for normalization here
        kspace_full, global_image_max, _, _ = self._load_kspace_and_attributes(file_id, slice_idx)

        # Apply mask to generate undersampled k-space using the new helper method
        kspace_masked = self._get_masked_kspace(kspace_full)

        # --- CRITICAL FIX: Convert native complex tensors to [..., 2] format --- due to ifft2c restrictions
        def to_fastmri_complex_format(kspace_tensor: torch.Tensor) -> torch.Tensor:
            """
            Converts native complex tensors (torch.complex64) to the [..., 2] 
            format required by older fastmri IFFT functions (ifft2c).
            """
            if kspace_tensor.is_complex():
                # Stack real and imaginary parts along a new last dimension
                return torch.stack((kspace_tensor.real, kspace_tensor.imag), dim=-1)
            # If it's already in the expected [..., 2] format, return as is
            return kspace_tensor

        # Apply the conversion to both k-space tensors
        kspace_full = to_fastmri_complex_format(kspace_full)
        kspace_masked = to_fastmri_complex_format(kspace_masked)

        # --- Transformation Steps (IFFT, Magnitude, SoS) ---
        
        # 1. Transform both full and masked k-space to image domain
        img_full_complex = ifft2c(kspace_full)
        img_masked_complex = ifft2c(kspace_masked)

        # 2. Calculate Sum-of-Squares (SoS) Magnitude (single channel)
        img_full_magnitude = complex_abs(img_full_complex).square()
        img_masked_magnitude = complex_abs(img_masked_complex).square()

        # 3. Center Crop
        img_full_magnitude = center_crop(img_full_magnitude, self.target_resolution)
        img_masked_magnitude = center_crop(img_masked_magnitude, self.target_resolution)

        # 5. CRITICAL: Calculate Slice-Specific Mean and Std for the *Target*
        # This is the normalization used in FastMRI for the target image Y
        
        # Calculate mean (mu) and standard deviation (sigma)
        mean = img_full_magnitude.mean()
        std = img_full_magnitude.std()
        std_epsilon = std + 1e-11

        img_full_norm = (img_full_magnitude - mean) / std_epsilon
        img_masked_norm = img_masked_magnitude / img_masked_magnitude.max()

        # 5. Final PyTorch Tensor Formatting: [C, H, W]
        X = img_masked_norm.unsqueeze(0).float()
        Y = img_full_norm.unsqueeze(0).float()

        # The mean and std must also be tensors for batching
        mean_tensor = torch.tensor(mean, dtype=torch.float32)
        std_tensor = torch.tensor(std, dtype=torch.float32)
        max_value_tensor = torch.tensor(img_full_magnitude.max(), dtype=torch.float32)
        
        return X, Y, mean_tensor, std_tensor, max_value_tensor

# --- 3. Pipeline B: K-Space Reconstruction Dataset (KSpaceReconDataset) ---

class KSpaceReconDataset(FastMRIBaseDataset):
    """
    Dataset for Pipeline B: K-Space Reconstruction.
    Returns: (Masked K-space complex tensor, Full K-space complex tensor)
    """
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        file_id, slice_idx, _ = self.samples[idx]
        
        # We only need the K-space norm for normalization here
        kspace_full, _, kspace_l2_norm, _ = self._load_kspace_and_attributes(file_id, slice_idx)
        
        # CRITICAL: Normalize K-space using the L2 norm attribute
        kspace_full_norm = kspace_full / kspace_l2_norm
        
        # Apply mask to generate undersampled k-space on the normalized data
        kspace_masked = self._get_masked_kspace(kspace_full_norm)
        
        # Output is the masked and full normalized K-space tensors
        # The tensor shape is [Coil, H, W] and the data type is complex
        X = kspace_masked
        Y = kspace_full_norm
        
        return X, Y


# --- 4. Pipeline C: Classification Dataset (ClassificationDataset) ---

class ClassificationDataset(FastMRIBaseDataset):
    """
    Dataset for Pipeline C: Classification.
    Returns: (Fully-sampled magnitude image, Binary label)
    
    NOTE: Classification only uses the fully sampled image, so no mask is applied here.
    """
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        file_id, slice_idx, label_int = self.samples[idx]
        
        # We only need the image max for normalization here
        kspace_full, global_image_max, _, _ = self._load_kspace_and_attributes(file_id, slice_idx)
        
        # --- Transformation Steps (IFFT, Magnitude, SoS) ---
        
        # 1. Transform to image domain
        img_full_complex = ifft2c(kspace_full)
        
        # 2. Calculate Sum-of-Squares (SoS) Magnitude
        img_full_magnitude = complex_abs(img_full_complex).square().sum(dim=0).sqrt()
        
        # 3. Center Crop
        #crop_size = center_crop_to_smallest(img_full_magnitude.shape)
        # img_full_magnitude = center_crop(img_full_magnitude, crop_size)
        img_full_magnitude = center_crop(img_full_magnitude, self.target_resolution)
        
        # 4. Normalization (using global_image_max from the 'max' attribute)
        img_full_norm = img_full_magnitude / global_image_max
        
        # 5. Final PyTorch Tensor Formatting: [C, H, W]
        X = img_full_norm.unsqueeze(0).float()
        L = torch.tensor(label_int).long()
        
        return X, L

