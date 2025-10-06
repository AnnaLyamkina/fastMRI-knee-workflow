import torch
from pathlib import Path
import sys
from fastmri.models import Unet

def load_fastmri_unet(weights_path: Path) -> Unet:
    """
    Loads the U-Net model architecture and weights.
    """
    print(f"Loading U-Net model from: {weights_path.name}...")
    
    model = Unet(in_chans=1, out_chans=1, chans=256, num_pool_layers=4, drop_prob=0.0)
    checkpoint = torch.load(weights_path, map_location=torch.device('cpu'))
    print("Checkpoint loaded")
    model.load_state_dict(checkpoint)
    print("state loaded")
   
    return model
