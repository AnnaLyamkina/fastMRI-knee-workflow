# FastMRI Classification Demo: Comparative Workflow
### 1. Project Idea and Scope

This project is a **small workflow demonstration** designed to achieve and **compare** initial classification results across two distinct data input scenarios using the FastMRI knee dataset and external clinical labels (FastMRI+).

The core idea is to benchmark the classification performance by comparing two inputs, both derived from **undersampled k-space data**:

1. **Pipeline A (U-Net Reconstruction / Baseline):** Uses the image reconstructed by first performing an **IFFT on the undersampled k-space** (zero-filled), and then using the **Pretrained U-Net** to refine the resulting artifact-ridden image. This image is then fed to the same **pretrained U-Net (acting as a CNN feature extractor)** for classification. This establishes the performance of the advanced reconstruction method.

2. **Pipeline B (K-Space Reconstruction / Comparison):** Uses a **second Pretrained U-Net** (trained for K-space processing) to directly reconstruct the undersampled k-space data. The resulting high-quality image is then fed to the same **Pretrained U-Net (acting as a CNN feature extractor)** for classification. This provides the K-space benchmark.

In both scenarios, we prioritize resource efficiency by using the pre-trained U-Net's encoding layers as a **fixed feature extractor (Backbone)**, training only a small, newly added **Classification Head** attached to it.

### Data Usage and Citation
Data used in the preparation of this project were obtained from the NYU fastMRI Initiative database ([fastmri.med.nyu.edu](URL)). The data is licensed solely for internal research and educational purposes. Commercial exploitation, monetization, distribution, or transfer of the dataset is strictly prohibited. **Therefore, the contents of the data/ folder are maintained locally, and no raw data or notebooks containing data references will be shared on GitHub or similar platforms.**

References:

1. [Knoll et al. Radiol Artif Intell. 2020 Jan 29;2(1):e190007. doi: 10.1148/ryai.2020190007.](URL)

2. [https://arxiv.org/abs/1811.08839](URL)

### 2. Project Folder Structure

The repository is organized to separate data, reusable code components, and experimental assets.
    
├── data/  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── raw/                     # Large original HDF5 files (train/val)  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── labels/                   # fastmri_plus_labels.csv (The ground  truth for classification)  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── processed/                # Any pre-processed metadata (e.g., image_metadata.json)    
|  
├── models/  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── pretrained/               # The downloaded, frozen U-Net weights   (unet_fastmri.pt)  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── checkpoints/              # Checkpoints for the small, trainable Classification Head  
│  
├── notebooks/                    # Exploration and prototyping sandbox (Jupyter)  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── 01_Data_Load_Preprocess.ipynb           # Initial k-space data loading, mask prototyping, and label mapping.  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── 02_Real_Space_Reconstruction.ipynb      # Executes IFFT to get fully reconstructed image input (Pipeline A Setup).  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── 03_Pipeline_A_Classification.ipynb      # Trains the Classification Head using the clean input (Pipeline A).  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── 04_KSpace_Reconstruction.ipynb          # Executes the U-Net reconstruction from undersampled k-space (Pipeline B Input Prep).  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── 05_Pipeline_B_Classification.ipynb      # Trains the Classification Head using the reconstructed images (Pipeline B).  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── 06_Comparative_Analysis.ipynb           # Compares final results from Pipeline A and Pipeline B  
│  
├── src/                          # Modularized, production-ready Python code  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── main_train.py             # Entry point for training the classification  head  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── main_evaluate.py          # Entry point for metric evaluation  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── models.py                 # Defines the UNetClassifier class and freezing logic  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── datasets.py               # Handles HDF5 reading and label mapping  
│  
├── results/                      # Saved metrics and visualizations from evaluation runs  
├── requirements.txt              # Environment dependencies (PyTorch, fastmri, etc.)  
└── README.md                     # This file  

