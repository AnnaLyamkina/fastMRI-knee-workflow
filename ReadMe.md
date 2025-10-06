# FastMRI Classification Demo: Comparative Workflow
### 1. Project Idea and Scope

This project is a **small workflow demonstration** designed to **achieve and compare** real space and k-space reconstruction of undersampled MRI data and relate initial classification results across two distinct reconstruction methods using the FastMRI knee dataset and external clinical labels (FastMRI+).

This comparative approach focuses on **tasked-based reconstruction (TBR)**, measuring the utility of each reconstruction method by its ultimate impact on diagnostic accuracy (Meniscus Tear detection) rather than relying solely on image quality metrics.


### 1.1 Core Pipeline Objectives:

- Pipeline A (Real-Space Reconstruction): The process involves adapting a pretrained U-Net to refine artifacts from a zero-filled, real-space magnitude image.

- Pipeline B (K-Space Reconstruction): This involves training a U-Net to directly reconstruct high-quality k-space data before conversion to a magnitude image via IFFT.

- Pipeline C (Classification): This final stage performs Meniscus Tear detection using a frozen CNN feature extractor and a trainable classification head.

### 1.2 Experimental Flows: Task-Based Reconstruction Benchmarks

The core idea is to benchmark the classification performance by comparing three distinct image processing flows, all derived from undersampled k-space data. Flows 1 and 2 act as benchmarks, while Flow 3 represents the End-to-End Optimization objective.

**1. Flow A→C (Real-Space Reconstruction Baseline):**

Image Source: Uses the image reconstructed by first performing an IFFT on the undersampled k-space (zero-filled), and then using the Pretrained U-Net (trained for Real-Space) to refine the resulting artifact-ridden image.

Classification: This high-quality image is fed to the same pretrained U-Net (acting as a CNN feature extractor) for classification. This establishes the performance of the advanced reconstruction method.

**2. Flow B→IFFT→C (K-Space Reconstruction Comparison):**

Image Source: Uses a second Pretrained U-Net (trained for K-space processing) to directly reconstruct the undersampled k-space data. The resulting high-quality k-space is then transformed via IFFT to create the final magnitude image.

Classification: This image is also fed to the same pretrained U-Net (acting as a CNN feature extractor) for classification. This provides the K-space benchmark.

Possible extension:

**3. Flow A→C (End-to-End Task-Driven Model - True TBR):**

Training Objective: The reconstruction U-Net (Pipeline A) and the Classification Head (Pipeline C) are trained simultaneously. The feature extractor (U-Net encoder) is not frozen.

Loss Function: A combined loss is used: 

$$ Loss_{total}​=λ⋅Loss_{Recon}​+Loss_{Clas} $$

Result: This provides the benchmark for true Task-Based Reconstruction, where the reconstruction weights are directly informed by the requirements of the classification task.

### 2. Data Usage and Citation
Data used in the preparation of this project were obtained from the NYU fastMRI Initiative database ([fastmri.med.nyu.edu](URL)). The data is licensed solely for internal research and educational purposes. Commercial exploitation, monetization, distribution, or transfer of the dataset is strictly prohibited. **Therefore, the contents of the data/ folder are maintained locally, and no raw data or notebooks containing data references will be shared on GitHub or similar platforms.**

References:

1. [Knoll et al. Radiol Artif Intell. 2020 Jan 29;2(1):e190007. doi: 10.1148/ryai.2020190007.](URL)

2. [https://arxiv.org/abs/1811.08839](URL)

### 3. Project Folder Structure

The repository is organized to separate data, reusable code components, and experimental assets.
    
├── data/  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── raw/                     # Large original HDF5 files (train/val)  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── labels/                   # fastmri_plus_labels.csv (The ground  truth for classification)  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── processed/                # Any pre-processed metadata (e.g., image_metadata.json)    
|  
├── models/  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── pretrained/               # The downloaded, frozen U-Net weights   
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── checkpoints/              # Checkpoints for the small, trainable Classification Head  
│  
├── notebooks/                    # Exploration and prototyping sandbox (Jupyter)  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── 01_Data_Load_Preprocess.ipynb           # Initial k-space data loading, mask prototyping, and label mapping.  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── 02_Real_Space_Reconstruction.ipynb      # Sets up a pretrained U-net for reconstruction in real domain (Pipeline A).  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── 03_Classification_Head_Baseline.ipynb      # Trains the Classification Head using the clean input from Pipeline A. (Pipeline C)  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── 04_KSpace_Reconstruction.ipynb          # Executes the U-Net reconstruction from undersampled k-space (Pipeline B).  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── 
*05_End_to_End_TBR_Training.  # Optional: performs end-to-end training of Pipeline A or B together with Pipeline C with combined loss to archieve a fully task-based reconstruction.*  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── 
06_Comparative_Analysis.ipynb           # Compares final results from Flows 1 and 2  (optionally also Flow 3)   
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

