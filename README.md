# Vision-Transformer-for-Image-Reconstruction
The project focuses on image reconstruction using Vision Transformers (ViTs) with masking, an innovative approach that adapts Transformer models from NLP to computer vision. Images are divided into patches, treated as sequential tokens, and partially masked to train the model to reconstruct missing patches. Unlike classical autoencoders that compress images into a latent space, ViTs use contextual information from visible patches to predict masked ones, capturing long-range dependencies. The project tests various masking strategies and compares ViT performance against an autoencoder baseline.

## Features
-Patch-based Processing: Divides 32x32 CIFAR-10 images into 4x4 patches for sequential processing.
-Vision Transformer: Implements a ViT with multi-head attention and feed-forward networks.
-Multiple Masking Strategies: Includes random, row-wise, column-wise, and importance-based masking.
-Flexible Experimentation: Supports different masking ratios (30%, 50%, 70%).
-Evaluation Metrics: Uses MSE, SSIM, and PSNR for quantitative evaluation, with visualization of original, masked, and reconstructed images.
-Baseline Comparison: Compares ViT performance with a classical autoencoder.
-Learning Rate Scheduling: Incorporates ReduceLROnPlateau for adaptive training.

## Requirements
Python 3.8+
TensorFlow 2.8+
NumPy
Matplotlib
Scikit-image
TQDM

## Dataset
The project uses the CIFAR-10 dataset, containing 60,000 32x32 RGB images (50,000 training, 10,000 testing) across 10 classes. The dataset is loaded and preprocessed using tf.keras.datasets.cifar10.load_data().

## Methodology

The project follows a structured methodology for image reconstruction with ViTs:
Preprocessing
Image Normalization: Pixel values are scaled to [0, 1] by dividing by 255.
Patch Extraction: Images are divided into 4x4 patches, resulting in 64 patches per image (each patch is 48-dimensional: 4x4x3 RGB channels).
Patch Flattening: Each patch is flattened and projected into a 128-dimensional embedding space.

### Masking Strategies

Four masking strategies are implemented to hide patches during training:
Random Masking: Randomly masks a specified percentage of patches (e.g., 50%).
Row-wise Masking: Masks entire rows of patches in the image grid.
Column-wise Masking: Masks entire columns of patches.
Importance-based Masking: Masks patches based on their importance, determined by gradients from a trained model.
Different masking ratios (30%, 50%, 70%) are tested to evaluate their impact on reconstruction quality.

### Model Architecture

The ViT model consists of:
Patch Embedding: Projects 48-dimensional patches to 128 dimensions.
Positional Embedding: Adds learnable position encodings to maintain spatial information.
Transformer Blocks: 4 blocks, each including:
Multi-head attention (4 heads)
Layer normalization
Feed-forward network (256 units, GELU activation)
Dropout (0.1)
Output Layer: Predicts the original 48-dimensional patches using a sigmoid activation.

### Training
The model is trained to reconstruct masked patches using the Mean Squared Error (MSE) loss.
The Adam optimizer is used with an initial learning rate of 0.001.
Learning rate scheduling (ReduceLROnPlateau) adjusts the learning rate based on validation loss.
Models are saved during training (best and final) in the saved_models directory.

### Evaluation
Metrics: MSE, SSIM, and PSNR are computed to assess reconstruction quality.
Visualization: Original, masked, and reconstructed images are visualized for qualitative analysis.
Baseline Comparison: The ViT's performance is compared to a classical autoencoder.
