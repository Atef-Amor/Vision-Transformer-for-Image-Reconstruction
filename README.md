# Vision-Transformer-for-Image-Reconstruction
% Defining document class and basic setup
\documentclass[12pt]{article}
\usepackage[utf8]{inputenc}
\usepackage{geometry}
\geometry{a4paper, margin=1in}
\usepackage{hyperref}
\hypersetup{
    colorlinks=true,
    linkcolor=blue,
    urlcolor=blue,
    citecolor=blue
}
\usepackage{tocloft}
\usepackage{enumitem}
\usepackage{amsmath}
\usepackage{booktabs}
\usepackage{parskip}
\usepackage{titling}
\usepackage{xcolor}

% Setting up title and author
\title{Vision Transformer for Image Reconstruction}
\author{}
\date{}

% Customizing table of contents
\renewcommand{\cftsecleader}{\cftdotfill{\cftdotsep}}
\renewcommand{\cftsecpagefont}{\normalfont}

% Begin document
\begin{document}

% Generating title
\maketitle

% Generating table of contents
\tableofcontents
\newpage

% Project Overview Section
\section{Project Overview}
This repository implements a Vision Transformer (ViT) model for image reconstruction using masking techniques, as part of a data mining project. The project applies self-supervised learning inspired by Natural Language Processing (NLP) to computer vision, leveraging the CIFAR-10 dataset. The ViT model reconstructs images by predicting masked patches, and its performance is compared to a classical autoencoder baseline using metrics such as Mean Squared Error (MSE), Structural Similarity Index (SSIM), and Peak Signal-to-Noise Ratio (PSNR).

The project focuses on image reconstruction using Vision Transformers (ViTs) with masking, an innovative approach that adapts Transformer models from NLP to computer vision. Images are divided into patches, treated as sequential tokens, and partially masked to train the model to reconstruct missing patches. Unlike classical autoencoders that compress images into a latent space, ViTs use contextual information from visible patches to predict masked ones, capturing long-range dependencies. The project tests various masking strategies and compares ViT performance against an autoencoder baseline.

% Features Section
\section{Features}
\begin{itemize}[leftmargin=*]
    \item \textbf{Patch-based Processing}: Divides 32x32 CIFAR-10 images into 4x4 patches for sequential processing.
    \item \textbf{Vision Transformer}: Implements a ViT with multi-head attention and feed-forward networks.
    \item \textbf{Multiple Masking Strategies}: Includes random, row-wise, column-wise, and importance-based masking.
    \item \textbf{Flexible Experimentation}: Supports different masking ratios (30\%, 50\%, 70\%).
    \item \textbf{Evaluation Metrics}: Uses MSE, SSIM, and PSNR for quantitative evaluation, with visualization of original, masked, and reconstructed images.
    \item \textbf{Baseline Comparison}: Compares ViT performance with a classical autoencoder.
    \item \textbf{Learning Rate Scheduling}: Incorporates \texttt{ReduceLROnPlateau} for adaptive training.
\end{itemize}

% Requirements Section
\section{Requirements}
\begin{itemize}[leftmargin=*]
    \item Python 3.8 or higher
    \item TensorFlow 2.8 or higher
    \item NumPy
    \item Matplotlib
    \item Scikit-image
    \item TQDM
\end{itemize}

% Installation Section
\section{Installation}
\begin{enumerate}[leftmargin=*]
    \item Clone the repository:
    \begin{verbatim}
git clone https://github.com/your-username/vision-transformer-reconstruction.git
cd vision-transformer-reconstruction
    \end{verbatim}
    \item Create and activate a virtual environment:
    \begin{verbatim}
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
    \end{verbatim}
    \item Install dependencies:
    \begin{verbatim}
pip install tensorflow numpy matplotlib scikit-image tqdm
    \end{verbatim}
\end{enumerate}

% Dataset Section
\section{Dataset}
The project uses the \textbf{CIFAR-10 dataset}, containing 60,000 32x32 RGB images (50,000 training, 10,000 testing) across 10 classes. The dataset is loaded and preprocessed using \texttt{tf.keras.datasets.cifar10.load\_data()}.

% Methodology Section
\section{Methodology}
The project follows a structured methodology for image reconstruction with ViTs:

\subsection{Preprocessing}
\begin{itemize}[leftmargin=*]
    \item \textbf{Image Normalization}: Pixel values are scaled to $[0, 1]$ by dividing by 255.
    \item \textbf{Patch Extraction}: Images are divided into 4x4 patches, resulting in 64 patches per image (each patch is 48-dimensional: $4 \times 4 \times 3$ RGB channels).
    \item \textbf{Patch Flattening}: Each patch is flattened and projected into a 128-dimensional embedding space.
\end{itemize}

\subsection{Masking Strategies}
Four masking strategies are implemented to hide patches during training:
\begin{itemize}[leftmargin=*]
    \item \textbf{Random Masking}: Randomly masks a specified percentage of patches (e.g., 50\%).
    \item \textbf{Row-wise Masking}: Masks entire rows of patches in the image grid.
    \item \textbf{Column-wise Masking}: Masks entire columns of patches.
    \item \textbf{Importance-based Masking}: Masks patches based on their importance, determined by gradients from a trained model.
\end{itemize}
Different masking ratios (30\%, 50\%, 70\%) are tested to evaluate their impact on reconstruction quality.

\subsection{Model Architecture}
The ViT model consists of:
\begin{itemize}[leftmargin=*]
    \item \textbf{Patch Embedding}: Projects 48-dimensional patches to 128 dimensions.
    \item \textbf{Positional Embedding}: Adds learnable position encodings to maintain spatial information.
    \item \textbf{Transformer Blocks}: 4 blocks, each including:
    \begin{itemize}
        \item Multi-head attention (4 heads)
        \item Layer normalization
        \item Feed-forward network (256 units, GELU activation)
        \item Dropout (0.1)
    \end{itemize}
    \item \textbf{Output Layer}: Predicts the original 48-dimensional patches using a sigmoid activation.
\end{itemize}

\subsection{Training}
\begin{itemize}[leftmargin=*]
    \item The model is trained to reconstruct masked patches using the Mean Squared Error (MSE) loss.
    \item The Adam optimizer is used with an initial learning rate of 0.001.
    \item Learning rate scheduling (\texttt{ReduceLROnPlateau}) adjusts the learning rate based on validation loss.
    \item Models are saved during training (best and final) in the \texttt{saved\_models} directory.
\end{itemize}

\subsection{Evaluation}
\begin{itemize}[leftmargin=*]
    \item \textbf{Metrics}: MSE, SSIM, and PSNR are computed to assess reconstruction quality.
    \item \textbf{Visualization}: Original, masked, and reconstructed images are visualized for qualitative analysis.
    \item \textbf{Baseline Comparison}: The ViT's performance is compared to a classical autoencoder.
\end{itemize}

% Usage Section
\section{Usage}
The project is implemented in a Jupyter Notebook (\texttt{ViT.ipynb}). Follow these steps to run it:

\begin{enumerate}[leftmargin=*]
    \item \textbf{Preprocess the Data}:
    \begin{verbatim}
(x_train, _), (x_test, _) = tf.keras.datasets.cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
train_patches = image_to_patches(x_train)
test_patches = image_to_patches(x_test)
    \end{verbatim}
    \item \textbf{Train the Model}:
    \begin{verbatim}
vit_model = create_vit_model(num_layers=4, embed_dim=128, num_heads=4, ff_dim=256, dropout_rate=0.1)
vit_model.compile(optimizer=Adam(0.001), loss=MeanSquaredError())
train_loss_history, val_loss_history = train_model(
    model=vit_model,
    train_patches=train_patches,
    mask_fn=random_masking,
    mask_ratio=0.5,
    epochs=50,
    batch_size=128,
    validation_split=0.1,
    save_dir='saved_models'
)
    \end{verbatim}
    \item \textbf{Evaluate the Model}:
    \begin{verbatim}
evaluate_model(vit_model, test_patches, random_masking, mask_ratio=0.5)
    \end{verbatim}
    \item \textbf{Experiment with Masking Ratios}: Modify the \texttt{mask\_ratio} parameter (e.g., 0.3, 0.5, 0.7) in the \texttt{train\_model} and \texttt{evaluate\_model} functions to test different masking ratios.
\end{enumerate}

% Results Section
\section{Results}
The ViT model outperforms the classical autoencoder across all evaluated metrics:

\begin{table}[h]
    \centering
    \begin{tabular}{lcccc}
        \toprule
        \textbf{Metric} & \textbf{Vision Transformer} & \textbf{Autoencoder} & \textbf{Difference (ViT - Autoencoder)} \\
        \midrule
        MSE & 0.0065 & 0.0088 & -0.0024 (ViT better) \\
        SSIM & 0.8047 & 0.6480 & 0.1567 (ViT better) \\
        PSNR & 22.7501 & 21.0665 & 1.6836 (ViT better) \\
        \bottomrule
    \end{tabular}
    \caption{Performance comparison between Vision Transformer and Autoencoder.}
\end{table}

These results highlight the ViT's ability to capture global dependencies and reconstruct images more effectively than the autoencoder, particularly with a 50\% masking ratio. Experiments with different masking ratios (30\%, 70\%) can be conducted to further analyze performance trends.

% Contributing Section
\section{Contributing}
Contributions are welcome! To contribute:
\begin{enumerate}[leftmargin=*]
    \item Fork the repository.
    \item Create a feature branch (\texttt{git checkout -b feature-branch}).
    \item Commit your changes (\texttt{git commit -m 'Add feature'}).
    \item Push to the branch (\texttt{git push origin feature-branch}).
    \item Open a pull request.
\end{enumerate}
Please adhere to PEP 8 guidelines and include tests for new features.

% License Section
\section{License}
This project is licensed under the MIT License. See the \href{https://github.com/your-username/vision-transformer-reconstruction/blob/main/LICENSE}{LICENSE} file for details.

% End document
\end{document}
