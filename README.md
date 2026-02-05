# Maqam Identification using Diffusion SDE Methods

This project attempts to identify and classify **Maqam** (the system of melodic modes in Arabic music) using **Diffusion Models**, specifically **Stochastic DDPM (Denoising Diffusion Probabilistic Models)**. The stochastic nature of Arabic music requires more than a classic classifier; it demands a deeper understanding of its structure, making diffusion models an ideal candidate.

The goal of the project is to explore various methods for Maqam identification and classification using diffusion models, as well as advanced embeddings to capture the unique characteristics of Arabic music.

## Methods Explored:

### 1) **Conditional Score-Based Diffusion**:
   - This method involves conditioning the model on some additional information, such as the type of Maqam, to improve classification performance. The model learns to generate or classify Maqam music with higher accuracy by incorporating this conditioning.

### 2) **Unconditional Score-Based Diffusion**:
   - This method focuses on learning the distribution of Maqam music without additional conditioning. The model learns to generate or represent Maqam music from scratch in an unsupervised manner.

### 3) **Positional Ornamental Embedding**:
   - Arabic music is known for its ornamentation (trills, slides, microtonal variations). This method seeks to capture the **context** and **placement** of ornamental features across a performance, allowing the model to understand and replicate the fine details of Maqam music.

## Dependencies

Below are the required dependencies for running this project, including essential libraries for machine learning, audio processing, and diffusion model implementation.

### Install the following packages:

```bash
pip install torch torchvision torchaudio
pip install librosa
pip install numpy
pip install matplotlib
pip install mido
