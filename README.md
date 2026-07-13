# Arabic Maqam Identification via Generative Score-Matching and Regularized Latent Representations

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)

This repository contains the implementation, models, and evaluation frameworks for **Arabic Maqam Identification** using deep sequence modeling, generative diffusion scoring, and regularized latent joint representation models.

* **Author:** Charbel El Haddad  
* **Affiliation:** American University of Beirut (AUB) Research Department  

---

## Abstract

Arabic music relies on modal systems called Maqams, which present challenging classification problems due to microtonal interval patterns. We present a chronological study of multiple machine learning approaches for Maqam identification. Starting from sequential LSTM baselines, we explore a conditional convolutional diffusion classifier using a parameterized inner product space on the score function. We then investigate dual-network architectures combining convolutional autoencoders and classification layers, demonstrating that a regularized joint loss formulation yields state-of-the-art results (93.2% accuracy) over standard setups. Finally, we initiate a mechanistic interpretability study to isolate the receptive fields of specific neurons and kernels.

---

## Chronological Methodology & Architectures

### 1. Sequential LSTM Baseline
Our initial investigations began with a sequential Long Short-Term Memory (LSTM) network designed to classify pitch-class profiles extracted over time. LSTMs are effective at tracking localized pitch shifts, but struggle with global modal context when transpositions are present.
* **Accuracy:** 60.0%

### 2. Conditional Convolutional Diffusion Classifier
To leverage generative representations, we trained a conditional Convolutional Diffusion Model on spectrogram images $x$ conditioned on Maqam labels $y$. By feeding both the image and the label during training, the network learns to approximate the score function (the gradient of the log probability density):

$$
abla_{x_t} \log p_t(x_t \mid y)$$

During inference, computing the exact density is intractable. We approximate it by taking the inner product (cosine distance) between the noisy sample $x_t$ and the denoised sample $\hat{x}_{0|y,t}$, which is obtained by applying the neural network predictor $\psi_	heta(x_t, t, y)$:

$$	ext{Score}(y)  pprox \langle x_t, \hat{x}_{0|y,t} 
angle$$

To optimize classification, we defined a parameterized inner product space governed by a learnable metric matrix $A$:

$$	ext{Metric}(x, y) = x^T A y$$

We trained the metric matrix $A$ using stochastic gradient descent (SGD) to minimize the cross-entropy loss between the scores and the true labels. This approach reached an accuracy of 56.0%. To avoid over-optimizing and overfitting the generative metric, we pivoted to joint latent models.

### 3. Dual-Network Latent Space Models & Loss Formulations
We designed a dual-network framework consisting of a Convolutional Autoencoder (for feature extraction) and a Fully Connected Network (FCN) classifier operating on the compressed latent space. We compared three loss functions:

#### **Formulation A: Disjoint Reconstruction (66% Accuracy)**
The autoencoder is trained strictly to minimize reconstruction error. The latent representation is then fixed and used to train the FCN classifier. The separation limits the encoder's ability to retain classification-relevant microtonal structures.

#### **Formulation B: Pure Classification Loss (84% Accuracy)**
Both the convolutional encoder and the FCN classifier share the classification loss directly (Cross-Entropy loss), disregarding reconstruction quality. This encourages the model to extract class-discriminative features, improving performance significantly.

#### **Formulation C: Regularized Joint Loss (93.2% Accuracy)**
We formulated a joint regularization loss where the classifier minimizes classification error, while the autoencoder minimizes a weighted sum of reconstruction error and classification cross-entropy:

$$\mathcal{L} = \|x - \hat{x}\|^2 + \lambda \cdot \mathcal{L}_{CE}(y, \hat{y})$$

By using the classification task as a regularizer on the reconstruction loss, the encoder retains structural audio details while emphasizing pitch-critical modal intervals. This approach achieved our best accuracy of 93.2%.

---

## Performance Evaluation

The table below summarizes the comparison across all tested methodology/architecture paradigms:

| Methodology / Architecture | Loss Objective | Accuracy |
| :--- | :--- | :---: |
| **Baseline LSTM** | Pitch-class sequence classification | 60.0% |
| **Conditional Convolutional Diffusion** | Metric optimization on score matching | 56.0% |
| **Autoencoder + FCN (Formulation A)** | Unsupervised reconstruction only | 66.0% |
| **Autoencoder + FCN (Formulation B)** | Direct backpropagation of cross-entropy | 84.0% |
| **Autoencoder + FCN (Formulation C)** | Joint Reconstruction + Classification Regularizer | **93.2%** |

---

## Ongoing Work: Mechanistic Interpretability & Neuron Analysis

Following the success of the regularized autoencoder model (Formulation C), our current work focuses on opening the "black box" of the neural network. We are conducting mechanistic interpretability studies to analyze the specific activation patterns of individual kernels and neurons. By mapping activation profiles back to acoustic frequencies, we aim to understand how the network:
* Identifies microtonal steps (such as quarter-tones or neutral seconds).
* Groups complex melodies into specific Maqam families (e.g., Bayati, Rast, Hijaz).

*Note: Neuron ablation experiments and kernel mapping are currently in progress.*

---

## Citation

If you use this research or codebase in your work, please cite:

```bibtex
@article{elhaddad2026maqam,
  title={Arabic Maqam Identification via Generative Score-Matching and Regularized Latent Representations},
  author={El Haddad, Charbel},
  journal={AUB Research Department Working Paper},
  year={2026}
}
```
