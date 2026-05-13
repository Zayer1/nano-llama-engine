# Self-Attention from Scratch (Pure NumPy)

A pedagogical, from-scratch implementation of a Transformer Self-Attention block and its full backpropagation calculus in pure Python/NumPy. 

## Overview
Modern deep learning frameworks like PyTorch and TensorFlow abstract away the complex calculus and gradient flow of neural networks. While `import torch.nn as nn` is standard for production, relying exclusively on auto-grad engines can leave engineers with a shallow understanding of the underlying architecture.

The goal of this "toy model" is to prove fundamental comprehension of the linear algebra and multivariable calculus that powers modern Large Language Models (LLMs). This repository contains a fully functional forward and backward pass of a Single-Head Self-Attention mechanism, written entirely from scratch without any ML libraries.

## Key Mathematical Implementations

This implementation explicitly handles the mathematical routing of gradients that are normally hidden behind framework abstractions:

1. **The Softmax Jacobian:** Deriving the gradient of the Softmax function is notoriously complex because it forms a Jacobian matrix (since every output depends on every input). This code manually reduces the piecewise derivative (where i=j and i is different from j) into a fully vectorized NumPy operation: 
   dZ_i = S_i (E_i - sum_j E_j S_j)
2. **Matrix Transposition for Gradient Flow:** Explicitly demonstrating why transposed matrices are required during the backward pass (e.g., `dW_q = sentence_embedding.T @ dQ`). Transposing the input matrix maps the blame from individual errors directly back to the original input features that caused them, summing the total blame across the sequence length.
3. **The Total Derivative Rule:** Tracing the upstream gradients from the Queries, Keys, and Values back into a single unified gradient for the original `sentence_embedding`, demonstrating the multivariable chain rule in action.
4. **Context vs. Sequence:** Utilizing a Causal Language Modeling approach by isolating the context vector of the final token (`context_vector[-1:]`) to predict the next sequential word.

## Architecture
The neural network consists of:
* A trainable Vocabulary Embedding matrix (`E`)
* A Self-Attention Layer (`W_q`, `W_k`, `W_v`)
* A Feed-Forward Hidden Layer (`W1`, `b1`)
* A Softmax Output Layer (`W2`, `b2`) trained via Categorical Cross-Entropy.

## Usage
The entire architecture is contained within a single, highly commented file to maximize readability. 

To run the training loop:
```bash
python Attention.py
