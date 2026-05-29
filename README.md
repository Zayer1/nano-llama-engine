# The Nano-Llama Engine
*A 110K Parameter Autoregressive Character-Level Language Model built from absolute scratch.*

![GitHub last commit](https://img.shields.io/github/last-commit/Zayer1/nano-llama-engine)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)

Welcome to the **Nano-Llama Engine**. This repository is not a wrapper around an existing model. It is a complete, deep-dive architectural recreation of a Generative Pre-trained Transformer (GPT), built manually from the ground up to understand the underlying calculus and matrix mathematics of Large Language Models.

## The Architecture
This model implements the core mechanics of modern LLMs (like Llama 3 and GPT-4) at a microscopic scale:
- **Rotary Positional Embeddings (RoPE)**
- **SwiGLU Activation Functions**
- **Multi-Head Causal Self-Attention**
- **RMSNorm (Root Mean Square Normalization)**
- **KV-Caching & Autoregressive Inference**

---

## The 5 Volumes of Progression

This repository is structured educationally into 5 distinct volumes, showing the evolution from raw math to a full web-based Generative AI.

### [Volume 1: NumPy Math](volume_1_numpy_math/)
*The fundamental linear algebra and multivariate calculus.*
We build the Transformer block (Self-Attention, SwiGLU, RMSNorm) using pure NumPy. The primary goal was to manually derive the backpropagation and gradient flow for complex mechanisms without relying on automated differentiation.

### [Volume 2: PyTorch Automaton](volume_2_pytorch_automaton/)
*Scaling the architecture with GPU acceleration.*
We take the mathematical intuition proven in Volume 1 and translate the exact same architecture into PyTorch's `nn.Module`. Here we introduce the Adam optimizer, batched data processing, and GPU tensors.

### [Volume 3: The Inference Engine](volume_3_inference_engine/)
*Giving the Automaton a voice.*
We write the generation loop. We transition the model from "training mode" into a fully autonomous text generator, handling token decoding and context window sliding so the model can generate text autoregressively.

### [Volume 4: The Shakespeare Scale](volume_4_shakespeare_scale/)
*Training the brain.*
We build a dynamic character-level vocabulary and a persistent training loop. The model is trained on a 1MB dataset of William Shakespeare's works, successfully learning to spell English words and understand basic grammar entirely from scratch.

### [Volume 5: The Showcase Interface](volume_5_showcase/)
*Visualizing the neural network.*
A custom Flask API and glassmorphic Web UI. Instead of just printing text to a terminal, this interface dynamically graphs the Softmax probabilities of the neural network's thought process in real-time as it generates text.

---

## How to Run the Showcase (Volume 5)

If you want to see the Neural Network calculate probabilities live in your browser:

1. Clone this repository.
2. Install the requirements:
   ```bash
   pip install torch flask
   ```
3. Run the Flask Server:
   ```bash
   python volume_5_showcase/app.py
   ```
4. Open your web browser and navigate to `http://127.0.0.1:5000`.

*(Note: The trained `shakespeare_gpt.pth` weights are not included in this repository due to GitHub file size limits. To use the UI, you must first run `python volume_4_shakespeare_scale/training_loop.py` to train your own local weights!)*
