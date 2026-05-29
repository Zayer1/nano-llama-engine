# Transformer from Scratch

Welcome to this repository. This project documents the complete evolution of building a Large Language Model architecture, split into two major volumes.

## [Volume 1: NumPy Math](volume_1_numpy_math/README.md)
*The fundamental linear algebra and multivariate calculus.*

In Volume 1, we build the Transformer block (Self-Attention, SwiGLU, Pre-LayerNorm) from absolute scratch using pure NumPy. The primary goal is to manually derive the backpropagation and gradient flow for complex mechanisms like the Softmax Jacobian, variance routing, and Causal Masking, without relying on automated differentiation.

[Go to Volume 1 ->](volume_1_numpy_math/)

## [Volume 2: PyTorch Automaton](volume_2_pytorch_automaton/README.md)
*Scaling the architecture with GPU acceleration.*

In Volume 2, we take the mathematical intuition proven in Volume 1 and translate the exact same architecture into PyTorch's `nn.Module`. Here we introduce production-grade components like the Adam optimizer, batched data processing, Residual Connections, and KV Caching to train a functional "NanoGPT".

[Go to Volume 2 ->](volume_2_pytorch_automaton/)

## [Volume 3: The Inference Engine](volume_3_inference_engine/README.md)
*Giving the Automaton a voice.*

In Volume 3, we write the generation loop. We transition the model from "training mode" into a fully autonomous text generator, handling token decoding, temperature scaling, and context window sliding so the model can talk back.

*(In Progress)*
