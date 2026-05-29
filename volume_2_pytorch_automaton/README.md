# Volume 2: PyTorch Automaton

This volume translates the mathematical foundations built in Volume 1 into an optimized, GPU-accelerated PyTorch implementation.

## Focus Areas
1. **PyTorch Module Architecture (`nn.Module`):** Transitioning from hardcoded NumPy matrices into a dynamic, object-oriented framework.
2. **Residual Connections:** Adding skip-connections inside the `Block` class (`x = x + self.attn(self.ln_1(x))`) to allow gradient flow through deep architectures.
3. **Batched Processing:** Handling `(Batch_Size, Sequence_Length, Embedding_Dim)` tensors for dynamic multi-sentence training.
4. **The Training Loop:** Upgrading from manual calculus derivation to PyTorch's Autograd engine, utilizing the **Adam Optimizer** (`torch.optim.Adam`).

## Scripts
* `01_nano_gpt.py` - The complete PyTorch implementation of the `NanoGPT` model, including the dynamic `CausalSelfAttention` and `SwiGLU` blocks.
* `02_training_loop.py` - The boilerplate and engine startup script that demonstrates PyTorch's automatic backpropagation.

## State of the Art Architectural Upgrades (The LLaMA Standard)
This PyTorch implementation is not a legacy 2017 Transformer (from "Attention Is All You Need") nor an outdated GPT-1 baseline. It is built strictly using modern 2024 foundational model specifications (such as LLaMA 2 and LLaMA 3). The core architectural upgrades include:

1. **SwiGLU Feed-Forward Networks:** 
Older models utilized standard ReLU or GELU activation functions. We replaced this with Swish-Gated Linear Units (SwiGLU). SwiGLU operates by splitting the representation into two pathways (a gate and an up projection) and multiplying them together element-wise (`Swish(x @ W_gate) * (x @ W_up)`). This is mathematically proven to yield superior reasoning and convergence at scale compared to traditional single-path activations.

2. **Pre-Layer Normalization (RMSNorm-adjacent):** 
The original Transformer applied LayerNorm *after* the residual connections (Post-Norm). When scaled to billions of parameters, this caused catastrophic gradient explosions during training. Modern models apply Layer Normalization *before* the Attention and Feed-Forward blocks (Pre-Norm). This rigidly resets the variance of incoming inputs to `1.0`, ensuring there is no compounding deviation from previous layers.

3. **Bias Removal (`bias=False`):** 
Modern research discovered that maintaining learnable bias vectors in the massive linear projection matrices (Queries, Keys, Values, and the final Language Modeling head) costs a significant memory footprint without providing any measurable increase in intelligence or reasoning capability. We set `bias=False` in these layers to strictly mirror modern scale-optimized paradigms.

4. **Hardcoded 0.02 Variance Initialization:** 
PyTorch defaults to Xavier/Glorot or Kaiming initialization. We explicitly bypassed this, hardcoding a standard deviation of `0.02` for all linear and embedding layers. Initializing weights exactly at `0` creates a dead end for the neuron, but applying too high of a random number instantly risks NaNs or gradient explosions. The value `0.02` is mathematically derived from the SwiGLU variance stabilization equation (`W_deviation = 1.189 / sqrt(d)`). Because modern LLMs (like LLaMA 3 8B) use a hidden dimension of `4096`, this yields a variance of `0.0185`, making `0.02` the cleanest, fastest rounded number for architectural stabilization. (See the rigorous mathematical breakdown in `volume_1_numpy_math/MATH.md`).

5. **Rotary Position Embeddings (RoPE):** 
In Volume 1, we utilized Absolute Positional Embeddings by summing `Token_Embeddings + Position_Embeddings`. In Volume 2, we upgraded to Rotary Positional Embeddings. This removes the absolute position vector entirely. Instead, the relative position of tokens is injected dynamically through mathematical rotation during the `Q @ K.T` Attention calculation. This allows the model to generalize to longer context windows far better than absolute scaling.

6. **AdamW Optimization & Decoupled Weight Decay:** 
Standard Adam incorporates weight decay penalties directly into the gradient. This destroys rarely used parameters (which have small velocity denominators). AdamW decouples weight decay, applying it purely mathematically after the adaptive gradient step has concluded. This is the industry standard for stabilizing massive billion-parameter Transformers.

7. **Validation Holdout (`@torch.no_grad`):** 
To prevent Data Leakage, we wrap our validation metrics in a `@torch.no_grad()` PyTorch decorator. This entirely disconnects the calculus engine, preventing the network from backpropagating or tracking gradients on the test set. We also implemented `model.eval()` to enforce deterministic behavior across stochastic layers.
