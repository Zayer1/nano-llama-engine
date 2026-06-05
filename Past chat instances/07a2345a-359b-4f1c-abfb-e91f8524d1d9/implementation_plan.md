# RoPE (Rotary Position Embeddings) Architectural Upgrade

The web version is 100% correct. I apologize—I missed the final, most complex boss of modern architectures. Absolute positional embeddings (`x = tok_emb + pos_emb`) are a relic of the 2017 era. Modern models like LLaMA 2/3 utilize **Rotary Position Embeddings (RoPE)**.

To make this a true State of the Art PyTorch engine, we must rip out the absolute embeddings and mathematically rotate the Queries and Keys inside the Attention Block.

## Proposed Changes

### 1. `01_nano_gpt.py`

#### [MODIFY] `01_nano_gpt.py`
* **Delete Absolute Embeddings:** Inside the `NanoGPT` class, we will delete `self.positional_embedding = nn.Embedding(...)`. We will also delete the `pos_emb` variable and the `x = tok_emb + pos_emb` addition in the `forward()` function.
* **Inject RoPE Function:** We will create a pedagogical, real-number PyTorch function `apply_rotary_emb(q, k)`. Instead of using PyTorch's native C++ complex numbers (which are difficult to read), we will explicitly code the mathematical rotation: creating the `theta` frequencies, repeating them, shifting the `Q` and `K` vectors (`[-x2, x1]`), and applying the `cos()` and `sin()` waves.
* **Update Attention Block:** Inside `CausalSelfAttention.forward()`, right after we split `Q` and `K` into multiple heads, we will pass them through `apply_rotary_emb(q, k)` before they undergo the dot-product multiplication.

### 2. Documentation Updates

#### [MODIFY] `volume_1_numpy_math/MATH.md`
* We will append **Section 11: Rotary Position Embeddings (RoPE)**. 
* This section will contain the explicit mathematical breakdown of how RoPE applies a 2D rotation matrix to pairs of features, and how the dot product of two rotated vectors mathematically derives their relative distance ($m - n$).

#### [MODIFY] `volume_2_pytorch_automaton/README.md`
* We will update the "State of the Art Architectural Upgrades" to include RoPE as the 5th and final modern upgrade, proudly cementing the architecture as a true 2024 LLaMA implementation.

## User Review Required
> [!IMPORTANT]
> The RoPE math is extremely complex. It involves shifting tensor dimensions and applying sine/cosine frequencies to pairs of numbers. Are you comfortable taking on this final mathematical challenge? If so, approve this plan and I will begin the architectural refactoring.
