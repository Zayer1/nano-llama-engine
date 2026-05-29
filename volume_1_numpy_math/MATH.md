# The Mathematics of the Transformer

This document details the multivariate calculus and linear algebra derivations implemented in this repository. 

## 1. Cross-Entropy Loss & The Final Softmax
The final layer of the network predicts the next word using a Softmax activation followed by Categorical Cross-Entropy Loss.

**Forward Pass:**
`Prediction (A) = Softmax(Z)`
`Loss (L) = -sum(Target * log(A))`

**Backward Pass:**
The beautiful thing about combining Cross-Entropy Loss with Softmax is that their combined derivative simplifies into a clean subtraction:
`dL/dZ = Prediction - Target`
*(In our code: `dZ2 = A2 - target`)*

## 2. Linear Layers (Feed-Forward)
For any standard linear transformation `Z = X @ W + b`:

**Backward Pass:**
1. **Weights:** We transpose the input to match matrix dimensions.
   `dL/dW = X.T @ dL/dZ`
2. **Bias:** We sum the gradients across the sequence dimension.
   `dL/db = sum(dL/dZ, axis=0)`
3. **Input (to pass backward to earlier layers):** We transpose the weights.
   `dL/dX = dL/dZ @ W.T`

## 3. The Softmax Jacobian (Inside Attention)
Inside the attention mechanism, Softmax converts raw scores into attention weights. Unlike the final output layer, this Softmax is *not* paired with Cross-Entropy Loss, meaning we must manually derive its Jacobian matrix.

`S[i] = exp(z[i]) / sum(exp(z))`

There are two cases for the derivative `dS[i] / dz[j]`:
1. **When i = j:** `dS[i] / dz[i] = S[i] * (1 - S[i])`
2. **When i != j:** `dS[i] / dz[j] = -S[i] * S[j]`

In our code, we vectorize this into a single operation using the incoming upstream gradient `E` (which is `d_attention_weights`):
`d_scores = attention_weights * (E - sum(E * attention_weights, axis=-1))`

## 4. Attention Scores (Matrix Transposition)
The scores are calculated via matrix multiplication of Queries and Keys:
`Scores = Q @ K.T`

Given the upstream gradient `dScores`, we calculate the gradients for `Q` and `K` as follows:
`dQ = dScores @ K`
`dK = dScores.T @ Q`

**Why the transpose?** We must transpose `dScores` when calculating `dK` because `K` itself was transposed in the forward pass (`K.T`). 

## 5. Multi-Head Concatenation
When we split our embedding into multiple heads, we process them in parallel and concatenate them:
`Context = concatenate(Head1, Head2, axis=-1)`

During backpropagation, the gradient `d_context_vector` is simply sliced back into its respective components:
`d_head1 = d_context_vector[:, :head_dim]`
`d_head2 = d_context_vector[:, head_dim:]`

## 6. Causal Masking Gradients
The Causal Mask is an upper-triangular matrix of `-1e9` added to the scores before Softmax. 
Because it is a static constant matrix with **no learnable parameters**, its derivative is exactly **0**. We completely ignore the mask during the backward pass.

## 7. SwiGLU Activation Derivatives
The SwiGLU Feed-Forward network introduces a Swish-activated gate:
`Z_gate = X @ W_gate + b_gate`
`S_gate = Sigmoid(Z_gate)`
`A_gate = Z_gate * S_gate`

And the element-wise multiplication core:
`Merged = A_gate * Z_up`

During backpropagation, we use the product rule to split the `Merged` error into the two pathways:
`d_Z_up = d_Merged * A_gate`
`d_A_gate = d_Merged * Z_up`

## 8. Layer Normalization (Pre-LN)
Layer Normalization forces the variance of features to 1.0 and mean to 0.0 before Attention.
`x_norm = (x - mean) / std`
`out = gamma * x_norm + beta`

Because `x` splits into three paths (`x_norm`, `var`, `mean`), we must derive all three pathways and sum them:
1. **Direct Path (via `x_norm`):** 
   `d_x_norm = d_out * gamma`
2. **Variance Path:** 
   `d_var = sum(d_x_norm * -0.5 * x_norm / std^2)`
3. **Mean Path:** 
   `d_mean = sum(d_x_norm * -1 / std)`

**The Total Derivative for Input X:**
By applying the Chain Rule across all three bridges, we merge the gradients:
`d_x = d_x_norm * (1/std) + d_mean * (1/N) + d_var * (2 * x_norm * std / N)`

## 9. Variance Scaling (Xavier Initialization)
When generating random weight matrices from a standard normal distribution (`variance = 1.0`), matrix multiplication causes the variance to grow linearly by the input dimension `N`:
`Variance(X @ W) = N`

To prevent massive exploding gradients (where `N=4096` would result in NaN), we scale the random weights so the output variance stays strictly at `1.0`. Because variance scales quadratically, we divide the numbers by the square root of `N`:
`W = np.random.randn(...) / np.sqrt(N)`

## 10. Modern LLM Variance (SwiGLU & 0.02 Initialization)

The reason we apply a Gaussian initialization with `mean = 0.0` and `std = 0.02` is crucial for network stability. If we initiate weights as exactly `0`, that is a dead end for the entire neuron. However, if we apply a random number and that number is too high, we instantly expose the network to a potential `NaN` or gradient explosion. Thus, we initialize at `0.0` with a standard deviation of `0.02` to ensure there is safe "wiggle room" initially.

The choice of `0.02` is not random. It is mathematically derived by comparing modern activations against traditional weight normalization methods:

### Traditional Methods (Xavier vs. Kaiming)
Traditionally, there are two primary methods for weight initialization:
1. **Xavier (Glorot):** Designed for `tanh` and `sigmoid`. It automatically scales the standard deviation to `sqrt(2 / (N_in + N_out))`. In a random initiation, the gradients deviate at the rate of `1 / sqrt(d_in)`.
2. **Kaiming (He):** Designed for `ReLU` and `GELU`. Because `ReLU` mercilessly cancels out half the variance (everything below 0), the initialization variance must be doubled to keep the signal from collapsing in deep layers (which would shrink the gradients and impact the neurons). This scales the standard deviation to `1.414 / sqrt(d_in)`.

### Deriving the SwiGLU Variance
SwiGLU operates differently: `SwiGLU(x) = (Swish(xW_gate) * xW_up)W_down`

Let `A = Swish(xW_gate)` and `B = xW_up`. Assuming a variance of `1.0` and a mean of `0.0`:
* `Var(B) = d * W_deviation^2`
* `Var(A) = 0.5 * d * W_deviation^2`

Because SwiGLU multiplies these together element-wise (`A * B`), the variance scales quadratically:
`Var(A * B) = Var(A) * Var(B)`
`Var(A * B) = (0.5 * d * W_deviation^2) * (d * W_deviation^2)`
`Var(SwiGLU) = 0.5 * d^2 * W_deviation^4`

If we want to preserve a clean variance of exactly `1.0` (`Var(SwiGLU) = 1`), we solve for `W_deviation`:
`1 = 0.5 * d^2 * W_deviation^4`
`W_deviation = 1.189 / sqrt(d)`

Comparing the three:
* **Glorot:** `1.000 / sqrt(d)`
* **Kaiming:** `1.414 / sqrt(d)`
* **SwiGLU:** `1.189 / sqrt(d)`

From this, one might conclude that Glorot is the closest and safest traditional method for SwiGLU.

### Why Modern LLMs Hardcode 0.02
Modern Large Language Models bypass all three of these dynamic calculations and instead settle for a hardcoded `0.02`. 

They can do this because of a massive architectural safeguard: **Pre-Layer Normalization (RMSNorm)**. Layer Normalization is placed directly *before* the feed-forward network to forcibly reset the variance of the incoming input back to `1.0`, ensuring there is no compounding deviation occurred in previous layers.

With the incoming variance safely locked at `1.0`, let's plug the hidden dimension (`d`) of modern foundational models (like LLaMA 2 (7B) or LLaMA 3 (8B)) into our derived SwiGLU equation. These models have a hidden dimension of `d = 4096`.

`W_deviation = 1.189 / sqrt(4096)`
`W_deviation = 1.189 / 64`
`W_deviation = 0.01857`

This number is extremely close to `0.02`. Systems Engineers picked **0.02** because it is the cleanest, fastest rounded number that provides perfect variance stabilization for massive-scale architectures.
