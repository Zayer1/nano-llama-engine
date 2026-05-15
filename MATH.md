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

---
*Work in Progress: Layer Normalization and Residual Connections derivations will be added soon.*
