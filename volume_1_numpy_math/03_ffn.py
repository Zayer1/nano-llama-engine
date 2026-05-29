import numpy as np

np.random.seed(42)
lr = 0.1

# ==========================================
# 1. THE DATA PIPELINE (Your Code)
# ==========================================
vocab = {"the": 0, "cat": 1, "sat": 2, "on": 3, "mat": 4}
sentence = ["the", "cat", "sat"]



# Tokenize
tokens = [vocab[word] for word in sentence]


# Embedding Matrix
vocab_size = len(vocab)
embedding_dim = 6 #we update this to 6

# ==========================================
# POSITIONAL EMBEDDINGS (The Time Stamps)
# ==========================================
# The Problem: Matrix multiplication doesn't care about order. 
# If 'cat' = [1, 2] and 'dog' = [3, 4], their dot product is always 11. 
# Whether the sentence is "cat chases dog" or "dog chases cat", the math is 100% identical.
# The network is completely blind to time.
# 
# The Solution: We create a dedicated "Position Vector" for each slot in the sentence.
# Position 1 gets its own vector, Position 2 gets its own vector, etc.
# Instead of hard-coding these vectors using complex math (like the original paper), 
# we start them as random numbers and let the calculus figure out the optimal values!
MAX_CONTEXT_WINDOW = 10
P = np.random.randn(MAX_CONTEXT_WINDOW, embedding_dim)

# ==========================================
# THE CAUSAL MASK (The "Blindfold")
# ==========================================
# Problem: When predicting a word, the AI shouldn't be allowed to look at future words.
# Solution: In our scores matrix (Q @ K.T), Row 'i' is the Query word, and Col 'j' is the Key word.
# If j > i, it means the Key word comes AFTER the Query word in the sentence (The Future).
# 
# 1. np.ones: Creates a grid of 1s.
# 2. np.triu(..., k=1): Targets the upper triangle. It keeps 1s exactly where j > i, and zeroes out the rest.
# 3. * -1e9: Turns all those "future" 1s into Negative Infinity.
# 
# Later, when Softmax processes this mask, the math of e^(-infinity) will crush all future 
# probabilities down to exactly 0.000%, physically making it impossible for the AI to "see" ahead.
mask = np.triu(np.ones((len(sentence), len(sentence))), k = 1) * -1e9
#We can use np.inf, mathematically that'd be the same, however when we deal with computer
#calculation, that infinity might cause NaN when applied.

num_heads = 2 #we're building 2 parallel heads
head_dim = embedding_dim // num_heads #Each head gets 3 dimensions
E = np.random.randn(vocab_size, embedding_dim)


# ==========================================
# 2. THE NEURAL NETWORK 
# ==========================================

def sigmoid(z):
    """
    Applies the Sigmoid activation function.
    Math: f(z) = 1 / (1 + e^(-z))
    
    Parameters:
    z : np.ndarray
        The input tensor (e.g., Z_gate).
        
    Returns:
    np.ndarray
        The activated output, with values bound between 0 and 1. Same shape as z.
    """
    # Cap the values between -500 and 500 to prevent exp() overflows
    z_safe = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z_safe))


def softmax(z):
    """
    Applies the Softmax function to convert raw scores into a probability distribution.
    Math: f(z_i) = e^(z_i) / Σ(e^(z_j))
    
    Parameters:
    z : np.ndarray
        The input raw scores (e.g., attention scores or logits). Shape: (..., N)
        
    Returns:
    np.ndarray
        The probability distribution where the sum across the last axis is 1.0. Same shape as z.
    """
    # Subtract the maximum value from every row to prevent np.exp() from blowing up to infinity
    z_shifted = z - np.max(z, axis = -1, keepdims = True)
    exp_z = np.exp(z_shifted)
    return exp_z/np.sum(exp_z, axis = -1, keepdims = True)

#Here we start out process to apply layer normalization
#As we go about the training process of a transformer, eventually we will encounter
#initially random inputs that are too large, which might put our calculating process in jeorpardy.
#Thus, we apply Layer Normalization to ensure that the inputs won't be too big, nor too small that
#the model cannot learn
#Let's start off by declaring gamma and beta
#In essence, gamma amplifies an input just in case it is important by multiplying the value
#Whereas beta shifts the entire vector left or right, so the activation layer can finish it off
gamma = np.ones((1, embedding_dim))
beta = np.zeros((1,embedding_dim)) #the reason the dimensions are all 1,embedding_dim
#here is so that the layer norm won't calculate the info inside in a dot scale
#It's simply here to mediatate the informations by scaling, not calculating
def layernorm_forward(x):
    """
    Applies Pre-Layer Normalization to stabilize deep network activations.
    Math: out = γ * ((x - μ) / √(σ² + ε)) + β
    
    Parameters:
    x : np.ndarray
        The input tensor to be normalized. Shape: (Sequence_Length, Embedding_Dim)
        
    Returns:
    out : np.ndarray
        The scaled and shifted normalized output. Shape: (Sequence_Length, Embedding_Dim)
    x_norm : np.ndarray
        Cached normalized input for backpropagation. Shape: (Sequence_Length, Embedding_Dim)
    var : np.ndarray
        Cached variance (σ²) for backpropagation. Shape: (Sequence_Length, 1)
    mean : np.ndarray
        Cached mean (μ) for backpropagation. Shape: (Sequence_Length, 1)
    """
    # 1. Calculate Mean (μ): μ = 1/H * Σ x_i
    # Shape transition: (Seq_Len, Embed_Dim) -> (Seq_Len, 1)
    mean = np.mean(x, axis=-1, keepdims=True)
    # 2. Calculate Variance (σ²): σ² = 1/H * Σ (x_i - μ)²
    # Shape transition: (Seq_Len, Embed_Dim) -> (Seq_Len, 1)
    var = np.var(x, axis = -1, keepdims=True)
    # 3. Normalize (adding 1e-5 to prevent dividing by zero)
    std = np.sqrt(var + 1e-5)
    x_norm = (x - mean) / std
    # 4. Scale and Shift
    out = gamma * x_norm + beta
    return out, x_norm, var, mean

def layernorm_backprop(d_out, x_norm, var):
    """
    Computes the gradients for Layer Normalization during backpropagation.
    Calculates the exact total derivative with respect to x, γ, and β.
    
    Parameters:
    d_out : np.ndarray
        The upstream gradient flowing into the LayerNorm output. Shape: (Seq_Len, Embed_Dim)
    x_norm : np.ndarray
        The cached normalized input from the forward pass. Shape: (Seq_Len, Embed_Dim)
    var : np.ndarray
        The cached variance from the forward pass. Shape: (Seq_Len, 1)
        
    Returns:
    d_x : np.ndarray
        The gradient flowing back into the input tensor. Shape: (Seq_Len, Embed_Dim)
    d_gamma : np.ndarray
        The gradient for the scale parameter. Shape: (1, Embed_Dim)
    d_beta : np.ndarray
        The gradient for the shift parameter. Shape: (1, Embed_Dim)
    """
    #Read all the way down to see why I already have d_out, it is actually the new
    #sum of d_sentence_embedding
    N = x_norm.shape[-1]
    std = np.sqrt(var + 1e-5)
    
    # 1. The Learnable Weights (Summed across the sequence: axis=0)
    d_gamma = np.sum(d_out * x_norm, axis=0, keepdims=True)
    d_beta = np.sum(d_out, axis=0, keepdims=True)
    
    # 2. The 3 Bridges
    # Derivation for x_norm: d_x_norm = d_out * d_out/d_x_norm = d_out * gamma
    d_x_norm = d_out * gamma
    # Derivation for variance: d_var = d_x_norm * d_x_norm/d_var = d_x_norm * -0.5 * (x_norm * std) / std^3
    d_var = np.sum(d_x_norm * -0.5 * (x_norm * std) / (std**3), axis=-1, keepdims=True)
    # Derivation for mean: d_mean = d_x_norm * d_x_norm/d_mean = d_x_norm * (-1 / std)
    d_mean = np.sum(d_x_norm * (-1.0 / std), axis=-1, keepdims=True)
    
    # 3. The Grand Finale!
    # Derivation for the total error on x: d_x = d_x_norm*(1/std) + d_mean*(1/N) + d_var*(2*(x-mean)/N)
    d_x = d_x_norm * (1.0 / std) + d_mean * (1.0 / N) + d_var * ((2.0 * x_norm * std) / N)
    
    return d_x, d_gamma, d_beta


#1. SwiGLU feed-forward network
ffn_hidden_states = embedding_dim * 2 #Expand to 12 dims, this is so that each head gets more dims
#to calculate without getting bottle neck, since as we can see earlier, there are 5 words yet only 6 dimensions

#Gate matrix, this determines what information can pass through
W_gate = np.random.randn(embedding_dim, ffn_hidden_states) / np.sqrt(embedding_dim)
b_gate = np.zeros((1, ffn_hidden_states))

#Up matrix, this carries actual information
W_up = np.random.randn(embedding_dim, ffn_hidden_states) / np.sqrt(embedding_dim)
b_up = np.zeros((1, ffn_hidden_states))

#Down matrix, this compress the information back to the original size of when it got in
#This is because the other layers only accept up to 6 dims, so we must compress it to calculate further
W_down = np.random.randn(ffn_hidden_states, embedding_dim) / np.sqrt(ffn_hidden_states)
b_down = np.zeros((1, embedding_dim))

#2. Language modeling head
W_lm = np.random.randn(embedding_dim, vocab_size) / np.sqrt(embedding_dim) 
#this so that the model can understand from the actual
#words of a human written language, also this can server for future output modeling
b_lm = np.zeros((1, vocab_size))

#Layer 1: Self-attention weights
W_q = np.random.randn(embedding_dim, embedding_dim) / np.sqrt(embedding_dim)
W_k = np.random.randn(embedding_dim, embedding_dim) / np.sqrt(embedding_dim)
W_v = np.random.randn(embedding_dim, embedding_dim) / np.sqrt(embedding_dim)

for epoch in range(1000):
    # -------------------------------------------------------------------------
    # THE ADDITION: Word Meaning + Time Stamp
    # -------------------------------------------------------------------------
    # Why do we ADD (E + P) instead of gluing them together (concatenating)?
    # 1. Math Cost: Gluing a 6-number word and a 6-number position makes a 12-number array. 
    #    This would double the size of all our matrices (W_q, W_k, W_v) and make training incredibly slow.
    # 
    # 2. How Addition works here: Imagine a 2D graph. The word "Cat" is at coordinates (x=100, y=100).
    #    If we add the Position 1 vector (x=1, y=0), Cat becomes (101, 100).
    #    If we add the Position 2 vector (x=0, y=1), Cat becomes (100, 101).
    #    When this new mixed coordinate hits our weight matrix (W_q), the matrix multiplication 
    #    is mathematically capable of separating the massive base number (100) from the tiny 
    #    shift (+1). The network decodes both "What is this word?" and "Where is it?" at the exact same time!
    # Note: We slice P[:len(tokens)] so it matches the current sentence length!
    sentence_embedding = E[tokens] + P[:len(tokens)]
    sentence_embedding_norm, x_norm_ln, var_ln, mean_ln = layernorm_forward(sentence_embedding)

    #From that we generate the values for Q, K, V. These are deterministic projections, the weights (W_q, W_k, W_v) are what we randomized
    Q = sentence_embedding_norm @ W_q #Shape (3,6)
    K = sentence_embedding_norm @ W_k #Shape (3,6)
    V = sentence_embedding_norm @ W_v #Shape (3,6)

    #Split Q,K and V into multiple heads
    #Head 1 gets the first 3 columns
    Q1 = Q[:, :head_dim]
    K1 = K[:, :head_dim]
    V1 = V[:, :head_dim]
    #Head 2 gets the second 3 columns
    Q2 = Q[:, head_dim:]
    K2 = K[:, head_dim:]
    V2 = V[:, head_dim:]

    #Head 1 processing
    scores1 = Q1 @ K1.T + mask # The Bouncer slaps -1e9 on future coordinates
    weights1 = softmax(scores1) # Softmax converts -1e9 to exactly 0.0% attention
    context1 = weights1 @ V1 #(3,3)

    #Head 2 processing
    scores2 = Q2 @ K2.T + mask # The Bouncer slaps -1e9 on future coordinates
    weights2 = softmax(scores2) # Softmax converts -1e9 to exactly 0.0% attention
    context2 = weights2 @ V2 #(3,3)

    #Now we glue those 2 heads together
    context_vector = np.concatenate((context1, context2), axis = -1)
    
    #swiGLU foward-pass, this applies to all tokens: shape (3,6)
    Z_gate = context_vector @ W_gate + b_gate #(3,12)
    Z_up = context_vector @ W_up + b_up #(3, 12)

    #The swish activation: Z * Sigmoid(Z)
    #Let's explain a bit about the Swish activation
    #In standard ReLU, when you apply activation function, it is mercilessly accurate
    #During the random generation of value, there can be negatives, and after the calculation, the negative values may yet stay
    #Naturally, we would want to omit the negative values out, thus ReLU would just cut them off clean, even if they're only slightly negative
    #If the output of a neuron is 0 too often, the backpropagation of said neuron will keep resulting in 0
    #Thus rendering the AI completely neutralized
    #This is why we add in the swish function, mathematically, it is essentially:
    #f(x) = x * (1/(1+np.exp(-x)))
    #Like ReLU, if the number is positive, it'll pass through clean
    #If the number is slightly negative, swish doesn't crush it to 0, instead it makes a tiny negative curve
    #The backpropagation, which we will manually derive later, will be Error * (a tiny fraction)
    S_gate = sigmoid(Z_gate)
    A_gate = Z_gate * S_gate

    #Element-wise multiplication (the core of SwiGLU)
    #First off, the reason we use normal multiplication here is that we're calculating the coherence, or rather
    #How different these elements impact each other
    #In this exact instance, the reason they're multiplied is so that the elements that pass through
    #the gates will only affects the dictionary's definition of itself
    Merged = A_gate * Z_up #(3,12)

    #Compress back down
    Z_down = Merged @ W_down + b_down #(3,6)

    #Language modeling head
    #We grab the token's representation to predict the next word
    sentence_vector = Z_down[-1:] #(1,6)

    # -------------------------------------------------------------------------
    # DEBUG: TENSOR SHAPE TRACKER
    # -------------------------------------------------------------------------
    DEBUG_SHAPES = True # Set to True to watch the dimensions morph!
    if epoch == 0 and DEBUG_SHAPES:
        print("\n--- FORWARD PASS SHAPES ---")
        print(f"sentence_embedding: {sentence_embedding.shape} | Expected: ({len(sentence)}, {embedding_dim})")
        print(f"Q, K, V matrices:   {Q.shape} | Expected: ({len(sentence)}, {embedding_dim})")
        print(f"Split Heads (Q1):   {Q1.shape} | Expected: ({len(sentence)}, {head_dim})")
        print(f"Attention Scores:   {scores1.shape} | Expected: ({len(sentence)}, {len(sentence)})")
        print(f"Context Vector:     {context_vector.shape} | Expected: ({len(sentence)}, {embedding_dim})")
        print(f"SwiGLU Z_gate/up:   {Z_gate.shape} | Expected: ({len(sentence)}, {ffn_hidden_states})  <-- Expansion!")
        print(f"SwiGLU Z_down:      {Z_down.shape} | Expected: ({len(sentence)}, {embedding_dim})   <-- Compression!")
        print(f"Sentence Vector:    {sentence_vector.shape} | Expected: (1, {embedding_dim})")
        print("---------------------------\n")

    Z_lm = sentence_vector @ W_lm + b_lm
    A_lm = softmax(Z_lm)

    # The Corrected Loss Calculation
    target = np.array([[0.0,0.0,0.0,0.0,1.0]]) #Shape: (1,5)

    loss = -np.sum(target *np.log(A_lm +1e-9))

    # -------------------------------------------------------------------------
    # BACKPROPAGATION
    # -------------------------------------------------------------------------
    
    # 1. Backprop through Language Modeling Head
    # User's Math: dL/dZ_lm = dL/dA_lm * dA_lm/dZ_lm = A_lm - target
    d_Z_lm = A_lm - target # Shape: (1, 5)
    
    # User's Math: dW_lm = dZ_lm * sentence_vector (Matrix form: sentence_vector.T @ d_Z_lm)
    d_W_lm = sentence_vector.T @ d_Z_lm # (6, 1) @ (1, 5) -> (6, 5)
    # User's Math: db_lm = dZ_lm (Summed across batch)
    d_b_lm = np.sum(d_Z_lm, axis=0, keepdims=True) # (1, 5)
    
    # User's Math: d_sentence_vector = dL/dZ_lm * dZ_lm/d_sentence_vector (where dZ_lm/d_sentence_vector = W_lm)
    d_sentence_vector = d_Z_lm @ W_lm.T # (1, 5) @ (5, 6) -> (1, 6)
    
    # 2. Backprop through Z_down
    # User's Math: dZ_down = d_sentence_vector * d_sentence_vector/dZ_down (where d_sentence_vector/dZ_down = 1 for the last token)
    d_Z_down = np.zeros_like(Z_down) # Shape: (3, 6)
    d_Z_down[-1:] = d_sentence_vector
    
    # 3. Backprop through SwiGLU FFN (Down Projection)
    # User's Math: dW_down = Merged * dZ_down (Matrix form: Merged.T @ d_Z_down)
    d_W_down = Merged.T @ d_Z_down # (12, 3) @ (3, 6) -> (12, 6)
    # User's Math: db_down = dZ_down (Summed across batch)
    d_b_down = np.sum(d_Z_down, axis=0, keepdims=True) # (1, 6)
    # User's Math: d_Merged = dZ_down * dZ_down/d_Merged (where dZ_down/d_Merged = W_down)
    d_Merged = d_Z_down @ W_down.T # (3, 6) @ (6, 12) -> (3, 12)
    
    # 4. Element-wise Multiplication (Core of SwiGLU)
    # User's Math: d_Merged/dA_gate = Z_up -> dL/dA_gate = dL/d_Merged * d_Merged/dA_gate
    d_A_gate = d_Merged * Z_up # Element-wise: (3, 12)
    # User's Math: Z_up (intended dZ_up) = d_Merged * A_gate (Derived using product rule)
    d_Z_up = d_Merged * A_gate # Element-wise: (3, 12)
    
    # 5. Up Projection
    # User's Math: dW_up = d_Merged * A_gate * context_vector (Matrix form: context_vector.T @ d_Z_up)
    d_W_up = context_vector.T @ d_Z_up # (6, 3) @ (3, 12) -> (6, 12)
    # User's Math: db_up = dZ_up (Summed across batch)
    d_b_up = np.sum(d_Z_up, axis=0, keepdims=True) # (1, 12)
    
    # 6. Gate Projection and Swish Activation
    # User's Math: dA_gate/dZ_gate = S_gate + Z_gate*S_gate*(1-S_gate)
    d_Z_gate = d_A_gate * (S_gate + Z_gate * S_gate * (1 - S_gate)) # (3, 12)
    
    # User's Math: dW_gate = dZ_gate * context_vector (Matrix form: context_vector.T @ d_Z_gate)
    d_W_gate = context_vector.T @ d_Z_gate # (6, 3) @ (3, 12) -> (6, 12)
    # User's Math: db_gate = dZ_gate (Summed across batch)
    d_b_gate = np.sum(d_Z_gate, axis=0, keepdims=True) # (1, 12)
    
    # 7. Error reaching the context vector (Recombination)
    # User's Math: d_context_vector = dZ_gate + dZ_up (Passing backwards through W_gate and W_up first)
    d_context_vector_gate = d_Z_gate @ W_gate.T # (3, 12) @ (12, 6) -> (3, 6)
    d_context_vector_up = d_Z_up @ W_up.T # (3, 12) @ (12, 6) -> (3, 6)
    d_context_vector = d_context_vector_gate + d_context_vector_up # (3, 6)

    #Now we split the blame
    #First 3 columns of error to head 1
    d_context1 = d_context_vector[:, :head_dim]
    #Second 3 columns of error to head 2
    d_context2 = d_context_vector[:, head_dim:]

    #Now we calculate the blame of head 1 and 2
    d_weights1 = d_context1 @ V1.T
    dV1 = weights1.T @ d_context1
    d_weights2 = d_context2 @ V2.T
    dV2 = weights2.T @ d_context2
    # Note on Causal Mask: Because the mask is a static constant (no weights), 
    # its derivative is 0. We literally throw it away, so the backward pass requires ZERO changes!
    #d_scores =  Ei*Si*(1-Si) - sum(Ej*Si*Sj)
    #S is attention weights, E is d_attention_weights
    d_scores1 = weights1 * (d_weights1 - np.sum(d_weights1 * weights1, axis = -1, keepdims = True))
    d_scores2 = weights2 * (d_weights2 - np.sum(d_weights2 * weights2, axis = -1, keepdims = True))

    #Next up, we find the blame for Q1 and K1
    #dQ = dL/dQ = dL/d_scores * d_scores/dQ
    dQ1 = d_scores1 @ K1 #(3, 3) @ (3, 3) -> Shape: (3, 3)
    #dK = dL/dK = dL/d_scores * d_scores/dK
    dK1 = d_scores1.T @ Q1 #(3, 3) @ (3, 3) -> Shape: (3, 3)

    #And now, we find the blame for Q2 and K2
    dQ2 = d_scores2 @ K2 # (3, 3) @ (3, 3) -> Shape: (3, 3)
    dK2 = d_scores2.T @ Q2 # (3, 3) @ (3, 3) -> Shape: (3, 3)
    
    # --- RECOMBINE THE PARALLEL BRAINS ---
    # Glue the Head 1 and Head 2 errors back together side-by-side
    dQ = np.concatenate((dQ1, dQ2), axis = -1) # (3,3) + (3,3) -> Shape: (3, 6)
    dK = np.concatenate((dK1, dK2), axis = -1) # (3,3) + (3,3) -> Shape: (3, 6)
    dV = np.concatenate((dV1, dV2), axis = -1) # (3,3) + (3,3) -> Shape: (3, 6)

    #After that we do dW_q, dW_k and dW_v
    # sentence_embedding is shape (3, 6), so its transpose is (6, 3)
    dW_q = sentence_embedding.T @ dQ # (6, 3) @ (3, 6) -> Shape: (6, 6)
    #And the same goes for the other 2
    dW_k = sentence_embedding.T @ dK # (6, 3) @ (3, 6) -> Shape: (6, 6)
    dW_v = sentence_embedding.T @ dV # (6, 3) @ (3, 6) -> Shape: (6, 6)
    #Note: the reason we apply transpose here is because of the content within the matrices themselves
    #sentence_embedding is a matrix with it's row 1 as features for the word "the"
    #dQ's first colum is error for word 1, 2 and 3
    #So if we forgot the transposition, word 1's feature would be multiplied with word 1's, 2's and 3's errors, which would make 0 sense
    #Same case for dK and dV

    #Now we calculate the blame of the sentence embedding
    # dQ is shape (3, 6), W_q is shape (6, 6), so W_q.T is (6, 6)
    d_sentence_embedding_q = dQ @ W_q.T # (3, 6) @ (6, 6) -> Shape: (3, 6)
    #Calculate similarly for the others
    d_sentence_embedding_k = dK @ W_k.T # (3, 6) @ (6, 6) -> Shape: (3, 6)
    d_sentence_embedding_v = dV @ W_v.T # (3, 6) @ (6, 6) -> Shape: (3, 6)

    #Now we wrap it all up with a simple addition. The reason it's not seperate and instead addition here is
    #because all those 3 are connected into the full on Q, K and V calculation. However deriving them
    #all at once is mathematically impossible with 3 different variables, and we need a total derivative here
    # Adding three (3, 6) matrices together results in a single (3, 6) matrix.
    d_sentence_embedding_norm = d_sentence_embedding_q + d_sentence_embedding_k + d_sentence_embedding_v
    #This is essentially our d_out for the layer normalization, basically, the
    #backprop of layer norm is the derivation of it's output, and since we applied
    #the norm into the sentence_embedding, the derivative here essentially becomes
    #the derivative of the layer's output
    d_sentence_embedding, d_gamma, d_beta = layernorm_backprop(d_sentence_embedding_norm, x_norm_ln, var_ln)

    #Update rule
    gamma = gamma - lr * d_gamma
    beta = beta - lr * d_beta
    
    W_gate = W_gate - lr * d_W_gate
    b_gate = b_gate - lr * d_b_gate
    W_up = W_up - lr * d_W_up
    b_up = b_up - lr * d_b_up
    W_down = W_down - lr * d_W_down
    b_down = b_down - lr * d_b_down
    
    W_lm = W_lm - lr * d_W_lm
    b_lm = b_lm - lr * d_b_lm
    
    W_q = W_q - lr * dW_q
    W_k = W_k - lr * dW_k
    W_v = W_v - lr * dW_v
    
    for i, token in enumerate(tokens):
        E[token] = E[token] - lr * d_sentence_embedding[i]
        #We also update the positional index too
        P[i] = P[i] - lr * d_sentence_embedding[i]

    #Every 10 steps printing out progess
    if epoch % 10 == 0:
        print(f"Epoch {epoch} , Loss {loss:.6f} , Prediction: {A_lm[0][4]:.4f}")

print("\nFinal Attention Weights for 'sat':")

print(weights1[-1])

print("\nFinal Attention Weights for 'sat' (Head 2):")
print(weights2[-1])

print("\nFinal Attention Weights (Head 1):")
print(weights1)

print("\nFinal Attention Weights (Head 2):")
print(weights2)

# ==========================================
# 3. EXPORT WEIGHTS FOR INFERENCE
# ==========================================
print("\nSaving trained weights to 'weights.npz'...")
np.savez("weights.npz", 
         E=E, P=P, gamma=gamma, beta=beta, 
         W_gate=W_gate, b_gate=b_gate, 
         W_up=W_up, b_up=b_up, 
         W_down=W_down, b_down=b_down, 
         W_lm=W_lm, b_lm=b_lm, 
         W_q=W_q, W_k=W_k, W_v=W_v)
print("Done!")
