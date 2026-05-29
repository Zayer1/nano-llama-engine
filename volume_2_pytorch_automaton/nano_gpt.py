# ==========================================
# VOLUME 2: THE PYTORCH AUTOMATON
# ==========================================
import torch
import torch.nn as nn
from torch.nn import modules 
from torch.nn import functional as F

vocab = {"the": 0, "cat": 1, "sat": 2, "on": 3, "mat": 4, "<END>": 5}
vocab_size = len(vocab)
embedding_dim = 6
num_heads = 3
head_dim = embedding_dim // num_heads
max_context_window = 10

class SwiGLU(nn.Module):
    def __init__(self, embedding_dim, ffn_hidden_states):
        super().__init__()
        
        # -------------------------------------------------------------
        # VOLUME 1 MAPPING: W_gate, b_gate & W_up, b_up (See 03_ffn.py Lines 193-198)
        # -------------------------------------------------------------
        # Rather than managing two separate matrices, we can instantiate two Linear layers.
        # W_gate determines what information passes; W_up carries the actual representation.
        self.w_gate = nn.Linear(embedding_dim, ffn_hidden_states)
        self.w_up = nn.Linear(embedding_dim, ffn_hidden_states)
        
        # -------------------------------------------------------------
        # VOLUME 1 MAPPING: W_down, b_down (See 03_ffn.py Lines 202-203)
        # -------------------------------------------------------------
        # Compresses the expanded hidden states back to the original embedding dimension.
        self.w_down = nn.Linear(ffn_hidden_states, embedding_dim)

    def forward(self, x):
        # -------------------------------------------------------------
        # VOLUME 1 MAPPING: Merged = A_gate * Z_up (See 03_ffn.py Line 286)
        # -------------------------------------------------------------
        # F.silu() is the PyTorch equivalent of the Swish activation function.
        # It handles the element-wise multiplication natively and optimally.
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        merged = gate * up
        
        # -------------------------------------------------------------
        # VOLUME 1 MAPPING: Z_down = Merged @ W_down + b_down (See 03_ffn.py Line 289)
        # -------------------------------------------------------------
        return self.w_down(merged)

# =============================================================
# DOCUMENTARY DEEP DIVE: PYTORCH MAGIC METHODS & ENGINE ROUTING
# =============================================================
# How does Python know to trigger our code? It intercepts syntax.
# 
# 1. THE CREATION TRIGGER: `model = NanoGPT()`
#    When Python sees parentheses `()` attached to a Class name, it intercepts them.
#    It automatically triggers the `__init__` (dunder init) method.
#    Calling `super().__init__()` is what invites the parent class (nn.Module) into our 
#    empty object to set up PyTorch's hidden plumbing (like the `self._parameters` dictionary).
# 
# 2. THE EXECUTION TRIGGER: `logits = model(idx)`
#    When Python sees parentheses `()` attached to an Object, it intercepts them.
#    It automatically triggers a hidden function called `__call__`. 
#    PyTorch's `nn.Module` has a massive `__call__` function built-in. When triggered, 
#    it does background math and then secretly routes the data into our `forward()` function!
# =============================================================

# =============================================================
# DOCUMENTARY DEEP DIVE: ROTARY POSITION EMBEDDING (RoPE)
# =============================================================
# WHAT HAPPENS WITHOUT IT:
# By default, a Transformer is completely blind to word order. It processes all words in 
# parallel. Without positional data, "The cat sat" and "Sat cat the" look mathematically 
# identical to the Attention mechanism. 
#
# ABSOLUTE VS RELATIVE (THE EXTRAPOLATION PROBLEM):
# In Volume 1, we used "Absolute" embeddings (fixed slots). If a model learns "Dog" is 
# in Slot 1 and "Man" is in Slot 4, and later sees them in Slot 7 and 10, it panics. It 
# has to relearn that relationship from scratch.
#
# THE ROPE SOLUTION (SPINNING THE DIAL):
# RoPE solves this by treating the vectors like hands on a clock. Instead of adding a vector, 
# it ROTATES the word's embedding vector. Because Attention only cares about the ANGLE 
# BETWEEN two hands (the relative distance), the model instantly understands they are 
# exactly 3 words apart, whether they are at slots 1 & 4, or slots 1001 & 1004! This 
# allows the model to scale to documents infinitely longer than it was trained on.
#
# WHY WE PUT THIS FUNCTION HERE:
# The rotation speeds (frequencies) are pure, hardcoded math. They are not learned weights. 
# We build this function standalone so we can compute the frequencies exactly once and cache 
# them, rather than wasting GPU cycles recalculating constants on every forward pass.
# =============================================================
def precompute_theta_pos_frequencies(head_dim, seq_len):
    # RoPE rotates values on a 2D graph (a complex plane). To have an X and Y coordinate, 
    # we must group our embedding dimensions into pairs (Dim 0 & 1, Dim 2 & 3, etc.).
    # By stepping by 2 (0, 2, 4...), we generate exactly one rotation speed for each PAIR.
    theta_numerator = torch.arange(0, head_dim, 2).float()
    
    # =============================================================
    # DOCUMENTARY DEEP DIVE: FAST CLOCKS VS SLOW CLOCKS & MAGIC NUMBERS
    # =============================================================
    # Why this specific formula (10000^-2i/d)? It creates a spectrum of spinning speeds.
    # - The first dimensions (i=0) spin very FAST. They measure microscopic local context 
    #   (like adjectives next to nouns). If used for long distances, they "wrap around" 
    #   the circle and lose track (aliasing).
    # - The deep dimensions spin incredibly SLOWLY. They measure macroscopic long-range 
    #   context (like the theme of page 1 vs page 10). They spin so slowly they never wrap.
    # 
    # THE MATH BREAKDOWN:
    # 1. Why `(theta_numerator / head_dim)`? This is a normalization trick. It creates a 
    #    fraction that always slides from 0.0 to 1.0, regardless of model size. Whether 
    #    your model has 6 dimensions or 4096 dimensions, the exponent smoothly slides 
    #    across the exact same mathematical spectrum.
    # 
    # 2. Why `10000`? This is the "Base Stretch". 10000^0.0 = 1 (Fast). 10000^1.0 = 10000 (Slow). 
    #    10000 stretches the frequencies across a massive range. If you train a model with a 
    #    128,000-word context window (like LLaMA 3), 10000 isn't big enough! The slow clocks 
    #    wrap around before reaching the end of the book. To fix this, researchers crank 
    #    the base up to 500,000 to make the slow clocks spin even slower (RoPE Base Scaling).
    #    Since our NanoGPT only reads 10 words (max_context_window), 10000 is perfectly fine!
    # =============================================================
    # Calculate the frequency for each dimension pair: 10000^(-2i/d)
    theta = 1.0 / (10000 ** (theta_numerator / head_dim))
    
    # =============================================================
    # DOCUMENTARY DEEP DIVE: THE MAGIC OF THE DOT PRODUCT
    # =============================================================
    # Why rotate? Because of how Attention calculates scores: `q @ k.transpose()`
    # In linear algebra, a Dot Product mathematically measures the ANGLE between two vectors.
    # By rotating our vectors based on their physical position before they enter the Dot Product, 
    # the resulting Attention Score naturally reflects the RELATIVE angle (distance) between them! 
    # We don't have to write any extra code to calculate distance—the matrix math does it for us.
    # =============================================================
    
    # =============================================================
    # DOCUMENTARY DEEP DIVE: THE PHYSICAL ANCHOR (m)
    # =============================================================
    # We just calculated 'theta' (the speeds of our clocks). But a speed is useless 
    # if you don't know how long to drive! 
    # 'm' stands for the physical position index. It answers: Which word are we looking at?
    # 'seq_len' is our max context window (10). 
    # torch.arange(seq_len) generates a simple list: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].
    # This means Word 0 is at position 0, Word 1 is at position 1. 
    # In Step 4, we multiply this position by the speed. If the speed is 10 degrees per word, 
    # and we are at Word 3 (m=3), the math becomes 3 * 10 = 30 degrees. 
    # 'm' tells the model exactly how many "ticks" to turn the dial for each specific word!
    # =============================================================
    # Step 3: Create the physical position indices
    m = torch.arange(seq_len)
    
    # =============================================================
    # DOCUMENTARY DEEP DIVE: THE OUTER PRODUCT GRID
    # =============================================================
    # Step 4: Multiply Positions by Speeds to get the final Rotation Angles!
    # torch.outer takes our vertical column of positions (m) and horizontal row of 
    # speeds (theta) and multiplies them into a 2D grid (matrix) in one GPU blast.
    #
    # ROW 0 (Word 0): Multiplies Position 0 by all clocks. 0 * Speed = 0. 
    #                 (Word 0 gets absolutely no rotation).
    # ROW 1 (Word 1): Multiplies Position 1 by all clocks. 1 * Speed = Base Speed.
    #                 (Word 1 gets rotated exactly by the base speeds).
    # ROW 2 (Word 2): Multiplies Position 2 by all clocks. 2 * Speed = Double Speed.
    #                 (Word 2 gets rotated twice as much as Word 1).
    # 
    # The resulting 'freqs' matrix contains the exact, final rotation angle (in radians)
    # needed for every single dimension pair, for every single word in the sentence.
    # =============================================================
    freqs = torch.outer(m, theta).float()
    
    # Print the shape so we can track the tensor geometry!
    print(f"\n[RoPE] Precomputed Frequencies Shape (Seq_Len, Head_Dim/2): {freqs.shape}")
    
    return freqs

# =============================================================
# DOCUMENTARY DEEP DIVE: COMPLEX NUMBERS & THE UNPACK OPERATOR (*)
# =============================================================
# To rotate a vector, we need a 2D coordinate (X, Y). PyTorch handles this using Complex 
# Numbers, where X is the Real part and Y is the Imaginary part.
# 
# Right now, our Queries (xq) and Keys (xk) end in 6 raw, real numbers. We need to 
# slice that 6 into 3 pairs of 2, and then convert those pairs into Complex Numbers.
#
# THE UNPACK OPERATOR (*xq.shape[:-1]):
# `xq.shape[:-1]` gets every dimension EXCEPT the final one (e.g., Batch, Heads, Time).
# The `*` symbol is the Python unpack operator. It deletes the tuple brackets and 
# pastes the numbers directly into the function. 
# It tells PyTorch: "I don't care how many dimensions come before the end. Keep them 
# exactly the same. Just take the final dimension and chop it into pairs of 2 (-1, 2)."
# This makes the function perfectly modular, whether we pass in a 3D or 5D tensor!
# =============================================================
def apply_rotary_position_embeddings(xq, xk, freqs):
    # Part 1: Reshape the last dimension into pairs, and cast to PyTorch Complex Numbers
    xq_complex = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_complex = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    # =============================================================
    # DOCUMENTARY DEEP DIVE: EULER'S CHEAT CODE (DE FACTO LINEAR MATH)
    # =============================================================
    # Why use Complex Numbers instead of normal [X, Y] coordinates?
    # Because rotating an [X, Y] coordinate requires clunky matrix math:
    # New X = (X * cos) - (Y * sin)
    # New Y = (X * sin) + (Y * cos)
    # 
    # By casting our pairs as Complex Numbers (Real + Imaginary), we can use Euler's 
    # formula. Complex multiplication automatically executes that entire sine/cosine 
    # matrix formula at the hardware level! We turn clunky geometry into a simple, 
    # hyper-fast linear multiplication (A * B).
    # 
    # torch.polar(abs, angle) creates the complex rotation gear. The radius is 1.0 
    # because we only want to spin the vectors, not stretch them!
    # =============================================================
    
    # Part 2: Convert the real angles (freqs) into complex rotation gears
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    
    # =============================================================
    # DOCUMENTARY DEEP DIVE: THE REVERSE CAST & THE SMASH
    # =============================================================
    # 1. THE MULTIPLICATION: `xq_complex * freqs_complex` physically executes the 
    #    rotation. Every 2D coordinate spins by its assigned angle.
    # 2. THE REVERSE CAST: The rest of the neural network cannot read complex numbers.
    #    `view_as_real` strips away the imaginary 'i' and splits the coordinate back 
    #    into a standard pair of floats: [X, Y]. Shape is now (Batch, Heads, Time, 3, 2).
    # 3. THE SMASH: We need a flat list of 6 numbers, not 3 pairs of 2.
    #    `.flatten(3)` tells PyTorch: "Leave indices 0 (Batch), 1 (Heads), and 2 (Time) 
    #    alone! Start at index 3, and smash everything after it together." It multiplies 
    #    the 3 and 2 together, returning our original flat shape: (Batch, Heads, Time, 6).
    # 4. THE SAFETY NET: `type_as(xq)` ensures that if the input was a fast 16-bit float, 
    #    we don't accidentally return a slow 32-bit float and crash the network.
    # =============================================================
    xq_out = torch.view_as_real(xq_complex * freqs_complex).flatten(3)
    xk_out = torch.view_as_real(xk_complex * freqs_complex).flatten(3)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)

class NanoGPT(nn.Module):
    def __init__(self):
        # =============================================================
        # DOCUMENTARY DEEP DIVE: ARCHITECTURE INHERITANCE (`super().__init__()`)
        # =============================================================
        # PyTorch's `nn.Module` serves as the foundational base class (the engine chassis)
        # for all neural network components, providing memory management, parameter tracking, 
        # and backpropagation routing. 
        #
        # By defining `def __init__(self):`, the base setup function is technically overwritten.
        # Calling `super().__init__()` pauses the custom initialization, reaches up to the 
        # parent class (`nn.Module`), and executes its setup function first. This ensures the 
        # PyTorch backend is fully booted and tracking before any custom layers are instantiated.
        # If this call is omitted or moved to the bottom, PyTorch will crash when attempting 
        # to register the custom layers.
        # =============================================================
        super().__init__()
        
        # -------------------------------------------------------------
        # VOLUME 1 MAPPING: E = np.random.randn(...) (See 03_ffn.py Line 56)
        # -------------------------------------------------------------
        # nn.Embedding is the PyTorch equivalent of the 'E' matrix. 
        # Under the hood, it initializes a learnable weight matrix of shape (vocab_size, embedding_dim).
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # We instantiate a single Transformer Block. (In a real model, we would loop and create 12 or 24 of these!)
        self.block = Block(embedding_dim, num_heads, max_context_window, embedding_dim * 2)
        self.final_ln = nn.LayerNorm(embedding_dim)
        
        # -------------------------------------------------------------
        # VOLUME 1 MAPPING: W_lm = np.random.randn(...) (See 03_ffn.py Line 206)
        # -------------------------------------------------------------
        # nn.Linear performs the matrix multiplication: (Input @ Weights) + Bias.
        # Note: PyTorch automatically transposes the weights internally, requiring only (input_dim, output_dim).
        # Bias is set to False to directly mirror the pure dot-product projection established in Volume 1.
        self.lm_head = nn.Linear(embedding_dim, vocab_size, bias=False)
        #When python hits apply(), Pytorch's backend engine aggressively crawls through every single
        #components in the model, if it finds a component, it calls _init_weights inspector
        #if it hands a layernorm, the isinstance check fails and it skips
        #if it hands a linear layer, the inspector overwrite the memory with 0.02 standard deviation
        #As for why 0.02, explanation is down below
        #This is basically just a 2nd safeguard just in case of some components slipping through the
        #Layernorm, for smaller model, it'd be fine, but for bigger model, if even in the slightest
        #chance there's a bug, the consequence would be catastrophic
        self.apply(self._init_weights)

    def _init_weights(self, module):
    #1. If the module is an embedding layer or a linear layer:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            #Reach into its physical memory and overwrite the weights with a normal distribution
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)
            #The reason we apply mean = 0.0 and std = 0.02 here is because if we initiate weights as 0, that's
            #a dead end for the entire neuron, yet if we apply a random number and that number is too high
            #we instantly expose ourselves to a potential NaN or gradient explosion
            #thus we make it at just 0.0 with standard distribution of 0.02 to ensure there's wiggle room initially
            #The reason the number 0.02 is chosen isn't random either, let's go back to the standard
            #weights normalization
            #We have 2 methods traditionally: Xavier (Glorot) and He (Kaiming)
            #Glorot is for tanh/sigmoid: automatically scales standard deviation to np.sqrt(2/(N_in + N_out))
            #Kaiming is for ReLU, GELU: automatically scales standard deviation to np.squrt(2/N_in)
            #For Glorot, in the random initiation, the gradients deviate at the rate of 1/np.sqrt(d_in)
            #For Kaiming, it's a bit different because ReLU cancels out half the variance that are below 0
            #we double the initialization variance in order to keep the signal from collapsing in the 
            #deeper layers (this can mean the gradients get shrunk, thus impacting the neurons): 1.414/np.sqrt(d_in)
            #SwiGLU operates as such: SwiGLU(x) (Swish(xW)@xV)W2
            #Let A = Swish(xW) and B = xV, assuming variance = 1 and a mean of 0.
            #Var(B) = d * W_deviation ** 2
            #Var(A) = 1/2 * d * W_deviation ** 2
            #Var(A @ B) = Var(A) * Var(B) = 1/2 * d**2 * W_deviation**4 = Var(SwiGLU)
            #Let Var = 1, we have W_deviation = 1.189/np.sqrt(d)
            #Glorot: 1/sqrt(d); Kaiming: 1.414/sqrt(d), SwiGLU deivation: 1.189/sqrt(d) -> Safely conclude that Glorot is better
            #However, modern LLM completely bypasses this and instead settles for 0.02 hardcoded.
            #They can do this because of a safeguard: Pre-layer norm (RMSNorm), this is like the 
            #Layer normalization we have above that's placed directly before feed-forward to reset incoming
            #variance of incoming input to 1.0, thus ensuring there's to deviation orrcured in previous layer
            #As for why they use 0.02, let's plug it into the standard deviation of SwiGLU we just did
            #W_deviation = 1.189/np.sqrt(d). If we look at foundational models specs like Llama 2(7B) or
            #Llama 3(8B), the hidden dimension is 4096. Plug that in and we have the standard deviation at
            #0.0185, extremely close to 0.02. They picked 0.02 because it's the fastest rounded number for
            #variance stabilization.
            #If this Linear layer happens to have a bias vector, we put it to zero
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        # =============================================================
        # DOCUMENTARY DEEP DIVE: BATCH (B) & TIME (T)
        # =============================================================
        # In Volume 1, a single sentence was processed at a time (e.g., [0, 1, 2]).
        # The sequence length (3) is referred to as "Time" (T).
        # PyTorch is optimized for GPUs, requiring multiple sentences to be processed simultaneously.
        # It expects a 2D matrix of stacked sentences, where the number of rows is the "Batch Size" (B).
        # Therefore, idx shape is (B, T). The Batch dimension simply rides along in parallel during all math operations.
        B, T = idx.shape
        
        # -------------------------------------------------------------
        # VOLUME 1 MAPPING: E[tokens] (See 03_ffn.py Line 231)
        # -------------------------------------------------------------
        # Extracts the row vectors corresponding to the input tokens.
        # This transforms the 2D matrix (B, T) into a 3D tensor (B, T, C).
        # C (Channels) is the Embedding Dimension (6 in this case).
        tok_emb = self.token_embedding(idx) 
        
        # -------------------------------------------------------------
        # FAREWELL TO ABSOLUTE POSITIONS (VOLUME 1 DEPRECATED)
        # -------------------------------------------------------------
        # In Volume 1, we added absolute positional embeddings here: x = tok_emb + pos_emb
        # Since we upgraded to RoPE (Rotary Position Embeddings), we completely skip 
        # that step! The raw token embeddings flow directly into the Transformer Block, 
        # and position is mathematically injected later during the Attention Dot Product.
        x = tok_emb # Shape: (Batch_Size, Sequence_Length, Embedding_Dim)
        
        # Pass the embeddings through the Transformer Block
        x = self.block(x)
        
        # A final LayerNorm is applied before the Language Model head to stabilize the final predictions.
        x = self.final_ln(x)
        
        # -------------------------------------------------------------
        # VOLUME 1 MAPPING: Z_lm = sentence_vector @ W_lm + b_lm (See 03_ffn.py Line 311)
        # -------------------------------------------------------------
        # In Volume 1, only the final token's vector was passed to the LM head to predict a single next word.
        # Here, the entire sequence matrix is passed through the Linear layer at once,
        # returning the next-word predictions for every single time-step simultaneously.
        logits = self.lm_head(x) # Shape: (Batch_Size, Sequence_Length, Vocab_Size)
        
        # -------------------------------------------------------------
        # THE LOSS CALCULATION (CROSS ENTROPY)
        # -------------------------------------------------------------
        # If targets (the answer key) are provided, we calculate the grade.
        loss = None
        if targets is not None:
            # PyTorch's Cross Entropy function requires 2D matrices, so we smash 
            # Batch and Time together into a single flat list of predictions.
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)
            
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        # idx is (Batch, Time) array of indices in the current context
        for _ in range(max_new_tokens):
            # Crop idx to the max_context_window so position embeddings/RoPE don't overflow
            idx_cond = idx[:, -max_context_window:]
            
            # Get the predictions
            logits, _ = self(idx_cond)
            
            # Focus only on the last time step
            logits = logits[:, -1, :] # becomes (Batch, Vocab_Size)
            
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Sample from the distribution (or we could just use argmax for greedy)
            # Let's use argmax (greedy decoding) since our model is very small and trained on a tiny string
            idx_next = torch.argmax(probs, dim=-1, keepdim=True) # (Batch, 1)
            
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (Batch, Time+1)
            
        return idx

class CausalSelfAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, max_context_window):
        super().__init__()
        # =============================================================
        # DOCUMENTARY DEEP DIVE: THE DYNAMIC ALLOCATOR
        # =============================================================
        # In Volume 1, manual slices were hardcoded (e.g., Q1 = Q[:, :3]).
        # If the number of heads changed, the slicing logic had to be rewritten entirely.
        # PyTorch utilizes mathematical division to dynamically allocate slice sizes.
        # If C=6 and num_heads=2, head_dim becomes 3. 
        # The .view() function below will use this `3` to dynamically slice the tensor,
        # making the architecture fully modular.
        # =============================================================
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        
        # =============================================================
        # DOCUMENTARY DEEP DIVE: REGISTER_BUFFER VS VARIABLES
        # =============================================================
        # We precompute the RoPE frequencies (the clock speeds) once during initialization 
        # because the math is heavily constant. We don't want to waste GPU cycles doing it 
        # on every single forward pass.
        # 
        # Why `register_buffer` instead of `self.freqs = freqs`?
        # When you call `model.to('cuda')`, PyTorch sweeps through the model and moves all 
        # nn.Layers to the GPU. It ignores normal python variables, leaving them on the CPU!
        # If your Queries (on GPU) try to multiply with freqs (on CPU), the model crashes.
        # `register_buffer` officially bolts the tensor to the model's chassis. It tells 
        # PyTorch: "This is not a trainable weight (no gradients), but it is a permanent 
        # part of the hardware. Wherever the model goes, this tensor goes with it."
        # =============================================================
        freqs = precompute_theta_pos_frequencies(self.head_dim, max_context_window)
        self.register_buffer("freqs", freqs)

        # -------------------------------------------------------------
        # VOLUME 1 MAPPING: W_q, W_k, W_v (See 03_ffn.py Lines 212-214)
        # -------------------------------------------------------------
        # We define three separate Linear layers to project our inputs into Queries, Keys, and Values.
        # Note: In highly optimized production models, these three are often merged into a single massive matrix for speed,
        # but separating them mirrors the mathematical derivation perfectly.
        self.w_q = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.w_k = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.w_v = nn.Linear(embedding_dim, embedding_dim, bias=False)
        
        # -------------------------------------------------------------
        # THE MISSING PIECE: The Output Projection (W_o)
        # -------------------------------------------------------------
        # In Volume 1, the heads were simply concatenated and passed directly to the FFN.
        # A true Transformer uses W_o to blend the concatenated heads back together mathematically.
        self.w_o = nn.Linear(embedding_dim, embedding_dim, bias=False)
        
        # =============================================================
        # DOCUMENTARY DEEP DIVE: THE CAUSAL MASK & BROADCASTING
        # =============================================================
        # In Vol 1: mask = np.triu(np.ones((T, T)), k=1) * -1e9
        # In PyTorch, doing that on the fly is too slow. Instead, we build a MASSIVE mask 
        # (e.g., 1024 x 1024) once during initialization and save it to the GPU memory.
        # 
        # 1. torch.tril(): Creates the lower-triangle of 1s (bottom-left) and 0s (top-right).
        # 2. .view(1, 1, max_C, max_C): The Conservation of Elements rule! 1*1*1024*1024 = 1024*1024.
        #    Adding a `1` doesn't create data, it just adds a new set of mathematical brackets:
        #    [10] is 1D. [[10]] is 2D. [[[10]]] is 3D. The total elements remain exactly the same!
        #    By adding the two `1`s, we tell PyTorch this is a 4D tensor with 1 Batch and 1 Head. 
        #    During math, PyTorch will "Broadcast" (instantly copy-paste) this mask across 
        #    all 32 Batches and 12 Heads without taking up extra RAM!
        # 3. register_buffer: Tells the GPU this is a permanent state, not a learnable weight.
        # =============================================================
        mask = torch.tril(torch.ones(max_context_window, max_context_window))
        self.register_buffer("mask", mask.view(1, 1, max_context_window, max_context_window))

    # =============================================================
    # DOCUMENTARY DEEP DIVE: `SELF` AS A DYNAMIC MEMORY POINTER
    # =============================================================
    # In a full-scale architecture, multiple identical Attention blocks are instantiated in a loop.
    # Without the `self` instance reference, the codebase would require uniquely named 
    # forward functions for every single layer (e.g., forward_layer_1, forward_layer_2) 
    # to guarantee the correct weight matrices are accessed.
    # 
    # `self` acts as a dynamic memory pointer. When data flows into Layer 1, the execution 
    # environment translates `layer_1.forward(x)` into `forward(self=layer_1, x)`. 
    # The pointer binds strictly to Layer 1's isolated memory address, ensuring that 
    # `self.w_q(x)` accesses Layer 1's specific Q-matrix. When data enters Layer 2, 
    # the pointer updates, allowing a single generic blueprint to route data perfectly 
    # across infinite unique layers.
    # =============================================================
    def forward(self, x):
        # =============================================================
        # DOCUMENTARY DEEP DIVE: THE 3D TENSOR (B, T, C)
        # =============================================================
        # In Volume 1, a single sentence was processed at a time: shape (Time, Channels).
        # PyTorch relies on 3D Tensors to maximize GPU parallel processing.
        #
        # C (Channels): The embedding dimension (e.g., 6 parameters defining a single token).
        #               The term "Channels" originates from PyTorch's computer vision legacy (RGB).
        #               During initialization, these are random values. During training, backpropagation 
        #               sculpts these parameters to represent semantic meaning in continuous vector space.
        #               Tokens with similar contexts map to similar vectors.
        #
        # T (Time):     The sequence length (the number of tokens in the context window).
        # 
        # B (Batch):    The number of independent sequences processed simultaneously.
        #               The Batch dimension stacks B distinct (T, C) matrices into a 3D block.
        # 
        # Visualizing (B, T, C) = (32, 3, 6): A tensor containing 32 independent sequences, 
        # each composed of 3 tokens, where every token is defined by 6 embedding channels.
        # =============================================================
        B, T, C = x.shape
        
        # 1. Linear Projections
        q = self.w_q(x) # (B, T, C)
        k = self.w_k(x) # (B, T, C)
        v = self.w_v(x) # (B, T, C)
        
        # =============================================================
        # DOCUMENTARY DEEP DIVE: .VIEW() AND .TRANSPOSE()
        # =============================================================
        # 1. THE FOLD (.view)
        # For C=6 and num_heads=2, .view(B, T, 2, 3) folds the 6 features into 2 rows of 3.
        # Head 1 is allocated the first 3 features. Head 2 is allocated the last 3 features.
        # This physically isolates different learned representations into separate parallel computational paths.
        #
        # Note on `-1`: PyTorch allows using `-1` for a single dimension (e.g., .view(B, T, 2, -1)) 
        # to automatically infer the shape. This is an architectural hazard. If a shape mismatch occurs 
        # in the batch size, PyTorch will silently fold the matrix incorrectly without raising an error. 
        # Explicitly defining `self.head_dim` forces strict dimensional validation.
        #
        # 2. THE SWAP (.transpose)
        # PyTorch's matrix multiplication (@) strictly operates on the LAST TWO dimensions. 
        # The current shape is (Batch, Time, Heads, Head_Dim). A direct multiplication would multiply across Heads.
        # .transpose(1, 2) swaps Time and Heads, resulting in (Batch, Heads, Time, Head_Dim).
        # The GPU processes the first two dimensions (Batch * Heads) as independent parallel universes, 
        # executing the matrix multiplication exclusively on (Time @ Head_Dim).
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # =============================================================
        # DOCUMENTARY DEEP DIVE: THE DYNAMIC SLICE ([:T])
        # =============================================================
        # `self.freqs` was generated for the absolute max_context_window (e.g., 10 words).
        # So it has 10 rows of angles. But what if the user types a sentence that is only 
        # 3 words long? T = 3. 
        # If you try to multiply a 3-word Query matrix by a 10-word Frequency grid, 
        # PyTorch will instantly crash due to a shape mismatch.
        # 
        # `self.freqs[:T]` dynamically slices the massive precomputed grid on the fly. 
        # It says: "If the current sentence is only 3 words long, just grab the first 
        # 3 rows of angles from the grid." This guarantees perfect shape matching!
        # =============================================================
        q, k = apply_rotary_position_embeddings(q, k, self.freqs[:T])
        # =============================================================
        # DOCUMENTARY DEEP DIVE: THE INNER DIMENSION COLLISION
        # =============================================================
        # Right now, q and k have the exact same shape: (..., T, Head_Dim).
        # If we blindly multiplied them (q @ k), PyTorch would try to compute:
        # (T, Head_Dim) @ (T, Head_Dim)
        # This is mathematically illegal because the inner dimensions (Head_Dim and T) do not match!
        #
        # To fix this, we flip `k` on its side using .transpose(-2, -1), changing it to (Head_Dim, T).
        # Now the math becomes: (T, Head_Dim) @ (Head_Dim, T).
        # The inner dimensions (Head_Dim) perfectly match, collapse, and vanish, leaving us 
        # with the (T, T) Attention Grid (Words looking at other Words)!
        # =============================================================
        scores = q @ k.transpose(-2, -1)
        
        # Divides by the square root of the head dimension to prevent the variance from exploding.
        # (This replicates the Xavier initialization logic from Volume 1, applied dynamically here).
        scores = scores / (self.head_dim ** 0.5)
        
        # =============================================================
        # DOCUMENTARY DEEP DIVE: MASKED_FILL & INFINITY
        # =============================================================
        # 1. THE SLICE: self.mask[:, :, :T, :T]
        #    The mask is 1024x1024. If our sentence is only 3 words (T=3), we slice out a 3x3 grid.
        #    `[:, :,` means grab ALL Batches and ALL Heads. `:T, :T` means grab a TxT grid.
        # 2. THE BOOLEAN: == 0
        #    This turns the 3x3 grid into a True/False map. Wherever it sees a 0 (the future), it marks `True`.
        # 3. THE VALUE: float('-inf')
        #    Everywhere the map is `True`, it punches a hole in `scores` and fills it with Negative Infinity.
        #    (In Volume 1, -1e9 was utilized to prevent `NaN` corruption during multiplication. 
        #    PyTorch's C++ Softmax implementation is hyper-optimized to safely ingest literal `-inf` 
        #    without triggering NaN gradients during backpropagation.)
        # =============================================================
        scores = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        
        # =============================================================
        # DOCUMENTARY DEEP DIVE: SOFTMAX & VALUE INTEGRATION
        # =============================================================
        # VOLUME 1 MAPPING: weights1 = softmax(scores1) (See 03_ffn.py Line 251)
        # VOLUME 1 MAPPING: context1 = weights1 @ V1 (See 03_ffn.py Line 252)
        # 
        # F.softmax converts the raw scores into a probability distribution (0.0 to 1.0).
        # The `-inf` values generated by the mask are mathematically crushed to exactly 0.0.
        #
        # Matrix Multiplication (`@`) ignores the Batch and Head dimensions, computing:
        # (Time, Time) @ (Time, Head_Dim). The inner dimensions (Time) match and collapse, 
        # producing an output shape of (Batch, Head, Time, Head_Dim).
        # Conceptually, the probability weights are multiplied by the physical features (v), 
        # resulting in a context-aware token representation.
        # =============================================================
        weights = F.softmax(scores, dim=-1)
        context = weights @ v
        
        # =============================================================
        # DOCUMENTARY DEEP DIVE: RECOMBINING THE PARALLEL UNIVERSES
        # =============================================================
        # VOLUME 1 MAPPING: np.concatenate((context1, context2), axis=-1) (See 03_ffn.py Line 260)
        # 
        # The rest of the network (LayerNorm, SwiGLU) expects a solid (B, T, C) tensor. 
        # The parallel heads must be concatenated back together.
        #
        # 1. .transpose(1, 2): Swaps Head and Time to (Batch, Time, Head, Head_Dim).
        #    This ensures all heads belonging to a specific token are grouped adjacently.
        # 2. .contiguous(): PyTorch's transpose operations do not physically move data in RAM; 
        #    they only update memory pointers. `.contiguous()` forces the GPU to physically 
        #    realign the bytes side-by-side in memory. This is required before reshaping.
        # 3. .view(B, T, C): Smashes the Head and Head_Dim dimensions back into the original 
        #    Channels dimension (C). This perfectly reverses the slice made at the start of the block.
        # =============================================================
        context = context.transpose(1, 2).contiguous().view(B, T, C)
        
        # 7. Final Output Projection
        return self.w_o(context)

# =============================================================
# DOCUMENTARY DEEP DIVE: THE RESEARCH SCIENTIST VS. SYSTEMS ENGINEER
# =============================================================
# This `Block` class is the pure mathematical truth of the Transformer (The Research Scientist's job).
# It defines exactly how data flows through Attention, Norms, and SwiGLU.
# To build a massive corporate model (like LLaMA or GPT-4), we literally just copy-paste this 
# exact Block 96 times in a loop. 
# 
# The corporate moat isn't a secret algorithm; it's the Systems Engineers who write complex 
# networking code to split this single Block's math across 10,000 GPUs without the computers catching fire.
# Mathematically, this Block is the State of the Art.
# =============================================================
class Block(nn.Module):
    def __init__(self, embedding_dim, num_heads, max_context_window, ffn_hidden_states):
        super().__init__()
        
        # -------------------------------------------------------------
        # VOLUME 1 MAPPING: layernorm_forward(x) (See 03_ffn.py Line 111 & 232)
        # -------------------------------------------------------------
        # nn.LayerNorm computes the mean and variance dynamically.
        # The learnable gamma and beta vectors from Volume 1 are handled entirely under the hood.
        self.ln_1 = nn.LayerNorm(embedding_dim)
        self.attn = CausalSelfAttention(embedding_dim, num_heads, max_context_window)
        
        self.ln_2 = nn.LayerNorm(embedding_dim)
        self.ffn = SwiGLU(embedding_dim, ffn_hidden_states)

    def forward(self, x):
        # =============================================================
        # DOCUMENTARY DEEP DIVE: THE "LOCKED ROOM" OF ATTENTION
        # =============================================================
        # Notice how LayerNorm (ln_1, ln_2) and SwiGLU (ffn) operate. They don't know what "Heads" are.
        # They ONLY operate on the solid, unbroken `C` dimension (e.g., the full 6 numbers).
        # 
        # The Attention mechanism (self.attn) is a locked room. 
        # Data enters as a solid `C`. Inside the room, it gets chopped into 12 heads, the parallel
        # universe math happens, and then it is GLUED BACK TOGETHER into a solid `C` before leaving.
        # LayerNorm and SwiGLU never see the heads. They just process the solid block.
        # =============================================================
        
        # -------------------------------------------------------------
        # THE MISSING PIECE: Residual Connections
        # -------------------------------------------------------------
        # In Volume 1, the data flowed in a straight line: x -> Norm -> Attention -> FFN.
        # Here, we add the output of the attention mechanism BACK into the original input (x + ...).
        # This creates an alternative pathway for gradients to flow during backpropagation,
        # preventing the vanishing gradient problem in deep networks.
        
        # Pathway 1: Input + Attention(Norm(Input))
        x = x + self.attn(self.ln_1(x))
        
        # Pathway 2: Input + SwiGLU(Norm(Input))
        x = x + self.ffn(self.ln_2(x))
        
        return x

# --- TESTING BLOCK ---
if __name__ == "__main__":
    model = NanoGPT()
    print("PyTorch Automaton Initialized!")
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\n--- INITIATING FORWARD PASS TEST ---")
    # Generate 2 fake sentences, each 3 words long. (Tokens must be between 0 and vocab_size-1)
    # The vocab_size is 6 (from our vocab dictionary at the top).
    dummy_idx = torch.randint(0, vocab_size, (2, 3))
    print(f"Input Shape (Batch, Time): {dummy_idx.shape}")
    
    # Push the data through the engine. This triggers model.forward()
    logits, loss = model(dummy_idx)
    
    print(f"Output Shape (Batch, Time, Vocab_Size): {logits.shape}")
    print("\nIf you see (2, 3, 6) above, the Automaton's tensor calculus is flawless.")

