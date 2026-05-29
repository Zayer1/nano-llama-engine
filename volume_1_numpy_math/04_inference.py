import numpy as np

# ==========================================
# 0. THE ARCHITECTURE UPGRADE
# ==========================================
# To generate text autoregressively, the sequence length will grow over time.
# "the cat sat" (3) -> "the cat sat on" (4) -> "the cat sat on mat" (5).
# If we only initialize 3 positional embeddings (like in 03_ffn.py), the 
# network will crash on the 4th word. We must initialize a "Context Window".
MAX_CONTEXT_WINDOW = 10
embedding_dim = 6
num_heads = 2
head_dim = embedding_dim // num_heads

np.random.seed(42)
vocab = {"the": 0, "cat": 1, "sat": 2, "on": 3, "mat": 4, "<END>": 5}
inverse_vocab = {v: k for k, v in vocab.items()}
vocab_size = len(vocab)

# ==========================================
# 1. LOAD TRAINED WEIGHTS
# ==========================================
try:
    print("Loading trained weights from 'weights.npz'...")
    weights = np.load("weights.npz")
    E = weights["E"]
    P = weights["P"]
    gamma = weights["gamma"]
    beta = weights["beta"]
    W_gate = weights["W_gate"]
    b_gate = weights["b_gate"]
    W_up = weights["W_up"]
    b_up = weights["b_up"]
    W_down = weights["W_down"]
    b_down = weights["b_down"]
    W_lm = weights["W_lm"]
    b_lm = weights["b_lm"]
    W_q = weights["W_q"]
    W_k = weights["W_k"]
    W_v = weights["W_v"]
    print("Weights loaded successfully!\n")
except FileNotFoundError:
    print("CRITICAL ERROR: 'weights.npz' not found. Please run 'python core/03_ffn.py' first to train the network.")
    exit()

# Utility Math Functions
def sigmoid(z):
    z_safe = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z_safe))

def softmax(z):
    z_shifted = z - np.max(z, axis=-1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

def layernorm_forward(x):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    std = np.sqrt(var + 1e-5)
    x_norm = (x - mean) / std
    return gamma * x_norm + beta

# ==========================================
# 2. THE MODULAR FORWARD PASS
# ==========================================
# We wrap your exact math inside a function so we can call it repeatedly.
def generate_next_token(tokens):
    """
    Executes a single forward pass of the Transformer block to predict the next token.
    """
    seq_len = len(tokens)
    
    # 1. Embeddings + Position
    # Notice we only slice the Positional Embeddings up to the current sequence length!
    sentence_embedding = E[tokens] + P[:seq_len]
    sentence_embedding_norm = layernorm_forward(sentence_embedding)

    # 2. Attention Projections
    Q = sentence_embedding_norm @ W_q
    K = sentence_embedding_norm @ W_k
    V = sentence_embedding_norm @ W_v

    Q1, Q2 = Q[:, :head_dim], Q[:, head_dim:]
    K1, K2 = K[:, :head_dim], K[:, head_dim:]
    V1, V2 = V[:, :head_dim], V[:, head_dim:]

    # The Dynamic Causal Mask! 
    # It must adapt to whatever the current sequence length is.
    mask = np.triu(np.ones((seq_len, seq_len)), k=1) * -1e9

    # Head 1
    scores1 = Q1 @ K1.T + mask
    weights1 = softmax(scores1)
    context1 = weights1 @ V1

    # Head 2
    scores2 = Q2 @ K2.T + mask
    weights2 = softmax(scores2)
    context2 = weights2 @ V2

    context_vector = np.concatenate((context1, context2), axis=-1)
    
    # 3. SwiGLU FFN
    Z_gate = context_vector @ W_gate + b_gate
    Z_up = context_vector @ W_up + b_up
    
    S_gate = sigmoid(Z_gate)
    A_gate = Z_gate * S_gate
    Merged = A_gate * Z_up
    Z_down = Merged @ W_down + b_down

    # 4. Language Modeling Head
    # We only care about predicting the NEXT word, so we only grab the very last token's state
    last_token_vector = Z_down[-1:] 
    
    Z_lm = last_token_vector @ W_lm + b_lm
    probabilities = softmax(Z_lm)
    
    # Get the index of the highest probability word
    predicted_token_id = np.argmax(probabilities, axis=-1)[0]
    return predicted_token_id

# ==========================================
# 3. THE AUTOREGRESSIVE INFERENCE LOOP
# ==========================================
if __name__ == "__main__":
    print("--- INITIATING AUTOREGRESSIVE GENERATION ---")

    # Start with a prompt
    current_sentence = ["the", "cat"]
    print(f"Prompt: {' '.join(current_sentence)}")
    
    # Convert words to integer tokens
    current_tokens = [vocab[word] for word in current_sentence]
    
    max_new_tokens = 5
    
    for step in range(max_new_tokens):
        # 1. Pass the sequence through the Neural Network
        next_token_id = generate_next_token(current_tokens)
        
        # 2. Decode the ID back to an English word
        next_word = inverse_vocab[next_token_id]
        
        # 3. APPEND the new word to the sentence (The Autoregressive Loop!)
        current_sentence.append(next_word)
        current_tokens.append(next_token_id)
        
        print(f"Step {step + 1}: Generated '{next_word}' -> Current Sequence: {' '.join(current_sentence)}")
        
        # If the model predicts the end of the thought, we break the loop early.
        if next_word == "<END>":
            print("\nModel reached <END>. Stopping generation.")
            break
            
        # Stop if we hit the context window limit to avoid crashing P
        if len(current_tokens) >= MAX_CONTEXT_WINDOW:
            print("\nReached max Context Window size. Stopping generation.")
            break

    print(f"\nFinal Generated Output: {' '.join(current_sentence)}")
