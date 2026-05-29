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
embedding_dim = 3
E = np.random.randn(vocab_size, embedding_dim)

# ==========================================
# 2. THE NEURAL NETWORK 
# ==========================================

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def softmax(z):
    exp_z = np.exp(z)
    return exp_z/np.sum(exp_z, axis = 1, keepdims = True)
# --- LAYER 1: HIDDEN STATE ---
# Connects 3 input features to 5 hidden neurons
W1 = np.random.randn(3, 5) # Shape: (3, 5)
b1 = np.zeros((1, vocab_size)) # Shape: (1, 5)
W2 = np.random.randn(5, vocab_size) # Shape: (5, 5)
b2 = np.zeros((1, vocab_size)) # Shape: (1, 1)

#Layer 1: Self-attention weights
W_q = np.random.randn(embedding_dim, embedding_dim)
W_k = np.random.randn(embedding_dim, embedding_dim)
W_v = np.random.randn(embedding_dim, embedding_dim)

for epoch in range(1000):
    # The Lookup (3, 3)
    sentence_embedding = E[tokens]

    #From that we generate the random values for Q, K, V. This is random so the training loop can have a start somewhere
    Q = sentence_embedding @ W_q #Shape (3,3)
    K = sentence_embedding @ W_k #Shape (3,3)
    V = sentence_embedding @ W_v #Shape (3,3)

    #Next we calculate the scores, it is basically calculating the value with which we can calculate the
    #"context" between different words
    scores = Q @ K.T # Shape: (3, 3) @ (3, 3) -> (3, 3)
    #Note: We apply transposition here because Q and K have the same meaning in terms of calculation
    #If we don't use transposition, matrix Q and K will be multiple columns with rows, which won't make any sense given the
    #original idea
    attention_weights = softmax(scores) #Shape(3,3)
    
    #Finally, we calculate the context
    context_vector = attention_weights @ V #(3,3) @ (3,3) -> (3,3)

    #GPT2 model use the last word as context for the next word, thus we use that as calculation
    sentence_vector = context_vector[-1:] #This basically grabs the last row, shape (1,3)

    #Layer 1
    Z1 = sentence_vector @ W1 + b1 # (1, 3) @ (3, 5) + (1, 5) -> Shape: (1, 5)
    A1 = sigmoid(Z1) # Shape: (1, 5)

    #LAYER 2: THE OUTPUT LAYER
    # Task: We need to compress the (1, 5) hidden state down to a (1, 1) final prediction.
    Z2 = A1 @ W2 + b2 # (1, 5) @ (5, 5) + (1, 1) -> Shape: (1, 5)
    A2 = softmax(Z2) # Shape: (1, 5)

    # The Corrected Loss Calculation
    target = np.array([[0.0,0.0,0.0,1.0,0.0]]) #Shape: (1,5)

    # Notice how 'target' is only multiplied by the first log term now
    #loss = -(target * np.log(A2 + 1e-9) + (1 - target) * np.log(1 - A2 + 1e-9))
    loss = -np.sum(target *np.log(A2 +1e-9))

    #Backpropagation
    #dZ2 = dL/dZ = dL/dA * dA/dZ
    #dZ2 = -(target*(1/(A2 + 1e-9)) + (1-target)* (-1/(1-A2+1e-9))) * (A2*(1-A2))
    dZ2 = A2 - target # (1, 5) - Scalar -> Shape: (1, 5)

    #dW2 = dL/dW2 = dL/dZ2 * dZ2/dW2
    dW2 = A1.T @ dZ2 # (5, 1) @ (1, 5) -> Shape: (5, 5)

    #db2 = dL/dB2 = dL/dZ2 * dZ2/db2
    db2 = np.sum(dZ2, axis = 0, keepdims = True) # Shape: (1, 5)
    #dZ1 = dL/dZ1 = dL/dA1 * dA1/dZ1
    #dL/dA1 = dL/dZ2 * dZ2/dA1
    dZ1 = dZ2 @ W2.T * A1 * (1-A1) # (1, 5) @ (5, 5) * (1, 5) -> Shape: (1, 5)

    #dW1 = dL/dW1 = dL/dZ1 * dZ1/dW1
    #dZ1/dW1 = sentence_vector
    dW1 = sentence_vector.T @ dZ1 # (3, 1) @ (1, 5) -> Shape: (3, 5)

    #db1 = dL/db1 = dL/dZ1 * dZ1/db1
    #dZ1/db1 = 1
    db1 = np.sum(dZ1, axis = 0, keepdims = True) # Shape: (1, 5)

    #Tracing the blame back to the beginning
    #d_sentence_vector = dL/d_sentence_vector = dL/dZ1 * dZ1/d_sentence_vector
    d_sentence_vector = dZ1 @ W1.T # Shape: (1, 5) @ (5, 3) = (1, 3)

    #We did not yet engineer that the blame to be split equally
    #d_sentence_embedding = dL/d_sentence_embedding = dL/dZ1 * dZ1/d_sentence_embedding
    #dZ1/d_sentence_embedding = dZ1/d_sentence_vector * d_sentence_vector/d_sentence_embedding
    #d_sentence_vector/d_sentence_embedding = 1/len(tokens)
    #d_sentence_embedding = (d_sentence_vector * 1/len(tokens)) * (np.ones((len(tokens),embedding_dim)))

    #Backpropagation for attention algorithm:

    #Blame for context vector calculation
    #d_context_vector = dL/d_context_vector = dL/dSvec * dSvec/dC
    #dSvec/dC = M.T (M here stands for matrix, also the formal actual formula for Svec is actually M @ C)
    #dL/dSvec = dL/dZ1 * dZ1/dSvec = dZ1.W1
    #dZ1/dSvec = W1.T
    d_context_vector = np.zeros((len(tokens), embedding_dim)) #shape (3,3); this is basically making an empty placeholder matrix for storing the value
    d_context_vector[-1:] = d_sentence_vector #Actual formula woud be dZ1 @ W1.T @ M.T
    
    #d_attention_weights = dL/dC * dC/dAweights
    d_attention_weights = d_context_vector @ V.T #(3,3) @ (3,3) -> (3,3)

    #dV = dL/dV = dL/dC * dC/dV
    dV = attention_weights.T @ d_context_vector #(3,3) @ (3,3) -> (3,3)
    
    #3. Now we derive Softmax
    #Basically, we calculate d_scores, this one is the toughest of the bunch.
    #First off, we start off with the overview, d_scores = dL/d_scores = dL/d_attention_weights * d_attention_weights/d_score
    #d_attention_weights/d_scores would be the derivative of the softmax function, it's derivative is a Jacobian Matrix
    #attention_weights = softmax(scores)
    #Softmaxfunction = e^zi / np.sum(e^zk)
    #We'll split this into 2 cases, case A and B, for case A, let's assume that i=j, and for case B, i is different from j
    #Case A would be Si(1-Si), obviously it's not that simple, but you'll have to derive yourself
    #Case B would be -Si*Sj
    #The combined would be Si(1-Si) - Si*Sj
    #Next would be the d_attention_weights, normally we'd inject this in at the end of the function, but for the sake of softmax derivative
    #that can't be done sinc we're realing with a Jacobian matrix here
    #function would be Ei*Si*(1-Si) - sum(Ej*Si*Sj), the reason there's a sum here is because i isn't the same as j, so we need to add it all up
    #the end function would be Si(Ei - sum of (Ej * Si)) however the sum here has changed, it's in the condition of all j instead of just i different from j
    #Now you can proudly inject that in
    #S is attention weights, E is d_attention_weights
    d_scores = attention_weights * (d_attention_weights - np.sum(d_attention_weights * attention_weights, axis = -1, keepdims = True))
    #If you did the math manually, you'd be surprised why there's not a single index here
    #The index is actually there, however numpy executes it for us on the background
    #The exact formula would be in c++:
    '''for (int i = 0; i < sequence_length; i++) {
    for (int j = 0; j < sequence_length; j++) {
        // out[i][j] = S_i * (E_i - sum_term)
        out_ptr[i * stride + j] = S_ptr[i * stride + j] * E_result_ptr[i * stride + j];
    }
}'''

    #Next up, we find the blame for Q and K
    #dQ = dL/dQ = dL/d_scores * d_scores/dQ
    dQ = d_scores @ K #(3,3) @ (3,3) -> (3,3)
    #dK = dL/dK = dL/d_scores * d_scores/dK
    dK = d_scores.T @ Q #(3,3) @ (3,3) -> (3,3), we use transpose here because we transpose K back when
    #we were calculating scores (scores = Q @ K.T) so when we calculate the loss backwards, we also have to transpose it

    #After that we do dW_q, dW_k and dW_v
    #dW_q = dL/dW_q = dL/dQ * dQ/dW_q
    dW_q = sentence_embedding.T @ dQ
    #And the same goes for the other 2
    dW_k = sentence_embedding.T @ dK
    dW_v = sentence_embedding.T @ dV
    #Note: the reason we apply transpose here is because of the content within the matrices themselves
    #sentence_embedding is a matrix with it's row 1 as features for the word "the"
    #dQ's first colum is error for word 1, 2 and 3
    #So if we forgot the transposition, word 1's feature would be multiplied with word 1's, 2's and 3's errors, which would make 0 sense
    #Same case for dK and dV

    #Now we calculate the blame of the sentence embedding
    #d_sentence_embedding = dL/d_sentence_embedding = dL/dQ * dQ/d_sentence_embedding
    d_sentence_embedding_q = dQ @ W_q.T #(3,3) for all
    #Calculate similarly for the others
    d_sentence_embedding_k = dK @ W_k.T
    d_sentence_embedding_v = dV @ W_v.T

    #Now we wrap it all up with a simple addition. The reason it's not seperate and instead addition here is
    #because all those 3 are connected into the full on Q, K and V calculation. However deriving them
    #all at once is mathematically impossible with 3 different variables, and we need a total derivative here
    d_sentence_embedding = d_sentence_embedding_q + d_sentence_embedding_k + d_sentence_embedding_v

    #Update rule
    W1 = W1 - lr * dW1
    b1 = b1 - lr * db1
    W2 = W2 - lr*dW2
    b2 = b2 - lr * db2
    W_q = W_q - lr * dW_q
    W_k = W_k - lr * dW_k
    W_v = W_v - lr * dW_v
    
    for i, token in enumerate(tokens):
        E[token] = E[token] - lr * d_sentence_embedding[i]

    #Every 10 steps printing out progess
    if epoch % 10 == 0:
        print(f"Epoch {epoch} , Loss {loss:.6f} , Prediction: {A2[0][3]:.4f}")

print("\nFinal Attention Weights for 'sat':")
print(attention_weights[-1])