import torch
import torch.nn.functional as F

# Import the architecture we just built!
from nano_gpt import NanoGPT, vocab, max_context_window

# =============================================================
# HYPERPARAMETERS
# =============================================================
batch_size = 16
eval_iters = 20

# =============================================================
# 1. THE TOY DATASET
# =============================================================
# We need a continuous stream of text. We will use a repeating sequence 
# of our limited vocabulary so the model can learn the "grammar" rules.
text_data = "the cat sat on the mat <END> the mat sat on the cat <END> " * 20

# Tokenize: Convert the raw text into our integer IDs.
tokens = [vocab[word] for word in text_data.split()]
data = torch.tensor(tokens, dtype=torch.long)

# =============================================================
# DOCUMENTARY DEEP DIVE: THE TRAIN/VALIDATION SPLIT
# =============================================================
# We split the data: 80% for training (memorizing), 20% for validation (testing).
# If the model performs well on the validation set, we prove it has "Grokked" 
# the underlying rules of our grammar, rather than just brute-force memorizing it!
# =============================================================
n = int(0.8 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split, batch_size):
    # Select the correct dataset split
    dataset = train_data if split == 'train' else val_data
    
    # =============================================================
    # DOCUMENTARY DEEP DIVE: torch.randint(high, size)
    # =============================================================
    # Argument 1 (high): `len(dataset) - max_context_window`. PyTorch expects a raw 
    # number (0-Dimensional) for the maximum limit. No comma needed.
    # Argument 2 (size): `(batch_size,)`. PyTorch expects the geometric shape of the 
    # output to be a Tuple. The trailing comma forces Python to treat the single 
    # number `batch_size` as a 1-Dimensional Tuple shape, rather than just doing math.
    # =============================================================
    ix = torch.randint(len(dataset) - max_context_window, (batch_size,))
    
    # X (Inputs): The sentences the model gets to read.
    x = torch.stack([dataset[i : i+max_context_window] for i in ix])
    
    # Y (Targets): The exact same sentences, but shifted 1 word into the future!
    # If X is "the cat sat", Y must be "cat sat on".
    y = torch.stack([dataset[i+1 : i+1+max_context_window] for i in ix])
    
    return x, y

# =============================================================
# 2. THE CALCULUS ENGINE (OPTIMIZER)
# =============================================================
# We pull our NanoGPT architecture from the void and place it in memory.
model = NanoGPT()

# We need an Optimizer. AdamW is the industry standard for Transformers.
# It looks at the slopes (gradients) calculated by backpropagation and 
# takes a physical "step" downhill for every single weight in the model.
# The learning rate (1e-3) is the size of the step. Too big, and it jumps 
# over the valley. Too small, and it takes forever to reach the bottom.
#For more information about AdamW and it's legacy, Adam, please read adamw_theory_notes.model

# =============================================================
# DOCUMENTARY DEEP DIVE: BUFFERS VS NO_GRAD
# =============================================================
# Why do we use @torch.no_grad() here instead of register_buffer?
# 
# 1. Buffers (Nouns/Data): register_buffer is used to bolt permanent, non-trainable 
#    data (like a Causal Triangle Mask) directly to the chassis of the model. 
#    If the model moves to the GPU, the buffer moves with it. 
#    We do not use buffers for the 'losses' variable because 'losses' is temporary 
#    scratchpad data. We don't want to save test scores permanently to our hard drive.
#
# 2. @torch.no_grad() (Verbs/Engine): This is a Python decorator that acts as a 
#    master switch. During this test, our data still flows through the W_q, W_k, W_v 
#    weight matrices (which MUST be trainable Parameters, not buffers). 
#    Normally, PyTorch would automatically calculate gradients the second the data 
#    touches those matrices. 
#    @torch.no_grad() tells the PyTorch C++ backend: "I know these are trainable 
#    Parameters, but temporarily shut down the calculus engine for the next 2 seconds 
#    so we don't accidentally learn from the test data (Data Leakage)."
#
# 3. Why out = {} (Dictionary vs Vector): We test two distinct datasets ('train' and 'val').
#    If we stored the results in a vector like [1.45, 1.62], we would have to remember
#    that index 0 is train and index 1 is val. This causes indexing bugs. By using a 
#    dictionary, we return {"train": 1.45, "val": 1.62}, allowing us to safely call 
#    losses['val'] later without guessing the index.
# =============================================================

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)

        for k in range(eval_iters):
            #grab a batch
            X, Y = get_batch(split, batch_size)

            #forward pass (get the logits and the loss)
            logits, loss = model(X,Y)

            #Store the raw number (item) of the loss into our tensor
            losses[k] = loss.item()

        #Calculate the mean of all those losses and store it in our dictionary
        out[split] = losses.mean()

    #Switch the model back to Training mode so it can learn again
    model.train()

    return out

#4. The training loop

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

max_iters = 5000 #5000 steps
eval_interval = 500 #Check the test score every 500 steps

print("Start the engine...")

for iter in range(max_iters):
    #1. The test (checking the validation score periodically)
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {iter}: Train Loss: {losses['train']:.4f}, Val Loss: {losses['val']:.4f}")

    #2. The data (grab a batch of training data)
    xb, yb = get_batch('train', batch_size)

    #3. The forward pass
    logits, loss = model(xb, yb)

    #4. The backward pass
    #Step A: Empty the gradient buckets from the last loop
    optimizer.zero_grad(set_to_none=True)

    #Step B: Calculate the new grads (chain rule backwards)
    loss.backward()

    #Step C: Take the physical step downhill (AdamW decoupled weight decay)
    optimizer.step()

print("Training complete")

# Save the trained brain (weights) so we can load it in Volume 3!
torch.save(model.state_dict(), 'nano_gpt.pth')
print("Model saved to nano_gpt.pth")
