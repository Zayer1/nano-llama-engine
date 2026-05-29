import os
import torch
import torch.nn.functional as F

from nano_gpt import NanoGPT, max_context_window

# =============================================================
# HYPERPARAMETERS
# =============================================================
batch_size = 32
eval_iters = 100
max_iters = 5000
eval_interval = 500
learning_rate = 1e-3

# Setup device (Use GPU if available!)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Training on: {device.upper()}")

# =============================================================
# 1. THE DATASET & DYNAMIC VOCABULARY
# =============================================================
current_dir = os.path.dirname(__file__)
input_path = os.path.join(current_dir, 'input.txt')

with open(input_path, 'r', encoding='utf-8') as f:
    text = f.read()

# Build the character-level vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Build the Translators
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

# -------------------------------------------------------------
# UNPACKED (NON-LAMBDA) VERSIONS FOR CLARITY:
# -------------------------------------------------------------
# def encode(text):
#     result = []
#     for char in text:
#         result.append(stoi[char]) 
#     return result
#
# def decode(number_list):
#     result_letters = []
#     for number in number_list:
#         result_letters.append(itos[number])
#     return ''.join(result_letters) # Glues the letters together with no spaces
# -------------------------------------------------------------
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Convert entire text to PyTorch tensor
data = torch.tensor(encode(text), dtype=torch.long)

# Train / Validation Split (90% Train, 10% Val)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# =============================================================
# 2. THE BATCH LOADER
# =============================================================
def get_batch(split):
    data_source = train_data if split == 'train' else val_data
    # Grab random starting indices
    ix = torch.randint(len(data_source) - max_context_window, (batch_size,))
    
    # Slice the chunks for X and Y
    x = torch.stack([data_source[i:i+max_context_window] for i in ix])
    y = torch.stack([data_source[i+1:i+max_context_window+1] for i in ix])
    
    # Move to GPU if available
    x, y = x.to(device), y.to(device)
    return x, y

# =============================================================
# 3. THE EVALUATOR
# =============================================================
@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval() # Turn off learning
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() # Turn learning back on
    return out

# =============================================================
# 4. INITIALIZE THE AUTOMATON
# =============================================================
model = NanoGPT()
model.to(device)

print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f} Million")

# =============================================================
# 5. THE OPTIMIZER (ADAMW)
# =============================================================
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# =============================================================
# 6. THE TRAINING LOOP
# =============================================================
print("\nIgniting the engine... (This will take longer than Volume 2!)")

for iter in range(max_iters):
    
    # Every few hundred steps, check our grade and generate some text!
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss(model)
        print(f"Step {iter:4d}: Train Loss: {losses['train']:.4f}, Val Loss: {losses['val']:.4f}")
        
        # --- GENERATE A TEXT SAMPLE ---
        model.eval()
        context = torch.zeros((1, 1), dtype=torch.long, device=device) # Start with just a newline (0)
        # Generate 100 characters to see how smart it is getting
        sample_indices = model.generate(context, max_new_tokens=100)[0].tolist()
        print(f"\n[SAMPLE] -> {decode(sample_indices)}\n")
        model.train()

    # Get a batch of Shakespeare
    xb, yb = get_batch('train')
    
    # Forward Pass
    logits, loss = model(xb, yb)
    
    # Backward Pass (Calculus)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    
    # Update Weights
    optimizer.step()

print("\nTraining Complete!")

# Save the brain!
save_path = os.path.join(current_dir, 'shakespeare_gpt.pth')
torch.save(model.state_dict(), save_path)
print(f"Model saved to {save_path}")
