from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn.functional as F
import os
from nano_gpt import NanoGPT

# 1. Initialize the FastAPI server (The "Steering Wheel")
app = FastAPI(title="Nano-Llama API", description="Production API for custom autoregressive transformer")

# Setup device (Use GPU if available, otherwise CPU)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[SYSTEM] Starting inference server on device: {device}")

# =============================================================
# 2. LOAD THE DYNAMIC VOCABULARY
# =============================================================
# We need to load the exact same vocabulary mapping we used during training.
current_dir = os.path.dirname(__file__)
input_path = os.path.join(current_dir, 'input.txt')

with open(input_path, 'r', encoding='utf-8') as f:
    text = f.read()

# Create character-to-integer (stoi) and integer-to-character (itos) mappings
chars = sorted(list(set(text)))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi.get(c, 0) for c in s]

# =============================================================
# 3. INITIALIZE THE ENGINE (Your Neural Network)
# =============================================================
model = NanoGPT()
model.to(device)

weights_path = os.path.join(current_dir, 'shakespeare_gpt.pth')
if os.path.exists(weights_path):
    # Load the physical math (weights) you trained into the engine block
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    print("[SYSTEM] Neural Network weights loaded into GPU memory successfully!")
else:
    print("[SYSTEM] WARNING: shakespeare_gpt.pth not found. Model will hallucinate raw noise.")

# Put the model in evaluation mode (turns off Dropout layers so math is deterministic)
model.eval()

# =============================================================
# 4. DEFINE THE DATA SCHEMA
# =============================================================
# This defines the exact JSON structure we expect from the outside world.
class GenerateRequest(BaseModel):
    prompt: str

# =============================================================
# 5. THE API ENDPOINT (Where JSON meets Math)
# =============================================================
@app.post("/generate")
@torch.no_grad() # Disable gradient tracking to save massive amounts of RAM during inference
def generate(request: GenerateRequest):
    print(f"\n[API] Received POST request with prompt: '{request.prompt}'")
    
    # Step A: Convert the raw string into a list of integers
    idx = encode(request.prompt)
    if len(idx) == 0:
        idx = [0] # Fallback if empty
        
    # Step B: Convert the list of integers into a PyTorch Tensor
    # Shape becomes (Batch=1, SequenceLength=T)
    idx_tensor = torch.tensor(idx, dtype=torch.long, device=device).unsqueeze(0)
    print(f"[MATH] Converted prompt to Tensor. Shape: {idx_tensor.shape}")
    
    # Step C: Crop the context window
    # We only feed the last 256 tokens into the engine (our block_size)
    idx_cond = idx_tensor[:, -256:]
    print(f"[MATH] Cropped Context Window. Shape fed to engine: {idx_cond.shape}")
    
    # Step D: Forward Pass through the Neural Network
    # This runs the self-attention blocks and matrix multiplications!
    logits, _ = model(idx_cond)
    print(f"[MATH] Forward pass complete. Logits Shape: {logits.shape}")
    
    # Step E: Pluck the last prediction
    # We only care about the prediction for the VERY NEXT character.
    # We slice out the last position in the sequence dimension.
    logits = logits[:, -1, :]
    print(f"[MATH] Sliced last prediction. Logits Shape: {logits.shape}")
    
    # Step F: Convert Logits to Probabilities using Softmax
    probs = F.softmax(logits, dim=-1)[0] # Shape: (65,)
    
    # Step G: Sample the highest probability
    next_idx = torch.argmax(probs).item()
    next_char = itos.get(next_idx, "?")
    
    # Step H: Extract the Top 5 probabilities to return to the user
    top_probs, top_indices = torch.topk(probs, 5)
    
    top_5 = []
    for p, i in zip(top_probs.tolist(), top_indices.tolist()):
        char_label = itos.get(i, "?")
        if char_label == " ":
            char_label = "SPACE"
        elif char_label == "\n":
            char_label = "ENTER"
            
        top_5.append({
            "char": char_label,
            "prob": round(p * 100, 2)
        })
        
    print(f"[API] Predicted next character: '{next_char}'. Returning JSON.")
    
    # Step I: Return the data to the outside world as JSON
    return {
        "next_char": next_char,
        "top_5": top_5
    }
