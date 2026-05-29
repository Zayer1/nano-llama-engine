
from flask import Flask, request, jsonify, render_template  # pyrefly: ignore [missing-import]
import torch
import torch.nn.functional as F
import os
from nano_gpt import NanoGPT

app = Flask(__name__)

# Setup device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# =============================================================
# 1. LOAD THE DYNAMIC VOCABULARY
# =============================================================
current_dir = os.path.dirname(__file__)
input_path = os.path.join(current_dir, 'input.txt')

with open(input_path, 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi.get(c, 0) for c in s]
decode = lambda l: ''.join([itos.get(i, "?") for i in l])

# =============================================================
# 2. INITIALIZE THE MODEL
# =============================================================
model = NanoGPT()
model.to(device)

weights_path = os.path.join(current_dir, 'shakespeare_gpt.pth')
if os.path.exists(weights_path):
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    print("[SYSTEM] Shakespeare weights loaded successfully!")
else:
    print("[SYSTEM] WARNING: shakespeare_gpt.pth not found. Model will hallucinate.")

model.eval()

# =============================================================
# 3. ROUTES
# =============================================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
@torch.no_grad()
def predict():
    data = request.get_json()
    prompt = data.get('prompt', '')
    
    # 1. Translate string to numbers
    idx = encode(prompt)
    if len(idx) == 0:
        idx = [0] # Fallback to a space/newline if empty
        
    idx_tensor = torch.tensor(idx, dtype=torch.long, device=device).unsqueeze(0)
    
    # 2. Crop to context window
    idx_cond = idx_tensor[:, -256:]
    
    # 3. Forward Pass
    logits, _ = model(idx_cond)
    
    # 4. Pluck the last prediction
    logits = logits[:, -1, :]
    
    # 5. Convert to Probabilities
    probs = F.softmax(logits, dim=-1)[0] # Shape: (65,)
    
    # 6. Sample the next character
    next_idx = torch.argmax(probs).item()
    next_char = itos.get(next_idx, "?")
    
    # 7. Get the Top 5 probabilities for the UI Visualizer
    top_probs, top_indices = torch.topk(probs, 5)
    
    top_5 = []
    for p, i in zip(top_probs.tolist(), top_indices.tolist()):
        char_label = itos.get(i, "?")
        # Make spaces visible in the UI
        if char_label == " ":
            char_label = "SPACE"
        elif char_label == "\n":
            char_label = "ENTER"
            
        top_5.append({
            "char": char_label,
            "prob": round(p * 100, 2) # Convert to percentage
        })
        
    return jsonify({
        "next_char": next_char,
        "top_5": top_5
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
