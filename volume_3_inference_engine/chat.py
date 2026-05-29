import os
import sys
import torch

# 1. LOAD THE ARCHITECTURE FROM VOLUME 2
current_dir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))
from volume_2_pytorch_automaton.nano_gpt import NanoGPT, vocab

# Create a reverse dictionary so we can translate numbers back into English
reverse_vocab = {i: word for word, i in vocab.items()}

# 2. INITIALIZE THE MODEL & LOAD WEIGHTS
model = NanoGPT()
weights_path = os.path.abspath(os.path.join(current_dir, '..', 'volume_2_pytorch_automaton', 'nano_gpt.pth'))

if os.path.exists(weights_path):
    model.load_state_dict(torch.load(weights_path, weights_only=True))
    print("[SYSTEM] Trained weights loaded successfully!")
else:
    print("[SYSTEM] WARNING: nano_gpt.pth not found. Model will hallucinate.")

# Switch model to Evaluation Mode (Turn off Dropout, etc.)
model.eval()

# 3. TEXT ENCODING & DECODING
def encode(text):
    words = text.strip().lower().split()
    # Map words to numbers. If word is unknown, we just default to 0 to prevent a crash.
    return [vocab.get(w, 0) for w in words]

def decode(indices):
    return " ".join([reverse_vocab.get(i, "<UNK>") for i in indices])

# 4. THE INTERACTIVE CHAT LOOP
print("\n=============================================")
print("VOLUME 3: THE INFERENCE ENGINE IS ONLINE")
print("=============================================")
print(f"Available Vocabulary: {list(vocab.keys())}")
print("Type a prompt using the words above, or 'quit' to exit.\n")

while True:
    prompt = input("User Prompt > ")
    if prompt.lower() == "quit":
        break
        
    # Translate text to numbers
    idx = encode(prompt)
    if len(idx) == 0:
        continue
        
    # Convert to 2D PyTorch Tensor: Shape (Batch=1, Time)
    idx_tensor = torch.tensor(idx, dtype=torch.long).unsqueeze(0)
    
    # Run the Inference Engine! Generate up to 5 new words.
    generated_indices = model.generate(idx_tensor, max_new_tokens=5)
    
    # Extract the 1D sequence from the batch dimension
    generated_list = generated_indices[0].tolist()
    
    # Translate numbers back to text
    final_text = decode(generated_list)
    print(f"Automaton   > {final_text}\n")
