import os
import sys
import torch

# 1. LOAD THE ARCHITECTURE FROM VOLUME 2
current_dir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))
from volume_2_pytorch_automaton.nano_gpt import NanoGPT, vocab

# =============================================================
# DOCUMENTARY DEEP DIVE: THE DICTIONARY FLIP
# =============================================================
# In Volume 2, we created vocab = {"the": 0, "cat": 1} to translate 
# English into Numbers so the math engine could process it.
# Now, the AI is going to output numbers (e.g., [1]), and we need 
# to translate that back into English so humans can read it. 
# We flip the dictionary backward so it becomes {0: "the", 1: "cat"}.
# =============================================================
reverse_vocab = {i: word for word, i in vocab.items()}

# 2. INITIALIZE THE MODEL & LOAD WEIGHTS
# =============================================================
# DOCUMENTARY DEEP DIVE: THE BRAIN SURGERY
# =============================================================
# 1. We build an empty NanoGPT architectural shell (matrices with random numbers).
# 2. We use torch.load() to grab the nano_gpt.pth file from the hard drive.
# 3. We use load_state_dict() to physically inject the trained numbers into the empty shell.
# 4. We call model.eval() to tell PyTorch: "We are taking a test. Do not learn anything. 
#    Turn off training-only features like Dropout."
# =============================================================
model = NanoGPT()
weights_path = os.path.abspath(os.path.join(current_dir, '..', 'volume_2_pytorch_automaton', 'nano_gpt.pth'))

if os.path.exists(weights_path):
    model.load_state_dict(torch.load(weights_path, weights_only=True))
    print("[SYSTEM] Trained weights loaded successfully!")
else:
    print("[SYSTEM] WARNING: nano_gpt.pth not found. Model will hallucinate.")

model.eval()

# =============================================================
# DOCUMENTARY DEEP DIVE: THE TRANSLATORS
# =============================================================
# encode("the cat") uses the dictionary to return the math list [0, 1].
# decode([0, 1]) uses the reverse dictionary to return the string "the cat".
# =============================================================
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

# =============================================================
# DOCUMENTARY DEEP DIVE: THE AUTOREGRESSIVE ENGINE
# =============================================================
# When you type "the", here is the exact sequence of events:
# 1. Translate to Math: We `encode` your text into a python list: [0]
# 2. Convert to Tensor: PyTorch doesn't read python lists. We convert it to 
#    a 2D Tensor torch.tensor([[0]]). The shape is (Batch=1, Time=1).
# 3. Generate: We feed [[0]] into model.generate(). The model looks at 0, 
#    predicts 1, appends it to the list to make [[0, 1]], and repeats.
# 4. Translate to English: We `decode` the final list of numbers back into a string!
# =============================================================
while True:
    prompt = input("User Prompt > ")
    if prompt.lower() == "quit":
        break
        
    # 1. Translate text to numbers
    idx = encode(prompt)
    if len(idx) == 0:
        continue
        
    # 2. Convert to 2D PyTorch Tensor: Shape (Batch=1, Time)
    idx_tensor = torch.tensor(idx, dtype=torch.long).unsqueeze(0)
    
    # 3. Run the Inference Engine! Generate up to 5 new words.
    generated_indices = model.generate(idx_tensor, max_new_tokens=5)
    
    # Extract the 1D sequence from the batch dimension
    generated_list = generated_indices[0].tolist()
    
    # 4. Translate numbers back to text
    final_text = decode(generated_list)
    print(f"Automaton   > {final_text}\n")
