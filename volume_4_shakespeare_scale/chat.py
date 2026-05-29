import os
import torch

from nano_gpt import NanoGPT

# Setup device (Use GPU if available!)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# =============================================================
# 1. LOAD THE DYNAMIC VOCABULARY
# =============================================================
current_dir = os.path.dirname(__file__)
input_path = os.path.join(current_dir, 'input.txt')

if not os.path.exists(input_path):
    print("[ERROR] input.txt not found. Cannot build vocabulary.")
    exit()

with open(input_path, 'r', encoding='utf-8') as f:
    text = f.read()

# =============================================================
# DOCUMENTARY DEEP DIVE: THE CHARACTER-LEVEL TOKENIZER
# =============================================================
# 1. THE MATHEMATICAL FILTER
# `set(text)` acts as a mathematical filter on the 1MB Shakespeare string. 
# It instantly strips out every duplicate, leaving exactly 65 unique characters.
# We sort them so the dictionary is perfectly consistent every time.
chars = sorted(list(set(text)))

# 2. THE TWO-WAY DICTIONARY
# Python doesn't reverse-engineer tokenization. We explicitly build a two-way street!
# stoi (String-to-Integer) maps Letters to Numbers (e.g., {"a": 13})
# itos (Integer-to-String) maps Numbers to Letters (e.g., {13: "a"})
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

# 3. THE LAMBDA TRANSLATORS
# `lambda` is just Python's shortcut for a one-line function. 
# encode("cat") loops over the letters, looks up "c", "a", "t" in `stoi`, and returns [12, 39, 50].
# decode([12, 39, 50]) looks up the numbers in `itos`, returning ["c", "a", "t"].
# Finally, ''.join() uses an empty string (no spaces) to glue ["c", "a", "t"] into "cat".
# 
# -------------------------------------------------------------
# UNPACKED (NON-LAMBDA) VERSIONS FOR CLARITY:
# -------------------------------------------------------------
# def encode(text):
#     result = []
#     for char in text:
#         number = stoi.get(char, 0) # Fallback to 0 if character isn't found
#         result.append(number)
#     return result
#
# def decode(number_list):
#     result_letters = []
#     for number in number_list:
#         letter = itos.get(number, "?") # Fallback to "?" if number isn't found
#         result_letters.append(letter)
#     return ''.join(result_letters) # Glues the letters together with no spaces
# =============================================================
encode = lambda s: [stoi.get(c, 0) for c in s]
decode = lambda l: ''.join([itos.get(i, "?") for i in l])

# =============================================================
# 2. INITIALIZE THE MODEL & LOAD WEIGHTS
# =============================================================
# =============================================================
# DOCUMENTARY DEEP DIVE: THE BRAIN SURGERY
# =============================================================
# 1. We build an empty NanoGPT architectural shell (matrices with random numbers).
# 2. We use `model.to(device)` to push the empty shell onto the GPU if you have one.
# 3. We use `torch.load()` to grab the `shakespeare_gpt.pth` file from the hard drive.
#    (`map_location=device` ensures the weights go straight to the GPU to match the shell).
# 4. We use `load_state_dict()` to physically inject the trained numbers into the empty shell.
# 5. We call `model.eval()` to tell PyTorch: "We are taking a test. Do not learn anything. 
#    Turn off training-only features like Dropout."
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
# 3. THE INTERACTIVE CHAT LOOP
# =============================================================
print("\n=============================================")
print("VOLUME 4: THE SHAKESPEARE INFERENCE ENGINE")
print("=============================================")
print("Type a starting prompt, or 'quit' to exit.\n")

# =============================================================
# DOCUMENTARY DEEP DIVE: THE AUTOREGRESSIVE ENGINE
# =============================================================
# When you type a prompt (e.g., "O Romeo"), here is the exact sequence of events:
# 1. Translate to Math: We `encode` your text into a python list of numbers: [43, 0, 50, ...]
# 2. Convert to Tensor: PyTorch doesn't read python lists. We convert it to a 
#    2D Tensor and push it to the GPU. The shape is (Batch=1, Time).
# 3. Generate: We feed the tensor into model.generate(). The model looks at the text, 
#    predicts the next letter, appends it to the list, and repeats 500 times.
# 4. Translate to English: We `decode` the final list of numbers back into a string!
# =============================================================
while True:
    prompt = input("Prompt > ")
    if prompt.lower() == "quit":
        break
        
    idx = encode(prompt)
    if len(idx) == 0:
        continue
        
    idx_tensor = torch.tensor(idx, dtype=torch.long, device=device).unsqueeze(0)
    
    # Generate 500 characters
    print("\n[GENERATING...]\n")
    generated_indices = model.generate(idx_tensor, max_new_tokens=500)
    
    generated_list = generated_indices[0].tolist()
    final_text = decode(generated_list)
    
    print("====================")
    print(final_text)
    print("====================\n")
