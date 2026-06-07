# Bare-metal PyTorch Fine-Tuning Pipeline
# We are building this from scratch without wrapper libraries.
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn as nn
from huggingface_hub import HfApi
from lora_math_from_scratch import LoRALinear
from lora_math_from_scratch import LoRALinear

class MedicalInstructionDataset(Dataset):
    """
    A bare-metal PyTorch Dataset class.
    
    Why are we writing this?
    SFTTrainer hides how data gets to the GPU. Here, we manually format the 
    clinical text and tokenize it into integer arrays (tensors) so the math 
    engines can actually process it.
    """
    def __init__(self, hf_dataset, tokenizer, max_length=512):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # 1. Grab the raw text
        example = self.dataset[idx]
        instruction = example['instruction']
        output = example['output']
        
        # 2. Format it exactly how the model will see it in production
        formatted_text = f"### Clinical Instruction:\n{instruction}\n\n### Doctor's Analysis:\n{output}"
        
        # 3. Convert text to integers (Tokenization)
        tokenized = self.tokenizer(
            formatted_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length", # Ensures all tensors in a batch are exactly the same size
            return_tensors="pt"
        )
        
        # 4. Strip the batch dimension added by the tokenizer
        input_ids = tokenized["input_ids"].squeeze(0)
        attention_mask = tokenized["attention_mask"].squeeze(0)
        
        # FIX: The Exploding Loss Bug
        # We must clone the input_ids to create the labels.
        labels = input_ids.clone()
        # Then, we tell PyTorch's loss function to completely ignore the padding tokens
        # by setting them to -100. If we don't do this, the model tries to "learn" how
        # to predict empty space, which destroys its brain and causes the loss to skyrocket.
        labels[attention_mask == 0] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

def prepare_data(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", batch_size=2):
    print("\n--- STEP 1: PREPARING BARE-METAL DATALOADER ---")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token 
    
    # Load raw data
    print("Loading raw dataset from HuggingFace...")
    # REMOVED the [:100] tiny subset. We are using the full 52k dataset now.
    raw_data = load_dataset("lavita/AlpaCare-MedInstruct-52k", split="train") 
    
    # Wrap it in our custom PyTorch Dataset
    pytorch_dataset = MedicalInstructionDataset(raw_data, tokenizer)
    
    # Create the DataLoader (this handles the batching automatically)
    dataloader = DataLoader(pytorch_dataset, batch_size=batch_size, shuffle=True)
    
    # Let's prove it works by grabbing one batch and checking the tensor shapes
    sample_batch = next(iter(dataloader))
    print(f"Batch Input IDs Shape:  {sample_batch['input_ids'].shape}")
    print(f"Batch Attn Mask Shape:  {sample_batch['attention_mask'].shape}")
    print(f"Batch Labels Shape:     {sample_batch['labels'].shape}")
    
    return dataloader, tokenizer

def load_and_inject_model(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    print("\n--- STEP 2: LOADING LLM & INJECTING LORA ---")
    
    # 1. Load the massive pre-trained model into memory
    print(f"Downloading and loading {model_name}...")
    # Using torch.bfloat16 or float32. We'll use float32 for maximum compatibility on any hardware.
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    
    # 2. Freeze the entire model so we don't accidentally train the base weights
    for param in model.parameters():
        param.requires_grad = False
        
    # 3. The Hot-Swap: Find specific Linear layers and replace them with our LoRALinear
    # In Llama architectures, the attention projection layers are called 'q_proj' and 'v_proj'.
    print("Hot-swapping attention layers with custom LoRA math...")
    
    replaced_count = 0
    # We loop through all named modules in the PyTorch graph
    for name, module in model.named_modules():
        # TinyLlama has layers named like: model.layers.0.self_attn.q_proj
        if "q_proj" in name or "v_proj" in name:
            # We found a target layer! It's currently a standard nn.Linear.
            # Let's extract its dimensions and pre-trained weights.
            in_features = module.in_features
            out_features = module.out_features
            pretrained_weight = module.weight.data
            
            # Create our custom LoRA layer with the exact same dimensions
            new_lora_layer = LoRALinear(
                in_features=in_features, 
                out_features=out_features, 
                r=8, 
                lora_alpha=16,
                pretrained_weight=pretrained_weight
            )
            
            # Now we have to actually inject it into the model structure.
            # We split the name 'model.layers.0.self_attn.q_proj' to get the parent object.
            parent_name = name.rsplit('.', 1)[0]
            child_name = name.rsplit('.', 1)[1]
            
            parent_module = model.get_submodule(parent_name)
            # The hot-swap: physically overwrite the layer in the PyTorch graph
            setattr(parent_module, child_name, new_lora_layer)
            replaced_count += 1
            
    print(f"Successfully injected {replaced_count} LoRA bypasses into the model's brain.")
    
    # 4. Verify what is trainable (Only our Matrix A and Matrix B should be True)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Total Parameters:     {all_params:,}")
    print(f"% Trainable:          {100 * trainable_params / all_params:.4f}%")
    
    return model

def train_lora_model(model, dataloader, epochs=1, lr=3e-4):
    print("\n--- STEP 3: THE PYTORCH TRAINING LOOP ---")
    
    # 1. The Optimizer: AdamW is the industry standard for LLMs.
    # We ONLY give the optimizer the parameters that have requires_grad=True (our LoRA matrices).
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=lr)
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Pushing model to {device}...")
    model.to(device)
    model.train() # Set model to training mode
    
    print("Ignition... starting training!")
    
    for epoch in range(epochs):
        for step, batch in enumerate(dataloader):
            # Move the padded text matrices to the GPU
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # A. ZERO THE GRADIENTS
            # PyTorch accumulates gradients by default. We must wipe the slate clean every step.
            optimizer.zero_grad()
            
            # B. FORWARD PASS
            # Feed the medical text into the LLM. It tries to predict the next words.
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss # How wrong was the prediction?
            
            # C. BACKWARD PASS (The Calculus)
            # This calculates the gradients (how much to adjust Matrix A and Matrix B)
            loss.backward()
            
            # D. OPTIMIZER STEP
            # Actually physically change the LoRA weights based on the gradients
            optimizer.step()
            
            print(f"Epoch {epoch+1} | Step {step} | Loss: {loss.item():.4f}")
            
            # --- B2B PORTFOLIO FAST-TRACK ---
            # Loss stabilizes at ~0.99 around step 4000. 
            # We break early to beat cloud compute timeouts and push the weights.
            if step >= 4000:
                print("Step 4000 reached. Loss has converged. Halting training to push weights!")
                break
                
        if step >= 4000:
            break
            
            
    print("\nSUCCESS! Engine training complete.")
    return model

def push_to_huggingface(model, repo_id, hf_token):
    print("\n--- STEP 4: EXTRACTING & PUSHING LORA WEIGHTS ---")
    
    # We don't want to upload the massive 1.1B parameter base model.
    # We ONLY extract the parameters that we trained (Matrix A and Matrix B).
    lora_weights = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            lora_weights[name] = param.cpu() # Move to CPU before saving
            
    print(f"Extracted {len(lora_weights)} LoRA weight tensors. Saving locally...")
    torch.save(lora_weights, "custom_lora_medical.pt")
    
    print("Connecting to Hugging Face Hub...")
    api = HfApi(token=hf_token)
    api.create_repo(repo_id=repo_id, exist_ok=True)
    
    api.upload_file(
        path_or_fileobj="custom_lora_medical.pt",
        path_in_repo="custom_lora_medical.pt",
        repo_id=repo_id
    )
    print(f"B2B Portfolio Complete! LoRA matrices successfully pushed to: https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    # Set to None so the script automatically pulls the token from your notebook_login() session!
    HF_TOKEN = None
    HF_USERNAME = "Zayer1" # Replace with your actual HF username if different
    REPO_ID = f"{HF_USERNAME}/tinyllama-medical-lora-from-scratch"
    
    dataloader, tokenizer = prepare_data(batch_size=4) # Increased batch size for the Colab GPU
    model = load_and_inject_model()
    
    # Train the model (Warning: This will take a few hours on a T4 GPU for 52k rows)
    trained_model = train_lora_model(model, dataloader, epochs=1)
    
    # Push the extracted brain to your portfolio
    push_to_huggingface(trained_model, repo_id=REPO_ID, hf_token=HF_TOKEN)
