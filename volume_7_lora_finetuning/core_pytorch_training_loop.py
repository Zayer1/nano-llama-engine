import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def format_medical_prompt(example):
    instruction = example['instruction']
    output = example['output']
    return {"text": f"### Clinical Instruction:\n{instruction}\n\n### Doctor's Analysis:\n{output}"}

def run_my_manual_pipeline():
    print("="*50)
    print("STARTING CUSTOM MANUAL B2B MEDICAL LORA PIPELINE")
    print("="*50)

    # 1. LOAD MODEL
    print("\n[Step 1] Loading Base Engine...")
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token 
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quant_config, device_map="auto")
    model = prepare_model_for_kbit_training(model)

    # 2. INJECT LORA
    print("\n[Step 2] Injecting LoRA...")
    lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_config)

    # 3. LOAD DATASET
    print("\n[Step 3] Loading Dataset...")
    dataset = load_dataset("lavita/AlpaCare-MedInstruct-52k", split="train[:50]")
    dataset = dataset.map(format_medical_prompt)

    # =====================================================================
    # YOUR TURN: WRITE THE MANUAL TRAINING LOOP BELOW
    # =====================================================================
    print("\n[Step 4] Configuring Optimizer and Dataloader...")
    
    # TODO: 1. Create a custom collate function to tokenize the text into PyTorch tensors
    
    # TODO: 2. Create the PyTorch DataLoader
    
    # TODO: 3. Initialize the Optimizer (torch.optim.AdamW)
    
    # TODO: 4. Write the Epoch Loop
    
        # TODO: 5. Write the Batch Loop
        
            # TODO: 6. Forward Pass (Calculate Loss)
            
            # TODO: 7. Backward Pass (loss.backward())
            
            # TODO: 8. Optimizer Step
            
            # TODO: 9. Zero Gradients
            
    # =====================================================================

    print("\n[Step 5] Saving the manual adapter...")
    model.save_pretrained("./my_medical_lora")
    tokenizer.save_pretrained("./my_medical_lora")

if __name__ == "__main__":
    run_my_manual_pipeline()
