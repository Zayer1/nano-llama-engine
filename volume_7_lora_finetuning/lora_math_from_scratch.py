import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LoRALinear(nn.Module):
    """
    Low-Rank Adaptation (LoRA) implemented from scratch.
    """
    def __init__(self, in_features: int, out_features: int, r: int = 8, lora_alpha: int = 16, pretrained_weight: torch.Tensor = None):
        super().__init__()
        
        # 1. THE MASSIVE FOUNDATION (Frozen)
        # If we hot-swap into a real LLM, we MUST copy its pre-trained brain over.
        if pretrained_weight is not None:
            self.weight = nn.Parameter(pretrained_weight.clone())
        else:
            self.weight = nn.Parameter(torch.randn(out_features, in_features))
            
        self.weight.requires_grad = False 
        
        # 2. THE LORA STEERING WHEEL (Trainable)
        self.r = r
        
        # --------------------------------------------------------------------------------
        # --- THE VOLUME KNOB (SCALING) ---
        # --------------------------------------------------------------------------------
        # The scaling factor is mathematically defined as: lora_alpha / r
        # 
        # Why divide by r?
        # This stabilizes the math. If we increase the rank 'r' to make the model smarter,
        # the raw sum of (A * B) gets larger. By dividing by 'r', we ensure the 
        # gradients don't explode when we change the architecture size.
        # 
        # What is lora_alpha?
        # It is the "Volume Knob" for the fine-tuning. If it is too high, the model forgets 
        # English and just screams medical gibberish (Catastrophic Forgetting). 
        # The B2B industry standard rule is lora_alpha = 2 * r.
        self.scaling = lora_alpha / r
        
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        
        # --------------------------------------------------------------------------------
        # --- SYMMETRY BREAKING & STEP 0 STABILITY ---
        # --------------------------------------------------------------------------------
        # Matrix A is filled with random numbers. Matrix B is filled with perfect ZEROS.
        # 
        # Why zeros? 
        # At Step 0 of training, (x @ A @ B) equals exactly ZERO. 
        # Final Output = base_out + 0.
        # This guarantees our layer acts exactly like the original pre-trained Llama 
        # before training begins, avoiding an instant garbage-output shock to the loss function.
        # 
        # But doesn't a ZERO matrix kill the learning (Dead Neuron problem)? 
        # No! Because Matrix A is random, the input into Matrix B (x @ A) is non-zero.
        # By the Chain Rule of calculus: Gradient_B = (Input to B)^T @ Error_Signal. 
        # Because the input to B is non-zero, the calculated Gradient_B is non-zero.
        # 
        # On Step 1, the AdamW Optimizer executes: 
        # New_B = 0.0 - (learning_rate * Gradient_B).
        # Matrix B wakes up instantly, symmetry is broken, and the bypass begins steering.
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The Math: Output = (x @ W.T) + (x @ A.T @ B.T) * scaling
        """
        # Step 1: Calculate the base output through the massive frozen W matrix.
        # Because W is frozen, this preserves 100% of the model's original intelligence (grammar, logic).
        base_out = F.linear(x, self.weight)
        
        # Step 2: Calculate the bottleneck pass. 
        # First, compress the 4096 dimensions into the tiny bottleneck 'r' (e.g., 8).
        compressed_x = F.linear(x, self.lora_A)
        # Next, decompress those 8 dimensions back out to 4096.
        lora_out = F.linear(compressed_x, self.lora_B)
        
        # Step 3: Scale the bottleneck and ADD them together.
        # The frozen base engine (grammar) and the tiny steering wheel (medical knowledge) are combined.
        scaled_out = lora_out * self.scaling + base_out
        
        return scaled_out 

if __name__ == "__main__":
    print("Testing LoRA Math...")
    layer = LoRALinear(in_features=4096, out_features=4096, r=8)
    dummy_x = torch.randn(2, 10, 4096)
    
    out = layer(dummy_x)
    print(f"Input shape:  {dummy_x.shape}")
    print(f"Output shape: {out.shape}")
    if out.shape == dummy_x.shape:
        print("MATH SUCCESS! The shape is perfectly preserved.")
