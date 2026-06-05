import torch
import math

# Let's create a tiny weight matrix (just 2 numbers)
W_manual = torch.tensor([0.5, -0.5], dtype=torch.float32)
W_pytorch = torch.tensor([0.5, -0.5], dtype=torch.float32, requires_grad=True)

# Let's pretend the backpropagation engine just handed us a slope (gradient)
gradient = torch.tensor([0.1, -0.2], dtype=torch.float32)

# ==========================================
# 1. THE BOTTOM-UP MANUAL MATH (Volume 1 Style)
# ==========================================
lr = 1e-3
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
weight_decay = 0.01

# Start momentum and velocity at 0
m = torch.tensor([0.0, 0.0], dtype=torch.float32)
v = torch.tensor([0.0, 0.0], dtype=torch.float32)

# AdamW Math for Step t=1
m = beta1 * m + (1 - beta1) * gradient
v = beta2 * v + (1 - beta2) * (gradient ** 2)

# Bias correction for time step t=1
m_hat = m / (1 - beta1**1)
v_hat = v / (1 - beta2**1)

# Apply Weight Decay and update weight
W_manual = W_manual - lr * (m_hat / (torch.sqrt(v_hat) + epsilon) + weight_decay * W_manual)

# ==========================================
# 2. THE PYTORCH BLACK BOX (Volume 2 Style)
# ==========================================
optimizer = torch.optim.AdamW([W_pytorch], lr=lr, betas=(beta1, beta2), eps=epsilon, weight_decay=weight_decay)

# Inject our fake gradient into the PyTorch tensor so it doesn't have to calculate it
W_pytorch.grad = gradient.clone()

# Press the Gas Pedal
optimizer.step()

print("=========================================")
print(f"Manual Raw Math:   {W_manual}")
print(f"PyTorch Black Box: {W_pytorch.data}")
print("=========================================")
