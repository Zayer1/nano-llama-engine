# Medical Engine Inference (Testing the AI)

You successfully trained the adapter! The heavy lifting is done. Now we need to actually talk to it and see what you built.

To test the model, we need an **Inference Script**. This script will run on Colab and do the following:

## Proposed Pipeline:
1. **Load the Frozen Base Engine:** Load the exact same `TinyLlama/TinyLlama-1.1B-Chat-v1.0` model in 4-bit quantization, just like we did for training.
2. **Mount the Custom Brain:** We will load the PeftModel using your custom `fine_tuned_LoRA` adapter folder. This attaches your trained Matrix A and Matrix B bypasses to the base model.
3. **Format the Prompt:** We will take a raw medical symptom (e.g., "Patient presents with severe chest pain and shortness of breath") and format it using the exact `### Clinical Instruction:\n` format the model was trained on.
4. **Generate Inference:** We will pass the prompt into the GPU, let the model compute the response, and decode the output.

## User Review Required

> [!IMPORTANT]
> The inference script will require you to define a system prompt and a generation config (max_new_tokens, temperature).
> Are you ready to build the inference script and see your creation speak? Approve this plan and we will write the final file.
