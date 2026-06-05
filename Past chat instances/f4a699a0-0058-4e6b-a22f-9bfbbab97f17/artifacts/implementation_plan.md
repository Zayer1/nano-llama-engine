# Volume 6: The Researcher's Bare Minimum Plumbing

As an AI Researcher (AI RE/R), your primary focus is on novel architectures, math, and CUDA kernels. You do not want to become a DevOps engineer. However, to fund your research via high-paying B2B contracts, you must know how to hand your model to a startup's engineering team in a usable format.

This volume teaches the **absolute bare minimum plumbing** required to maintain your integrity while proving you can ship production code. 

## User Review Required
Please review this philosophical and technical division of labor. If you agree with this approach, approve the plan and we will execute!

## Proposed Changes & Division of Labor

We will isolate the plumbing into a new directory: `e:\Antigravity\Antigravity\Projects\volume_6_production_api`

### Phase 1: Infrastructure Boilerplate (Done by Agent)
I will handle the repetitive DevOps tasks so you don't have to waste brain power on them:
1. **Directory Setup:** Create `volume_6_production_api` and copy your core research files (`nano_gpt.py`, `shakespeare_gpt.pth`, `input.txt`) into it.
2. **[NEW] `requirements.txt`:** Create the minimal dependency lockfile (`fastapi`, `uvicorn`, `torch`).
3. **[NEW] `Dockerfile`:** Write the configuration to put your engine inside an unbreakable container. This is how you hand the model to a startup's DevOps team so they can handle the cloud architecture.
4. **[NEW] `README.md`:** Brief instructions for the CTO on how to run your container.

### Phase 2: The Core API "Steering Wheel" (Done by User)
You will manually write the only piece of plumbing a Researcher actually needs to understand: the API endpoint.
- **[NEW] `main.py`:** I will provide the raw code for a FastAPI server. You will copy/paste it and review the logic. The code will be heavily documented (per your rules) explaining how a JSON string is parsed, converted into a PyTorch tensor, and fed into your engine. 

## Verification Plan
1. We will install the requirements and run the FastAPI server locally.
2. We will test the API endpoint to ensure your math correctly processes the incoming JSON request.
3. Once verified, you have officially learned everything required to fund your research, and you can begin your Gaming Megasprint!
