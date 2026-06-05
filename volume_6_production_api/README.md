# Volume 6: Production API (Dockerized)

This directory contains the production-ready API for the Nano-Llama-Engine, containerized via Docker.

## How to Run

1. Build the Docker image:
   ```bash
   docker build -t nano-gpt-api .
   ```

2. Run the Docker container:
   ```bash
   docker run -p 8000:8000 nano-gpt-api
   ```

3. The API will be available at `http://localhost:8000`. You can send a POST request to `http://localhost:8000/generate` with a JSON payload:
   ```json
   {
       "prompt": "O Romeo, Romeo! wherefore art thou "
   }
   ```
