# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies (ignoring cache to keep image small)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code, weights, and vocab
COPY . .

# Expose port 8000 for the FastAPI server
EXPOSE 8000

# Run uvicorn when the container launches
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
