# Use a CUDA base image
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set the working directory
WORKDIR /code

# Install Python and other necessary packages
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the code to the working directory
COPY . /code

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Install ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

RUN python3 required_downloads.py

CMD ["ollama", "serve"]
