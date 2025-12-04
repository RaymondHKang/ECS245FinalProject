# .devcontainer/Dockerfile
FROM nvcr.io/nvidia/tensorflow:24.12-tf2-py3

# Set a consistent workdir inside the container
WORKDIR /workspaces/ECS245FinalProject

# Install Python dependencies *during build* for reproducibility
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Optional: install extra OS tools you commonly need
# RUN apt-get update && apt-get install -y git vim && rm -rf /var/lib/apt/lists/*
