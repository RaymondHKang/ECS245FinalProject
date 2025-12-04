# .devcontainer/Dockerfile
FROM python:3.11-slim

# System setup: you can trim this if you don't need build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
 && rm -rf /var/lib/apt/lists/*

# Workdir inside container
WORKDIR /workspaces/ECS245FinalProject

# Install Python deps at build time
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt