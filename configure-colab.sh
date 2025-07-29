#!/bin/bash
pip install "pip<24"
# Clone the repository
git clone https://github.com/haofanwang/inswapper.git
cd inswapper

# Install Git LFS
git lfs install

# Clone the Hugging Face model
git clone https://huggingface.co/spaces/sczhou/CodeFormer

# Move back to the parent directory
cd ..

# Install Python dependencies
pip install onnxruntime-gpu
pip install -r requirements_versions.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Copy directories
echo "Copying basicsr"
cp -r inswapper/CodeFormer/CodeFormer/basicsr /usr/local/lib/python*/dist-packages
echo "Copying facelib"
cp -r inswapper/CodeFormer/CodeFormer/facelib /usr/local/lib/python*/dist-packages

# Create a directory for checkpoints
mkdir -p inswapper/checkpoints

# Download the ONNX model
gdown --id 1krOLgjW2tAPaqV-Bw4YALz0xT5zlb5HF -O inswapper/checkpoints/inswapper_128.onnx

# Setup InstantID
mkdir -p InstantID/models/antelopev2
# Descargar el archivo ZIP con gdown (sin usar --id porque está deprecated)
gdown "https://drive.google.com/uc?id=18wEUfMNohBJ4K3Ly5wpTejPfDzp-8fI8" -O /content/Fooocus-inswapper/InstantID/models/antelopev2/antelopev2.zip
unzip /content/Fooocus-inswapper/InstantID/models/antelopev2/antelopev2.zip -d /content/Fooocus-inswapper/InstantID/models/antelopev2/

