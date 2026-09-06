#!/bin/bash

echo "🚀 Iniciando configuración de entorno..."

# 1. Ajustes iniciales
pip install "pip<24"
git lfs install

# 2. Descargar CodeFormer dentro de inswapper (verifica si ya existe para no duplicar)
mkdir -p inswapper
cd inswapper
if [ ! -d "CodeFormer" ]; then
    echo "📥 Clonando CodeFormer..."
    git clone https://huggingface.co/spaces/sczhou/CodeFormer
fi
cd ..

# 3. Instalar dependencias
echo "📦 Instalando requerimientos de Fooocus..."
pip install -r requirements_versions.txt

echo "📦 Instalando PyTorch..."
# Nota: cu118 es para CUDA 11.8. Funciona bien, pero si tu contenedor usa CUDA 12.1, 
# la propia instalación base de Fooocus suele ajustarlo automáticamente.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 4. Solución inteligente para basicsr y facelib
echo "Copiando basicsr y facelib a la ruta correcta de Python..."
# Esto le pregunta a Python exactamente dónde guarda sus librerías
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")

cp -r inswapper/CodeFormer/CodeFormer/basicsr "$SITE_PACKAGES/"
cp -r inswapper/CodeFormer/CodeFormer/facelib "$SITE_PACKAGES/"

# 5. Descarga de modelo Inswapper
echo "📥 Descargando modelo Inswapper..."
mkdir -p inswapper/checkpoints
# $PWD detecta dinámicamente tu carpeta actual, evitando el uso de /content/
wget -c https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx -O "$PWD/inswapper/checkpoints/inswapper_128.onnx"

# 6. Limpieza de carpetas no utilizadas
echo "🧹 Limpiando directorios innecesarios..."
rm -rf "$PWD/InstantID"
rm -rf "$PWD/photomaker"

echo "✅ Instalación y configuración completadas exitosamente."