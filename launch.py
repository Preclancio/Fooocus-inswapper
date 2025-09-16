import os
import ssl
import sys
import platform
import fooocus_version

# --- Setup inicial ---
print('[System ARGV]', sys.argv)

root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root)
os.chdir(root)

# Habilitar fallback en MPS (solo Mac)
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# Fijar puerto de Gradio si no existe
os.environ.setdefault("GRADIO_SERVER_PORT", "7865")

# Evitar errores SSL en Colab / entornos no certificados
ssl._create_default_https_context = ssl._create_unverified_context

print(f"Python {sys.version}")
print(f"Fooocus version: {fooocus_version.version}")

# --- Cargar argumentos ---
from args_manager import args
if args.gpu_device_id is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_device_id)
    print("Set device to:", args.gpu_device_id)

if args.hf_mirror is not None:
    os.environ['HF_MIRROR'] = str(args.hf_mirror)
    print("Set hf_mirror to:", args.hf_mirror)

# --- Configuración de Fooocus ---
from modules import config
from modules.model_loader import load_file_from_url
from modules.hash_cache import init_cache

os.environ["U2NET_HOME"] = config.path_inpaint
os.environ["GRADIO_TEMP_DIR"] = config.temp_path

# Limpieza opcional de carpeta temporal
from modules.launch_util import delete_folder_content
if config.temp_path_cleanup_on_launch:
    print(f'[Cleanup] Deleting temp dir: {config.temp_path}')
    delete_folder_content(config.temp_path, '[Cleanup] ')

# --- Descarga de modelos si es necesario ---
vae_approx_filenames = [
    ('xlvaeapp.pth', 'https://huggingface.co/lllyasviel/misc/resolve/main/xlvaeapp.pth'),
    ('vaeapp_sd15.pth', 'https://huggingface.co/lllyasviel/misc/resolve/main/vaeapp_sd15.pt'),
    ('xl-to-v1_interposer-v4.0.safetensors',
     'https://huggingface.co/mashb1t/misc/resolve/main/xl-to-v1_interposer-v4.0.safetensors')
]

def download_models():
    for file_name, url in vae_approx_filenames:
        load_file_from_url(url=url, model_dir=config.path_vae_approx, file_name=file_name)
    load_file_from_url(
        url='https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_expansion.bin',
        model_dir=config.path_fooocus_expansion,
        file_name='pytorch_model.bin'
    )

download_models()

config.update_files()
init_cache(config.model_filenames, config.paths_checkpoints, config.lora_filenames, config.paths_loras)

# --- Lanzar webui ---
from webui import *
    
