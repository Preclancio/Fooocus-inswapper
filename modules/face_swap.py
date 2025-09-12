import sys
from PIL import Image
import numpy as np
import cv2

sys.path.append('../inswapper')
from inswapper.swapper import process

def perform_face_swap(
    images,
    only_codeformer=False,
    inswapper_source_image=None,
    inswapper_source_image_indicies=None,
    inswapper_target_image_indicies=None,
    codeformer_enabled=False,
    codeformer_fidelity=0.5,
    codeformer_alpha=50,
    codeformer_upscale=2,
    exclude_mouth=False   # 🔥 NUEVO: si es True, solo se aplica CodeFormer
):
    swapped_images = []
    resize_min_resolution = True
    codeformer_alpha = codeformer_alpha / 100  # convertir a rango [0–1]

    # Inicializar CodeFormer si se va a usar
    if codeformer_enabled or only_codeformer:
        from inswapper.restoration import face_restoration, check_ckpts, set_realesrgan, torch, ARCH_REGISTRY

        check_ckpts()
        upsampler = set_realesrgan()
        device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        print(f"CodeFormer en uso con dispositivo: {device}")

        codeformer_net = ARCH_REGISTRY.get("CodeFormer")(
            dim_embd=512,
            codebook_size=1024,
            n_head=8,
            n_layers=9,
            connect_list=["32", "64", "128", "256"],
        ).to(device)

        ckpt_path = "CodeFormer/CodeFormer/weights/CodeFormer/codeformer.pth"
        checkpoint = torch.load(ckpt_path, map_location=device)["params_ema"]
        codeformer_net.load_state_dict(checkpoint)
        codeformer_net.eval()

    for item in images:
        # 🔥 Si only_codeformer está activado, no hacemos swap
        if not only_codeformer and inswapper_source_image is not None:
            source_image = Image.fromarray(inswapper_source_image)
            result_image = process(
                [source_image], 
                item, 
                inswapper_source_image_indicies, 
                inswapper_target_image_indicies,
                "../inswapper/checkpoints/inswapper_128.onnx",
                exclude_mouth
            )
            result_image = np.array(result_image)
        else:
            # Si no hacemos swap, usamos la imagen original tal cual
            result_image = np.array(item)

        # Escalar si es necesario
        if resize_min_resolution:
            h, w = result_image.shape[:2]
            if min(h, w) < 512:
                scale = 512 / min(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                result_image = cv2.resize(result_image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        # Aplicar CodeFormer si está activado o si estamos en modo only_codeformer
        if codeformer_enabled or only_codeformer:
            restored = cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR)
            restored = face_restoration(
                restored, 
                True, 
                True, 
                codeformer_upscale,
                codeformer_fidelity,
                upsampler,
                codeformer_net,
                device
            )

            # Igualar tamaños si difieren
            if restored.shape[:2] != result_image.shape[:2]:
                restored = cv2.resize(restored, (result_image.shape[1], result_image.shape[0]))

            result_image = cv2.addWeighted(restored, codeformer_alpha, result_image, 1 - codeformer_alpha, 0)

        swapped_images.append(result_image)

    return swapped_images
