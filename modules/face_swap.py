import sys
from PIL import Image
import numpy as np
import cv2

sys.path.append('../inswapper')
from inswapper.swapper import process

def perform_face_swap(
    images,
    inswapper_source_image,
    inswapper_source_image_indicies,
    inswapper_target_image_indicies,
    codeformer_enabled=False,
    codeformer_fidelity=0.5,
    codeformer_alpha=50,
    exclude_mouth=False
):
    swapped_images = []
    resize_min_resolution = True
    codeformer_alpha=codeformer_alpha/100     # <--- NUEVO: porcentaje de CodeFormer

    # Si se usa CodeFormer, inicialízalo una sola vez
    if codeformer_enabled:
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
        source_image = Image.fromarray(inswapper_source_image)
        print(f"Inswapper: Source indices: {inswapper_source_image_indicies}")
        print(f"Inswapper: Target indices: {inswapper_target_image_indicies}")      

        result_image = process(
            [source_image], 
            item, 
            inswapper_source_image_indicies, 
            inswapper_target_image_indicies,
            "../inswapper/checkpoints/inswapper_128.onnx",
            exclude_mouth
        )

        result_image = np.array(result_image)

        # Escalar si es necesario
        if resize_min_resolution:
            h, w = result_image.shape[:2]
            if min(h, w) < 512:
                scale = 512 / min(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                result_image = cv2.resize(result_image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        # Aplicar CodeFormer si está activado
        if codeformer_enabled:
            # Restaurar con CodeFormer
            restored = cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR)
            restored = face_restoration(
                restored, 
                True, 
                True, 
                1, 
                codeformer_fidelity,
                upsampler,
                codeformer_net,
                device
            )

            # Mezclar restaurado y original con alpha
            result_image = cv2.addWeighted(restored, codeformer_alpha, result_image, 1 - codeformer_alpha, 0)

        swapped_images.append(result_image)

    return swapped_images
