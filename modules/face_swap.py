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
    exclude_mouth=False
):
    swapped_images = []
    resize_min_resolution = True
    alpha = codeformer_alpha / 100  # convertir a [0–1]

    # 🔹 Inicializar CodeFormer (solo si está activado)
    codeformer_net, upsampler, device = None, None, None
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
        print("CodeFormer cargado ✅")

    # 🔹 Inicializar GPEN siempre
    sys.path.append("../inswapper/GPEN")
    from inswapper.GPEN.face_enhancement import FaceEnhancement
    faceenhancer = FaceEnhancement(
        size=512,
        model="GPEN-512",
        channel_multiplier=2,
        base_dir="./inswapper/GPEN"
    )
    print("GPEN FaceEnhancement cargado ✅")

    for item in images:
        # 🔹 Primero face swap si no estamos solo con restauración
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
            result_image = np.array(item)

        # 🔹 Escalar si es necesario
        if resize_min_resolution:
            h, w = result_image.shape[:2]
            if min(h, w) < 512:
                scale = 512 / min(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                result_image = cv2.resize(result_image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        # 🔹 Restauración
        if codeformer_enabled or only_codeformer:
            # --- Paso 1: Restauración con CodeFormer ---
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
            if restored.shape[:2] != result_image.shape[:2]:
                restored = cv2.resize(restored, (result_image.shape[1], result_image.shape[0]))
            result_image = cv2.addWeighted(restored, alpha, result_image, 1 - alpha, 0)

            # --- Paso 2: Pasar resultado por GPEN ---
            restored_gpen, _, _ = faceenhancer.process(result_image)
            if restored_gpen.shape[:2] != result_image.shape[:2]:
                restored_gpen = cv2.resize(restored_gpen, (result_image.shape[1], result_image.shape[0]))
            result_image = cv2.addWeighted(restored_gpen, alpha, result_image, 1 - alpha, 0)

        else:
            # --- Solo GPEN por defecto ---
            restored, _, _ = faceenhancer.process(result_image)
            if restored.shape[:2] != result_image.shape[:2]:
                restored = cv2.resize(restored, (result_image.shape[1], result_image.shape[0]))
            result_image = cv2.addWeighted(restored, alpha, result_image, 1 - alpha, 0)

        swapped_images.append(result_image)

    return swapped_images
