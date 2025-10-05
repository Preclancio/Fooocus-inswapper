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
    codeformer_enabled=False,  # TRUE = GPEN + CodeFormer, FALSE = Solo CodeFormer
    codeformer_fidelity=0.5,
    codeformer_alpha=50,
    codeformer_upscale=2,
    exclude_mouth=False,
    use_selfie_enhancer=True,
    res_percentage=150
):
    swapped_images = []
    resize_min_resolution = True
    alpha = codeformer_alpha / 100  # convertir a [0–1]

    # 🔹 Inicializar CodeFormer (siempre se usa)
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

    # 🔹 Inicializar GPEN (solo si codeformer_enabled=True)
    faceenhancer = None
    if codeformer_enabled:
        sys.path.append("../inswapper/GPEN")
        from inswapper.GPEN.face_enhancement import FaceEnhancement
        
        # Elegir modelo según use_selfie_enhancer
        if use_selfie_enhancer:
            # Modelo selfie - 1024px (alta resolución)
            faceenhancer = FaceEnhancement(
                size=1024,
                model="GPEN-BFR-1024",
                channel_multiplier=2,
                base_dir="./inswapper/GPEN"
            )
            print("GPEN Selfie Enhancer (1024px) cargado ✅")
        else:
            # Modelo normal - 512px
            faceenhancer = FaceEnhancement(
                size=512,
                model="GPEN-512",
                channel_multiplier=2,
                base_dir="./inswapper/GPEN"
            )
            print("GPEN Normal Enhancer (512px) cargado ✅")

    for item in images:
        # Convertir a numpy array si es necesario
        if isinstance(item, Image.Image):
            item = np.array(item)
            
        # 🔹 PREPROCESAMIENTO: Aumentar resolución (150% por defecto)
        if res_percentage != 100:
            scale_factor = res_percentage / 100.0
            original_shape = item.shape
            item = cv2.resize(item, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
            print(f"Imagen redimensionada: {original_shape[:2]} → {item.shape[:2]} (escala: {scale_factor}x)")

        # 🔹 Primero face swap si no estamos solo con restauración
        if not only_codeformer and inswapper_source_image is not None:
            source_image = Image.fromarray(inswapper_source_image)
            result_image = process(
                [source_image],
                Image.fromarray(item),
                inswapper_source_image_indicies,
                inswapper_target_image_indicies,
                "../inswapper/checkpoints/inswapper_128.onnx",
                exclude_mouth
            )
            result_image = np.array(result_image)
        else:
            result_image = item.copy()

        # 🔹 Escalar si es necesario (solo si la imagen es muy pequeña)
        if resize_min_resolution:
            h, w = result_image.shape[:2]
            if min(h, w) < 512:
                scale = 512 / min(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                result_image = cv2.resize(result_image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        # 🔹 RESTAURACIÓN - LÓGICA SIMPLIFICADA
        if codeformer_enabled:
            # --- CASO 1: GPEN + CODEFORMER ---
            print("Usando GPEN + CodeFormer para restauración")
            
            # PASO 1: Aplicar GPEN primero
            restored_gpen, _, _ = faceenhancer.process(result_image)
            if restored_gpen.shape[:2] != result_image.shape[:2]:
                restored_gpen = cv2.resize(restored_gpen, (result_image.shape[1], result_image.shape[0]))
            
            # Mezclar GPEN con original
            result_with_gpen = cv2.addWeighted(restored_gpen, alpha, result_image, 1 - alpha, 0)

            # PASO 2: Aplicar CodeFormer sobre el resultado de GPEN
            restored_codeformer = cv2.cvtColor(result_with_gpen, cv2.COLOR_RGB2BGR)
            restored_codeformer = face_restoration(
                restored_codeformer,
                True,
                True,
                codeformer_upscale,
                codeformer_fidelity,
                upsampler,
                codeformer_net,
                device
            )
            
            # Redimensionar si es necesario
            if restored_codeformer.shape[:2] != result_with_gpen.shape[:2]:
                restored_codeformer = cv2.resize(restored_codeformer, (result_with_gpen.shape[1], result_with_gpen.shape[0]))
            
            # Mezcla final: CodeFormer + Resultado-GPEN
            result_image = cv2.addWeighted(restored_codeformer, alpha, result_with_gpen, 1 - alpha, 0)

        else:
            # --- CASO 2: SOLO CODEFORMER (por defecto) ---
            print("Usando SOLO CodeFormer para restauración")
            
            # Aplicar solo CodeFormer
            restored_codeformer = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
            restored_codeformer = face_restoration(
                restored_codeformer,
                True,
                True,
                codeformer_upscale,
                codeformer_fidelity,
                upsampler,
                codeformer_net,
                device
            )
            
            # Redimensionar si es necesario
            if restored_codeformer.shape[:2] != result_image.shape[:2]:
                restored_codeformer = cv2.resize(restored_codeformer, (result_image.shape[1], result_image.shape[0]))
            
            # Aplicar mezcla con alpha
            result_image = cv2.addWeighted(restored_codeformer, alpha, result_image, 1 - alpha, 0)

        swapped_images.append(result_image)

    return swapped_images
