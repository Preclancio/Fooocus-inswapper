import cv2
import numpy as np
from PIL import Image
import torch
from transformers import pipeline

# Variable global para no recargar el modelo en cada inferencia
depth_pipe = None


def get_depth_pipeline():
    global depth_pipe
    if depth_pipe is None:
        device = 0 if torch.cuda.is_available() else -1
        depth_pipe = pipeline(
            task="depth-estimation",
            model="depth-anything/Depth-Anything-V2-Large-hf",
            device=device,
        )
    return depth_pipe


def centered_canny(x: np.ndarray, canny_low_threshold, canny_high_threshold):
    assert isinstance(x, np.ndarray)
    assert x.ndim == 2 and x.dtype == np.uint8

    y = cv2.Canny(x, int(canny_low_threshold), int(canny_high_threshold))
    y = y.astype(np.float32) / 255.0
    return y


def centered_canny_color(x: np.ndarray, canny_low_threshold, canny_high_threshold):
    assert isinstance(x, np.ndarray)
    assert x.ndim == 3 and x.shape[2] == 3

    result = [
        centered_canny(x[..., i], canny_low_threshold, canny_high_threshold)
        for i in range(3)
    ]
    result = np.stack(result, axis=2)
    return result


def pyramid_canny_color(x: np.ndarray, canny_low_threshold, canny_high_threshold):
    assert isinstance(x, np.ndarray)
    assert x.ndim == 3 and x.shape[2] == 3

    H, W, C = x.shape
    acc_edge = None

    for k in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        Hs, Ws = int(H * k), int(W * k)
        small = cv2.resize(x, (Ws, Hs), interpolation=cv2.INTER_AREA)
        edge = centered_canny_color(
            small, canny_low_threshold, canny_high_threshold
        )
        if acc_edge is None:
            acc_edge = edge
        else:
            acc_edge = cv2.resize(
                acc_edge,
                (edge.shape[1], edge.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
            acc_edge = acc_edge * 0.75 + edge * 0.25

    return acc_edge


def norm255(x, low=4, high=96):
    assert isinstance(x, np.ndarray)
    assert x.ndim == 2 and x.dtype == np.float32

    v_min = np.percentile(x, low)
    v_max = np.percentile(x, high)

    x -= v_min
    x /= v_max - v_min

    return x * 255.0


def canny_pyramid(x, canny_low_threshold, canny_high_threshold):
    color_canny = pyramid_canny_color(
        x, canny_low_threshold, canny_high_threshold
    )
    result = np.sum(color_canny, axis=2)
    return norm255(result, low=1, high=99).clip(0, 255).astype(np.uint8)


def cpds(x):
    # Obtener el pipeline (se inicializa solo la primera vez)
    pipe = get_depth_pipeline()

    # Fooocus maneja imágenes en formato RGB/BGR NumPy array
    # Lo convertimos a PIL Image para pasarlo a Hugging Face Transformers
    pil_img = Image.fromarray(x)

    # Inferencia del Depth Anything V2 Large
    result = pipe(pil_img)
    depth_pil = result["depth"]

    # Convertimos de vuelta a NumPy uint8 (2 dimensiones: H, W)
    depth_np = np.array(depth_pil, dtype=np.uint8)

    # Si por alguna razón la imagen generada difiere en tamaño, ajustamos a la original
    if depth_np.shape[:2] != x.shape[:2]:
        depth_np = cv2.resize(
            depth_np, (x.shape[1], x.shape[0]), interpolation=cv2.INTER_CUBIC
        )

    return depth_np
