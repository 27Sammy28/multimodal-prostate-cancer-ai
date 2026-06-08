import numpy as np
from PIL import Image


HE_REF = np.array([[0.5626, 0.2159], [0.7201, 0.8012], [0.4062, 0.5581]], dtype=np.float32)
MAX_CONC_REF = np.array([1.9705, 1.0308], dtype=np.float32)


def reinhard_normalize(image, target_mean=(180, 145, 170), target_std=(35, 25, 30)):
    array = np.asarray(image.convert("RGB")).astype(np.float32)
    mean = array.reshape(-1, 3).mean(axis=0)
    std = array.reshape(-1, 3).std(axis=0)
    normalized = (array - mean) / np.maximum(std, 1e-6)
    normalized = normalized * np.asarray(target_std) + np.asarray(target_mean)
    return Image.fromarray(np.clip(normalized, 0, 255).astype(np.uint8))


def rgb_to_od(rgb, light_intensity=255.0, beta=0.15):
    rgb = np.asarray(rgb).astype(np.float32)
    rgb = np.maximum(rgb, 1.0)
    optical_density = -np.log(rgb / light_intensity)
    return optical_density[~np.any(optical_density < beta, axis=1)]


def estimate_stain_matrix_macenko(image, alpha=1.0, beta=0.15):
    rgb = np.asarray(image.convert("RGB")).reshape((-1, 3))
    optical_density = rgb_to_od(rgb, beta=beta)
    if len(optical_density) < 10:
        return HE_REF.copy()
    _, _, vt = np.linalg.svd(optical_density, full_matrices=False)
    plane = optical_density @ vt[:2].T
    angles = np.arctan2(plane[:, 1], plane[:, 0])
    min_angle, max_angle = np.percentile(angles, [alpha, 100 - alpha])
    stains = np.array(
        [
            vt[:2].T @ np.array([np.cos(min_angle), np.sin(min_angle)]),
            vt[:2].T @ np.array([np.cos(max_angle), np.sin(max_angle)]),
        ]
    ).T
    stains = stains / np.maximum(np.linalg.norm(stains, axis=0, keepdims=True), 1e-8)
    if stains[0, 0] < stains[0, 1]:
        stains = stains[:, [1, 0]]
    return stains.astype(np.float32)


def macenko_normalize(image, target_stain_matrix=HE_REF, target_max_concentration=MAX_CONC_REF, alpha=1.0, beta=0.15):
    rgb = np.asarray(image.convert("RGB")).astype(np.float32)
    flat_rgb = rgb.reshape((-1, 3))
    source_stain_matrix = estimate_stain_matrix_macenko(image, alpha=alpha, beta=beta)
    optical_density = -np.log(np.maximum(flat_rgb, 1.0) / 255.0)
    concentrations, *_ = np.linalg.lstsq(source_stain_matrix, optical_density.T, rcond=None)
    source_max = np.percentile(concentrations, 99, axis=1)
    concentrations *= (np.asarray(target_max_concentration) / np.maximum(source_max, 1e-8))[:, None]
    normalized_od = target_stain_matrix @ concentrations
    normalized_rgb = 255.0 * np.exp(-normalized_od.T)
    normalized_rgb = normalized_rgb.reshape(rgb.shape)
    return Image.fromarray(np.clip(normalized_rgb, 0, 255).astype(np.uint8))
