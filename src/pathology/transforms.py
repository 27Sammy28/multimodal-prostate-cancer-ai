import numpy as np


def pil_to_numpy(image):
    return np.asarray(image.convert("RGB"))


def normalize_image(image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    array = pil_to_numpy(image).astype(np.float32) / 255.0
    return (array - np.asarray(mean)) / np.asarray(std)


def center_crop(image, size):
    width, height = image.size
    crop_width, crop_height = (size, size) if isinstance(size, int) else size
    left = max((width - crop_width) // 2, 0)
    top = max((height - crop_height) // 2, 0)
    return image.crop((left, top, left + crop_width, top + crop_height))
