"""
Shared utility functions for image loading and conversion.
"""

import numpy as np
import cv2


def rgb2gray(img):
    """Convert an RGB image to grayscale using standard luminance weights."""
    return 0.2989 * img[..., 0] + 0.5870 * img[..., 1] + 0.1140 * img[..., 2]


def load_image(path, grayscale=False):
    """Load an image via OpenCV. Returns BGR by default, grayscale if requested."""
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    img = cv2.imread(path, flag)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    return img


def normalize_uint8(img):
    """Normalize image to [0, 255] uint8 range."""
    img_float = img.astype(np.float64)
    if img_float.max() > 0:
        img_float = img_float / img_float.max()
    return np.asarray(img_float * 255, dtype=np.uint8)


def bgr_to_rgb(img):
    """Convert OpenCV BGR image to RGB for matplotlib display."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
