"""8-bit safe grayscale loading and the exact XRV preprocessing path."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from xray_lab.dicom import is_dicom, read_dicom

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_SIDE = 4096
MAX_PIXELS = 16 * 1024 * 1024


def read_grayscale_image(data: bytes, *, allowed_formats=("PNG", "JPEG", "WEBP"),
                         max_bytes: int = MAX_UPLOAD_BYTES,
                         max_side: int = MAX_SIDE,
                         max_pixels: int = MAX_PIXELS) -> tuple[np.ndarray, dict]:
    """Decode 8-bit grayscale bytes from an in-memory upload.

    Only single-frame 8-bit grayscale images are accepted; RGB is allowed only
    when the three channels are identical, so color is never silently dropped.
    """
    if not data:
        raise ValueError("Empty upload.")
    if len(data) > max_bytes:
        raise ValueError(f"Image exceeds the {max_bytes} byte limit.")
    try:
        image = Image.open(BytesIO(data))
    except (Image.UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"Cannot decode this file as an image: {exc}") from exc
    with image:
        image_format = image.format
        mode = image.mode
        if image_format not in allowed_formats:
            raise ValueError(
                f"Unsupported format {image_format!r}; allowed: {', '.join(allowed_formats)}."
            )
        if mode not in {"L", "RGB"}:
            raise ValueError(
                f"Expected an 8-bit grayscale image; got {image_format}/{mode}. "
                "16-bit, DICOM, palette, and alpha inputs need a separate loader."
            )
        if getattr(image, "n_frames", 1) != 1:
            raise ValueError("Animated/multi-frame images are not supported.")
        width, height = image.size
        if min(width, height) < 2:
            raise ValueError("Image is too small.")
        if max(width, height) > max_side:
            raise ValueError(f"Image side exceeds {max_side} pixels.")
        if width * height > max_pixels:
            raise ValueError(f"Image exceeds {max_pixels} pixels.")
        pixels = np.array(image)
    if pixels.ndim == 3:
        if image_format == "WEBP":
            # Lossy WebP grayscale can carry small chroma noise. Accept only a
            # near-gray image; genuine color still fails.
            spread = max(
                int(np.abs(pixels[..., 0].astype(np.int16)
                           - pixels[..., 1].astype(np.int16)).max()),
                int(np.abs(pixels[..., 0].astype(np.int16)
                           - pixels[..., 2].astype(np.int16)).max()),
            )
            if spread > 8:
                raise ValueError("RGB channels differ; do not silently discard color information.")
        elif not (np.array_equal(pixels[..., 0], pixels[..., 1])
                  and np.array_equal(pixels[..., 0], pixels[..., 2])):
            raise ValueError("RGB channels differ; do not silently discard color information.")
        pixels = pixels[..., 0].copy()
    if pixels.ndim != 2 or pixels.dtype != np.uint8 or min(pixels.shape) < 2:
        raise ValueError("Expected a nonempty 2D uint8 grayscale image.")
    if pixels.min() == pixels.max():
        raise ValueError("Constant image: inspect the source before running inference.")
    return pixels, {
        "format": image_format,
        "original_mode": mode,
        "shape_hw": list(pixels.shape),
        "dtype": str(pixels.dtype),
        "pixel_min": int(pixels.min()),
        "pixel_max": int(pixels.max()),
        "grayscale_policy": "L, or identical RGB channels; no conversion",
        "orientation_policy": "as stored; no flip, rotation, or inversion",
    }


def read_image(data: bytes, *, allowed_formats=("PNG", "JPEG", "WEBP", "DICOM"),
               max_bytes: int = MAX_UPLOAD_BYTES, max_side: int = MAX_SIDE,
               max_pixels: int = MAX_PIXELS) -> tuple[np.ndarray, dict]:
    """Decode PNG, JPEG, or single-frame projection DICOM by content."""
    formats = {name.upper() for name in allowed_formats}
    if "DICOM" in formats and is_dicom(data):
        return read_dicom(data, max_bytes=max_bytes, max_side=max_side,
                          max_pixels=max_pixels)
    image_formats = tuple(name for name in ("PNG", "JPEG", "WEBP") if name in formats)
    if not image_formats:
        raise ValueError("No supported image format is allowed.")
    return read_grayscale_image(data, allowed_formats=image_formats,
                                max_bytes=max_bytes, max_side=max_side,
                                max_pixels=max_pixels)


def read_grayscale_png(path: Path) -> tuple[np.ndarray, dict]:
    return read_grayscale_image(Path(path).read_bytes(), allowed_formats=("PNG",))


def preprocess_pixels(pixels: np.ndarray, metadata: dict) -> tuple[torch.Tensor, dict]:
    """Apply XRV normalization, center crop, and resize to one 224 tensor."""
    import torch
    import torchxrayvision as xrv

    metadata = dict(metadata)
    height, width = pixels.shape
    side = min(height, width)
    # Match XRV's integer rounding exactly, including odd-sized images.
    left, top = width // 2 - side // 2, height // 2 - side // 2
    normalized = xrv.datasets.normalize(pixels, maxval=255)
    cropped = xrv.datasets.XRayCenterCrop()(normalized[None, ...])
    resized = xrv.datasets.XRayResizer(224, engine="skimage")(cropped)
    if resized.shape != (1, 224, 224) or resized.dtype != np.float32:
        raise RuntimeError(f"Unexpected preprocessed shape/dtype: {resized.shape}/{resized.dtype}")
    if not np.isfinite(resized).all() or resized.min() < -1024.01 or resized.max() > 1024.01:
        raise RuntimeError("Invalid preprocessed pixels.")
    tensor = torch.from_numpy(np.ascontiguousarray(resized)).unsqueeze(0)
    metadata.update({
        "normalization": "(2 * (float32(pixel) / 255) - 1) * 1024",
        "crop_xyxy_exclusive": [left, top, left + side, top + side],
        "resize": {"implementation": "xrv.datasets.XRayResizer", "engine": "skimage",
                   "size": 224, "mode": "constant", "preserve_range": True,
                   "order": "skimage default (1 for float32)",
                   "anti_aliasing": "skimage default (enabled when downsampling)"},
        "tensor_shape_nchw": list(tensor.shape), "tensor_dtype": str(tensor.dtype),
        "tensor_min": tensor.min().item(), "tensor_max": tensor.max().item(),
    })
    return tensor, metadata


def preprocess_png(path: Path) -> tuple[torch.Tensor, dict]:
    pixels, metadata = read_grayscale_png(path)
    return preprocess_pixels(pixels, metadata)
