"""Read a single-frame projection DICOM into 8-bit grayscale.

The loader applies Rescale Slope/Intercept, the DICOM window if present (or an
explicit 1–99 percentile fallback), and inverts MONOCHROME1. Only whitelisted
technical fields are returned: patient names, IDs, and free-text fields never
leave this module.
"""

from io import BytesIO

import numpy as np
import pydicom
from pydicom.errors import InvalidDicomError

DICOM_MAX_BYTES = 25 * 1024 * 1024
DICOM_MAX_SIDE = 4096
DICOM_MAX_PIXELS = 16 * 1024 * 1024
DEIDENTIFICATION_WARNING = (
    "Raw DICOM can contain burned-in identifiers or identifying metadata. "
    "This service does not de-identify the image; it returns only whitelisted "
    "technical fields and never stores the upload."
)
REJECTED_MODALITIES = {"CT", "MR", "US", "PT", "NM", "SR", "MG"}


def is_dicom(data: bytes) -> bool:
    return len(data) >= 132 and data[128:132] == b"DICM"


def _first(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value


def _float(value, default=None):
    value = _first(value)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def read_dicom(data: bytes, *, max_bytes: int = DICOM_MAX_BYTES,
               max_side: int = DICOM_MAX_SIDE,
               max_pixels: int = DICOM_MAX_PIXELS) -> tuple[np.ndarray, dict]:
    if not data:
        raise ValueError("Empty DICOM upload.")
    if len(data) > max_bytes:
        raise ValueError(f"DICOM exceeds the {max_bytes} byte limit.")
    try:
        dataset = pydicom.dcmread(BytesIO(data))
    except (InvalidDicomError, OSError) as exc:
        raise ValueError("This file is not a valid DICOM Part 10 file.") from exc

    modality = str(getattr(dataset, "Modality", "")).strip().upper()
    if modality in REJECTED_MODALITIES:
        raise ValueError(
            f"Modality {modality!r} is not a projection radiograph; "
            "upload a chest X-ray (CR or DX).")
    photometric = str(getattr(dataset, "PhotometricInterpretation", "")).upper()
    if photometric not in {"MONOCHROME1", "MONOCHROME2"}:
        raise ValueError(
            f"Unsupported photometric interpretation {photometric!r}; "
            "only single-channel MONOCHROME1/2 images are accepted.")
    if int(getattr(dataset, "SamplesPerPixel", 1)) != 1:
        raise ValueError("Only single-channel DICOM images are accepted.")
    frames = int(getattr(dataset, "NumberOfFrames", 1))
    if frames != 1:
        raise ValueError(
            f"Multi-frame DICOM is not supported (NumberOfFrames={frames}); "
            "select a single projection frame and convert it explicitly.")
    rows = int(getattr(dataset, "Rows", 0))
    columns = int(getattr(dataset, "Columns", 0))
    if rows < 2 or columns < 2:
        raise ValueError("DICOM image is too small.")
    if max(rows, columns) > max_side or rows * columns > max_pixels:
        raise ValueError("DICOM image exceeds the pixel dimensions limit.")

    transfer_syntax = getattr(getattr(dataset, "file_meta", None),
                              "TransferSyntaxUID", None)
    transfer_name = str(getattr(transfer_syntax, "name", "unknown"))
    try:
        raw = dataset.pixel_array
    except Exception as exc:  # noqa: BLE001 - decoder plugins raise many types
        raise ValueError(
            f"Cannot decode DICOM pixel data ({transfer_name}); a compatible "
            "decoder plugin may be required for this transfer syntax."
        ) from exc
    if raw.ndim != 2:
        raise ValueError(f"Expected a single 2D frame, got shape {raw.shape}.")

    slope = _float(getattr(dataset, "RescaleSlope", 1), 1.0)
    intercept = _float(getattr(dataset, "RescaleIntercept", 0), 0.0)
    values = raw.astype(np.float32) * slope + intercept

    center = _float(getattr(dataset, "WindowCenter", None))
    width = _float(getattr(dataset, "WindowWidth", None))
    if center is not None and width is not None and width > 0:
        low, high = center - width / 2.0, center + width / 2.0
        window_source = "dicom_window"
    else:
        if float(values.min()) == float(values.max()):
            raise ValueError("Constant DICOM pixels: inspect the source before inference.")
        low, high = (float(value) for value in np.percentile(values, [1, 99]))
        if high <= low:
            low, high = float(values.min()), float(values.max())
            window_source = "auto_min_max"
        else:
            window_source = "auto_percentile_1_99"
        center = width = None

    gray = (values - low) / (high - low) * 255.0
    gray = np.clip(gray, 0.0, 255.0).round().astype(np.uint8)
    if photometric == "MONOCHROME1":
        gray = 255 - gray
    if gray.min() == gray.max():
        raise ValueError("DICOM window produced a constant image; inspect the source.")

    metadata = {
        "format": "DICOM",
        "original_mode": photometric,
        "shape_hw": [int(gray.shape[0]), int(gray.shape[1])],
        "dtype": str(gray.dtype),
        "pixel_min": int(gray.min()),
        "pixel_max": int(gray.max()),
        "grayscale_policy": (
            "Rescale Slope/Intercept, window to 8-bit, MONOCHROME1 inverted"),
        "orientation_policy": "as stored; no flip, rotation, or inversion",
        "dicom": {
            "transfer_syntax_uid": str(transfer_syntax) if transfer_syntax else None,
            "transfer_syntax_name": transfer_name,
            "modality": modality or None,
            "view_position": str(getattr(dataset, "ViewPosition", "") or "") or None,
            "body_part": str(getattr(dataset, "BodyPartExamined", "") or "") or None,
            "rows": rows,
            "columns": columns,
            "frames": frames,
            "bits_allocated": int(getattr(dataset, "BitsAllocated", 0)),
            "bits_stored": int(getattr(dataset, "BitsStored", 0)),
            "pixel_representation": int(getattr(dataset, "PixelRepresentation", 0)),
            "photometric_interpretation": photometric,
            "rescale_slope": slope,
            "rescale_intercept": intercept,
            "window_center": center,
            "window_width": width,
            "window_low": low,
            "window_high": high,
            "window_source": window_source,
            "voi_lut_sequence_present": hasattr(dataset, "VOILUTSequence"),
            "patient_identifiers_returned": False,
            "deidentification_warning": DEIDENTIFICATION_WARNING,
        },
    }
    return gray, metadata
