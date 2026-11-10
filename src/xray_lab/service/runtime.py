"""Load one checkpoint once and score in-memory uploads."""

from datetime import datetime, timezone
import hashlib
from pathlib import Path
import threading
import time

import numpy as np
import torch
import torchxrayvision as xrv

from xray_lab.inference import (
    MODEL_ID,
    enable_raw_logits,
    load_model,
    predict_logits,
    prediction_rows,
    real_label_indices,
)
from xray_lab.preprocessing import MAX_UPLOAD_BYTES, preprocess_pixels, read_image
from xray_lab.provenance import sha256_file
from xray_lab.service.explain import base64_png, cam_stats, preview_png, render_overlay
from xray_lab.visualization import cam_to_image, gradcam_multi

from xray_lab.service.common import (
    DISCLAIMER,
    EXPLANATION_NOTE,
    NoSaliencyError,
    UnknownLabelError,
)


class ModelRuntime:
    """A single, loaded model with a serialized forward pass."""

    def __init__(self, model, *, model_id: str, thresholds: torch.Tensor,
                 weights_path: Path | None = None,
                 max_upload_bytes: int = MAX_UPLOAD_BYTES):
        self.model = model.eval()
        self.model_id = model_id
        self.full_labels = list(model.pathologies)
        self.indices = real_label_indices(model)
        self.labels = [self.full_labels[index] for index in self.indices]
        self.thresholds = thresholds[self.indices]
        self.weights_path = Path(weights_path) if weights_path else None
        self.weights_sha256 = (sha256_file(self.weights_path)
                               if self.weights_path and self.weights_path.is_file() else None)
        self.weights_size = (self.weights_path.stat().st_size
                             if self.weights_path and self.weights_path.is_file() else None)
        self.parameter_count = sum(parameter.numel() for parameter in self.model.parameters())
        registry = xrv.models.model_urls.get(model_id, {})
        self.source_url = registry.get("weights_url")
        self.training_description = registry.get("description", "not provided")
        self.max_upload_bytes = max_upload_bytes
        self.loaded_at = datetime.now(timezone.utc)
        self._lock = threading.Lock()

    @classmethod
    def from_model(cls, model, *, model_id: str, weights_path: Path | None = None,
                   max_upload_bytes: int = MAX_UPLOAD_BYTES) -> "ModelRuntime":
        allow_missing = len(real_label_indices(model)) != len(model.pathologies)
        thresholds = enable_raw_logits(model, allow_missing=allow_missing)
        return cls(model, model_id=model_id, thresholds=thresholds,
                   weights_path=weights_path, max_upload_bytes=max_upload_bytes)

    @classmethod
    def load(cls, model_id: str = MODEL_ID, weights_root: Path | str = "weights",
             max_upload_bytes: int = MAX_UPLOAD_BYTES) -> "ModelRuntime":
        model = load_model(model_id, str(weights_root))
        path = Path(model.weights_filename_local)
        return cls.from_model(model, model_id=model_id, weights_path=path,
                             max_upload_bytes=max_upload_bytes)

    def card(self) -> dict:
        return {
            "id": self.model_id,
            "source_url": self.source_url,
            "training_description": self.training_description,
            "sha256": self.weights_sha256,
            "size_bytes": self.weights_size,
            "parameter_count": self.parameter_count,
            "labels": self.labels,
            "output_count": len(self.labels),
            "operating_thresholds": self.thresholds.tolist(),
            "preprocessing": {
                "normalization": "(2 * (float32(pixel) / 255) - 1) * 1024",
                "crop": "center square (XRV integer rounding)",
                "resize": "XRV XRayResizer, 224, skimage",
                "input": "1 x 1 x 224 x 224 float32",
            },
            "xrv_version": xrv.__version__,
            "max_upload_bytes": self.max_upload_bytes,
            "disclaimer": DISCLAIMER,
        }

    def uptime_seconds(self) -> float:
        return (datetime.now(timezone.utc) - self.loaded_at).total_seconds()

    def predict(self, data: bytes, *, filename: str | None = None) -> dict:
        started = time.perf_counter()
        image_sha256 = hashlib.sha256(data).hexdigest()
        pixels, image_metadata = read_image(
            data, max_bytes=self.max_upload_bytes)
        preview = (base64_png(preview_png(pixels))
                   if image_metadata["format"] == "DICOM" else None)
        decoded = time.perf_counter()
        tensor, preprocessing = preprocess_pixels(pixels, image_metadata)
        preprocessed = time.perf_counter()
        with self._lock:
            logits_full = predict_logits(self.model, tensor, self.full_labels)
        inferred = time.perf_counter()
        rows = self._score_rows(logits_full)
        return {
            "model": {
                "id": self.model_id,
                "sha256": self.weights_sha256,
                "parameter_count": self.parameter_count,
                "training_description": self.training_description,
            },
            "image": {
                "filename": filename,
                "sha256": image_sha256,
                "format": image_metadata["format"],
                "mode": image_metadata["original_mode"],
                "shape_hw": image_metadata["shape_hw"],
                "bytes": len(data),
                "dicom": image_metadata.get("dicom"),
                "preview_png_base64": preview,
            },
            "preprocessing": preprocessing,
            "predictions": rows,
            "timing_ms": {
                "decode": round((decoded - started) * 1000, 2),
                "preprocess": round((preprocessed - decoded) * 1000, 2),
                "inference": round((inferred - preprocessed) * 1000, 2),
                "total": round((inferred - started) * 1000, 2),
            },
            "warning": DISCLAIMER,
        }

    def _score_rows(self, logits_full: torch.Tensor) -> list[dict]:
        logits = logits_full[:, self.indices]
        rows = prediction_rows(logits, self.labels, self.thresholds)
        ranked = sorted(rows, key=lambda row: row["sigmoid_score"], reverse=True)
        ranks = {row["label"]: index + 1 for index, row in enumerate(ranked)}
        for row in rows:
            row["rank"] = ranks[row["label"]]
        return rows

    def explain(self, data: bytes, *, label: str, filename: str | None = None,
                alpha: float = 0.45) -> dict:
        """Compute Grad-CAM for one output and return a PNG overlay in original geometry."""
        started = time.perf_counter()
        image_sha256 = hashlib.sha256(data).hexdigest()
        pixels, image_metadata = read_image(
            data, max_bytes=self.max_upload_bytes)
        preview = (base64_png(preview_png(pixels))
                   if image_metadata["format"] == "DICOM" else None)
        decoded = time.perf_counter()
        tensor, preprocessing = preprocess_pixels(pixels, image_metadata)
        preprocessed = time.perf_counter()
        if label not in self.labels:
            raise UnknownLabelError(
                f"Unknown label {label!r}; choose one of {', '.join(self.labels)}.")
        target_position = self.labels.index(label)
        full_index = self.indices[target_position]

        with self._lock:
            logits_full = predict_logits(self.model, tensor, self.full_labels)
            rows = self._score_rows(logits_full)
            control = min((row for row in rows if row["label"] != label),
                          key=lambda row: row["sigmoid_score"])
            control_index = self.indices[self.labels.index(control["label"])]
            cams, cam_logits, gradient_norms = gradcam_multi(
                self.model, tensor, [full_index, control_index])
        explained = time.perf_counter()
        if not torch.allclose(cam_logits, logits_full, rtol=1e-6, atol=1e-6):
            raise RuntimeError("Explanation forward did not reproduce the scoring logits.")

        cam_image = cam_to_image(
            cams[0], preprocessing["crop_xyxy_exclusive"], preprocessing["shape_hw"])
        if float(cam_image.max()) <= 0.0:
            raise NoSaliencyError("Grad-CAM produced no signal for this image and label.")
        overlay = render_overlay(pixels, cam_image, alpha=alpha)
        statistics = cam_stats(cam_image, preprocessing["crop_xyxy_exclusive"])
        rendered = time.perf_counter()

        target = next(row for row in rows if row["label"] == label)
        difference = float(np.abs(cams[0] - cams[1]).mean())
        return {
            "model": {
                "id": self.model_id,
                "sha256": self.weights_sha256,
                "parameter_count": self.parameter_count,
                "training_description": self.training_description,
            },
            "image": {
                "filename": filename,
                "sha256": image_sha256,
                "format": image_metadata["format"],
                "mode": image_metadata["original_mode"],
                "shape_hw": image_metadata["shape_hw"],
                "bytes": len(data),
                "dicom": image_metadata.get("dicom"),
                "preview_png_base64": preview,
            },
            "target": {
                "label": label,
                "index": full_index,
                "sigmoid_score": target["sigmoid_score"],
                "rank": target["rank"],
            },
            "method": (
                "Grad-CAM of the raw target logit on model.features "
                "(last 7x7 feature map), mapped through the recorded crop and resize."
            ),
            "overlay_png_base64": base64_png(overlay),
            "cam": statistics,
            "geometry": {
                "crop_xyxy_exclusive": preprocessing["crop_xyxy_exclusive"],
                "original_shape_hw": preprocessing["shape_hw"],
                "canvas_shape_hw": [int(cam_image.shape[0]), int(cam_image.shape[1])],
            },
            "class_targeting": {
                "control_label": control["label"],
                "control_sigmoid_score": control["sigmoid_score"],
                "mean_abs_difference": difference,
                "gradient_norms": gradient_norms,
                "note": "A nonzero difference shows the map depends on the chosen "
                        "class; it does not establish correctness.",
            },
            "timing_ms": {
                "decode": round((decoded - started) * 1000, 2),
                "preprocess": round((preprocessed - decoded) * 1000, 2),
                "explain": round((explained - preprocessed) * 1000, 2),
                "render": round((rendered - explained) * 1000, 2),
                "total": round((rendered - started) * 1000, 2),
            },
            "warning": f"{DISCLAIMER} {EXPLANATION_NOTE}",
        }
