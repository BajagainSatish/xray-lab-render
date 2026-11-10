"""Low-memory ONNX Runtime service backend for the Render free tier."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import threading
import time
from datetime import datetime, timezone

import numpy as np
import onnxruntime as ort
from skimage.transform import resize

from xray_lab.preprocessing import MAX_UPLOAD_BYTES, read_image
from xray_lab.service.common import (
    DISCLAIMER,
    EXPLANATION_NOTE,
    NoSaliencyError,
    UnknownLabelError,
)
from xray_lab.service.explain import base64_png, cam_stats, preview_png, render_overlay
from xray_lab.provenance import sha256_file


def _sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-value)))


def _op_norm(score: float, threshold: float) -> float:
    if score < threshold:
        return score / (threshold * 2)
    return 1.0 - (1.0 - score) / ((1.0 - threshold) * 2)


def _normalize_cam(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    peak = float(values.max())
    if peak > 0:
        return np.clip(values / peak, 0.0, 1.0)
    magnitude = np.abs(values)
    peak = float(magnitude.max())
    if peak <= 0:
        raise NoSaliencyError("CAM produced no signal for this image and label.")
    return np.clip(magnitude / peak, 0.0, 1.0)


class OnnxRuntime:
    """Duck-typed replacement for ModelRuntime with no torch import."""

    def __init__(self, model_path: Path, meta_path: Path, *, model_id: str | None = None,
                 max_upload_bytes: int = MAX_UPLOAD_BYTES):
        self.model_path = Path(model_path)
        self.meta = json.loads(Path(meta_path).read_text())
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(self.model_path), sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.labels = list(self.meta["labels"])
        self.thresholds = np.asarray(self.meta["thresholds"], dtype=np.float32)
        self.cam_weights = np.asarray(self.meta["cam_weights"], dtype=np.float32)
        self.model_id = model_id or self.meta["id"]
        self.weights_sha256 = self.meta.get("sha256") or sha256_file(self.model_path)
        self.parameter_count = int(self.meta["parameter_count"])
        self.training_description = self.meta.get("training_description", "")
        self.source_url = self.meta.get("source_url")
        self.max_upload_bytes = max_upload_bytes
        self.loaded_at = datetime.now(timezone.utc)
        self._lock = threading.Lock()

    def card(self) -> dict:
        return {
            "id": self.model_id,
            "source_url": self.source_url,
            "training_description": self.training_description,
            "sha256": self.weights_sha256,
            "size_bytes": self.model_path.stat().st_size,
            "parameter_count": self.parameter_count,
            "labels": self.labels,
            "output_count": len(self.labels),
            "operating_thresholds": self.thresholds.tolist(),
            "preprocessing": {
                "normalization": "(2 * (float32(pixel) / 255) - 1) * 1024",
                "crop": "center square",
                "resize": "skimage resize 224, anti-aliasing",
                "input": "1 x 1 x 224 x 224 float32",
            },
            "xrv_version": "onnx-runtime",
            "max_upload_bytes": self.max_upload_bytes,
            "disclaimer": DISCLAIMER,
        }

    def uptime_seconds(self) -> float:
        return (datetime.now(timezone.utc) - self.loaded_at).total_seconds()

    def _preprocess(self, pixels: np.ndarray) -> tuple[np.ndarray, dict]:
        height, width = pixels.shape
        side = min(height, width)
        left = width // 2 - side // 2
        top = height // 2 - side // 2
        normalized = ((pixels.astype(np.float32) / 255.0) * 2.0 - 1.0) * 1024.0
        crop = normalized[top:top + side, left:left + side]
        resized = resize(crop, (224, 224), order=1, mode="constant",
                         preserve_range=True, anti_aliasing=True).astype(np.float32)
        tensor = resized[None, None, :, :]
        metadata = {
            "normalization": "(2 * (float32(pixel) / 255) - 1) * 1024",
            "crop_xyxy_exclusive": [left, top, left + side, top + side],
            "resize": {"implementation": "skimage.transform.resize", "size": 224,
                       "anti_aliasing": True},
            "tensor_shape_nchw": list(tensor.shape),
            "tensor_dtype": str(tensor.dtype),
            "tensor_min": float(tensor.min()),
            "tensor_max": float(tensor.max()),
        }
        return tensor, metadata

    def _rows(self, logits: np.ndarray) -> list[dict]:
        scores = 1.0 / (1.0 + np.exp(-logits))
        rows = []
        for index, label in enumerate(self.labels):
            score = float(scores[index])
            threshold = float(self.thresholds[index])
            rows.append({
                "index": index,
                "label": label,
                "logit": float(logits[index]),
                "sigmoid_score": score,
                "xrv_rescaled_score": _op_norm(score, threshold),
                "rank": 0,
            })
        ranked = sorted(rows, key=lambda row: row["sigmoid_score"], reverse=True)
        ranks = {row["label"]: index + 1 for index, row in enumerate(ranked)}
        for row in rows:
            row["rank"] = ranks[row["label"]]
        return rows

    def _image_block(self, data: bytes, pixels: np.ndarray, metadata: dict,
                     started: float) -> tuple[dict, float]:
        preview = (base64_png(preview_png(pixels))
                   if metadata["format"] == "DICOM" else None)
        decoded = time.perf_counter()
        return {
            "filename": metadata.get("filename"),
            "sha256": hashlib.sha256(data).hexdigest(),
            "format": metadata["format"],
            "mode": metadata["original_mode"],
            "shape_hw": metadata["shape_hw"],
            "bytes": len(data),
            "dicom": metadata.get("dicom"),
            "preview_png_base64": preview,
        }, decoded

    def predict(self, data: bytes, *, filename: str | None = None) -> dict:
        started = time.perf_counter()
        pixels, metadata = read_image(data, max_bytes=self.max_upload_bytes)
        metadata["filename"] = filename
        image, decoded = self._image_block(data, pixels, metadata, started)
        tensor, preprocessing = self._preprocess(pixels)
        preprocessed = time.perf_counter()
        with self._lock:
            logits, _features = self.session.run(
                None, {self.input_name: tensor})
        inferred = time.perf_counter()
        return {
            "model": {
                "id": self.model_id,
                "sha256": self.weights_sha256,
                "parameter_count": self.parameter_count,
                "training_description": self.training_description,
            },
            "image": image,
            "preprocessing": preprocessing,
            "predictions": self._rows(logits[0]),
            "timing_ms": {
                "decode": round((decoded - started) * 1000, 2),
                "preprocess": round((preprocessed - decoded) * 1000, 2),
                "inference": round((inferred - preprocessed) * 1000, 2),
                "total": round((inferred - started) * 1000, 2),
            },
            "warning": DISCLAIMER,
        }

    def _cam(self, features: np.ndarray, label_index: int) -> np.ndarray:
        weights = self.cam_weights[label_index]
        values = np.sum(weights[:, None, None] * features, axis=0)
        return _normalize_cam(values)

    def _cam_to_image(self, cam: np.ndarray, crop: list[int],
                      shape: tuple[int, int]) -> np.ndarray:
        x1, y1, x2, y2 = crop
        side = x2 - x1
        resized = resize(cam, (side, side), order=1, mode="constant",
                         preserve_range=True, anti_aliasing=False)
        canvas = np.zeros(shape, dtype=np.float32)
        canvas[y1:y2, x1:x2] = resized
        return np.clip(canvas, 0.0, 1.0)

    def explain(self, data: bytes, *, label: str, filename: str | None = None,
                alpha: float = 0.45) -> dict:
        started = time.perf_counter()
        pixels, metadata = read_image(data, max_bytes=self.max_upload_bytes)
        metadata["filename"] = filename
        image, decoded = self._image_block(data, pixels, metadata, started)
        tensor, preprocessing = self._preprocess(pixels)
        preprocessed = time.perf_counter()
        if label not in self.labels:
            raise UnknownLabelError(
                f"Unknown label {label!r}; choose one of {', '.join(self.labels)}.")
        index = self.labels.index(label)
        with self._lock:
            logits, features = self.session.run(None, {self.input_name: tensor})
        rows = self._rows(logits[0])
        target = rows[index]
        cam = self._cam(features[0], index)
        cam_image = self._cam_to_image(
            cam, preprocessing["crop_xyxy_exclusive"], pixels.shape)
        overlay = render_overlay(pixels, cam_image, alpha=alpha)
        rendered = time.perf_counter()
        statistics = cam_stats(cam_image, preprocessing["crop_xyxy_exclusive"])

        control_row = min((row for row in rows if row["label"] != label),
                          key=lambda row: row["sigmoid_score"])
        control_cam = self._cam(features[0], control_row["index"])
        difference = float(np.abs(cam - control_cam).mean())

        return {
            "model": {
                "id": self.model_id,
                "sha256": self.weights_sha256,
                "parameter_count": self.parameter_count,
                "training_description": self.training_description,
            },
            "image": image,
            "target": {
                "label": label,
                "index": index,
                "sigmoid_score": target["sigmoid_score"],
                "rank": target["rank"],
            },
            "method": "class activation map (linear classifier weights)",
            "overlay_png_base64": base64_png(overlay),
            "cam": statistics,
            "geometry": {
                "crop_xyxy_exclusive": preprocessing["crop_xyxy_exclusive"],
                "original_shape_hw": list(pixels.shape),
                "canvas_shape_hw": list(pixels.shape),
            },
            "class_targeting": {
                "control_label": control_row["label"],
                "control_sigmoid_score": control_row["sigmoid_score"],
                "mean_abs_difference": difference,
                "gradient_norms": [0.0, 0.0],
                "note": "Gradient-free CAM; control map uses classifier weights.",
            },
            "timing_ms": {
                "decode": round((decoded - started) * 1000, 2),
                "preprocess": round((preprocessed - decoded) * 1000, 2),
                "explain": round((rendered - preprocessed) * 1000, 2),
                "render": round((rendered - preprocessed) * 1000, 2),
                "total": round((rendered - started) * 1000, 2),
            },
            "warning": EXPLANATION_NOTE,
        }
