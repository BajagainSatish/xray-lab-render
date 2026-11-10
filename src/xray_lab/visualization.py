"""Score plots and Grad-CAM overlays with explicit geometry mapping.

Grad-CAM is computed from the gradient of a raw target logit with respect to
the last convolutional feature map (``model.features``, 7x7 for a 224 input).
The resulting map is coarse: it is not a lesion mask and must not be read as
one.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def normalize_cam(cam: np.ndarray) -> np.ndarray:
    """Scale a 2D saliency map to [0, 1]; a constant map becomes all zeros."""
    cam = np.asarray(cam, dtype=np.float32)
    if cam.ndim != 2 or cam.size == 0:
        raise ValueError("CAM must be a nonempty 2D array.")
    if not np.isfinite(cam).all():
        raise ValueError("CAM contains non-finite values.")
    low, high = float(cam.min()), float(cam.max())
    if high - low < 1e-12:
        return np.zeros_like(cam, dtype=np.float32)
    return (cam - low) / (high - low)


def cam_to_image(cam: np.ndarray, crop_xyxy: list[int],
                 original_shape_hw: list[int]) -> np.ndarray:
    """Map a square CAM back into the original image canvas.

    XRV preprocesses by center-cropping the original to a square and resizing it
    to 224. Inverting that means resizing the CAM to the crop side and pasting
    it at the recorded crop offset; pixels outside the crop stay zero because
    the model never saw them.
    """
    cam = np.asarray(cam, dtype=np.float32)
    if cam.ndim != 2 or cam.shape[0] != cam.shape[1] or cam.shape[0] == 0:
        raise ValueError("CAM must be a nonempty square 2D array.")
    if len(crop_xyxy) != 4 or len(original_shape_hw) != 2:
        raise ValueError("Expected crop [x1, y1, x2, y2] and original [height, width].")
    x1, y1, x2, y2 = (int(value) for value in crop_xyxy)
    height, width = (int(value) for value in original_shape_hw)
    side = x2 - x1
    if side <= 0 or y2 - y1 != side:
        raise ValueError(f"Crop must be a nonempty square, got {crop_xyxy}.")
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
        raise ValueError(f"Crop {crop_xyxy} lies outside a {height}x{width} image.")
    resized = F.interpolate(
        torch.from_numpy(cam)[None, None], size=(side, side),
        mode="bilinear", align_corners=False,
    )[0, 0].numpy()
    canvas = np.zeros((height, width), dtype=np.float32)
    canvas[y1:y2, x1:x2] = resized
    return canvas


def gradcam_multi(model, tensor: torch.Tensor,
                  class_indices: list[int]) -> tuple[list[np.ndarray], torch.Tensor, list[float]]:
    """Return normalized CAMs for several classes from one forward pass.

    A forward hook clones the last feature map because XRV applies an in-place
    ReLU afterwards. Backward passes share that graph with ``retain_graph``.
    """
    if tensor.ndim != 4 or tensor.shape[0] != 1 or tensor.shape[1] != 1:
        raise ValueError("Expected a single-channel NCHW tensor with batch size 1.")
    if model.op_threshs is not None or model.apply_sigmoid:
        raise ValueError("Raw logits are required; disable XRV output transformations first.")
    if not hasattr(model, "features"):
        raise ValueError("Model has no 'features' module to hook.")
    if not class_indices or len(set(class_indices)) != len(class_indices):
        raise ValueError("class_indices must be a nonempty list of unique indices.")

    captured: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        captured["activation"] = output
        return output.clone()

    handle = model.features.register_forward_hook(hook)
    try:
        logits = model(tensor)
        if logits.shape != (1, len(model.pathologies)) or not torch.isfinite(logits).all():
            raise ValueError("Model output shape/values do not match the pathologies.")
        if any(not 0 <= index < logits.shape[1] for index in class_indices):
            raise ValueError("A class index is outside the model output range.")
        activation = captured["activation"]
        gradients = []
        for position, index in enumerate(class_indices):
            gradient = torch.autograd.grad(
                logits[0, index], activation,
                retain_graph=position < len(class_indices) - 1,
            )[0]
            if gradient.shape != activation.shape:
                raise RuntimeError("Gradient and activation shapes differ.")
            gradients.append(gradient)
    finally:
        handle.remove()

    size = (int(tensor.shape[2]), int(tensor.shape[3]))
    cams = []
    for gradient in gradients:
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=size, mode="bilinear", align_corners=False)
        cams.append(normalize_cam(cam[0, 0].detach().cpu().numpy()))
    return cams, logits.detach().cpu(), [float(gradient.norm().item()) for gradient in gradients]


def gradcam(model, tensor: torch.Tensor, class_index: int) -> tuple[np.ndarray, torch.Tensor, float]:
    """Convenience wrapper for one class; see :func:`gradcam_multi`."""
    cams, logits, norms = gradcam_multi(model, tensor, [class_index])
    return cams[0], logits, norms[0]


def plot_scores(rows: list[dict], *, sample_id: str, official_labels: list[str],
                model_id: str, run_id: str, output_path: Path) -> Path:
    """Horizontal bar chart of all 18 uncalibrated sigmoid scores."""
    ranked = sorted(rows, key=lambda row: row["sigmoid_score"], reverse=True)
    labels = [row["label"] for row in ranked]
    scores = [row["sigmoid_score"] for row in ranked]
    colors = ["#d97925" if label in official_labels else "#4878a8" for label in labels]

    figure, axis = plt.subplots(figsize=(8, 5.8))
    axis.barh(range(len(scores)), scores, color=colors)
    axis.set_yticks(range(len(scores)))
    axis.set_yticklabels(labels, fontsize=8)
    axis.invert_yaxis()
    axis.set_xlim(0, 1)
    axis.set_xlabel("sigmoid score (ranking only, not a clinical probability)")
    axis.set_title(f"{sample_id} — all 18 model outputs\n{model_id} · run {run_id}",
                   fontsize=10)
    axis.grid(axis="x", alpha=0.25)
    note = ", ".join(official_labels) if official_labels else "none recorded"
    figure.text(0.01, 0.01,
                f"NIH reference label(s): {note}. Orange marks model outputs that match "
                "an NIH label.\nScores are uncalibrated and do not sum to 1; no "
                "positive/negative decision is implied.",
                fontsize=7, va="bottom")
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def render_cam_figure(original: np.ndarray, cam_image: np.ndarray, *, sample_id: str,
                      model_id: str, target_label: str, target_score: float,
                      run_id: str, output_path: Path, cmap: str = "inferno",
                      alpha: float = 0.45) -> Path:
    """Three panels: original, CAM, overlay; all with the original geometry."""
    if original.shape != cam_image.shape:
        raise ValueError("Original and CAM geometry differ.")
    figure, axes = plt.subplots(1, 3, figsize=(12, 4.4))
    axes[0].imshow(original, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("original (release pixels)")
    image = axes[1].imshow(cam_image, cmap=cmap, vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM (relative saliency)")
    axes[2].imshow(original, cmap="gray", vmin=0, vmax=255)
    axes[2].imshow(cam_image, cmap=cmap, vmin=0, vmax=1, alpha=alpha)
    axes[2].set_title(f"overlay (alpha={alpha})")
    for axis in axes:
        axis.axis("off")
    figure.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04,
                    label="normalized gradient weight")
    figure.suptitle(
        f"{sample_id} · {target_label} (sigmoid {target_score:.3f})\n"
        f"{model_id} · run {run_id} · layer: last conv feature map (7×7)",
        fontsize=10,
    )
    figure.text(0.01, 0.01,
                "Coarse saliency from gradients of the raw target logit; "
                "not a lesion mask, not a diagnosis, and not evidence of correctness.",
                fontsize=7, va="bottom")
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def plot_comparison_bars(rows_primary: list[dict], rows_secondary: list[dict], *,
                         sample_id: str, primary_id: str, secondary_id: str,
                         run_id: str, official_labels: list[str],
                         output_path: Path) -> Path:
    """Grouped bars for the labels both checkpoints actually predict."""
    secondary = {row["label"]: row for row in rows_secondary}
    shared = [row for row in rows_primary if row["label"] in secondary]
    shared.sort(key=lambda row: row["sigmoid_score"], reverse=True)
    if not shared:
        raise ValueError("The two score tables share no labels.")
    labels = [row["label"] for row in shared]
    primary_scores = [row["sigmoid_score"] for row in shared]
    secondary_scores = [secondary[label]["sigmoid_score"] for label in labels]
    positions = np.arange(len(labels))
    height = 0.38

    figure, axis = plt.subplots(figsize=(8.5, 6.2))
    axis.barh(positions - height / 2, primary_scores, height, label=primary_id,
              color="#d97925")
    axis.barh(positions + height / 2, secondary_scores, height, label=secondary_id,
              color="#4878a8")
    axis.set_yticks(positions)
    axis.set_yticklabels(labels, fontsize=8)
    axis.invert_yaxis()
    axis.set_xlim(0, 1)
    axis.set_xlabel("sigmoid score (uncalibrated; not directly comparable as probability)")
    axis.set_title(f"{sample_id} — shared labels, two checkpoints\nrun {run_id}", fontsize=10)
    axis.grid(axis="x", alpha=0.25)
    axis.legend(fontsize=8, loc="lower right")
    note = ", ".join(official_labels) if official_labels else "none recorded"
    figure.text(0.01, 0.01,
                f"NIH reference label(s): {note}. Models were trained on different "
                "collections; these are rankings, not comparable probabilities.",
                fontsize=7, va="bottom")
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def plot_comparison_scatter(points: list[dict], *, primary_id: str, secondary_id: str,
                            run_id: str, output_path: Path) -> Path:
    """One dot per image/label pair; the diagonal means equal scores."""
    samples = sorted({point["sample_id"] for point in points})
    colors = plt.get_cmap("tab10").colors
    figure, axis = plt.subplots(figsize=(6.4, 6.0))
    for index, sample_id in enumerate(samples):
        subset = [point for point in points if point["sample_id"] == sample_id]
        axis.scatter([point["primary"] for point in subset],
                     [point["secondary"] for point in subset],
                     s=22, alpha=0.85, color=colors[index % len(colors)],
                     label=sample_id)
    axis.plot([0, 1], [0, 1], color="#888888", linewidth=1, linestyle="--")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_xlabel(f"{primary_id} sigmoid score")
    axis.set_ylabel(f"{secondary_id} sigmoid score")
    axis.set_title(f"Shared-label scores (dashed line = identical)\nrun {run_id}", fontsize=10)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7, title="image", title_fontsize=7)
    figure.text(0.01, 0.01,
                "Uncalibrated scores from models trained on different collections; "
                "distance from the line shows disagreement, not which model is right.",
                fontsize=7, va="bottom")
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path
