"""XRV score handling: keep logits, sigmoid scores, and rescaled scores distinct."""

import torch
import torchxrayvision as xrv

MODEL_ID = "densenet121-res224-all"
WEIGHTS_URL = (
    "https://github.com/mlmed/torchxrayvision/releases/download/v1/"
    "nih-pc-chex-mimic_ch-google-openi-kaggle-densenet121-d121-tw-"
    "lr001-rot45-tr15-sc15-seed0-best.pt"
)

SECONDARY_MODEL_ID = "densenet121-res224-nih"
SECONDARY_WEIGHTS_URL = (
    "https://github.com/mlmed/torchxrayvision/releases/download/v1/"
    "nih-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt"
)

# Reviewed checkpoint URLs; a silent upstream change must be noticed, not loaded.
EXPECTED_WEIGHTS = {
    MODEL_ID: WEIGHTS_URL,
    SECONDARY_MODEL_ID: SECONDARY_WEIGHTS_URL,
}


def load_model(model_id: str, weights_root: str):
    """Load one reviewed XRV checkpoint and check its registry URL."""
    if model_id not in EXPECTED_WEIGHTS:
        raise ValueError(f"Unreviewed model id: {model_id}")
    registry = xrv.models.model_urls[model_id]
    if registry["weights_url"] != EXPECTED_WEIGHTS[model_id]:
        raise RuntimeError("The upstream checkpoint URL changed; review the source before loading.")
    return xrv.models.DenseNet(
        weights=model_id, cache_dir=f"{weights_root}/{model_id}"
    ).cpu().eval()


def real_label_indices(model) -> list[int]:
    """Indices of non-empty outputs; XRV pads short label lists with ''."""
    return [index for index, label in enumerate(model.pathologies) if label]


def enable_raw_logits(model, *, allow_missing: bool = False) -> torch.Tensor:
    """Disable normalization AFTER loading; XRV installs thresholds in __init__.

    Some XRV checkpoints pad their label list to 18 with empty strings and NaN
    thresholds. Those slots carry no meaning; ``allow_missing`` permits them
    while still requiring valid thresholds for every real output.
    """
    thresholds = model.op_threshs
    labels = model.pathologies
    if thresholds is None or thresholds.shape != (len(labels),):
        raise ValueError("Expected one operating threshold per model output.")
    real = real_label_indices(model)
    if not real:
        raise ValueError("Model has no labelled outputs.")
    if not allow_missing and len(real) != len(labels):
        raise ValueError("Model has blank output slots; pass allow_missing=True and select real labels.")
    if not torch.isfinite(thresholds[real]).all():
        raise ValueError("Real outputs must have finite operating thresholds.")
    if not ((thresholds[real] > 0) & (thresholds[real] < 1)).all():
        raise ValueError("Real outputs must have thresholds in (0, 1).")
    if allow_missing:
        blank = [index for index, label in enumerate(labels) if not label]
        if any(not torch.isnan(thresholds[index]) for index in blank):
            raise ValueError("Blank output slots should carry NaN thresholds.")
    thresholds = thresholds.detach().cpu().clone()
    model.op_threshs = None
    model.apply_sigmoid = False
    return thresholds


def predict_logits(model, tensor: torch.Tensor, labels: list[str]) -> torch.Tensor:
    if model.op_threshs is not None or model.apply_sigmoid:
        raise ValueError("Raw logits are required; disable XRV output transformations first.")
    with torch.inference_mode():
        logits = model(tensor).detach().cpu()
    if logits.shape != (1, len(labels)) or not torch.isfinite(logits).all():
        raise ValueError("Model output shape/values do not match the ordered label list.")
    return logits


def prediction_rows(logits: torch.Tensor, labels: list[str], thresholds: torch.Tensor) -> list[dict]:
    if logits.shape != (1, len(labels)) or thresholds.shape != (len(labels),):
        raise ValueError("Score and label dimensions do not match.")
    if not labels or any(not label for label in labels) or len(set(labels)) != len(labels):
        raise ValueError("Each output must have one unique, nonempty trained label.")
    scores = torch.sigmoid(logits)  # Exactly once, on raw logits.
    rescaled = xrv.models.op_norm(scores, thresholds)
    if not all(torch.isfinite(t).all() for t in (logits, scores, rescaled)):
        raise ValueError("Non-finite prediction values.")
    return [
        {"index": i, "label": label, "logit": logits[0, i].item(),
         "sigmoid_score": scores[0, i].item(),
         "xrv_rescaled_score": rescaled[0, i].item()}
        for i, label in enumerate(labels)
    ]
