"""Constants and errors shared by the torch and ONNX runtimes."""

DISCLAIMER = (
    "Research demonstration only. Scores are uncalibrated, models were trained on "
    "public data that overlaps these sources, and no output is a diagnosis or a "
    "clinical probability."
)
EXPLANATION_NOTE = (
    "Correctness of the heatmap is not established. It shows gradient sensitivity "
    "at a 7x7 feature map, upsampled to the image; it is not a lesion mask and "
    "does not prove the model looked at the disease."
)


class UnknownLabelError(ValueError):
    """Raised when an explanation asks for a label the model does not output."""


class NoSaliencyError(ValueError):
    """Raised when Grad-CAM produced no usable gradient signal."""
