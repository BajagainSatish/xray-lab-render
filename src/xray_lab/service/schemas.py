"""Response contracts shared by the API and future clients."""

from pydantic import BaseModel, Field


class Health(BaseModel):
    status: str = Field(examples=["ok"])
    version: str
    model_id: str
    model_loaded: bool
    uptime_seconds: float
    code_commit: str | None = None
    working_tree_dirty: bool | None = None


class ModelSummary(BaseModel):
    id: str
    sha256: str | None
    parameter_count: int
    training_description: str


class ImageInfo(BaseModel):
    filename: str | None
    sha256: str
    format: str
    mode: str
    shape_hw: list[int]
    bytes: int
    dicom: dict | None = None
    preview_png_base64: str | None = None


class Timing(BaseModel):
    decode: float
    preprocess: float
    inference: float
    total: float


class Score(BaseModel):
    index: int
    label: str
    logit: float
    sigmoid_score: float
    xrv_rescaled_score: float
    rank: int


class PredictionResponse(BaseModel):
    request_id: str
    model: ModelSummary
    image: ImageInfo
    preprocessing: dict
    predictions: list[Score]
    timing_ms: Timing
    warning: str


class DemoImage(BaseModel):
    sample_id: str
    stratum: str
    label: str
    finding_labels: str
    view_position: str
    sha256: str
    url: str
    thumbnail_url: str | None = None


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    detail: ErrorDetail


class ExplainedTarget(BaseModel):
    label: str
    index: int
    sigmoid_score: float
    rank: int


class CamStats(BaseModel):
    shape_hw: list[int]
    min: float
    max: float
    mean: float
    fraction_above_half_in_crop: float
    high_region_bbox_xyxy: list[int] | None


class CamGeometry(BaseModel):
    crop_xyxy_exclusive: list[int]
    original_shape_hw: list[int]
    canvas_shape_hw: list[int]


class ClassTargeting(BaseModel):
    control_label: str
    control_sigmoid_score: float
    mean_abs_difference: float
    gradient_norms: list[float]
    note: str


class ExplainTiming(BaseModel):
    decode: float
    preprocess: float
    explain: float
    render: float
    total: float


class ExplanationResponse(BaseModel):
    request_id: str
    model: ModelSummary
    image: ImageInfo
    target: ExplainedTarget
    method: str
    overlay_png_base64: str
    cam: CamStats
    geometry: CamGeometry
    class_targeting: ClassTargeting
    timing_ms: ExplainTiming
    warning: str


class ExternalImage(BaseModel):
    id: str
    split: str
    image_id: str
    label: int
    label_text: str
    mask_area: int
    mask_area_fraction: float
    has_mask: bool
    url: str
    thumbnail_url: str
    mask_url: str | None = None
    overlay_url: str | None = None


class ExternalImagesResponse(BaseModel):
    count: int
    test_count: int
    validation_count: int
    positive_count: int
    items: list[ExternalImage]


class ExternalPredictRequest(BaseModel):
    image_id: str = Field(min_length=1, max_length=64, examples=["test-0001"])


class ExternalPrediction(BaseModel):
    label: str
    sigmoid_score: float
    logit: float
    rank: int
    threshold: float
    positive: bool


class ExternalReference(BaseModel):
    label: int
    label_text: str
    mask_area: int
    mask_area_fraction: float


class ExternalPredictionResponse(BaseModel):
    request_id: str
    image: ExternalImage
    reference: ExternalReference
    prediction: ExternalPrediction
    match: bool
    threshold_source: str
    predictions: list[Score]
    model: ModelSummary
    timing_ms: Timing
    warning: str
