"""FastAPI application factory for the X-ray inference service."""

import json
from pathlib import Path
import re
from typing import TYPE_CHECKING
import subprocess
import uuid

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from xray_lab.provenance import sha256_file
from xray_lab.service.common import DISCLAIMER, NoSaliencyError, UnknownLabelError
if TYPE_CHECKING:
    from xray_lab.service.runtime import ModelRuntime

from xray_lab.service.schemas import (
    DemoImage,
    ErrorResponse,
    ExplanationResponse,
    ExternalImage,
    ExternalImagesResponse,
    ExternalPredictRequest,
    ExternalPredictionResponse,
    Health,
    PredictionResponse,
)

VERSION = "0.1.0"
STATIC_DIR = Path(__file__).resolve().parent / "static"
ALLOWED_CONTENT_TYPES = {
    "image/png", "image/jpeg", "image/jpg", "application/dicom",
    "application/octet-stream", "",
}


def load_gallery(project_root: Path, manifest_path: Path) -> dict[str, dict]:
    manifest = json.loads(manifest_path.read_text())
    gallery: dict[str, dict] = {}
    for item in manifest["images"]:
        path = project_root / item["relative_path"]
        if not path.is_file():
            raise RuntimeError(f"Gallery image is missing: {path}")
        if sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"Gallery image differs from its manifest: {path}")
        thumbnail = (project_root / item["thumbnail_relative_path"]
                     if item.get("thumbnail_relative_path") else None)
        if thumbnail is not None:
            if not thumbnail.is_file():
                raise RuntimeError(f"Gallery thumbnail is missing: {thumbnail}")
            if sha256_file(thumbnail) != item["thumbnail_sha256"]:
                raise RuntimeError(f"Gallery thumbnail differs from its manifest: {thumbnail}")
        gallery[item["sample_id"]] = {
            "path": path,
            "thumbnail": thumbnail,
            "stratum": item["stratum"],
            "label": item["label_list"][0] if item["label_list"] else "unknown",
            "finding_labels": item["finding_labels"],
            "view_position": item["view_position"],
            "sha256": item["sha256"],
        }
    if not gallery:
        raise RuntimeError("Gallery manifest contains no images.")
    return gallery


def code_state(root: Path) -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                                capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=root,
                                    check=True, capture_output=True, text=True).stdout.strip())
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def read_upload(file: UploadFile, runtime: ModelRuntime) -> tuple[bytes, str]:
    content_type = (file.content_type or "").lower()
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail={
            "code": "unsupported_media_type",
            "message": f"Unsupported content type {content_type!r}; send PNG or JPEG.",
        })
    data = file.file.read(runtime.max_upload_bytes + 1)
    if len(data) > runtime.max_upload_bytes:
        raise HTTPException(status_code=413, detail={
            "code": "payload_too_large",
            "message": f"Upload exceeds {runtime.max_upload_bytes} bytes.",
        })
    return data, Path(file.filename or "upload").name[:200]


def latest_validation_run(results_root: Path) -> Path | None:
    candidates = sorted((Path(results_root)).glob("external_siim_*"),
                        key=lambda path: path.name)
    for candidate in reversed(candidates):
        if (candidate / "run.json").is_file():
            return candidate
    return None


def load_validation(root: Path) -> tuple[Path, dict] | None:
    run_dir = latest_validation_run(root)
    if run_dir is None:
        return None
    return run_dir, json.loads((run_dir / "run.json").read_text())



def load_external_gallery(project_root: Path, manifest_path: Path) -> dict[str, dict]:
    """Load the optional browsable SIIM gallery without making the API depend on it."""
    if not manifest_path.is_file():
        return {}
    manifest = json.loads(manifest_path.read_text())
    gallery: dict[str, dict] = {}
    for item in manifest.get("items", []):
        image_path = project_root / item["image_relative_path"]
        thumbnail_path = project_root / item["thumbnail_relative_path"]
        if not image_path.is_file() or not thumbnail_path.is_file():
            continue
        if sha256_file(image_path) != item.get("image_sha256"):
            continue
        mask_path = (project_root / item["mask_relative_path"]
                     if item.get("mask_relative_path") else None)
        overlay_path = (project_root / item["overlay_relative_path"]
                        if item.get("overlay_relative_path") else None)
        gallery[item["id"]] = {
            "id": item["id"],
            "split": item["split"],
            "image_id": item["image_id"],
            "label": int(item["label"]),
            "label_text": item["label_text"],
            "mask_area": int(item["mask_area"]),
            "mask_area_fraction": float(item["mask_area_fraction"]),
            "image_path": image_path,
            "thumbnail_path": thumbnail_path,
            "mask_path": mask_path if mask_path and mask_path.is_file() else None,
            "overlay_path": (overlay_path
                             if overlay_path and overlay_path.is_file() else None),
        }
    return gallery


def external_payload(item: dict) -> dict:
    base = f"/v1/validation/pneumothorax/images/{item['id']}"
    return {
        "id": item["id"],
        "split": item["split"],
        "image_id": item["image_id"],
        "label": item["label"],
        "label_text": item["label_text"],
        "mask_area": item["mask_area"],
        "mask_area_fraction": item["mask_area_fraction"],
        "has_mask": item["mask_path"] is not None,
        "url": base,
        "thumbnail_url": f"{base}/thumbnail",
        "mask_url": f"{base}/mask" if item["mask_path"] else None,
        "overlay_url": f"{base}/overlay" if item["overlay_path"] else None,
    }


def create_app(runtime: "ModelRuntime", *, project_root: Path, gallery_manifest: Path,
               cors_origins: list[str] | None = None,
               validation_root: Path | None = None,
               external_gallery_manifest: Path | None = None,
               version: str = VERSION) -> FastAPI:
    project_root = Path(project_root)
    validation_root = Path(validation_root) if validation_root else project_root / "results"
    gallery = load_gallery(project_root, Path(gallery_manifest))
    external_manifest = (Path(external_gallery_manifest) if external_gallery_manifest
                         else project_root / "data" / "manifests" /
                         "siim-pneumothorax-gallery-v1.json")
    external_gallery = load_external_gallery(project_root, external_manifest)
    code_commit, working_tree_dirty = code_state(project_root)
    app = FastAPI(
        title="X-ray Lab Inference API",
        version=version,
        description=(
            "Upload a single 8-bit grayscale chest X-ray (PNG or JPEG) and receive "
            "18 uncalibrated multilabel scores. This is a research demonstration, "
            "not a diagnostic device. Uploads are processed in memory and are not "
            "stored by the service."
        ),
        openapi_tags=[
            {"name": "meta", "description": "Service status and contracts"},
            {"name": "inference", "description": "Image scoring"},
            {"name": "demo", "description": "Documented gallery for demonstrations"},
            {"name": "validation", "description": "External validation results"},
        ],
    )
    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        supplied = request.headers.get("X-Request-ID", "")
        request_id = (supplied if re.fullmatch(r"[A-Za-z0-9_-]{8,64}", supplied)
                      else uuid.uuid4().hex)
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    @app.get("/", tags=["meta"])
    def root() -> dict:
        return {
            "service": "xray-lab-inference",
            "version": version,
            "app": "/app/",
            "docs": "/docs",
            "openapi": "/openapi.json",
            "health": "/health",
            "predict": "/v1/predict",
            "demo_images": "/v1/demo-images",
            "validation": "/v1/validation/pneumothorax",
            "validation_images": "/v1/validation/pneumothorax/images",
            "warning": DISCLAIMER,
        }

    @app.get("/health", response_model=Health, tags=["meta"])
    def health() -> dict:
        return {
            "status": "ok",
            "version": version,
            "model_id": runtime.model_id,
            "model_loaded": True,
            "uptime_seconds": round(runtime.uptime_seconds(), 3),
            "code_commit": code_commit,
            "working_tree_dirty": working_tree_dirty,
        }

    @app.get("/v1/models", tags=["meta"])
    def model_card() -> dict:
        return runtime.card()

    @app.get("/v1/demo-images", response_model=list[DemoImage], tags=["demo"])
    def demo_images() -> list[dict]:
        return [
            {
                "sample_id": sample_id,
                "stratum": item["stratum"],
                "label": item["label"],
                "finding_labels": item["finding_labels"],
                "view_position": item["view_position"],
                "sha256": item["sha256"],
                "url": f"/v1/demo-images/{sample_id}",
                "thumbnail_url": (f"/v1/demo-images/{sample_id}/thumbnail"
                                  if item["thumbnail"] else None),
            }
            for sample_id, item in sorted(gallery.items())
        ]

    @app.get("/v1/demo-images/{sample_id}", tags=["demo"],
             responses={404: {"model": ErrorResponse}})
    def demo_image(sample_id: str) -> FileResponse:
        item = gallery.get(sample_id)
        if item is None:
            raise HTTPException(status_code=404, detail={
                "code": "demo_image_not_found",
                "message": f"No demo image with id {sample_id!r}.",
            })
        return FileResponse(item["path"], media_type="image/png",
                            headers={"Cache-Control": "public, max-age=3600"})

    @app.get("/v1/demo-images/{sample_id}/thumbnail", tags=["demo"],
             responses={404: {"model": ErrorResponse}})
    def demo_thumbnail(sample_id: str) -> FileResponse:
        item = gallery.get(sample_id)
        if item is None or item["thumbnail"] is None:
            raise HTTPException(status_code=404, detail={
                "code": "demo_thumbnail_not_found",
                "message": f"No thumbnail for demo image {sample_id!r}.",
            })
        return FileResponse(item["thumbnail"], media_type="image/png",
                            headers={"Cache-Control": "public, max-age=3600"})

    @app.get("/v1/validation/pneumothorax", tags=["validation"],
             responses={404: {"model": ErrorResponse}})
    def validation_pneumothorax() -> dict:
        loaded = load_validation(validation_root)
        if loaded is None:
            raise HTTPException(status_code=404, detail={
                "code": "validation_not_run",
                "message": "No external validation run found. Run scripts/run_external_validation.py.",
            })
        run_dir, record = loaded
        base = "/v1/validation/pneumothorax/figures"
        record = dict(record)
        record["run_directory"] = run_dir.name
        record["figures"] = {
            name: {**meta, "url": f"{base}/{name}"}
            for name, meta in record.get("figures", {}).items()
        }
        return record

    @app.get("/v1/validation/pneumothorax/figures/{name}", tags=["validation"],
             responses={404: {"model": ErrorResponse}})
    def validation_figure(name: str) -> FileResponse:
        loaded = load_validation(validation_root)
        if loaded is None:
            raise HTTPException(status_code=404, detail={
                "code": "validation_not_run",
                "message": "No external validation run found.",
            })
        run_dir, record = loaded
        if name not in record.get("figures", {}):
            raise HTTPException(status_code=404, detail={
                "code": "validation_figure_not_found",
                "message": f"Unknown validation figure {name!r}.",
            })
        path = run_dir / name
        if not path.is_file():
            raise HTTPException(status_code=404, detail={
                "code": "validation_figure_missing",
                "message": f"Figure file is missing: {name}.",
            })
        return FileResponse(path, media_type="image/png",
                            headers={"Cache-Control": "public, max-age=300"})

    @app.get("/v1/validation/pneumothorax/images", response_model=ExternalImagesResponse,
             tags=["validation"], responses={404: {"model": ErrorResponse}})
    def validation_images() -> dict:
        if not external_gallery:
            raise HTTPException(status_code=404, detail={
                "code": "external_images_not_built",
                "message": "No external image gallery found. Run scripts/build_external_gallery.py.",
            })
        items = [external_payload(item) for item in external_gallery.values()]
        items.sort(key=lambda item: (item["split"] != "test", item["id"]))
        return {
            "count": len(items),
            "test_count": sum(1 for item in items if item["split"] == "test"),
            "validation_count": sum(1 for item in items if item["split"] == "validation"),
            "positive_count": sum(1 for item in items if item["label"] == 1),
            "items": items,
        }

    @app.get("/v1/validation/pneumothorax/images/{image_id}", tags=["validation"],
             responses={404: {"model": ErrorResponse}})
    def validation_image(image_id: str) -> FileResponse:
        item = external_gallery.get(image_id)
        if item is None:
            raise HTTPException(status_code=404, detail={
                "code": "external_image_not_found",
                "message": f"No external image with id {image_id!r}.",
            })
        return FileResponse(item["image_path"], media_type="image/png",
                            headers={"Cache-Control": "public, max-age=3600"})

    @app.get("/v1/validation/pneumothorax/images/{image_id}/thumbnail", tags=["validation"],
             responses={404: {"model": ErrorResponse}})
    def validation_image_thumbnail(image_id: str) -> FileResponse:
        item = external_gallery.get(image_id)
        if item is None:
            raise HTTPException(status_code=404, detail={
                "code": "external_image_not_found",
                "message": f"No external image with id {image_id!r}.",
            })
        return FileResponse(item["thumbnail_path"], media_type="image/png",
                            headers={"Cache-Control": "public, max-age=3600"})

    @app.get("/v1/validation/pneumothorax/images/{image_id}/mask", tags=["validation"],
             responses={404: {"model": ErrorResponse}})
    def validation_image_mask(image_id: str) -> FileResponse:
        item = external_gallery.get(image_id)
        if item is None or item["mask_path"] is None:
            raise HTTPException(status_code=404, detail={
                "code": "external_mask_not_found",
                "message": f"No reference mask for external image {image_id!r}.",
            })
        return FileResponse(item["mask_path"], media_type="image/png",
                            headers={"Cache-Control": "public, max-age=3600"})

    @app.get("/v1/validation/pneumothorax/images/{image_id}/overlay", tags=["validation"],
             responses={404: {"model": ErrorResponse}})
    def validation_image_overlay(image_id: str) -> FileResponse:
        item = external_gallery.get(image_id)
        if item is None or item["overlay_path"] is None:
            raise HTTPException(status_code=404, detail={
                "code": "external_overlay_not_found",
                "message": f"No reference overlay for external image {image_id!r}.",
            })
        return FileResponse(item["overlay_path"], media_type="image/png",
                            headers={"Cache-Control": "public, max-age=3600"})

    @app.post("/v1/validation/pneumothorax/predict",
              response_model=ExternalPredictionResponse, tags=["validation"],
              responses={404: {"model": ErrorResponse},
                         500: {"model": ErrorResponse}})
    def validation_predict(payload: ExternalPredictRequest, request: Request) -> dict:
        item = external_gallery.get(payload.image_id)
        if item is None:
            raise HTTPException(status_code=404, detail={
                "code": "external_image_not_found",
                "message": f"No external image with id {payload.image_id!r}.",
            })
        data = item["image_path"].read_bytes()
        body = runtime.predict(data, filename=item["id"] + ".png")
        row = next((row for row in body["predictions"]
                    if row["label"] == "Pneumothorax"), None)
        if row is None:
            raise HTTPException(status_code=500, detail={
                "code": "pneumothorax_label_missing",
                "message": "The loaded model does not expose a Pneumothorax output.",
            })
        loaded = load_validation(validation_root)
        if loaded is not None:
            _, record = loaded
            threshold = float(record["validation"]["threshold"])
            threshold_source = (
                "latest external validation run (Youden J on the validation split)")
        else:
            threshold = float(runtime.thresholds[runtime.labels.index("Pneumothorax")])
            threshold_source = "XRV operating threshold (no validation run found)"
        prediction_positive = bool(row["sigmoid_score"] >= threshold)
        return {
            "request_id": request.state.request_id,
            "image": external_payload(item),
            "reference": {
                "label": item["label"],
                "label_text": item["label_text"],
                "mask_area": item["mask_area"],
                "mask_area_fraction": item["mask_area_fraction"],
            },
            "prediction": {
                "label": row["label"],
                "sigmoid_score": float(row["sigmoid_score"]),
                "logit": float(row["logit"]),
                "rank": int(row["rank"]),
                "threshold": threshold,
                "positive": prediction_positive,
            },
            "match": prediction_positive == bool(item["label"]),
            "threshold_source": threshold_source,
            "predictions": body["predictions"],
            "model": body["model"],
            "timing_ms": body["timing_ms"],
            "warning": body["warning"],
        }

    @app.post("/v1/predict", response_model=PredictionResponse, tags=["inference"],
              responses={400: {"model": ErrorResponse},
                         413: {"model": ErrorResponse},
                         415: {"model": ErrorResponse}})
    def predict(request: Request, file: UploadFile = File(
            ..., description="8-bit grayscale chest X-ray, PNG or JPEG")) -> dict:
        request_id = request.state.request_id
        data, filename = read_upload(file, runtime)
        try:
            result = runtime.predict(data, filename=filename)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail={
                "code": "invalid_image",
                "message": str(exc),
            }) from exc
        return {"request_id": request_id, **result}

    @app.post("/v1/explain", response_model=ExplanationResponse, tags=["inference"],
              responses={400: {"model": ErrorResponse},
                         413: {"model": ErrorResponse},
                         415: {"model": ErrorResponse},
                         422: {"model": ErrorResponse}})
    def explain(request: Request,
                file: UploadFile = File(
                    ..., description="8-bit grayscale chest X-ray, PNG or JPEG"),
                label: str = Form(
                    ..., description="One exact model output label, e.g. Hernia")) -> dict:
        request_id = request.state.request_id
        data, filename = read_upload(file, runtime)
        try:
            result = runtime.explain(data, label=label, filename=filename)
        except UnknownLabelError as exc:
            raise HTTPException(status_code=400, detail={
                "code": "unknown_label",
                "message": str(exc),
            }) from exc
        except NoSaliencyError as exc:
            raise HTTPException(status_code=422, detail={
                "code": "no_saliency",
                "message": str(exc),
            }) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail={
                "code": "invalid_image",
                "message": str(exc),
            }) from exc
        return {"request_id": request_id, **result}

    if STATIC_DIR.is_dir():
        app.mount("/app", StaticFiles(directory=str(STATIC_DIR), html=True), name="ui")

    return app
