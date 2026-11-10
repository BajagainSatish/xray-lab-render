"""Serve the full FastAPI app with the low-memory ONNX Runtime backend."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import uvicorn

from xray_lab.service.app import create_app
from xray_lab.service.onnx_runtime import OnnxRuntime


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    runtime = OnnxRuntime(
        ROOT / "model.onnx",
        ROOT / "model-meta.json",
    )
    app = create_app(
        runtime,
        project_root=ROOT,
        gallery_manifest=ROOT / "data" / "render" / "nih-reference-library-v1.json",
        validation_root=ROOT / "results",
        external_gallery_manifest=(
            ROOT / "data" / "render" / "siim-pneumothorax-gallery-v1.json"),
        cors_origins=["*"],
        version="0.1.0",
    )
    print(f"ONNX model ready: {runtime.model_id} | {len(runtime.labels)} outputs",
          flush=True)
    print(f"UI: http://{args.host}:{args.port}/app/", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
