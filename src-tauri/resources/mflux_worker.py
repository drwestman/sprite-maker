#!/usr/bin/env python3
"""Serialized JSON-lines worker for the managed MFLUX Z-Image Turbo runtime."""

from __future__ import annotations

import contextlib
import gc
import hashlib
import json
import queue
import sys
import threading
from pathlib import Path
from typing import Any


RUNTIME_VERSION = "python3.11;mflux==0.19.1;mlx==0.32.0"
DEFAULT_STEPS = 9
DEFAULT_GUIDANCE = 0.0


def emit(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def fail(request_id: str, message: str) -> None:
    emit({"id": request_id, "type": "error", "error": message})


def checkpoint_key(repository: str, revision: str) -> str:
    return hashlib.sha256(f"{repository}@{revision}".encode("utf-8")).hexdigest()[:24]


def checkpoint_has_controlnet_metadata(directory: Path) -> bool:
    for path in directory.rglob("*.json"):
        try:
            if path.stat().st_size > 2_000_000:
                continue
            if "controlnet" in path.read_text(encoding="utf-8", errors="ignore").lower():
                return True
        except OSError:
            continue
    return any("controlnet" in path.as_posix().lower() for path in directory.rglob("*"))


class MfluxWorker:
    def __init__(self) -> None:
        self.model: Any = None
        self.model_key: tuple[str, str] | None = None
        self.cancel_requested = threading.Event()

    def unload_model(self) -> None:
        self.model = None
        self.model_key = None
        gc.collect()
        try:
            import mlx.core as mx

            clear_cache = getattr(mx, "clear_cache", None)
            if callable(clear_cache):
                clear_cache()
        except ImportError:
            pass

    def _model_class(self) -> Any:
        try:
            from mflux.models.z_image import ZImageTurbo

            return ZImageTurbo
        except ImportError:
            from mflux.models.z_image.variants.z_image import ZImage

            return ZImage

    def load_model(self, request: dict[str, Any]) -> None:
        repository = str(request["repository"]).strip()
        revision = str(request["revision"]).strip()
        checkpoint_root = Path(request["checkpointPath"]).expanduser().resolve()
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        key = (repository, revision)
        if self.model is not None and self.model_key == key:
            return
        self.unload_model()
        checkpoint = checkpoint_root / checkpoint_key(repository, revision)
        checkpoint.mkdir(parents=True, exist_ok=True)
        with contextlib.redirect_stdout(sys.stderr):
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=repository,
                revision=revision,
                local_dir=str(checkpoint),
                cache_dir=str(checkpoint_root / ".hub"),
            )
            model_class = self._model_class()
            from mflux.models.common.resolution.config_resolution import ConfigResolution

            kwargs: dict[str, Any] = {
                "quantize": 8,
                "model_path": str(checkpoint),
                "model_config": ConfigResolution.resolve_restricted(
                    "z-image-turbo",
                    "z-image-turbo",
                    model_path=str(checkpoint),
                ),
            }
            self.model = model_class(**kwargs)
        bits = getattr(self.model, "bits", getattr(self.model, "quantize", None))
        if str(bits) not in {"8", "8.0"}:
            self.unload_model()
            raise RuntimeError("Loaded checkpoint is not 8-bit quantized")
        model_config = getattr(self.model, "model_config", None)
        if (
            str(getattr(model_config, "model_name", "")).lower() != "tongyi-mai/z-image-turbo"
            or getattr(model_config, "controlnet_model", None) is not None
            or checkpoint_has_controlnet_metadata(checkpoint)
        ):
            self.unload_model()
            raise RuntimeError("Loaded checkpoint is not compatible with Z-Image Turbo")
        self.model_key = key

    def generate(self, request: dict[str, Any]) -> None:
        request_id = str(request["id"])
        self.cancel_requested.clear()
        self.load_model(request)
        if self.cancel_requested.is_set():
            emit({"id": request_id, "type": "cancelled"})
            return
        output_path = Path(request["outputPath"]).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image_path = request.get("imagePath")
        image_path = str(Path(image_path).expanduser().resolve()) if image_path else None
        steps = int(request.get("steps", DEFAULT_STEPS))
        guidance = float(request.get("guidance", DEFAULT_GUIDANCE))
        if guidance != 0.0:
            raise ValueError("Z-Image Turbo requires guidance 0")
        emit({"id": request_id, "type": "started", "steps": steps})
        with contextlib.redirect_stdout(sys.stderr):
            kwargs: dict[str, Any] = {
                "seed": int(request.get("seed", 42)),
                "prompt": str(request["prompt"]),
                "num_inference_steps": steps,
                "width": int(request["width"]),
                "height": int(request["height"]),
                "guidance": guidance,
            }
            if image_path is not None:
                kwargs["image_path"] = image_path
                kwargs["image_strength"] = request.get("imageStrength")
            image = self.model.generate_image(
                **kwargs,
            )
            image.save(str(output_path))
        emit({"id": request_id, "type": "completed", "outputPath": str(output_path)})

    def handle(self, request: dict[str, Any]) -> None:
        request_id = str(request.get("id", "unknown"))
        request_type = request.get("type")
        if request_type == "ping":
            emit({"id": request_id, "type": "pong", "runtimeVersion": RUNTIME_VERSION})
            return
        if request_type == "cancel":
            self.cancel_requested.set()
            emit({"id": request_id, "type": "cancelled"})
            return
        if request_type != "generate":
            fail(request_id, f"Unsupported worker request type: {request_type}")
            return
        try:
            self.generate(request)
        except KeyboardInterrupt:
            emit({"id": request_id, "type": "cancelled"})
        except Exception as error:  # noqa: BLE001 - worker must report request failures
            print(f"MFLUX worker request failed: {error}", file=sys.stderr)
            fail(request_id, str(error))


def reader(messages: queue.Queue[dict[str, Any]]) -> None:
    for line in sys.stdin:
        try:
            message = json.loads(line)
        except json.JSONDecodeError as error:
            fail("unknown", f"Invalid JSON request: {error}")
            continue
        if isinstance(message, dict):
            if message.get("type") == "cancel":
                worker.cancel_requested.set()
                try:
                    import _thread

                    _thread.interrupt_main()
                except RuntimeError:
                    pass
            else:
                messages.put(message)


worker = MfluxWorker()
messages: queue.Queue[dict[str, Any]] = queue.Queue()
threading.Thread(target=reader, args=(messages,), daemon=True).start()
while True:
    try:
        request = messages.get()
        worker.handle(request)
    except KeyboardInterrupt:
        continue
