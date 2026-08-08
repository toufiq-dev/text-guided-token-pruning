"""Image loading / preprocessing helpers.

``transformers`` is imported lazily inside functions so that unit tests and
lightweight demos (which use mocks) do not require a full model download.
"""

from __future__ import annotations

from typing import Any

from PIL import Image


def load_image(source: Any) -> Image.Image:
    """Load a PIL image from a path, URL, bytes, or existing PIL image."""
    if isinstance(source, Image.Image):
        return source.convert("RGB")
    if isinstance(source, (str, bytes, bytearray)):
        from io import BytesIO

        import requests

        if isinstance(source, str) and (source.startswith("http://") or source.startswith("https://")):
            resp = requests.get(source, timeout=60)
            resp.raise_for_status()
            return Image.open(BytesIO(resp.content)).convert("RGB")
        if isinstance(source, str):
            return Image.open(source).convert("RGB")
        return Image.open(BytesIO(source)).convert("RGB")
    raise TypeError(f"Unsupported image source type: {type(source)}")


def resize_for_model(image: Image.Image, size: tuple[int, int] = (384, 384)) -> Image.Image:
    """Center-crop-resize to a fixed square (matches SigLIP-384 preprocessing)."""
    w, h = image.size
    side = min(w, h)
    image = image.crop(((w - side) // 2, (h - side) // 2, (w + side) // 2, (h + side) // 2))
    return image.resize(size, Image.BILINEAR)


def get_image_processor(model_name: str, **kwargs: Any) -> Any:
    """Load the HuggingFace image processor for a model (lazy import)."""
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(model_name, trust_remote_code=True, **kwargs)


def process_image(
    image: Image.Image,
    processor: Any,
    model_name: str,
    size: tuple[int, int] | None = None,
) -> Any:
    """Preprocess a PIL image into the model's expected pixel values format."""
    if size is not None:
        image = resize_for_model(image, size)
    kwargs = {}
    if "Idefics3" in model_name or "SmolVLM" in model_name:
        kwargs["return_tensors"] = "pt"
        processed = processor(images=image, return_tensors="pt")
        return processed.get("pixel_values", processed)
    if "llava" in model_name.lower():
        return processor(images=image, return_tensors="pt")["pixel_values"]
    return processor(images=image, return_tensors="pt")["pixel_values"]
