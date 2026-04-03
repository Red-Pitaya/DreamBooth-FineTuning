from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "data" / "train" / "images"
SRC_CAPTION_DIR = ROOT / "data" / "train" / "captions"
OUT_DIR = ROOT / "data" / "train_preprocessed" / "images"
OUT_CAPTION_DIR = ROOT / "data" / "train_preprocessed" / "captions"
MANIFEST_PATH = ROOT / "data" / "manifests" / "preprocess_manifest.json"
TARGET_SIZE = 512
BACKGROUND_COLOR = (255, 255, 255)
ENHANCED_CAPTIONS: dict[str, str] = {
    "sksmiku_01": "a close-up photo of sksmiku figurine",
    "sksmiku_02": "a half-body photo of sksmiku figurine",
    "sksmiku_03": "a full-body front view photo of sksmiku figurine",
    "sksmiku_04": "a full-body back view photo of sksmiku figurine",
    "sksmiku_05": "a full-body three-quarter view photo of sksmiku figurine",
    "sksmiku_06": "a full-body studio photo of sksmiku figurine",
    "sksmiku_07": "a full-body photo of sksmiku figurine",
    "sksmiku_08": "a close-up photo of sksmiku figurine",
    "sksmiku_09": "a medium shot photo of sksmiku figurine",
    "sksmiku_10": "a full-body photo of sksmiku figurine",
}


# Crop boxes are (left, top, right, bottom) in source pixel coordinates.
# They follow the project plan: keep the figurine complete, reduce noisy
# background, and preserve key details like twintails, fan, and lifted leg.
CROP_BOXES: dict[str, tuple[int, int, int, int]] = {
    "sksmiku_01.png": (70, 35, 790, 755),
    "sksmiku_02.png": (65, 35, 790, 760),
    "sksmiku_03.png": (110, 20, 695, 800),
    "sksmiku_04.png": (95, 10, 705, 800),
    "sksmiku_05.png": (90, 20, 710, 800),
    "sksmiku_06.png": (95, 15, 705, 800),
    "sksmiku_07.jpg": (120, 130, 1085, 1160),
    "sksmiku_08.jpg": (135, 80, 1085, 1030),
    "sksmiku_09.jpg": (105, 90, 1060, 1045),
    "sksmiku_10.jpg": (140, 70, 1100, 1225),
}


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_CAPTION_DIR.mkdir(parents=True, exist_ok=True)


def validate_box(image: Image.Image, box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    left, top, right, bottom = box
    left = max(0, min(left, image.width - 2))
    top = max(0, min(top, image.height - 2))
    right = max(left + 1, min(right, image.width))
    bottom = max(top + 1, min(bottom, image.height))
    return left, top, right, bottom


def resize_with_padding(image: Image.Image) -> tuple[Image.Image, dict[str, object]]:
    width, height = image.size
    scale = min(TARGET_SIZE / width, TARGET_SIZE / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))

    resized = image.resize((resized_width, resized_height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (TARGET_SIZE, TARGET_SIZE), BACKGROUND_COLOR)

    offset_x = (TARGET_SIZE - resized_width) // 2
    offset_y = (TARGET_SIZE - resized_height) // 2
    canvas.paste(resized, (offset_x, offset_y))

    return canvas, {
        "cropped_size": [width, height],
        "resized_size": [resized_width, resized_height],
        "padding": {
            "left": offset_x,
            "top": offset_y,
            "right": TARGET_SIZE - resized_width - offset_x,
            "bottom": TARGET_SIZE - resized_height - offset_y,
        },
    }


def preprocess_image(src_path: Path, dst_path: Path, crop_box: tuple[int, int, int, int]) -> dict[str, object]:
    with Image.open(src_path) as image:
        image = image.convert("RGB")
        crop_box = validate_box(image, crop_box)
        cropped = image.crop(crop_box)
        output_image, resize_info = resize_with_padding(cropped)
        output_image.save(dst_path, quality=95)

    return {
        "source": str(src_path.relative_to(ROOT)),
        "output": str(dst_path.relative_to(ROOT)),
        "crop_box": list(crop_box),
        **resize_info,
        "target_size": [TARGET_SIZE, TARGET_SIZE],
    }


def write_caption(stem: str) -> None:
    dst = OUT_CAPTION_DIR / f"{stem}.txt"
    caption = ENHANCED_CAPTIONS.get(stem)

    if caption is None:
        src = SRC_CAPTION_DIR / f"{stem}.txt"
        caption = src.read_text(encoding="utf-8").strip()

    dst.write_text(caption + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs()
    manifest: list[dict[str, object]] = []

    for filename, crop_box in CROP_BOXES.items():
        src_path = SRC_DIR / filename
        if not src_path.exists():
            raise FileNotFoundError(f"Missing source image: {src_path}")

        dst_path = OUT_DIR / filename
        manifest.append(preprocess_image(src_path, dst_path, crop_box))
        write_caption(src_path.stem)

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Processed {len(manifest)} images into {OUT_DIR}")


if __name__ == "__main__":
    main()
