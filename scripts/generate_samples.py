from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline


ROOT = Path(__file__).resolve().parents[1]
RECORDS_DIR = ROOT / "outputs" / "records"
GEN_RECORD_PATH = RECORDS_DIR / "generation_runs.jsonl"


def load_config(config_path: Path) -> dict:
    return json.loads(config_path.read_text(encoding="utf-8"))


def record_run(config: dict, mode: str) -> None:
    RECORDS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "mode": mode,
        "config_path": "configs/generation_config.json",
        "base_model": config["base_model"],
        "lora_path": config["lora_path"],
        "output_dir": config["output_dir"],
        "num_prompts": len(config["prompts"]),
        "num_images_per_prompt": config["num_images_per_prompt"],
    }
    with GEN_RECORD_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args()

    config = load_config(ROOT / args.config)
    output_dir = ROOT / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    record_run(config, mode="print-only" if args.print_only else "run")

    if args.print_only:
        for prompt in config["prompts"]:
            print(prompt)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(config["base_model"], torch_dtype=dtype)
    pipe.load_lora_weights(str(ROOT / config["lora_path"]))
    pipe = pipe.to(device)

    generator = torch.Generator(device=device).manual_seed(config["seed"])

    for prompt_index, prompt in enumerate(config["prompts"], start=1):
        result = pipe(
            prompt=prompt,
            negative_prompt=config["negative_prompt"],
            num_inference_steps=config["num_inference_steps"],
            guidance_scale=config["guidance_scale"],
            num_images_per_prompt=config["num_images_per_prompt"],
            height=config["height"],
            width=config["width"],
            generator=generator,
        )

        for image_index, image in enumerate(result.images, start=1):
            image.save(output_dir / f"prompt_{prompt_index:02d}_img_{image_index:02d}.png")


if __name__ == "__main__":
    main()
