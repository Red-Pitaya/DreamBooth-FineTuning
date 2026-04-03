from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORDS_DIR = ROOT / "outputs" / "records"
TRAIN_RECORD_PATH = RECORDS_DIR / "training_runs.jsonl"


def load_config(config_path: Path) -> dict:
    return json.loads(config_path.read_text(encoding="utf-8"))


def ensure_layout(config: dict) -> None:
    Path(ROOT / config["output_dir"]).mkdir(parents=True, exist_ok=True)
    Path(ROOT / config["logging_dir"]).mkdir(parents=True, exist_ok=True)
    RECORDS_DIR.mkdir(parents=True, exist_ok=True)


def find_diffusers_script() -> str:
    common_candidates = [
        ROOT / "third_party" / "diffusers" / "examples" / "dreambooth" / "train_dreambooth_lora.py",
        ROOT / "diffusers" / "examples" / "dreambooth" / "train_dreambooth_lora.py",
        ROOT.parent / "diffusers" / "examples" / "dreambooth" / "train_dreambooth_lora.py",
    ]
    override = os.environ.get("DIFFUSERS_DREAMBOOTH_LORA_SCRIPT")
    if override:
        return override
    for candidate in common_candidates:
        if candidate.exists():
            return str(candidate)
    return "train_dreambooth_lora.py"


def build_command(config: dict) -> list[str]:
    script_path = config.get("dreambooth_script") or find_diffusers_script()
    cmd = [
        "accelerate",
        "launch",
        script_path,
        "--pretrained_model_name_or_path",
        config["pretrained_model_name_or_path"],
        "--instance_data_dir",
        str(ROOT / config["instance_data_dir"]),
        "--output_dir",
        str(ROOT / config["output_dir"]),
        "--instance_prompt",
        config["instance_prompt"],
        "--resolution",
        str(config["resolution"]),
        "--train_batch_size",
        str(config["train_batch_size"]),
        "--gradient_accumulation_steps",
        str(config["gradient_accumulation_steps"]),
        "--learning_rate",
        str(config["learning_rate"]),
        "--lr_scheduler",
        config["lr_scheduler"],
        "--lr_warmup_steps",
        str(config["lr_warmup_steps"]),
        "--max_train_steps",
        str(config["max_train_steps"]),
        "--checkpointing_steps",
        str(config["checkpointing_steps"]),
        "--rank",
        str(config["rank"]),
        "--seed",
        str(config["seed"]),
        "--mixed_precision",
        config["mixed_precision"],
        "--num_validation_images",
        str(config["num_validation_images"]),
        "--validation_prompt",
        config["validation_prompt"],
        "--validation_epochs",
        str(config["validation_epochs"]),
        "--logging_dir",
        str(ROOT / config["logging_dir"]),
        "--dataloader_num_workers",
        str(config["dataloader_num_workers"]),
    ]

    if config.get("gradient_checkpointing", False):
        cmd.append("--gradient_checkpointing")
    if config.get("train_text_encoder", False):
        cmd.append("--train_text_encoder")
    if config.get("enable_xformers_memory_efficient_attention", False):
        cmd.append("--enable_xformers_memory_efficient_attention")
    if config.get("use_8bit_adam", False):
        cmd.append("--use_8bit_adam")

    return cmd


def record_run(config: dict, command: list[str], mode: str) -> None:
    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "mode": mode,
        "run_name": config["run_name"],
        "config_path": "configs/dreambooth_lora_config.json",
        "instance_data_dir": config["instance_data_dir"],
        "caption_dir": config.get("caption_dir"),
        "output_dir": config["output_dir"],
        "logging_dir": config["logging_dir"],
        "command": command,
    }
    with TRAIN_RECORD_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args()

    config_path = ROOT / args.config
    config = load_config(config_path)
    ensure_layout(config)
    command = build_command(config)

    script_path = command[2]
    if "/" in script_path or "\\" in script_path:
        if not Path(script_path).exists():
            raise FileNotFoundError(
                f"DreamBooth training script not found: {script_path}\n"
                "Set `dreambooth_script` in the config or export "
                "DIFFUSERS_DREAMBOOTH_LORA_SCRIPT=/path/to/train_dreambooth_lora.py"
            )
    else:
        local_candidate = ROOT / script_path
        if not local_candidate.exists():
            raise FileNotFoundError(
                "Could not locate train_dreambooth_lora.py.\n"
                "This script is usually not included in the pip package alone.\n"
                "Use one of these options:\n"
                "1. export DIFFUSERS_DREAMBOOTH_LORA_SCRIPT=/path/to/diffusers/examples/dreambooth/train_dreambooth_lora.py\n"
                "2. set `dreambooth_script` in configs/dreambooth_lora_config.json\n"
                "3. clone the diffusers repo under ./third_party/diffusers/\n"
            )

    record_run(config, command, mode="print-only" if args.print_only else "run")

    printable = shlex.join(command)
    print(printable)

    if args.print_only:
        return

    subprocess.run(command, check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
