A small-scale, DreamBooth-style subject-driven image generation fine-tuning experiment. The goal is to complete the entire process: data collection → model fine-tuning → image generation → result analysis. This aims to understand how the generative model learns specific visual features and the impact of training parameter selection on the results.

# set environment
```bash
git clone https://github.com/huggingface/diffusers.git third_party/diffusers
export DIFFUSERS_DREAMBOOTH_LORA_SCRIPT=/home/data5/lyl/5740hw2/third_party/diffusers/examples/dreambooth/train_dreambooth_lora.py
ls /home/data5/lyl/5740hw2/third_party/diffusers/examples/dreambooth/train_dreambooth_lora.py

cd /home/data5/lyl/5740hw2/third_party/diffusers
python -m pip install -e ".[training]"
```

# train
```bash
CUDA_VISIBLE_DEVICES=4 python scripts/run_dreambooth_lora.py --config configs/dreambooth_lora_config.json
```

# generate
```bash
CUDA_VISIBLE_DEVICES=4 python scripts/generate_samples.py --config configs/generation_config.json
```
