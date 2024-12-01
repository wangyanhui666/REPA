#!/bin/bash
export NCCL_SOCKET_IFNAME=eth0
export WANDB_API_KEY="local-9fd3e86800565dfa4b2a5592ccfaf45ef59ec0c6"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1200 # Increase the timeout to 1200 seconds
# --gradient_accumulation_steps=4 \
# --gradient_checkpointing \
wandb login local-9fd3e86800565dfa4b2a5592ccfaf45ef59ec0c6 --relogin --host=https://microsoft-research.wandb.io
accelerate launch \
    --config_file ./config/default_config.yaml \
    train.py \
    --report-to="wandb" \
    --allow-tf32 \
    --mixed-precision="fp16" \
    --seed=0 \
    --path-type="linear" \
    --prediction="v" \
    --weighting="uniform" \
    --model="SiT-XL/2" \
    --enc-type="dinov2-vit-b" \
    --proj-coeff=0.5 \
    --encoder-depth=8 \
    --output-dir ~/guangtingsc/t2vg/dit/logs/debug_repa_256/1201_debug_repa_1 \
    --exp-name="linear-dinov2-b-enc8" \
    --data-dir="/home/t2vg-a100-G4-42/t2vgusw2/videos/imagenet/sd_latents/REPA_256"
