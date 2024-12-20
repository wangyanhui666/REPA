torchrun --nnodes=1 --nproc_per_node=4 generate.py \
  --model SiT-XL/2 \
  --num-fid-samples 50000 \
  --ckpt /home/t2vg-a100-G4-42/guangtingsc/t2vg/dit/logs/train_repa_256/1201_linear-dinov2-b-enc8_baseline_1/1201_linear-dinov2-b-enc8_baseline_1/checkpoints/0400000.pt \
  --sample-dir  /home/t2vg-a100-G4-42/guangtingsc/t2vg/dit/logs/train_repa_256/1201_linear-dinov2-b-enc8_baseline_1/1201_linear-dinov2-b-enc8_baseline_1/checkpoints_0400000\
  --vae "ema" \
  --path-type=linear \
  --encoder-depth=8 \
  --projector-embed-dims=768 \
  --per-proc-batch-size=64 \
  --mode=sde \
  --num-steps=250 \
  --cfg-scale=1.0 \
  --guidance-high=0.7 && \

echo "Done"
