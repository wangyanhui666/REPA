echo 'set up environment for distributed training...'
# for IB
export NCCL_IB_DISABLE=0
export NCCL_IB_PCI_RELAXED_ORDERING=1
export NCCL_SOCKET_IFNAME=eth0
export NCCL_NET_GDR_LEVEL=5
export NCCL_TOPO_FILE=/opt/microsoft/ndv4-topo.xml
export NCCL_TIMEOUT=1200  # Increase the timeout to 600 seconds
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1200 # Increase the timeout to 1200 seconds

echo "num_gpus"
echo $NUM_GPUS
WORLD_SIZE=$1
NUM_MACHINES=${2:-1}
accelerate launch \
    --config_file "config/sing_a100_config/default_config_${1}.yaml" \
    --num_processes ${WORLD_SIZE} \
    --num_machines ${NUM_MACHINES} \
    --machine_rank ${NODE_RANK} \
    --main_process_ip ${MASTER_ADDR} \
    --main_process_port ${MASTER_PORT} \
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
    --max-train-steps 500000 \
    --num-workers ${1} \
    --output-dir /mnt/t2vg/dit/logs/train_repa_256/ \
    --exp-name="1201_linear-dinov2-b-enc8_baseline_1" \
    --data-dir="/mnt/t2vgusw2_videos/imagenet/sd_latents/REPA_256"