python dataset_tools.py convert \
    --source="$HOME/datasets/imagenet/ILSVRC2012/train/train" \
    --dest="$HOME/datasets/imagenet/REPA_256/images" \
    --resolution=256x256 \
    --transform=center-crop-dhariwal && \

python dataset_tools.py encode --source="$HOME/datasets/imagenet/REPA_256/images" \
    --dest="$HOME/datasets/imagenet/REPA_256/vae-sd"