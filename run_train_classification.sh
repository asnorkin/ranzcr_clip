WORK_DIR=classification
LR=1e-2
EPOCHS=50
BATCH_SIZE=48
NUM_WORKERS=16

PROJECT=resnext50_32x4d

EXPERIMENT=${PROJECT}
GPUS=0,1

# Add repo root to PYTHONPATH for imports
export PYTHONPATH=.:${PYTHONPATH}

nohup python classification/train.py \
    --project=${PROJECT} \
    --work_dir=${WORK_DIR} \
    --experiment=${EXPERIMENT} \
    --batch_size=${BATCH_SIZE} \
    --num_workers=${NUM_WORKERS} \
    --lr=${LR} \
    --gpus=${GPUS} \
    --max_epochs=${EPOCHS} \
    --val_size=0.2 \
    --accelerator=ddp \
    --cache_images \
    &> ${EXPERIMENT}.log 2>&1 &
