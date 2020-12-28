WORK_DIR=classification
LR=1e-3
EPOCHS=20
BATCH_SIZE=48
NUM_WORKERS=32
PRECISION=16
VAL_SIZE=0.2

PROJECT=efficientnet_b0

EXPERIMENT=${PROJECT}_single_model
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
    --num_epochs=${EPOCHS} \
    --val_size=${VAL_SIZE} \
    --accelerator=ddp \
    --precision=${PRECISION} \
    --cache_images \
    &> ${EXPERIMENT}.log 2>&1 &
