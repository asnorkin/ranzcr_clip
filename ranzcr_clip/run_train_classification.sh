WORK_DIR=classification
LR=1e-3
EPOCHS=30
BATCH_SIZE=48
NUM_WORKERS=15
PRECISION=16

PROJECT=efficientnet_b0

EXPERIMENT=${PROJECT}_${LR}lr${EPOCHS}e${BATCH_SIZE}b${PRECISION}p
GPUS=0,1

# Train outputs dir
TRAIN_OUTPUTS_DIR=${WORK_DIR}/train_outputs
if [ ! -d ${TRAIN_OUTPUTS_DIR} ]; then
    mkdir ${TRAIN_OUTPUTS_DIR}
fi

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
    --val_type=single \
    --accelerator=ddp \
    --precision=${PRECISION} \
    --use_tta \
    &> ${TRAIN_OUTPUTS_DIR}/${EXPERIMENT}.log 2>&1 &
