WORK_DIR=segmentation
LR=1e-3
EPOCHS=50
BATCH_SIZE=14
ACC_BATCHES=2
NUM_WORKERS=15
PRECISION=16

PROJECT=unet64_512x512

EXPERIMENT=${PROJECT}_${LR}lr${EPOCHS}e${BATCH_SIZE}x${ACC_BATCHES}b${PRECISION}p
GPUS=0,1

# Train outputs dir
TRAIN_OUTPUTS_DIR=${WORK_DIR}/train_outputs
if [ ! -d ${TRAIN_OUTPUTS_DIR} ]; then
    mkdir ${TRAIN_OUTPUTS_DIR}
fi

# Add repo root to PYTHONPATH for imports
export PYTHONPATH=.:${PYTHONPATH}

nohup python segmentation/train.py \
    --project=${PROJECT} \
    --work_dir=${WORK_DIR} \
    --experiment=${EXPERIMENT} \
    --batch_size=${BATCH_SIZE} \
    --accumulate_grad_batches=${ACC_BATCHES} \
    --num_workers=${NUM_WORKERS} \
    --lr=${LR} \
    --gpus=${GPUS} \
    --num_epochs=${EPOCHS} \
    --accelerator=ddp \
    --precision=${PRECISION} \
    &> ${TRAIN_OUTPUTS_DIR}/${EXPERIMENT}.log 2>&1 &
