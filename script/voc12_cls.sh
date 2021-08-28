# NEED TO SET
GPU=0,1
DATASET_ROOT=PATH/TO/DATASET
WEIGHT_ROOT=PATH/TO/WEIGHT
SAVE_ROOT=PATH/TO/SAVE
SESSION=voc12_cls


# Default setting
DATASET=voc12
IMG_ROOT=${DATASET_ROOT}/JPEGImages
BACKBONE=resnet38_cls
BASE_WEIGHT=${WEIGHT_ROOT}/ilsvrc-cls_rna-a1_cls1000_ep-0001.params


# 1. train classification network
CUDA_VISIBLE_DEVICES=${GPU} python3 main.py \
    --dataset ${DATASET} \
    --train_list metadata/${DATASET}/train_aug.txt \
    --session ${SESSION} \
    --network network.${BACKBONE} \
    --data_root ${IMG_ROOT} \
    --weights ${BASE_WEIGHT} \
    --crop_size 448 \
    --max_iters 20000 \
    --batch_size 8 \
    --save_root ${SAVE_ROOT}


# 2. inference CAM
INFER_DATA=train # train / train_aug
TRAINED_WEIGHT=${SAVE_ROOT}/${SESSION}/checkpoint_cls.pth

CUDA_VISIBLE_DEVICES=${GPU} python3 infer.py \
    --dataset ${DATASET} \
    --infer_list metadata/${DATASET}/${INFER_DATA}.txt \
    --img_root ${IMG_ROOT} \
    --network network.${BACKBONE} \
    --weights ${TRAINED_WEIGHT} \
    --thr 0.20 \
    --n_gpus 2 \
    --n_processes_per_gpu 1 1 \
    --cam_png ${SAVE_ROOT}/${SESSION}/result/cam_png

# 3. evaluate CAM
GT_ROOT=${DATASET_ROOT}/SegmentationClassAug/

CUDA_VISIBLE_DEVICES=${GPU} python3 evaluate_png.py \
    --dataset ${DATASET} \
    --datalist metadata/${DATASET}/${INFER_DATA}.txt \
    --gt_dir ${GT_ROOT} \
    --save_path ${SAVE_ROOT}/${SESSION}/result/${INFER_DATA}.txt \
    --pred_dir ${SAVE_ROOT}/${SESSION}/result/cam_png