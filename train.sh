#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

GPUS="4"

ROOT="gc_loss"
# CONFIG='retinanet_obb_r50_fpn_gcl_6x_hrsc'
# CONFIG='retinanet_obb_r50_fpn_gcl_2x_ms_dota'
# CONFIG='retinanet_obb_r50_fpn_gcl_3x_uav_rod'
# CONFIG='retinanet_obb_r50_fpn_gcl_3x_ucas_aod'
# CONFIG='retinanet_obb_r50_fpn_gcl_3x_dior'
CONFIG='retinanet_obb_r50_fpn_gcl_6x_hrsc'
# CONFIG='retinanet_obb_r50_fpn_gcl_1x_ms_fair1m'
# CONFIG='retinanet_obb_r50_fpn_gcl_1x_ms_dota1_5'
# CONFIG='faster_rcnn_orpn_r50_fpn_gcl_3x_hrsc'
# CONFIG='faster_rcnn_roitrans_r50_fpn_gcl_3x_hrsc'


# ROOT="retinanet_obb"
# CONFIG='retinanet_obb_r50_fpn_3x_hrsc'
# CONFIG='retinanet_obb_r50_fpn_9x_msra_td500'


# ROOT="poly_iou_loss"
# CONFIG='retinanet_obb_r50_fpn_iouloss_1x_hrsc'
# CONFIG='faster_rcnn_orpn_r50_fpn_iouloss_3x_hrsc'
# CONFIG='faster_rcnn_roitrans_r50_fpn_iouloss_3x_hrsc'


# ROOT="oriented_rcnn"
# CONFIG='faster_rcnn_orpn_swinT_fpn_1x_dota10'
# CONFIG='faster_rcnn_orpn_swinT_fpn_1x_ms_dota10'
# CONFIG='faster_rcnn_orpn_r50_fpn_gcl_1x_dior'


# ROOT="roi_transformer"
# CONFIG='faster_rcnn_roitrans_r50_fpn_1x_dior'

# ROOT="s2anet"
# CONFIG='s2anet_r50_fpn_1x_dior'


## single GPU
# python tools/train.py configs/obb/$ROOT/$CONFIG.py

# Multiple GPU
./tools/dist_train.sh configs/obb/$ROOT/$CONFIG.py $GPUS

