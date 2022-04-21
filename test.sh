#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}


GPUS="4"

# ROOT="retinanet_obb"
# CONFIG='retinanet_obb_r50_fpn_3x_hrsc'
# CONFIG='retinanet_obb_r50_fpn_9x_msra_td500'


ROOT="gc_loss"
# CONFIG='s2anet_r50_fpn_gcl_3x_hrsc'
# CONFIG='s2anet_r50_fpn_gcl_1x_dior'
# CONFIG='s2anet_r50_fpn_gcl_3x_dior'
# CONFIG='s2anet_r50_fpn_gcl_1x_dota10'
# CONFIG='s2anet_r50_fpn_gcl_1x_dota15'
# CONFIG='s2anet_r50_fpn_gcl_1x_fair1m'
CONFIG='retinanet_obb_r50_fpn_gcl_6x_hrsc'
# CONFIG='retinanet_obb_r50_fpn_gcl_3x_dior'
# CONFIG='retinanet_obb_r50_fpn_gcl_3x_uav_rod'
# CONFIG='retinanet_obb_r50_fpn_gcl_3x_ucas_aod'
# CONFIG='retinanet_obb_r50_fpn_gcl_2x_ms_dota'
# CONFIG='retinanet_obb_r50_fpn_gcl_1x_ms_fair1m'
# CONFIG='retinanet_obb_r50_fpn_gcl_1x_ms_dota1_5'
# CONFIG='faster_rcnn_orpn_r50_fpn_gcl_3x_hrsc'
# CONFIG='faster_rcnn_orpn_r50_fpn_gcl_1x_dior'
# CONFIG='faster_rcnn_orpn_r50_fpn_gcl_1x_dota10'
# CONFIG='faster_rcnn_orpn_swinT_fpn_gcl_1x_ms_dota10'
# CONFIG='faster_rcnn_orpn_r50_fpn_gcl_1x_dota15'
# CONFIG='faster_rcnn_orpn_r50_fpn_gcl_1x_fair1m'
# CONFIG='faster_rcnn_roitrans_r50_fpn_gcl_3x_hrsc'
# CONFIG='faster_rcnn_roitrans_r50_fpn_gcl_1x_dior'
# CONFIG='faster_rcnn_roitrans_r50_fpn_gcl_1x_dota10'
# CONFIG='faster_rcnn_roitrans_r50_fpn_gcl_1x_dota15'
# CONFIG='faster_rcnn_roitrans_r50_fpn_gcl_1x_fair1m'


# ROOT="poly_iou_loss"
# CONFIG='s2anet_r50_fpn_iouloss_3x_hrsc'
# CONFIG='s2anet_r50_fpn_iouloss_1x_dior'
# CONFIG='s2anet_r50_fpn_iouloss_1x_dota10'
# CONFIG='s2anet_r50_fpn_iouloss_1x_dota15'
# CONFIG='s2anet_r50_fpn_iouloss_1x_fair1m'
# CONFIG='retinanet_obb_r50_fpn_iouloss_1x_hrsc'
# CONFIG='faster_rcnn_orpn_r50_fpn_iouloss_3x_hrsc'
# CONFIG='faster_rcnn_orpn_r50_fpn_iouloss_1x_dior'
# CONFIG='faster_rcnn_orpn_r50_fpn_iouloss_1x_dota10'
# CONFIG='faster_rcnn_orpn_r50_fpn_iouloss_1x_dota15'
# CONFIG='faster_rcnn_orpn_r50_fpn_iouloss_1x_fair1m'
# CONFIG='faster_rcnn_roitrans_r50_fpn_iouloss_3x_hrsc'
# CONFIG='faster_rcnn_roitrans_r50_fpn_iouloss_1x_dior'
# CONFIG='faster_rcnn_roitrans_r50_fpn_iouloss_1x_dota10'
# CONFIG='faster_rcnn_roitrans_r50_fpn_iouloss_1x_dota15'
# CONFIG='faster_rcnn_roitrans_r50_fpn_iouloss_1x_fair1m'

# ROOT="oriented_rcnn"
# CONFIG='faster_rcnn_orpn_r50_fpn_1x_dior'
# CONFIG='faster_rcnn_orpn_r50_fpn_3x_dior'
# CONFIG='faster_rcnn_orpn_swinT_fpn_1x_dota10'
# CONFIG='faster_rcnn_orpn_swinT_fpn_1x_ms_dota10'
# CONFIG='faster_rcnn_orpn_r50_fpn_1x_fair1m'


# ROOT="roi_transformer"
# CONFIG='faster_rcnn_roitrans_r50_fpn_1x_dior'
# CONFIG='faster_rcnn_roitrans_r50_fpn_3x_dior'


# ROOT="s2anet"
# CONFIG='s2anet_r50_fpn_1x_dior'


## single GPU
# python tools/test.py configs/obb/$ROOT/$CONFIG.py\
#           work_dirs/$CONFIG/latest.pth\
#            --format-only\
#            --options save_dir=submission/after_nms nproc=1


# Multiple GPU

./tools/dist_test.sh configs/obb/$ROOT/$CONFIG.py\
          work_dirs/$CONFIG/latest.pth\
          $GPUS\
          --eval mAP


# rm -rf submission
# ./tools/dist_test.sh configs/obb/$ROOT/$CONFIG.py\
#           work_dirs/$CONFIG/latest.pth\
#           $GPUS\
#            --format-only\
#            --options save_dir=submission/after_nms 

# nproc=1     





