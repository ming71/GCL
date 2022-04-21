#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}


# ROOT="poly_iou_loss"
# CONFIG='retinanet_obb_r50_fpn_griouloss_1x_dota'
# CONFIG='retinanet_obb_r50_fpn_iouloss_1x_dota'

# ROOT="oriented_rcnn"
# CONFIG='faster_rcnn_orpn_swinT_fpn_GRIoU_1x_ss1024_rr_dota10'


# ROOT="poly_iou_loss"
# CONFIG='faster_rcnn_roitrans_r50_fpn_iouloss_3x_hrsc'


ROOT="gc_loss"
CONFIG='retinanet_obb_r50_fpn_gcl_6x_hrsc'


# single GPU
rm demo/outputs/*

python demo/image_demo.py   data/HRSC2016/Test/100001417.jpg \
          configs/obb/$ROOT/$CONFIG.py\
          work_dirs/$CONFIG/latest.pth

# python demo/image_demo.py   data/split_ms_dota1_0/test/images/ \
#           configs/obb/$ROOT/$CONFIG.py\
#           work_dirs/$CONFIG/latest.pth




