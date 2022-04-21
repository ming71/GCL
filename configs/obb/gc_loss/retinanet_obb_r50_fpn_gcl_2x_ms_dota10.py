_base_ = './retinanet_obb_r50_fpn_gcl_2x_dota.py'

# dataset
dataset_type = 'DOTADataset'
data_root = 'data/split_ms_dota1_0/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadOBBAnnotations', with_bbox=True,
         with_label=True, with_poly_as_mask=True),
    dict(type='LoadDOTASpecialInfo'),
    dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True),
    dict(type='OBBRandomFlip', h_flip_ratio=0.5, v_flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='RandomOBBRotate', rotate_after_flip=True,
         angles=(0, 90), vert_rate=0.5, vert_cls=['roundabout', 'storage-tank']),
    dict(type='Pad', size_divisor=32),
    dict(type='DOTASpecialIgnore', ignore_size=2),
    dict(type='FliterEmpty'),
    dict(type='Mask2OBB', obb_type='obb'),
    dict(type='OBBDefaultFormatBundle'),
    dict(type='OBBCollect', keys=['img', 'gt_bboxes', 'gt_obboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipRotateAug',
        img_scale=[(1024, 1024)],
        h_flip=False,
        v_flip=False,
        rotate=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='OBBRandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='RandomOBBRotate', rotate_after_flip=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='OBBCollect', keys=['img']),
        ])
]

# training and testing settings
train_cfg = dict(
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.4,
        min_pos_iou=0,
        gpu_assign_thr=-1,
        ignore_iof_thr=-1,
        iou_calculator=dict(
            type='OBBOverlaps')),
    allowed_border=-1,
    pos_weight=-1,
    debug=False)
test_cfg = dict(
    nms_pre=2000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='obb_nms', iou_thr=0.1),
    max_per_img=2000)

        
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
checkpoint_config = dict(interval=3)

log_config = dict(
    interval=200,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
