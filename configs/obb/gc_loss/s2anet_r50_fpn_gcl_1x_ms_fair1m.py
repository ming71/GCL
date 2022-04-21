_base_ = [
    '../_base_/schedules/schedule_1x.py',
    '../../_base_/default_runtime.py'
]

# dataset
dataset_type = 'FAIR1MDataset'
data_root = 'data/split_ms_fair1m/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadOBBAnnotations', with_bbox=True,
         with_label=True, with_poly_as_mask=True),
    dict(type='Resize', img_scale=(800, 800), keep_ratio=True),
    dict(type='OBBRandomFlip', h_flip_ratio=0.5, v_flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='RandomOBBRotate', rotate_after_flip=True,
         angles=(0, 90), vert_rate=0.5),
    dict(type='Pad', size_divisor=32),
    dict(type='FliterEmpty'),
    dict(type='Mask2OBB', obb_type='obb'),
    dict(type='OBBDefaultFormatBundle'),
    dict(type='OBBCollect', keys=['img', 'gt_bboxes', 'gt_obboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipRotateAug',
        img_scale=[(800, 800)],
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

model = dict(
    type='S2ANet',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5),
    bbox_head=dict(
        type='S2AHead',
        feat_channels=256,
        align_type='AlignConv',
        heads=[
            dict(
                type='ODMHead',
                num_classes=37,
                in_channels=256,
                feat_channels=256,
                stacked_convs=2,
                anchor_generator=dict(
                    type='Theta0AnchorGenerator',
                    scales=[4],
                    ratios=[1.0],
                    strides=[8, 16, 32, 64, 128]),
                bbox_coder=dict(
                    type='OBB2OBBDeltaXYWHTCoder',
                    target_means=(0., 0., 0., 0., 0.),
                    target_stds=(1., 1., 1., 1., 1.)),
                reg_decoded_bbox=False,
                loss_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=1.0),
                loss_bbox=dict(
                    type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
            ),
            dict(
                type='ODMHead',
                num_classes=37,
                in_channels=256,
                feat_channels=256,
                stacked_convs=2,
                with_orconv=True,
                bbox_coder=dict(
                    type='OBB2OBBDeltaXYWHTCoder',
                    target_means=(0., 0., 0., 0., 0.),
                    target_stds=(1., 1., 1., 1., 1.)),
                reg_decoded_bbox=True,
                loss_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=1.0),
                loss_bbox=dict(
                    type='GCL',loss_weight=1.0),
            )
        ]
    )
)

# training and testing settings
train_cfg = [
    dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1,
            iou_calculator=dict(type='OBBOverlaps')),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1,
            iou_calculator=dict(type='OBBOverlaps')),
        allowed_border=-1,
        pos_weight=-1,
        debug=False
    ),
]
test_cfg = dict(
    skip_cls=[True, False],
    nms_pre=2000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='obb_nms', iou_thr=0.1),
    max_per_img=2000)




data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'train/annfiles/',
        img_prefix=data_root + 'train/images/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'test/annfiles/',
        img_prefix=data_root + 'test/images/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test/annfiles/',
        img_prefix=data_root + 'test/images/',
        pipeline=test_pipeline))

evaluation = None

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))
lr_config = dict(warmup_iters=1000, step=[9, 11])
runner = dict(max_epochs=12)

log_config = dict(
    interval=300,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
