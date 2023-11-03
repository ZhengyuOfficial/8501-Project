auto_scale_lr = dict(base_batch_size=512)
backend_args = dict(backend='local')
codec = dict(
    input_size=(
        192,
        256,
    ), type='RegressionLabel')
custom_hooks = [
    dict(type='SyncBuffersHook'),
]
data_mode = 'topdown'
data_root = 'data/pose/'
dataset_type = 'CocoDataset'
default_hooks = dict(
    checkpoint=dict(
        interval=10,
        rule='greater',
        save_best='coco/AP',
        type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(enable=False, type='PoseVisualizationHook'))
default_scope = 'mmpose'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(
    by_epoch=True, num_digits=6, type='LogProcessor', window_size=50)
model = dict(
    backbone=dict(num_classes=-1, type='AlexNet'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='PoseDataPreprocessor'),
    head=dict(
        decoder=dict(input_size=(
            192,
            256,
        ), type='RegressionLabel'),
        in_channels=256,
        loss=dict(type='RLELoss', use_target_weight=True),
        num_joints=17,
        type='RLEHead'),
    neck=dict(type='GlobalAveragePooling'),
    test_cfg=dict(flip_test=True, shift_coords=True),
    type='TopdownPoseEstimator')
optim_wrapper = dict(optimizer=dict(lr=0.001, type='Adam'))
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=500, start_factor=0.001, type='LinearLR'),
    dict(
        begin=0,
        by_epoch=True,
        end=20,
        gamma=0.1,
        milestones=[
            80,
            120,
        ],
        type='MultiStepLR'),
]
resume = False
test_cfg = dict()
test_dataloader = dict(
    batch_size=32,
    dataset=dict(
        ann_file='annotations/person_keypoints_val2017.json',
        bbox_file=
        'data/pose/person_detection_results/COCO_val2017_detections_AP_H_56_person.json',
        data_mode='topdown',
        data_prefix=dict(img='val2017/'),
        data_root='data/pose/',
        pipeline=[
            dict(type='LoadImage'),
            dict(type='GetBBoxCenterScale'),
            dict(input_size=(
                192,
                256,
            ), type='TopdownAffine'),
            dict(type='PackPoseInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(round_up=False, shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file='data/pose/annotations/person_keypoints_val2017.json',
    score_mode='bbox_rle',
    type='CocoMetric')
train_cfg = dict(by_epoch=True, max_epochs=20, val_interval=5)
train_dataloader = dict(
    batch_size=64,
    dataset=dict(
        ann_file='annotations/person_keypoints_train2017.json',
        data_mode='topdown',
        data_prefix=dict(img='train2017/'),
        data_root='data/pose/',
        pipeline=[
            dict(type='LoadImage'),
            dict(type='GetBBoxCenterScale'),
            dict(direction='horizontal', type='RandomFlip'),
            dict(type='RandomHalfBody'),
            dict(type='RandomBBoxTransform'),
            dict(input_size=(
                192,
                256,
            ), type='TopdownAffine'),
            dict(
                encoder=dict(input_size=(
                    192,
                    256,
                ), type='RegressionLabel'),
                type='GenerateTarget'),
            dict(type='PackPoseInputs'),
        ],
        type='CocoDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(direction='horizontal', type='RandomFlip'),
    dict(type='RandomHalfBody'),
    dict(type='RandomBBoxTransform'),
    dict(input_size=(
        192,
        256,
    ), type='TopdownAffine'),
    dict(
        encoder=dict(input_size=(
            192,
            256,
        ), type='RegressionLabel'),
        type='GenerateTarget'),
    dict(type='PackPoseInputs'),
]
val_cfg = dict()
val_dataloader = dict(
    batch_size=32,
    dataset=dict(
        ann_file='annotations/person_keypoints_val2017.json',
        bbox_file=
        'data/pose/person_detection_results/COCO_val2017_detections_AP_H_56_person.json',
        data_mode='topdown',
        data_prefix=dict(img='val2017/'),
        data_root='data/pose/',
        pipeline=[
            dict(type='LoadImage'),
            dict(type='GetBBoxCenterScale'),
            dict(input_size=(
                192,
                256,
            ), type='TopdownAffine'),
            dict(type='PackPoseInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(round_up=False, shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file='data/pose/annotations/person_keypoints_val2017.json',
    score_mode='bbox_rle',
    type='CocoMetric')
val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(input_size=(
        192,
        256,
    ), type='TopdownAffine'),
    dict(type='PackPoseInputs'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='PoseLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_dirs/alex'
