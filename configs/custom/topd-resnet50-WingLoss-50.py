auto_scale_lr = dict(base_batch_size=512)
backend_args = dict(backend='local')
codec = dict(
    input_size=(
        256,
        256,
    ), type='RegressionLabel')
custom_hooks = [
    dict(type='SyncBuffersHook'),
]
data_mode = 'topdown'
data_root = 'data/pose/WFLW/'
dataset_type = 'WFLWDataset'
default_hooks = dict(
    checkpoint=dict(
        interval=10, rule='less', save_best='NME', type='CheckpointHook'),
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
    backbone=dict(
        depth=50,
        init_cfg=dict(checkpoint='torchvision://resnet50', type='Pretrained'),
        type='ResNet'),
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
            256,
            256,
        ), type='RegressionLabel'),
        in_channels=2048,
        loss=dict(type='WingLoss', use_target_weight=True),
        num_joints=98,
        type='RegressionHead'),
    neck=dict(type='GlobalAveragePooling'),
    test_cfg=dict(flip_test=True, shift_coords=True),
    train_cfg=dict(),
    type='TopdownPoseEstimator')
optim_wrapper = dict(optimizer=dict(lr=0.0005, type='Adam'))
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=500, start_factor=0.001, type='LinearLR'),
    dict(
        begin=0,
        by_epoch=True,
        end=210,
        gamma=0.1,
        milestones=[
            170,
            200,
        ],
        type='MultiStepLR'),
]
resume = False
test_cfg = dict()
test_dataloader = dict(
    batch_size=32,
    dataset=dict(
        ann_file='annotations/face_landmarks_wflw_test.json',
        data_mode='topdown',
        data_prefix=dict(img='images/'),
        data_root='data/pose/WFLW/',
        pipeline=[
            dict(type='LoadImage'),
            dict(type='GetBBoxCenterScale'),
            dict(input_size=(
                256,
                256,
            ), type='TopdownAffine'),
            dict(type='PackPoseInputs'),
        ],
        test_mode=True,
        type='WFLWDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(round_up=False, shuffle=False, type='DefaultSampler'))
test_evaluator = dict(norm_mode='keypoint_distance', type='NME')
train_cfg = dict(by_epoch=True, max_epochs=50, val_interval=10)
train_dataloader = dict(
    batch_size=64,
    dataset=dict(
        ann_file='annotations/face_landmarks_wflw_train.json',
        data_mode='topdown',
        data_prefix=dict(img='images/'),
        data_root='data/pose/WFLW/',
        pipeline=[
            dict(type='LoadImage'),
            dict(type='GetBBoxCenterScale'),
            dict(direction='horizontal', type='RandomFlip'),
            dict(
                rotate_factor=60,
                scale_factor=[
                    0.75,
                    1.25,
                ],
                type='RandomBBoxTransform'),
            dict(input_size=(
                256,
                256,
            ), type='TopdownAffine'),
            dict(
                encoder=dict(input_size=(
                    256,
                    256,
                ), type='RegressionLabel'),
                type='GenerateTarget'),
            dict(type='PackPoseInputs'),
        ],
        type='WFLWDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(direction='horizontal', type='RandomFlip'),
    dict(
        rotate_factor=60,
        scale_factor=[
            0.75,
            1.25,
        ],
        type='RandomBBoxTransform'),
    dict(input_size=(
        256,
        256,
    ), type='TopdownAffine'),
    dict(
        encoder=dict(input_size=(
            256,
            256,
        ), type='RegressionLabel'),
        type='GenerateTarget'),
    dict(type='PackPoseInputs'),
]
val_cfg = dict()
val_dataloader = dict(
    batch_size=32,
    dataset=dict(
        ann_file='annotations/face_landmarks_wflw_test.json',
        data_mode='topdown',
        data_prefix=dict(img='images/'),
        data_root='data/pose/WFLW/',
        pipeline=[
            dict(type='LoadImage'),
            dict(type='GetBBoxCenterScale'),
            dict(input_size=(
                256,
                256,
            ), type='TopdownAffine'),
            dict(type='PackPoseInputs'),
        ],
        test_mode=True,
        type='WFLWDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(round_up=False, shuffle=False, type='DefaultSampler'))
val_evaluator = dict(norm_mode='keypoint_distance', type='NME')
val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(input_size=(
        256,
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
work_dir = './work_dirs/td-reg_res50_wingloss_8xb64-210e_wflw-256x256'
