weight = '/path/to/MOCHA_WORKDIR/06_checkpoints/multimocha_transfer/multimocha15k_studentwrap_for_checkpointloader.pth'
resume = False
evaluate = True
test_only = False
seed = 3857179
save_path = '/path/to/MOCHA_WORKDIR/05_runs/semantic_kitti/20260619_121153_semkitti_e20_from_multimocha15k_targetmocha00025_patch256_goose'
num_worker = 4
batch_size = 1
gradient_accumulation_steps = 1
batch_size_val = None
batch_size_test = None
epoch = 20
eval_epoch = 5
clip_grad = None
sync_bn = False
enable_amp = True
amp_dtype = 'float16'
empty_cache = False
empty_cache_per_epoch = False
find_unused_parameters = False
enable_wandb = True
wandb_project = 'pointcept'
wandb_key = None
mix_prob = 0.8
param_dicts = [dict(keyword='block', lr=2e-05)]
hooks = [
    dict(
        type='CheckpointLoader',
        keywords='module.student.backbone',
        replacement='module.backbone'),
    dict(type='IterationTimer', warmup_iter=2),
    dict(type='InformationWriter', log_interval=200),
    dict(type='SemSegEvaluator'),
    dict(type='CheckpointSaver', save_freq=None)
]
train = dict(type='DefaultTrainer')
test = dict(type='SemSegTester', verbose=True)
crop_h = 518
crop_w = 518
patch_size = 14
model = dict(
    type='TargetMOCHASegmentorV2',
    num_classes=19,
    backbone_out_channels=1386,
    backbone=dict(
        type='PT-v3m3',
        in_channels=9,
        order=('z', 'z-trans', 'hilbert', 'hilbert-trans'),
        stride=(2, 2, 2, 2),
        enc_depths=(3, 3, 3, 12, 3),
        enc_channels=(54, 108, 216, 432, 576),
        enc_num_head=(3, 6, 12, 24, 32),
        enc_patch_size=(256, 256, 256, 256, 256),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(54, 108, 216, 432),
        dec_num_head=(3, 6, 12, 24),
        dec_patch_size=(256, 256, 256, 256),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=False,
        upcast_attention=False,
        upcast_softmax=False,
        traceable=False,
        mask_token=False,
        enc_mode=True,
        freeze_encoder=False,
        rope_base=10,
        shift_coords=None,
        jitter_coords=1.1,
        rescale_coords=1.2),
    criteria=[
        dict(
            type='CrossEntropyLoss',
            weight=[
                3.1557, 8.7029, 7.8281, 6.1354, 6.3161, 7.9937, 8.9704,
                10.1922, 1.6155, 4.2187, 1.9385, 5.5455, 2.0198, 2.6261,
                1.3212, 5.1102, 2.5492, 5.8585, 7.3929
            ],
            loss_weight=1.0,
            ignore_index=-1),
        dict(
            type='LovaszLoss',
            mode='multiclass',
            loss_weight=1.0,
            ignore_index=-1)
    ],
    freeze_backbone=False,
    occ_loss_weight=0.0025,
    occ_scales=(0.2, 0.4, 0.8))
optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.02)
scheduler = dict(
    type='OneCycleLR',
    max_lr=[0.0001, 2e-05],
    pct_start=0.3,
    anneal_strategy='cos',
    div_factor=10.0,
    final_div_factor=1000.0)
dataset_type = 'SemanticKITTIColorNormalDataset'
data_root = '/path/to/MOCHA_WORKDIR/02_data/semantic_kitti'
ignore_index = -1
names = [
    'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle', 'person',
    'bicyclist', 'motorcyclist', 'road', 'parking', 'sidewalk', 'other-ground',
    'building', 'fence', 'vegetation', 'trunk', 'terrain', 'pole',
    'traffic-sign'
]
data = dict(
    num_classes=19,
    ignore_index=-1,
    names=[
        'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle', 'person',
        'bicyclist', 'motorcyclist', 'road', 'parking', 'sidewalk',
        'other-ground', 'building', 'fence', 'vegetation', 'trunk', 'terrain',
        'pole', 'traffic-sign'
    ],
    train=dict(
        type='SemanticKITTIColorNormalDataset',
        split='train',
        data_root=
        '/path/to/MOCHA_WORKDIR/02_data/semantic_kitti',
        transform=[
            dict(type='RandomScale', scale=[0.2, 0.2]),
            dict(
                type='RandomRotate',
                angle=[-1, 1],
                axis='z',
                center=[0, 0, 0],
                p=0.5),
            dict(type='RandomScale', scale=[0.9, 1.1]),
            dict(type='RandomFlip', p=0.5),
            dict(type='RandomJitter', sigma=0.001, clip=0.004),
            dict(
                type='GridSample',
                grid_size=0.01,
                hash_type='fnv',
                mode='train',
                return_grid_coord=True),
            dict(type='NormalizeColor'),
            dict(type='ToTensor'),
            dict(
                type='Collect',
                keys=('coord', 'grid_coord', 'segment'),
                feat_keys=('coord', 'color', 'normal'))
        ],
        test_mode=False,
        ignore_index=-1,
        loop=4),
    val=dict(
        type='SemanticKITTIColorNormalDataset',
        split='val',
        data_root=
        '/path/to/MOCHA_WORKDIR/02_data/semantic_kitti',
        transform=[
            dict(type='RandomScale', scale=[0.2, 0.2]),
            dict(type='Copy', keys_dict=dict(segment='origin_segment')),
            dict(
                type='GridSample',
                grid_size=0.01,
                hash_type='fnv',
                mode='train',
                return_grid_coord=True,
                return_inverse=True),
            dict(type='NormalizeColor'),
            dict(type='ToTensor'),
            dict(
                type='Collect',
                keys=('coord', 'grid_coord', 'segment', 'origin_segment',
                      'inverse'),
                feat_keys=('coord', 'color', 'normal'))
        ],
        test_mode=False,
        ignore_index=-1),
    test=dict(
        type='SemanticKITTIColorNormalDataset',
        split='val',
        data_root=
        '/path/to/MOCHA_WORKDIR/02_data/semantic_kitti',
        transform=[
            dict(type='RandomScale', scale=[0.2, 0.2]),
            dict(type='Copy', keys_dict=dict(segment='origin_segment')),
            dict(
                type='GridSample',
                grid_size=0.005,
                hash_type='fnv',
                mode='train',
                return_inverse=True),
            dict(type='NormalizeColor')
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type='GridSample',
                grid_size=0.01,
                hash_type='fnv',
                mode='test',
                return_grid_coord=True),
            crop=None,
            post_transform=[
                dict(type='ToTensor'),
                dict(
                    type='Collect',
                    keys=('coord', 'grid_coord', 'index'),
                    feat_keys=('coord', 'color', 'normal'))
            ],
            aug_transform=[[{
                'type': 'RandomRotateTargetAngle',
                'angle': [0],
                'axis': 'z',
                'center': [0, 0, 0],
                'p': 1
            }],
                           [{
                               'type': 'RandomRotateTargetAngle',
                               'angle': [0.5],
                               'axis': 'z',
                               'center': [0, 0, 0],
                               'p': 1
                           }],
                           [{
                               'type': 'RandomRotateTargetAngle',
                               'angle': [1],
                               'axis': 'z',
                               'center': [0, 0, 0],
                               'p': 1
                           }],
                           [{
                               'type': 'RandomRotateTargetAngle',
                               'angle': [1.5],
                               'axis': 'z',
                               'center': [0, 0, 0],
                               'p': 1
                           }],
                           [{
                               'type': 'RandomRotateTargetAngle',
                               'angle': [0],
                               'axis': 'z',
                               'center': [0, 0, 0],
                               'p': 1
                           }, {
                               'type': 'RandomScale',
                               'scale': [0.95, 0.95]
                           }],
                           [{
                               'type': 'RandomRotateTargetAngle',
                               'angle': [0.5],
                               'axis': 'z',
                               'center': [0, 0, 0],
                               'p': 1
                           }, {
                               'type': 'RandomScale',
                               'scale': [0.95, 0.95]
                           }],
                           [{
                               'type': 'RandomRotateTargetAngle',
                               'angle': [1],
                               'axis': 'z',
                               'center': [0, 0, 0],
                               'p': 1
                           }, {
                               'type': 'RandomScale',
                               'scale': [0.95, 0.95]
                           }],
                           [{
                               'type': 'RandomRotateTargetAngle',
                               'angle': [1.5],
                               'axis': 'z',
                               'center': [0, 0, 0],
                               'p': 1
                           }, {
                               'type': 'RandomScale',
                               'scale': [0.95, 0.95]
                           }],
                           [{
                               'type': 'RandomRotateTargetAngle',
                               'angle': [0],
                               'axis': 'z',
                               'center': [0, 0, 0],
                               'p': 1
                           }, {
                               'type': 'RandomScale',
                               'scale': [1.05, 1.05]
                           }],
                           [{
                               'type': 'RandomRotateTargetAngle',
                               'angle': [0.5],
                               'axis': 'z',
                               'center': [0, 0, 0],
                               'p': 1
                           }, {
                               'type': 'RandomScale',
                               'scale': [1.05, 1.05]
                           }],
                           [{
                               'type': 'RandomRotateTargetAngle',
                               'angle': [1],
                               'axis': 'z',
                               'center': [0, 0, 0],
                               'p': 1
                           }, {
                               'type': 'RandomScale',
                               'scale': [1.05, 1.05]
                           }],
                           [{
                               'type': 'RandomRotateTargetAngle',
                               'angle': [1.5],
                               'axis': 'z',
                               'center': [0, 0, 0],
                               'p': 1
                           }, {
                               'type': 'RandomScale',
                               'scale': [1.05, 1.05]
                           }], [{
                               'type': 'RandomFlip',
                               'p': 1
                           }]]),
        ignore_index=-1))
