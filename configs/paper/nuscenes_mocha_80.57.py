weight = '/path/to/MOCHA_WORKDIR/05_runs/transfer_mocha_v3/20260523_215115_mocha_v3_nomaelite_ftcont_e8_sweep3_bs1_goose/model/model_last.pth'
resume = False
evaluate = True
test_only = False
seed = 28489034
save_path = '/path/to/MOCHA_WORKDIR/05_runs/transfer_mocha_v3/20260525_114906_mocha_v3_nomaelite_ftcont_e5_sweep3_bs1_lr1e4_goose'
num_worker = 8
batch_size = 1
gradient_accumulation_steps = 1
batch_size_val = None
batch_size_test = None
epoch = 5
eval_epoch = 1
clip_grad = None
sync_bn = False
enable_amp = False
amp_dtype = 'float16'
empty_cache = False
empty_cache_per_epoch = False
find_unused_parameters = False
enable_wandb = True
wandb_project = 'pointcept'
wandb_key = None
mix_prob = 0.0
param_dicts = [dict(keyword='block', lr=0.0002)]
hooks = [
    dict(
        type='CheckpointLoader',
        keywords='module.backbone',
        replacement='module.backbone'),
    dict(type='IterationTimer', warmup_iter=2),
    dict(type='InformationWriter'),
    dict(type='SemSegEvaluator'),
    dict(type='CheckpointSaver', save_freq=None)
]
train = dict(type='DefaultTrainer')
test = dict(type='SemSegTester', verbose=True)
crop_h = 518
crop_w = 518
patch_size = 14
model = dict(
    type='DefaultSegmentorV2',
    num_classes=16,
    backbone_out_channels=1386,
    backbone=dict(
        type='PT-v3m3',
        in_channels=9,
        order=('z', 'z-trans', 'hilbert', 'hilbert-trans'),
        stride=(2, 2, 2, 2),
        enc_depths=(3, 3, 3, 12, 3),
        enc_channels=(54, 108, 216, 432, 576),
        enc_num_head=(3, 6, 12, 24, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
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
    freeze_backbone=False,
    criteria=[
        dict(type='CrossEntropyLoss', loss_weight=1.0, ignore_index=-1),
        dict(
            type='LovaszLoss',
            mode='multiclass',
            loss_weight=1.0,
            ignore_index=-1)
    ])
optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.005)
scheduler = dict(
    type='OneCycleLR',
    max_lr=[0.0001, 1e-05],
    pct_start=0.04,
    anneal_strategy='cos',
    div_factor=10.0,
    final_div_factor=100.0)
dataset_type = 'NuScenesImagePointDataset'
data_root = '/path/to/MOCHA_WORKDIR/02_data/nuscenes_pointcept_trainval'
ignore_index = -1
names = [
    'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle', 'motorcycle',
    'pedestrian', 'traffic_cone', 'trailer', 'truck', 'driveable_surface',
    'other_flat', 'sidewalk', 'terrain', 'manmade', 'vegetation'
]
data = dict(
    num_classes=16,
    ignore_index=-1,
    names=[
        'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
        'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
        'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
        'vegetation'
    ],
    train=dict(
        type='NuScenesImagePointDataset',
        split='train',
        data_root=
        '/path/to/MOCHA_WORKDIR/02_data/nuscenes_pointcept_trainval',
        if_sweep=True,
        sweeps_max=10,
        sweeps=3,
        sweep_gap=1,
        if_img=False,
        crop_h=518,
        crop_w=518,
        patch_size=14,
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
            dict(type='RandomJitter', sigma=0.0025, clip=0.01),
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
        loop=5),
    val=dict(
        type='NuScenesImagePointDataset',
        split='val',
        data_root=
        '/path/to/MOCHA_WORKDIR/02_data/nuscenes_pointcept_trainval',
        if_sweep=True,
        sweeps_max=10,
        sweeps=3,
        sweep_gap=1,
        if_img=False,
        crop_h=518,
        crop_w=518,
        patch_size=14,
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
        type='NuScenesImagePointDataset',
        split='val',
        if_sweep=True,
        sweeps_max=10,
        sweeps=3,
        sweep_gap=1,
        if_img=False,
        crop_h=518,
        crop_w=518,
        patch_size=14,
        data_root=
        '/path/to/MOCHA_WORKDIR/02_data/nuscenes_pointcept_trainval',
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
                'type': 'RandomScale',
                'scale': [0.9, 0.9]
            }], [{
                'type': 'RandomScale',
                'scale': [0.95, 0.95]
            }], [{
                'type': 'RandomScale',
                'scale': [1, 1]
            }], [{
                'type': 'RandomScale',
                'scale': [1.05, 1.05]
            }], [{
                'type': 'RandomScale',
                'scale': [1.1, 1.1]
            }],
                           [{
                               'type': 'RandomScale',
                               'scale': [0.9, 0.9]
                           }, {
                               'type': 'RandomFlip',
                               'p': 1
                           }],
                           [{
                               'type': 'RandomScale',
                               'scale': [0.95, 0.95]
                           }, {
                               'type': 'RandomFlip',
                               'p': 1
                           }],
                           [{
                               'type': 'RandomScale',
                               'scale': [1, 1]
                           }, {
                               'type': 'RandomFlip',
                               'p': 1
                           }],
                           [{
                               'type': 'RandomScale',
                               'scale': [1.05, 1.05]
                           }, {
                               'type': 'RandomFlip',
                               'p': 1
                           }],
                           [{
                               'type': 'RandomScale',
                               'scale': [1.1, 1.1]
                           }, {
                               'type': 'RandomFlip',
                               'p': 1
                           }]]),
        ignore_index=-1))
