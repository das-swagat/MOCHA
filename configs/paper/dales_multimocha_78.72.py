weight = '/path/to/MOCHA_WORKDIR/05_runs/dales_mocha/20260816_000732_job2286_dales_pointssm_multimocha_cont50_lr1e4_from_e50best/model/model_best.pth'
resume = False
evaluate = True
test_only = False
seed = 16530398
save_path = '/path/to/MOCHA_WORKDIR/05_runs/dales_mocha/20260816_092455_job2288_dales_pointssm_multimocha_stage3_cont50_lr5e5_from_stage2best'
num_worker = 8
batch_size = 1
gradient_accumulation_steps = 1
batch_size_val = None
batch_size_test = None
epoch = 50
eval_epoch = 50
clip_grad = 1.0
sync_bn = False
enable_amp = False
amp_dtype = 'float16'
empty_cache = False
empty_cache_per_epoch = False
find_unused_parameters = False
enable_wandb = True
wandb_project = 'pointcept'
wandb_key = None
mix_prob = 0.8
param_dicts = [dict(keyword='block', lr=1e-05)]
hooks = [
    dict(
        type='CheckpointLoader',
        keywords='module.student.backbone',
        replacement='module.backbone'),
    dict(type='IterationTimer', warmup_iter=2),
    dict(type='InformationWriter', log_interval=50),
    dict(type='SemSegEvaluator'),
    dict(type='CheckpointSaver', save_freq=1)
]
train = dict(type='DefaultTrainer')
test = dict(type='SemSegTester', verbose=True)
BASE = '/path/to/MOCHA_WORKDIR/03_configs/dales_mocha/dales_e10_grid002_baseline_goose.py'
crop_h = 518
crop_w = 518
patch_size = 14
model = dict(
    type='DefaultSegmentorV2',
    num_classes=8,
    backbone_out_channels=54,
    backbone=dict(
        type='PT-v3m3',
        in_channels=4,
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
        enc_mode=False,
        freeze_encoder=False,
        rope_base=10,
        shift_coords=None,
        jitter_coords=1.1,
        rescale_coords=1.2),
    criteria=[
        dict(type='CrossEntropyLoss', loss_weight=1.0, ignore_index=-1),
        dict(
            type='LovaszLoss',
            mode='multiclass',
            loss_weight=1.0,
            ignore_index=-1)
    ],
    freeze_backbone=False)
optimizer = dict(type='AdamW', lr=5e-05, weight_decay=0.005)
scheduler = dict(
    type='OneCycleLR',
    max_lr=[5e-05, 1e-05],
    pct_start=0.1,
    anneal_strategy='cos',
    div_factor=10.0,
    final_div_factor=100.0)
dataset_type = 'DALESPointSSMMOCHADataset'
data_root = '/path/to/MOCHA_WORKDIR/00_external/PointSSM_DALES_data/dales_noDrop_Normalized'
ignore_index = -1
names = [
    'ground', 'vegetation', 'cars', 'trucks', 'powerlines', 'fences', 'poles',
    'buildings'
]
data = dict(
    num_classes=8,
    ignore_index=-1,
    names=[
        'ground', 'vegetation', 'cars', 'trucks', 'powerlines', 'fences',
        'poles', 'buildings'
    ],
    train=dict(
        type='DALESPointSSMMOCHADataset',
        split='train',
        data_root=
        '/path/to/MOCHA_WORKDIR/00_external/PointSSM_DALES_data/dales_noDrop_Normalized',
        transform=[
            dict(
                type='RandomRotate',
                angle=[-1, 1],
                axis='z',
                center=[0, 0, 0],
                p=0.5),
            dict(type='RandomScale', scale=[0.9, 1.1]),
            dict(type='RandomFlip', p=0.5),
            dict(type='RandomJitter', sigma=0.005, clip=0.02),
            dict(
                type='GridSample',
                grid_size=0.02,
                hash_type='fnv',
                mode='train',
                return_grid_coord=True),
            dict(type='ToTensor'),
            dict(
                type='Collect',
                keys=('coord', 'grid_coord', 'segment'),
                feat_keys=('coord', 'strength'))
        ],
        test_mode=False,
        ignore_index=-1,
        loop=1),
    val=dict(
        type='DALESPointSSMMOCHADataset',
        split='test',
        data_root=
        '/path/to/MOCHA_WORKDIR/00_external/PointSSM_DALES_data/dales_noDrop_Normalized',
        transform=[
            dict(
                type='GridSample',
                grid_size=0.02,
                hash_type='fnv',
                mode='train',
                return_grid_coord=True),
            dict(type='ToTensor'),
            dict(
                type='Collect',
                keys=('coord', 'grid_coord', 'segment'),
                feat_keys=('coord', 'strength'))
        ],
        test_mode=False,
        ignore_index=-1),
    test=dict(
        type='DALESPointSSMMOCHADataset',
        split='test',
        data_root=
        '/path/to/MOCHA_WORKDIR/00_external/PointSSM_DALES_data/dales_noDrop_Normalized',
        transform=[
            dict(type='Copy', keys_dict=dict(segment='origin_segment')),
            dict(
                type='GridSample',
                grid_size=0.01,
                hash_type='fnv',
                mode='train',
                return_inverse=True)
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type='GridSample',
                grid_size=0.02,
                hash_type='fnv',
                mode='test',
                return_grid_coord=True),
            crop=None,
            post_transform=[
                dict(type='ToTensor'),
                dict(
                    type='Collect',
                    keys=('coord', 'grid_coord', 'index'),
                    feat_keys=('coord', 'strength'))
            ],
            aug_transform=[[{
                'type': 'RandomScale',
                'scale': [0.9, 0.9]
            }], [{
                'type': 'RandomScale',
                'scale': [0.95, 0.95]
            }], [{
                'type': 'RandomScale',
                'scale': [1.0, 1.0]
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
                               'scale': [1.0, 1.0]
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
DALES_NAMES = [
    'ground', 'vegetation', 'cars', 'trucks', 'powerlines', 'fences', 'poles',
    'buildings'
]
_k = 'test'
_c = dict(type='CrossEntropyLoss', loss_weight=1.0, ignore_index=-1)
k = 'test'
FULL = '/path/to/MOCHA_WORKDIR/02_data/dales_clean20m_v1'
h = dict(type='CheckpointSaver', save_freq=1)
