weight = '/path/to/MOCHA_WORKDIR/06_checkpoints/multimocha_transfer/multimocha15k_in4_studentwrap_for_checkpointloader.pth'
resume = False
evaluate = True
test_only = False
seed = 19661759
save_path = '/path/to/MOCHA_WORKDIR/05_runs/rellis3d/20260614_231907_rellis3d_e20_from_multimocha15k_targetmocha005_goose'
num_worker = 8
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
    num_classes=18,
    backbone_out_channels=54,
    backbone=dict(
        type='PT-v3m3',
        in_channels=4,
        order=('z', 'z-trans', 'hilbert', 'hilbert-trans'),
        stride=(2, 2, 2, 2),
        enc_depths=(3, 3, 3, 12, 3),
        enc_channels=(54, 108, 216, 432, 576),
        enc_num_head=(3, 6, 12, 24, 32),
        enc_patch_size=(512, 512, 512, 512, 512),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(54, 108, 216, 432),
        dec_num_head=(3, 6, 12, 24),
        dec_patch_size=(512, 512, 512, 512),
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
    freeze_backbone=False,
    occ_loss_weight=0.05,
    occ_scales=(0.2, 0.4, 0.8))
optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.02)
scheduler = dict(
    type='OneCycleLR',
    max_lr=[0.0001, 2e-05],
    pct_start=0.05,
    anneal_strategy='cos',
    div_factor=10.0,
    final_div_factor=1000.0)
dataset_type = 'DefaultDataset'
data_root = '/path/to/MOCHA_WORKDIR/02_data/rellis3d_pointcept'
ignore_index = -1
names = [
    'dirt', 'grass', 'tree', 'pole', 'water', 'sky', 'vehicle', 'object',
    'asphalt', 'building', 'log', 'person', 'fence', 'bush', 'concrete',
    'barrier', 'puddle', 'rubble'
]
data = dict(
    num_classes=18,
    ignore_index=-1,
    names=[
        'dirt', 'grass', 'tree', 'pole', 'water', 'sky', 'vehicle', 'object',
        'asphalt', 'building', 'log', 'person', 'fence', 'bush', 'concrete',
        'barrier', 'puddle', 'rubble'
    ],
    train=dict(
        type='DefaultDataset',
        split='train',
        data_root=
        '/path/to/MOCHA_WORKDIR/02_data/rellis3d_pointcept',
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
            dict(type='ToTensor'),
            dict(
                type='Collect',
                keys=('coord', 'grid_coord', 'segment'),
                feat_keys=('coord', 'strength'))
        ],
        test_mode=False,
        ignore_index=-1,
        loop=4),
    val=dict(
        type='DefaultDataset',
        split='val',
        data_root=
        '/path/to/MOCHA_WORKDIR/02_data/rellis3d_pointcept',
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
            dict(type='ToTensor'),
            dict(
                type='Collect',
                keys=('coord', 'grid_coord', 'segment', 'origin_segment',
                      'inverse'),
                feat_keys=('coord', 'strength'))
        ],
        test_mode=False,
        ignore_index=-1),
    test=dict(
        type='DefaultDataset',
        split='val',
        data_root=
        '/path/to/MOCHA_WORKDIR/02_data/rellis3d_pointcept',
        transform=[
            dict(type='RandomScale', scale=[0.2, 0.2]),
            dict(type='Copy', keys_dict=dict(segment='origin_segment')),
            dict(
                type='GridSample',
                grid_size=0.005,
                hash_type='fnv',
                mode='train',
                return_inverse=True)
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
                    feat_keys=('coord', 'strength'))
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
