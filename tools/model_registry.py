"""Canonical public model registry for MOCHA."""

HF_REPO = "das-swagat/MOCHA"

MODELS = {
    "nuscenes-mocha": {
        "family": "MOCHA",
        "dataset": "nuScenes",
        "classes": 16,
        "in_channels": 9,
        "features": "coord + color + normal",
        "grid_size": "0.01 m",
        "sweeps": 3,
        "paper_miou": 80.57,
        "checkpoint": "downstream/nuscenes/mocha/model_best.pth",
        "config": "downstream/nuscenes/mocha/config.py",
        "sha256": "b6487e1975e71f0983ae17a450d2b6f85d380f2eeff3954f49bc4aac1ade462d",
        "preprocessing": (
            "Use the released nuScenes pipeline: 3 LiDAR sweeps "
            "(sweep_gap=1), feat_keys=(coord, color, normal), "
            "and 0.01 m test voxelization."
        ),
    },

    "semkitti-multimocha": {
        "family": "MultiMOCHA",
        "dataset": "SemanticKITTI",
        "classes": 19,
        "in_channels": 9,
        "features": "coord + color + normal",
        "grid_size": "0.01 m",
        "sweeps": 1,
        "paper_miou": 59.26,
        "checkpoint": "downstream/semantickitti/multimocha/model_best.pth",
        "config": "downstream/semantickitti/multimocha/config.py",
        "sha256": "095fe10f8c7e08a1bdc41612087d6b9db8f3db79f4469c4d1ed71c0059f0f784",
        "preprocessing": (
            "Use the released SemanticKITTI pipeline with "
            "feat_keys=(coord, color, normal) and 0.01 m test voxelization."
        ),
    },

    "semkitti-targetmocha": {
        "family": "TargetMOCHA",
        "dataset": "SemanticKITTI",
        "classes": 19,
        "in_channels": 9,
        "features": "coord + color + normal",
        "grid_size": "0.01 m",
        "sweeps": 1,
        "paper_miou": 59.52,
        "checkpoint": "downstream/semantickitti/targetmocha/model_best.pth",
        "config": "downstream/semantickitti/targetmocha/config.py",
        "sha256": "f49795c543699cb94606e1bed92343d77c6958f3eb255883ac4c4b2e147fb830",
        "preprocessing": (
            "Use the released SemanticKITTI pipeline with "
            "feat_keys=(coord, color, normal) and 0.01 m test voxelization."
        ),
    },

    "toronto-multimocha": {
        "family": "MultiMOCHA",
        "dataset": "Toronto3D",
        "classes": 8,
        "in_channels": 4,
        "features": "coord + strength",
        "grid_size": "0.02 m",
        "sweeps": 1,
        "paper_miou": 58.42,
        "checkpoint": "downstream/toronto3d/multimocha/model_best.pth",
        "config": "downstream/toronto3d/multimocha/config.py",
        "sha256": "460f8b10074657a237d4af82ec7b70141e828424b586ed2b1f54ba3749fa48e7",
        "preprocessing": (
            "Use the released Toronto3D pipeline with "
            "feat_keys=(coord, strength) and 0.02 m test voxelization."
        ),
    },

    "toronto-targetmocha": {
        "family": "TargetMOCHA",
        "dataset": "Toronto3D",
        "classes": 8,
        "in_channels": 4,
        "features": "coord + strength",
        "grid_size": "0.02 m",
        "sweeps": 1,
        "paper_miou": 58.96,
        "checkpoint": "downstream/toronto3d/targetmocha/model_best.pth",
        "config": "downstream/toronto3d/targetmocha/config.py",
        "sha256": "002d7b228b9b04343beabe32d4a1dabb8ac4fea300421feae63cdaf608682211",
        "preprocessing": (
            "Use the released Toronto3D pipeline with "
            "feat_keys=(coord, strength) and 0.02 m test voxelization."
        ),
    },

    "rellis-multimocha": {
        "family": "MultiMOCHA",
        "dataset": "RELLIS-3D",
        "classes": 18,
        "in_channels": 4,
        "features": "coord + strength",
        "grid_size": "0.01 m",
        "sweeps": 1,
        "paper_miou": 26.67,
        "checkpoint": "downstream/rellis3d/multimocha/model_best.pth",
        "config": "downstream/rellis3d/multimocha/config.py",
        "sha256": "472aeef5033253ab0d1dc2dd039d0f254a54159ff9d45138e38c37e1de50f6b9",
        "preprocessing": (
            "Use the released RELLIS-3D pipeline with "
            "feat_keys=(coord, strength) and 0.01 m test voxelization."
        ),
    },

    "rellis-targetmocha": {
        "family": "TargetMOCHA",
        "dataset": "RELLIS-3D",
        "classes": 18,
        "in_channels": 4,
        "features": "coord + strength",
        "grid_size": "0.01 m",
        "sweeps": 1,
        "paper_miou": 26.78,
        "checkpoint": "downstream/rellis3d/targetmocha/model_best.pth",
        "config": "downstream/rellis3d/targetmocha/config.py",
        "sha256": "492b814c0acb98b339c88f52479bcd4d9b2c2d49139b58bae35461c6ec69431e",
        "preprocessing": (
            "Use the released RELLIS-3D pipeline with "
            "feat_keys=(coord, strength) and 0.01 m test voxelization."
        ),
    },

    "usl-multimocha": {
        "family": "MultiMOCHA",
        "dataset": "SemanticUSL",
        "classes": 19,
        "in_channels": 9,
        "features": "coord + color + normal",
        "grid_size": "0.01 m",
        "sweeps": 1,
        "paper_miou": 23.94,
        "checkpoint": "downstream/semanticusl/multimocha/model_best.pth",
        "config": "downstream/semanticusl/multimocha/config.py",
        "sha256": "cdc4530a6efd286ffd07e81fded58c55710e79f04e5afee70491c2206a1c7b41",
        "preprocessing": (
            "Use the released SemanticUSL pipeline with "
            "feat_keys=(coord, color, normal) and 0.01 m test voxelization."
        ),
    },

    "usl-targetmocha": {
        "family": "TargetMOCHA",
        "dataset": "SemanticUSL",
        "classes": 19,
        "in_channels": 9,
        "features": "coord + color + normal",
        "grid_size": "0.01 m",
        "sweeps": 1,
        "paper_miou": 24.51,
        "checkpoint": "downstream/semanticusl/targetmocha/model_best.pth",
        "config": "downstream/semanticusl/targetmocha/config.py",
        "sha256": "db91763d65bac4d132f126ef4c8a97e42270396f621f3e518a88f52f21c13ab5",
        "preprocessing": (
            "Use the released SemanticUSL pipeline with "
            "feat_keys=(coord, color, normal) and 0.01 m test voxelization."
        ),
    },

    "dales-multimocha": {
        "family": "MultiMOCHA",
        "dataset": "DALES",
        "classes": 8,
        "in_channels": 4,
        "features": "coord + strength",
        "grid_size": "0.02 m",
        "sweeps": 1,
        "paper_miou": 78.72,
        "checkpoint": "downstream/dales/multimocha/model_best.pth",
        "config": "downstream/dales/multimocha/config.py",
        "sha256": "570aac79089a1a119d2dc25950521975609ceb3aeeea791cec00ffea120030a2",
        "preprocessing": (
            "Use the released DALES pipeline with "
            "feat_keys=(coord, strength) and 0.02 m test voxelization."
        ),
    },

    "dales-targetmocha": {
        "family": "TargetMOCHA",
        "dataset": "DALES",
        "classes": 8,
        "in_channels": 4,
        "features": "coord + strength",
        "grid_size": "0.02 m",
        "sweeps": 1,
        "paper_miou": 78.69,
        "checkpoint": "downstream/dales/targetmocha/model_best.pth",
        "config": "downstream/dales/targetmocha/config.py",
        "sha256": "e37ad7c02e932a43ecaed8beeec9bf298d35eb86595b322e91ec95fbdb215359",
        "preprocessing": (
            "Use the released DALES pipeline with "
            "feat_keys=(coord, strength) and 0.02 m test voxelization."
        ),
    },
}

PRETRAINING_MODELS = {
    "mocha-pretrain": {
        "family": "MOCHA",
        "dataset": "nuScenes",
        "in_channels": 9,
        "features": "coord + color + normal",
        "sweeps": 1,
        "checkpoint": "pretraining/MOCHA/model_last.pth",
        "config": "pretraining/MOCHA/config.py",
        "sha256": "f240d9f08a66fc391a5c3443d0b0cdcb4bd2a0948635bdae6e984c1e61f7c551",
        "preprocessing": (
            "Pretraining checkpoint for representation initialization. "
            "Use the released MOCHA pretraining config and downstream "
            "fine-tuning pipeline; this is not a directly deployable "
            "semantic segmentation checkpoint."
        ),
    },

    "multimocha-15k-pretrain": {
        "family": "MultiMOCHA-15k",
        "dataset": "Multi-dataset outdoor LiDAR",
        "in_channels": 9,
        "features": "coord + color + normal",
        "sweeps": 1,
        "checkpoint": "pretraining/MultiMOCHA-15k/model_last.pth",
        "config": "pretraining/MultiMOCHA-15k/config.py",
        "sha256": "499f48d77f679c787f115d1696208369011648338bf82ecef2aa19b434a16a97",
        "preprocessing": (
            "Pretraining checkpoint for cross-domain initialization. "
            "Use the released MultiMOCHA-15k config and the appropriate "
            "target-dataset fine-tuning pipeline; this is not a directly "
            "deployable semantic segmentation checkpoint."
        ),
    },
}

ALL_MODELS = {**MODELS, **PRETRAINING_MODELS}
