# MOCHA Canonical Pretraining Models

| Model | Released checkpoint | Size | SHA256 | Initialization |
|---|---|---:|---|---|
| MOCHA | `pretraining/MOCHA/model_last.pth` | 2.05 GB | `f240d9f08a66fc391a5c3443d0b0cdcb4bd2a0948635bdae6e984c1e61f7c551` | MOCHA lineage initialized from the official Utonia pretraining checkpoint; see LICENSE-WEIGHTS.md for provenance and applicable terms. |
| MultiMOCHA-15k | `pretraining/MultiMOCHA-15k/model_last.pth` | 2.05 GB | `499f48d77f679c787f115d1696208369011648338bf82ecef2aa19b434a16a97` | Scratch initialization (weight = None). Does not descend from the Utonia pretrained checkpoint. |

The MultiMOCHA-50k scaling diagnostic is not part of the primary released model set.
