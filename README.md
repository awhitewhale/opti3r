# Opti3R


## Abstract

Attenuation and backscatter change underwater image appearance even when the scene geometry stays fixed. These changes can lead feed-forward models to predict inconsistent depth and shape. To reduce this effect, we propose Opti3R, which trains on Geometry-Locked Optical Pairing (GLOP): paired reference and optically perturbed images with consistent geometry targets. We further use teacher correspondences to supervise cross-view consistency with continuous visibility-confidence-reprojection (VCR) edge weighting, referred to as VCR Graph supervision (VCRG). Our proposed Axial-Channel Geometry Adapter (ACGA) adds identity-initialized multiscale residual blocks to the dense depth and point heads. These blocks take dense features as input and are optimized jointly with the GLOP and VCRG objectives. Raw inference achieves absolute relative depth errors of 0.0838 on SQUID and 0.0496 on FLSea-VI. GLOP gives lower reconstruction error than generic and unpaired augmentation. Opti3R also shows slower growth in geometry drift under held-out optical conditions, and higher VCR edge weights are associated with lower independent geometric disagreement. These results support geometry-locked optical pairing and continuously weighted cross-view supervision for underwater reconstruction.



## Repository layout

```text
.
├── train.py                    # training entry point
├── test.py                     # inference entry point
├── training/                   # data loading, losses, trainer, and configs
└── wat3r/                     # model and Opti3R head plug-ins
```

## Environment

Create the environment from `environment.yaml`, then install the training requirements:

```bash
conda env create -f environment.yaml
conda activate opti3r
pip install -e .
pip install -r training/requirements_train.txt
```

The exact PyTorch/CUDA build should match the machine used for training. A GPU with enough memory for the 518-pixel model is required for the reported configuration.

## Data preparation

The training configuration expects the following dataset root layout:

```text
DATA_ROOT/
├── canyons/
└── syn_haze/dtu_haze/
```

The real-water loader holds out `u_canyon`; the synthetic loader holds out `scan37,scan40`. Dataset files and evaluation annotations are not redistributed here. Set `DATA_ROOT` to a local copy with the corresponding layout.

## Reproduce the selected short training run

The selected run uses the SCSA plug-in in both dense heads, one GPU, a one-epoch/20-batch screening budget, and seed 42. The checkpoint records 15 optimizer updates because unlabeled warm-up batches are skipped before update 200. This anonymous package does not redistribute model checkpoints; provide a compatible pretrained initialization checkpoint through `checkpoint.resume_checkpoint_path`.

```bash
python train.py \
  --config reproduce_scsa_short \
  --set data_root=/absolute/path/to/DATA_ROOT \
  --set checkpoint.resume_checkpoint_path=/absolute/path/to/wat3r.pt
```

Training outputs are written under the configured `result/` directory. Change `log_save_dir` with another `--set` override if desired. The training code saves the model and optimizer state needed for resuming.

## Run inference without evaluation

`test.py` loads the selected student checkpoint and writes raw model predictions. 

```bash
python test.py \
  --checkpoint /absolute/path/to/opti3r_scsa_short.pt \
  --input /absolute/path/to/an/image/directory \
  --output outputs/example \
  --device cuda
```

The input directory should contain at least two RGB images. The command writes `depth.pt`, `depth_conf.pt`, `world_points.pt`, and `metadata.json`. Use `--smoke` instead of `--input` for a synthetic forward-pass check. The checkpoint must be supplied separately by the user; no checkpoint is included in this repository.