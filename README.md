# Physics-guided CVAE for aperiodic BOTDA codes

[简体中文](README.zh-CN.md)

A conditional variational autoencoder for generating aperiodic codes for Brillouin optical time-domain analysis (BOTDA). The model is conditioned on energy enhancement factor FE, code length and fading coefficient. An exponential EDFA envelope and FFT-based coding-gain evaluation are incorporated into the training objective.

The repository includes training and generation scripts and the complete training dataset of **6,079 binary codes**.

## Installation

Use Python 3.10 or later in a virtual environment:

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The scripts use CUDA when available, otherwise CPU. For GPU execution, install a PyTorch build compatible with the local driver.

## Dataset

The `data0/` directory contains six fading conditions: 0.00, 0.10, 0.20, 0.30, 0.40 and 0.50. Code lengths are 120, 180, 240, 300, 360, 450, 540, 630 and 720. Each text file stores one binary code. Directory names, filenames and code contents are preserved.

Training inputs are faded amplitude sequences; reconstruction targets are binary codes. Conditions are normalized as `[FE/700, L/1024, fading/0.5]`, with the training FE label calculated as the number of one bits. Sample counts are listed in [data0/README.md](data0/README.md).

## Training

```bash
python train.py --data_root data0 --checkpoint_dir checkpoints
```

Defaults: maximum length 1024, hidden dimension 1024, latent dimension 128, batch size 32, 900 epochs, AdamW with learning rate `1e-4`, and a 10% validation split. Hyperparameters are defined at the top of `train.py`.

Use `python train.py --resume` to continue from `checkpoints/universal_cvae_v2_last.pt`. The best model is selected by validation `Gc-Gr` and saved as `universal_cvae_v2_best.pt`. Historical best checkpoints and gain plots are saved in `checkpoints/history/` and `checkpoints/plots/`.

## Generation

After training, or after placing a trained `universal_cvae_v2_best.pt` in the checkpoint directory:

```bash
python generate.py --checkpoint_dir checkpoints --output_base outputs --fe_list 40 60 80 100 120 150 180 210 --fading_list 0.0 0.15
```

The code length is `L=3*FE`. Defaults use 10 rounds of 20,000 samples, temperature 0.5, and a `200*L`-point FFT. The number of one bits is determined by rounding the sum of predicted probabilities. Candidates are ranked by `Gc-Gr`, deduplicated and exported.

Parameters include `--n_sample`, `--n_rounds`, `--top_k`, `--temperature`, `--target_ratio` and `--amp_sum_tolerance`. The optional amplitude filter checks the ranked, deduplicated candidate subset; if no candidate passes, the original ranking is retained.

Outputs are grouped by FE, length and fading, with binary text files, per-condition `summary.csv` files and a combined `all_results.csv`. Repeating a condition updates its existing code files. Metrics include `Binary_FE` (one-bit count), `AmpSum` (faded amplitude sum), `Gc`, `Gr`, `Gr0` and `Diff=Gc-Gr`.

## Checks and packaging

```bash
python -m unittest discover -s tests -v
python train.py --help
python generate.py --help
python tools/build_release.py
```

The release archive includes source code, documentation and training data. Environments, checkpoints and generated outputs are excluded.
