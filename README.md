# FFT-LCNN for ASVspoof 2019 LA

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

## About

This project adapts the
[PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template)
to logical-access speech spoofing detection on the ASVspoof 2019 dataset.

## Installation

Follow these steps:

0. (Optional) Create and activate new environment using [`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) or `venv` ([`+pyenv`](https://github.com/pyenv/pyenv)).

   a. `conda` version:

   ```bash
   # create env
   conda create -n project_env python=PYTHON_VERSION

   # activate env
   conda activate project_env
   ```

   b. `venv` (`+pyenv`) version:

   ```bash
   # create env
   ~/.pyenv/versions/PYTHON_VERSION/bin/python3 -m venv project_env

   # alternatively, using default python version
   python3 -m venv project_env

   # activate env
   source project_env/bin/activate
   ```

1. Install all required packages

   ```bash
   pip install -r requirements.txt
   ```

2. Install `pre-commit`:
   ```bash
   pre-commit install
   ```

## How To Use

To train a model, run the following command:

```bash
python3 train.py -cn=CONFIG_NAME HYDRA_CONFIG_ARGUMENTS
```

Where `CONFIG_NAME` is a config from `src/configs` and `HYDRA_CONFIG_ARGUMENTS` are optional arguments.

To run inference (evaluate the model or save predictions):

```bash
python3 inference.py HYDRA_CONFIG_ARGUMENTS
```

## FFT-LCNN on ASVspoof 2019 LA

Attach the Kaggle dataset `awsaf49/asvpoof-2019-dataset`. The default config
expects its LA directory at:

```text
/kaggle/input/datasets/awsaf49/asvpoof-2019-dataset/LA/LA
```

Train FFT-LCNN with A-Softmax:

```bash
python3 train.py -cn=asvspoof
```

Use both available GPUs and upload the latest and best checkpoints to W&B:

```bash
python3 train.py -cn=asvspoof writer=asvspoof_wandb
```

Training uses balanced class sampling. The latest complete epoch is stored in
`saved/asvspoof_lcnn/checkpoint-latest.pth`; the checkpoint also contains the
optimizer and A-Softmax annealing state.

If Kaggle mounts the dataset elsewhere, override the root without changing the
configs:

```bash
ASVSPOOF_ROOT=/kaggle/input/your-dataset/LA python3 train.py -cn=asvspoof
```

Run evaluation and write `data/saved/asvspoof/eval/cm_scores.txt`:

```bash
python3 inference.py
```

The same inference run writes the two-column grading submission to
`data/saved/asvspoof/eval/submission.csv`.

## Credits

This repository is based on a [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
