
# PotSim: A Large-Scale Simulated Dataset for Benchmarking AI Techniques on Potato Crop
This repository contains the official implementation associated with this paper. The corresponding dataset is publicly available [here](https://doi.org/10.7910/DVN/GQMDOV).



# Description

---


## Features


---


## Requirements

To install the requirements:
```bash
conda env create -f environment.yml
conda activate potsim_env
```

Depending on the version of `CUDA` on your system, install `PyTorch v2.5.1` from official PyTorch source at [https://pytorch.org](https://pytorch.org/get-started/previous-versions/)

```bash
# Example for cuda-version 12.4
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
```

To allow on gpu metrics and display the model parameters clearly
```bash
pip install torchmetrics==1.7.1 torchinfo==1.8
```

If your system is not set up with `conda` package manager, then please visit [https://www.anaconda.com/download](https://www.anaconda.com/download/success#miniconda) to install `Miniconda` accoding to your operating system and then continue by installing the requirements from above.


## Usage run.py

The script supports two main commands: `train` and `test`.

### 1. Train a Model

```bash
python run.py train -tgt -m  [options]
```

**Arguments:**

| Argument                  | Type      | Required | Default        | Description                                                                 |
|---------------------------|-----------|----------|----------------|-----------------------------------------------------------------------------|
| `-tgt`, `--target`        | str       | Yes      |                | Target variable to predict. Choices: *see below*                            |
| `-m`, `--model`           | str       | Yes      |                | Model type to use. Choices: *see below*                                     |
| `-tdata`, `--train_dataset` | str     | No       | `train_split`  | Training dataset split                                                      |
| `-vdata`, `--val_dataset` | str       | No       | `val_split`    | Validation dataset split                                                    |
| `-bs`, `--batch_size`     | int       | No       | `256`          | Batch size                                                                  |
| `-lr`, `--learning_rate`  | float     | No       | `0.005`        | Learning rate                                                               |
| `-ep`, `--epochs`         | int       | No       | `100`          | Maximum number of epochs                                                    |
| `-sl`, `--seq_len`        | int       | No       | `15`           | Sequence length (for sequence models)                                       |
| `-d`, `--device`          | str       | No       | `None`         | Device: `cpu` or `cuda`                                                     |

**Example:**

```bash
python run.py train -tgt="NTotL1" -m="lstm" -tdata="train_split" -vdata="val_split" -bs=256 -lr=0.001 -ep=10 -sl=15 -d="cuda"
```

---

### 2. Test a Model

```bash
python run.py test -tgt  -m  -data  [options]
```

**Arguments:**

| Argument                  | Type      | Required | Default        | Description                                                                 |
|---------------------------|-----------|----------|----------------|-----------------------------------------------------------------------------|
| `-tgt`, `--target`        | str       | Yes      |                | Target variable to predict. Choices: *see below*                            |
| `-m`, `--model`           | str       | Yes      |                | Model type to use. Choices: *see below*                                     |
| `-data`, `--dataset`      | str       | Yes      |                | Dataset to run test on                                                      |
| `-mdir`, `--model_dir`    | str       | No       | `saves`        | Directory where trained models are saved (`outputs` or `saves`)             |
| `-bs`, `--batch_size`     | int       | No       | `256`          | Batch size                                                                  |
| `-sl`, `--seq_len`        | int       | No       | `15`           | Sequence length (for sequence models)                                       |
| `-d`, `--device`          | str       | No       | `None`         | Device: `cpu` or `cuda`                                                     |

**Example:**

```bash
python run.py test -tgt="NTotL1" -m="lstm" -data="test_split" -mdir="saves" -bs=256 -sl=15 -d="cuda"
```

---


## Notes

- Make sure your datasets are in the `.parquet` format and accessible by the script at `data` folder.
- For more details on available target variables and models, check the code or add a `--help` flag:

```bash
python run.py --help
python run.py train --help
python run.py test --help
```

---
