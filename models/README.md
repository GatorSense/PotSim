# PotSim: A Large-Scale Simulated Dataset for Benchmarking AI Techniques on Potato Crop

This repository is the official suplemental code for [PotSim Dataset](https://doi.org/10.7910/DVN/GQMDOV) and paper []()

## Model Architectures

### LinearRegression
A simple Linear Regression model represnting a linear function `y = wx + b`, where `w` is weight vector and `b` is  a bias vector. It uses PyTorch's `nn`.Linear module to implement transformation.

**Arguments:**

| Argument    | Type | Optional | Default | Description                      |
| ----------- | ---- | -------- | ------- | -------------------------------- |
| `input_dim` | int  | No       |         | Dimentionality of input features |

---

### MLP
A Multi-Layer Perceptron (MLP) model i.e. feedforward neural network using PyTorch's `nn.Module` with adjustable layers, hidden sizes, and dropout.

**Arguments:**

| Argument      | Type  | Optional | Default | Description                           |
| ------------- | ----- | -------- | ------- | ------------------------------------- |
| `input_dim`   | int   | No       |         | Number of input features              |
| `hidden_size` | int   | Yes      | `64`    | Size of hidden layers. Defaults to 64 |
| `num_layers`  | int   | Yes      | `2`     | Number of hidden layers               |
| `dropout`     | float | Yes      | `0.2`   | Dropout rate                          |

---

### CNN
A 1D Convolutional Neural Network model that consists three convolutional blocks followed by adaptive average pooling and a final linear layer for prediction. Each convulational block includes `Conv1D`, `ReLU` Activation, `BatchNorm`, and `Dropout` (except the last block).

**Arguments:**

| Argument      | Type  | Optional | Default | Description                               |
| ------------- | ----- | -------- | ------- | ----------------------------------------- |
| `input_dim`   | int   | No       |         | Number of input features (channels)       |
| `hidden_size` | int   | Yes      | `64`    | Number of filters in the first conv layer |
| `kernel_size` | int   | Yes      | `3`     | Size of the convolving kernel             |
| `padding`     | int   | Yes      | `1`     | Padding added to both sides of the input  |
| `dropout`     | float | Yes      | `0.2`   | Dropout rate                              |

---

### TCN


**Arguments:**
| Argument       | Type      | Optional | Default       | Description                                       |
| -------------- | --------- | -------- | ------------- | ------------------------------------------------- |
| `input_dim`    | int       | No       |               | Number of input features (channels)               |
| `num_channels` | List[int] | Yes      | `[32, 16, 8]` | Number of feature channels in each residual block |
| `kernel_size`  | int       | Yes      | `3`           | Size of the convolving kernel                     |
| `dropout`      | float     | Yes      | `0.2`         | Dropout rate                                      |

---

### LSTM

**Arguments:**
| Argument      | Type  | Optional | Default | Description                                 |
| ------------- | ----- | -------- | ------- | ------------------------------------------- |
| `input_dim`   | int   | No       |         | Number of input features                    |
| `hidden_size` | int   | Yes      | `64`    | Number of features in the hidden layer      |
| `num_layers`  | int   | Yes      | `2`     | Number of recurrent layers (stacked if > 1) |
| `dropout`     | float | Yes      | `0.2`   | Dropout rate                                |

---

### EncoderOnlyTransformer

**Arguments:**
| Argument      | Type  | Optional | Default | Description                                   |
| ------------- | ----- | -------- | ------- | --------------------------------------------- |
| `input_dim`   | int   | No       |         | Number of input features                      |
| `nhead`       | int   | Yes      | `4`     | Number of heads in MultiHeadAttention         |
| `num_layers`  | int   | Yes      | `2`     | Number of sub-encoder-layers in the encoder   |
| `d_model`     | int   | Yes      | `128`   | number of expected features as encoder inputs |
| `dropout`     | float | Yes      | `0.2`   | Dropout rate                                  |
| `max_seq_len` | int   | Yes      | `5000`  | Maximum length for Positional Engcoding       |
