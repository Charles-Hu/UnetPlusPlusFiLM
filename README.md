# UnetPlusPlusFiLM

A standalone PyTorch implementation of a five-level UNet++ model with FiLM
conditioning in each convolution block.

## Features

- Supports 2-D and 3-D convolution backends.
- Encodes one or more conditioning fields and injects them through FiLM layers.
- Keeps the model self-contained with only PyTorch and NumPy dependencies.

## Installation

Install from this repository:

```bash
pip install .
```

For editable development:

```bash
pip install -e .
```

## Usage

```python
import torch
from UnetPlusPlusFiLM import create_unetplusplus_film_model

model = create_unetplusplus_film_model(
    input_channels=1,
    base_num_features=30,
    num_classes=3,
    embedding_input_dims={"age": 1},
    embedding_dim=32,
    combined_embedding_dim=128,
)

image = torch.randn(2, 1, 128, 128)
age = torch.randn(2, 1)
outputs = model(image, {"age": age})
```

By default, `deep_supervision=True`, so `outputs` is a tuple of segmentation
maps. Set `deep_supervision=False` to return only the final prediction.

## Project Structure

- `create_model.py`: model factory at the package root.
- `model/`: UNet++ FiLM model definition and helper modules.
- `__init__.py`: package exports.

## Requirements

- Python 3.9+
- PyTorch
- NumPy
