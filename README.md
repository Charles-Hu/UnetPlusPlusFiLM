# UnetPlusPlusFiLM

A standalone PyTorch implementation of a five-level UNet++ model with FiLM
conditioning in each convolution block.

This implementation is modified from the official UNet++ implementation:
[MrGiovanni/UNetPlusPlus](https://github.com/MrGiovanni/UNetPlusPlus).

## Features

- Supports 2-D and 3-D convolution backends.
- Encodes one or more conditioning fields and injects them through FiLM layers.
- Optionally applies `LayerNorm` inside the conditioning embedding MLPs.
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
    embedding_input_dims={
        "age": 1,
        "sex": 1,
        "scanner": 3,
    },
    embedding_dim=32,
    combined_embedding_dim=128,
    embedding_use_norm=False,
    deep_supervision=True,
    convolutional_pooling=True,
    convolutional_upsampling=True,
    dropout_op_kwargs={"p": 0.5, "inplace": True},
    dropout_in_localization=False,
    final_nonlin=None,
)

image = torch.randn(2, 1, 128, 128)
conditions = {
    "age": torch.randn(2, 1),
    "sex": torch.tensor([[0.0], [1.0]]),
    "scanner": torch.randn(2, 3),
}
outputs = model(image, conditions)
```

By default, `deep_supervision=True`, so `outputs` is a tuple of segmentation
maps. Set `deep_supervision=False` to return only the final prediction.

By default, `final_nonlin=None`, so the model returns raw logits or raw
regression values. This is the safest default for both segmentation training
with `CrossEntropyLoss` and regression/SR tasks. If you need probabilities, pass
an explicit activation such as `final_nonlin=softmax_helper`, or apply
softmax/sigmoid outside the model during inference.

The example uses `convolutional_upsampling=True` because the retained UNet++
nested decoder expects the upsampled branch to also project channels to the
matching skip-connection width. Plain interpolation only changes spatial size
and is therefore disabled for this implementation.

With the default `num_pool=5` and `(2, 2)` or `(2, 2, 2)` pooling kernels, each
input spatial dimension should be divisible by `32`. For example, use 2-D input
sizes such as `128 x 128`, or 3-D patch sizes whose depth, height, and width are
all multiples of `32`. Otherwise, nested skip connections can fail with spatial
size mismatches during concatenation.

Key embedding parameters:

- `embedding_dim`: output size of each field-specific embedding MLP.
- `combined_embedding_dim`: output size after all field embeddings are combined.
- `embedding_use_norm`: whether to insert `LayerNorm` in the field-specific MLPs
  and the shared combiner MLP. Defaults to `False`.

Key dropout parameters:

- `dropout_op_kwargs`: keyword arguments passed to the selected dropout layer.
  With `conv_dim=2`, the factory uses `nn.Dropout2d`; with `conv_dim=3`, it uses
  `nn.Dropout3d`. The default is `{"p": 0.5, "inplace": True}`.
- `dropout_in_localization`: whether to keep dropout enabled in the nested
  decoder/localization paths. Defaults to `False`, which sets decoder dropout
  probability to `0.0` while keeping encoder and bottleneck dropout controlled
  by `dropout_op_kwargs`.

Key output activation parameter:

- `final_nonlin`: optional activation applied to each output head. Defaults to
  `None`, which returns raw logits/values. Use this for regression, SR,
  denoising, residual prediction, and segmentation losses that expect logits.

## FiLM in UNet++

The original UNet++ encoder-decoder graph is kept, including the nested skip
connections and deep supervision heads. FiLM is added inside the convolution
blocks rather than changing the UNet++ topology.

Each convolution block follows this order:

```text
Convolution -> Dropout -> Normalization -> FiLM -> Activation
```

For a feature map with `C` channels, the shared conditioning embedding is passed
through a small projection layer to produce `2 * C` values:

```text
[batch_size, combined_embedding_dim] -> [batch_size, 2 * C]
```

These values are split into channel-wise `scale` and `shift` tensors. They are
reshaped to broadcast across the spatial dimensions, so the same conditioning
information can modulate either 2-D features `[N, C, H, W]` or 3-D features
`[N, C, D, H, W]`.

Inside each convolution block, the normalized feature map is modulated as:

```text
features * (1 + scale) + shift
```

FiLM is applied to every `ConvDropoutNormNonlin` block used by the UNet++
context path, bottleneck, and nested decoder paths. This means the same
patient-, scanner-, or metadata-derived conditioning vector can influence both
low-level encoder features and high-level decoder features throughout the
network.

## Conditioning Embeddings

Conditioning variables are passed as a single dictionary whose keys match
`embedding_input_dims`. Each value must be a 2-D tensor with shape
`[batch_size, input_dim]`.

For each field, the model builds an independent embedding MLP:

```text
input_dim -> embedding_dim -> embedding_dim
```

With the default `embedding_use_norm=False`, each field uses:

```text
Linear -> SiLU -> Linear -> SiLU
```

With `embedding_use_norm=True`, each field uses:

```text
Linear -> LayerNorm -> SiLU -> Linear -> LayerNorm -> SiLU
```

Using the example above, `age`, `sex`, and `scanner` are encoded separately into
three tensors of shape `[batch_size, embedding_dim]`. These field embeddings are
then concatenated in the same key order as `embedding_input_dims`:

```text
[age_embedding, sex_embedding, scanner_embedding]
    -> [batch_size, num_fields * embedding_dim]
```

The concatenated vector is passed through a shared combiner MLP:

```text
num_fields * embedding_dim -> combined_embedding_dim -> combined_embedding_dim
```

The combiner also follows `embedding_use_norm`: when enabled, `LayerNorm` is
inserted after both combiner linear layers. The resulting
`[batch_size, combined_embedding_dim]` tensor is used as the shared conditioning
embedding for FiLM layers throughout the network.

## Project Structure

- `create_model.py`: model factory at the package root.
- `model/`: UNet++ FiLM model definition and helper modules.
- `__init__.py`: package exports.

## Requirements

- Python 3.9+
- PyTorch
- NumPy
