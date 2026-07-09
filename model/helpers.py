"""Helper modules for the standalone UNet++ FiLM implementation."""

from copy import deepcopy
from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F


def softmax_helper(x):
    return F.softmax(x, 1)


class InitWeights_He:
    def __init__(self, negative_slope=1e-2):
        self.negative_slope = negative_slope

    def __call__(self, module):
        if isinstance(module, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
            nn.init.kaiming_normal_(module.weight, a=self.negative_slope)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)


class FeatureWiseLinearModulation(nn.Module):
    """Project [N, D] embeddings to channel-wise scale and shift."""

    def __init__(self, embedding_dim, channels):
        super().__init__()
        self.channels = channels
        self.projection = nn.Sequential(nn.SiLU(), nn.Linear(embedding_dim, 2 * channels))

    def forward(self, features, embedding):
        if embedding.ndim != 2:
            raise ValueError(f"embedding must be 2-D, got shape {tuple(embedding.shape)}")
        if embedding.shape[0] != features.shape[0]:
            raise ValueError("features and embedding must have the same batch size")
        scale, shift = self.projection(embedding).to(features.dtype).chunk(2, dim=1)
        shape = (features.shape[0], self.channels) + (1,) * (features.ndim - 2)
        return features * (1 + scale.reshape(shape)) + shift.reshape(shape)


class ConditioningEmbedding(nn.Module):
    """Encode each raw field independently, then combine all field embeddings."""

    def __init__(self, input_dims, embedding_dim, combined_embedding_dim, use_norm=False):
        super().__init__()
        if not isinstance(use_norm, bool):
            raise TypeError(f"use_norm must be a bool, got {type(use_norm).__name__}")
        self.input_dims = dict(input_dims)
        self.keys = tuple(self.input_dims)
        self.use_norm = use_norm
        self.cond_embed = nn.ModuleDict(
            {
                key: self._make_field_embedding(input_dim, embedding_dim, use_norm)
                for key, input_dim in self.input_dims.items()
            }
        )
        self.combiner = self._make_combiner(
            len(self.keys) * embedding_dim, combined_embedding_dim, use_norm
        )

    @staticmethod
    def _linear_block(in_features, out_features, use_norm, activate=True):
        layers = [nn.Linear(in_features, out_features)]
        if use_norm:
            layers.append(nn.LayerNorm(out_features))
        if activate:
            layers.append(nn.SiLU())
        return layers

    @classmethod
    def _make_field_embedding(cls, input_dim, embedding_dim, use_norm):
        return nn.Sequential(
            *cls._linear_block(input_dim, embedding_dim, use_norm),
            *cls._linear_block(embedding_dim, embedding_dim, use_norm),
        )

    @classmethod
    def _make_combiner(cls, input_dim, combined_embedding_dim, use_norm):
        return nn.Sequential(
            *cls._linear_block(input_dim, combined_embedding_dim, use_norm),
            *cls._linear_block(
                combined_embedding_dim,
                combined_embedding_dim,
                use_norm,
                activate=False,
            ),
        )

    def forward(self, conditions, batch_size):
        if not isinstance(conditions, Mapping):
            raise TypeError(
                "conditions must be a mapping such as {'age': tensor}, "
                f"got {type(conditions).__name__}"
            )
        missing = [key for key in self.keys if key not in conditions]
        if missing:
            raise KeyError(f"missing conditioning fields: {missing}")

        encoded = []
        for key in self.keys:
            value = conditions[key]
            if not torch.is_tensor(value):
                raise TypeError(f"conditions[{key!r}] must be a tensor")
            expected = (batch_size, self.input_dims[key])
            if value.ndim != 2 or tuple(value.shape) != expected:
                raise ValueError(
                    f"conditions[{key!r}] must have shape {expected}, "
                    f"got {tuple(value.shape)}"
                )
            encoded.append(self.cond_embed[key](value))
        return self.combiner(torch.cat(encoded, dim=1))


class ConvDropoutNormNonlin(nn.Module):
    """Conv -> Dropout -> Norm -> FiLM -> activation."""

    def __init__(
        self,
        input_channels,
        output_channels,
        embedding_dim,
        conv_op=nn.Conv2d,
        conv_kwargs=None,
        norm_op=nn.BatchNorm2d,
        norm_op_kwargs=None,
        dropout_op=nn.Dropout2d,
        dropout_op_kwargs=None,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs=None,
    ):
        super().__init__()
        conv_kwargs = conv_kwargs or {
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "dilation": 1,
            "bias": True,
        }
        norm_op_kwargs = norm_op_kwargs or {"eps": 1e-5, "affine": True, "momentum": 0.1}
        dropout_op_kwargs = dropout_op_kwargs or {"p": 0.5, "inplace": True}
        nonlin_kwargs = nonlin_kwargs or {"negative_slope": 1e-2, "inplace": True}
        self.output_channels = output_channels
        self.conv = conv_op(input_channels, output_channels, **conv_kwargs)
        self.dropout = (
            dropout_op(**dropout_op_kwargs)
            if dropout_op is not None and dropout_op_kwargs.get("p", 0) > 0
            else None
        )
        self.instnorm = norm_op(output_channels, **norm_op_kwargs)
        self.emb_layers = FeatureWiseLinearModulation(embedding_dim, output_channels)
        self.lrelu = nonlin(**nonlin_kwargs)

    def forward(self, x, embedding):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.emb_layers(self.instnorm(x), embedding)
        return self.lrelu(x)


class StackedConvLayers(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        num_convs,
        embedding_dim,
        conv_op,
        conv_kwargs,
        norm_op,
        norm_op_kwargs,
        dropout_op,
        dropout_op_kwargs,
        nonlin,
        nonlin_kwargs,
        first_stride=None,
    ):
        super().__init__()
        self.output_channels = output_channels
        first_kwargs = deepcopy(conv_kwargs)
        if first_stride is not None:
            first_kwargs["stride"] = first_stride
        blocks = []
        for index in range(num_convs):
            blocks.append(
                ConvDropoutNormNonlin(
                    input_channels if index == 0 else output_channels,
                    output_channels,
                    embedding_dim,
                    conv_op,
                    first_kwargs if index == 0 else conv_kwargs,
                    norm_op,
                    norm_op_kwargs,
                    dropout_op,
                    dropout_op_kwargs,
                    nonlin,
                    nonlin_kwargs,
                )
            )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, embedding):
        for block in self.blocks:
            x = block(x, embedding)
        return x


class FiLMStage(nn.Module):
    """Two stacked FiLM layer groups used by bottleneck and decoder nodes."""

    def __init__(self, first, second):
        super().__init__()
        self.first = first
        self.second = second
        self.output_channels = second.output_channels

    def forward(self, x, embedding):
        return self.second(self.first(x, embedding), embedding)


class Upsample(nn.Module):
    def __init__(self, scale_factor, mode):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)
