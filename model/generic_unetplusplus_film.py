"""Standalone UNet++ model with FiLM conditioning."""

from copy import deepcopy
from collections.abc import Mapping

import numpy as np
import torch
from torch import nn

from .helpers import (
    ConditioningEmbedding,
    FiLMStage,
    InitWeights_He,
    StackedConvLayers,
    Upsample,
    softmax_helper,
)


class Generic_UNetPlusPlus_FiLM(nn.Module):
    """Standalone five-level UNet++ with FiLM in every convolution block."""

    MAX_NUM_FILTERS_3D = 320
    MAX_FILTERS_2D = 480

    def __init__(
        self,
        input_channels,
        base_num_features,
        num_classes,
        num_pool,
        embedding_input_dims,
        embedding_dim,
        combined_embedding_dim,
        embedding_use_norm=False,
        num_conv_per_stage=2,
        feat_map_mul_on_downscale=2,
        conv_op=nn.Conv2d,
        norm_op=nn.BatchNorm2d,
        norm_op_kwargs=None,
        dropout_op=nn.Dropout2d,
        dropout_op_kwargs=None,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs=None,
        deep_supervision=True,
        dropout_in_localization=False,
        final_nonlin=softmax_helper,
        weight_initializer=None,
        pool_op_kernel_sizes=None,
        conv_kernel_sizes=None,
        upscale_logits=False,
        convolutional_pooling=True,
        convolutional_upsampling=True,
        max_num_features=None,
        seg_output_use_bias=False,
    ):
        super().__init__()
        if not isinstance(embedding_input_dims, Mapping) or not embedding_input_dims:
            raise ValueError("embedding_input_dims must be a non-empty mapping")
        if any(not isinstance(dim, int) or dim <= 0 for dim in embedding_input_dims.values()):
            raise ValueError("all embedding input dimensions must be positive integers")
        if embedding_dim <= 0 or combined_embedding_dim <= 0:
            raise ValueError("embedding dimensions must be positive")
        if num_pool != 5:
            raise ValueError("the retained UNet++ nested forward graph requires num_pool=5")
        if num_conv_per_stage < 2:
            raise ValueError("num_conv_per_stage must be at least 2")
        if not convolutional_upsampling:
            raise NotImplementedError(
                "convolutional_upsampling=False currently causes channel mismatch in the "
                "retained UNet++ nested decoder graph. Use convolutional_upsampling=True."
            )

        norm_op_kwargs = norm_op_kwargs or {"eps": 1e-5, "affine": True, "momentum": 0.1}
        dropout_op_kwargs = dropout_op_kwargs or {"p": 0.5, "inplace": True}
        nonlin_kwargs = nonlin_kwargs or {"negative_slope": 1e-2, "inplace": True}
        self.embedding_input_dims = dict(embedding_input_dims)
        self.embedding_dim = embedding_dim
        self.combined_embedding_dim = combined_embedding_dim
        self.embedding_use_norm = embedding_use_norm
        self.conditioning_embedding = ConditioningEmbedding(
            self.embedding_input_dims,
            embedding_dim,
            combined_embedding_dim,
            use_norm=embedding_use_norm,
        )
        self.final_nonlin = final_nonlin
        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision
        self.upscale_logits = upscale_logits
        self.convolutional_pooling = convolutional_pooling
        self.convolutional_upsampling = convolutional_upsampling
        self.conv_op = conv_op

        if conv_op == nn.Conv2d:
            pool_op, transpconv, upsample_mode = nn.MaxPool2d, nn.ConvTranspose2d, "bilinear"
            pool_op_kernel_sizes = pool_op_kernel_sizes or [(2, 2)] * num_pool
            conv_kernel_sizes = conv_kernel_sizes or [(3, 3)] * (num_pool + 1)
            max_default = self.MAX_FILTERS_2D
        elif conv_op == nn.Conv3d:
            pool_op, transpconv, upsample_mode = nn.MaxPool3d, nn.ConvTranspose3d, "trilinear"
            pool_op_kernel_sizes = pool_op_kernel_sizes or [(2, 2, 2)] * num_pool
            conv_kernel_sizes = conv_kernel_sizes or [(3, 3, 3)] * (num_pool + 1)
            max_default = self.MAX_NUM_FILTERS_3D
        else:
            raise ValueError(f"unsupported convolution type: {conv_op}")

        self.input_shape_must_be_divisible_by = np.prod(pool_op_kernel_sizes, axis=0, dtype=np.int64)
        self.pool_op_kernel_sizes = pool_op_kernel_sizes
        self.conv_kernel_sizes = conv_kernel_sizes
        self.max_num_features = max_num_features or max_default
        self._layer_args = (
            combined_embedding_dim,
            conv_op,
            norm_op,
            norm_op_kwargs,
            dropout_op,
            dropout_op_kwargs,
            nonlin,
            nonlin_kwargs,
        )

        self.conv_blocks_context = nn.ModuleList()
        self.td = nn.ModuleList()
        in_features, out_features = input_channels, base_num_features
        for level in range(num_pool):
            stride = pool_op_kernel_sizes[level - 1] if level and convolutional_pooling else None
            self.conv_blocks_context.append(
                self._stack(in_features, out_features, num_conv_per_stage, level, stride)
            )
            if not convolutional_pooling:
                self.td.append(pool_op(pool_op_kernel_sizes[level]))
            in_features = out_features
            out_features = min(
                int(np.round(out_features * feat_map_mul_on_downscale)), self.max_num_features
            )

        bottleneck_out = (
            out_features if convolutional_upsampling
            else self.conv_blocks_context[-1].output_channels
        )
        bottleneck_stride = pool_op_kernel_sizes[-1] if convolutional_pooling else None
        self.conv_blocks_context.append(
            FiLMStage(
                self._stack(
                    in_features,
                    out_features,
                    num_conv_per_stage - 1,
                    num_pool,
                    bottleneck_stride,
                ),
                self._stack(out_features, bottleneck_out, 1, num_pool),
            )
        )

        localization_dropout = deepcopy(dropout_op_kwargs)
        if not dropout_in_localization:
            localization_dropout["p"] = 0.0
        self._layer_args = (
            combined_embedding_dim,
            conv_op,
            norm_op,
            norm_op_kwargs,
            dropout_op,
            localization_dropout,
            nonlin,
            nonlin_kwargs,
        )

        localization_paths, upsampling_paths = [], []
        encoder_features = bottleneck_out
        for z in range(5):
            loc_path, up_path, encoder_features = self._create_nest(
                z,
                num_pool,
                encoder_features,
                num_conv_per_stage,
                transpconv,
                upsample_mode,
            )
            localization_paths.append(loc_path)
            upsampling_paths.append(up_path)

        self.loc0, self.loc1, self.loc2, self.loc3, self.loc4 = localization_paths
        self.up0, self.up1, self.up2, self.up3, self.up4 = upsampling_paths

        self.seg_outputs = nn.ModuleList(
            conv_op(path[-1].output_channels, num_classes, 1, 1, 0, bias=seg_output_use_bias)
            for path in (self.loc0, self.loc1, self.loc2, self.loc3, self.loc4)
        )
        self.upscale_logits_ops = nn.ModuleList(nn.Identity() for _ in range(num_pool))
        if upscale_logits:
            cumulative = np.cumprod(np.vstack(pool_op_kernel_sizes), axis=0)[::-1]
            self.upscale_logits_ops = nn.ModuleList(
                Upsample(tuple(int(v) for v in cumulative[index + 1]), upsample_mode)
                for index in range(num_pool - 1)
            )

        self.apply(weight_initializer or InitWeights_He(1e-2))

    def _stack(self, in_channels, out_channels, count, level, stride=None):
        kernel = self.conv_kernel_sizes[level]
        kwargs = {
            "kernel_size": kernel,
            "stride": 1,
            "padding": tuple(1 if value == 3 else 0 for value in kernel),
            "dilation": 1,
            "bias": True,
        }
        (
            embedding_dim,
            conv_op,
            norm_op,
            norm_op_kwargs,
            dropout_op,
            dropout_op_kwargs,
            nonlin,
            nonlin_kwargs,
        ) = self._layer_args
        return StackedConvLayers(
            in_channels,
            out_channels,
            count,
            embedding_dim,
            conv_op,
            kwargs,
            norm_op,
            norm_op_kwargs,
            dropout_op,
            dropout_op_kwargs,
            nonlin,
            nonlin_kwargs,
            first_stride=stride,
        )

    def _create_nest(self, z, num_pool, final_features, conv_count, transpconv, mode):
        locations, upsamplers = nn.ModuleList(), nn.ModuleList()
        first_output = None
        for u in range(z, num_pool):
            from_down = final_features
            from_skip = self.conv_blocks_context[-(2 + u)].output_channels
            concat_features = from_skip * (2 + u - z)
            first_output = from_skip if first_output is None else first_output
            if u != num_pool - 1 and not self.convolutional_upsampling:
                final_features = self.conv_blocks_context[-(3 + u)].output_channels
            else:
                final_features = from_skip
            if self.convolutional_upsampling:
                kernel = self.pool_op_kernel_sizes[-(u + 1)]
                upsamplers.append(transpconv(from_down, from_skip, kernel, kernel, bias=False))
            else:
                upsamplers.append(Upsample(self.pool_op_kernel_sizes[-(u + 1)], mode))
            level = num_pool - (u + 1)
            locations.append(
                FiLMStage(
                    self._stack(concat_features, from_skip, conv_count - 1, level),
                    self._stack(from_skip, final_features, 1, level),
                )
            )
        return locations, upsamplers, first_output

    def _ctx(self, index, x, embedding):
        if index and not self.convolutional_pooling:
            x = self.td[index - 1](x)
        return self.conv_blocks_context[index](x, embedding)

    def forward(self, x, conditions):
        embedding = self.conditioning_embedding(conditions, x.shape[0])
        outputs = []
        x0_0 = self._ctx(0, x, embedding)
        x1_0 = self._ctx(1, x0_0, embedding)
        x0_1 = self.loc4[0](torch.cat([x0_0, self.up4[0](x1_0)], 1), embedding)
        outputs.append(self.final_nonlin(self.seg_outputs[-1](x0_1)))

        x2_0 = self._ctx(2, x1_0, embedding)
        x1_1 = self.loc3[0](torch.cat([x1_0, self.up3[0](x2_0)], 1), embedding)
        x0_2 = self.loc3[1](torch.cat([x0_0, x0_1, self.up3[1](x1_1)], 1), embedding)
        outputs.append(self.final_nonlin(self.seg_outputs[-2](x0_2)))

        x3_0 = self._ctx(3, x2_0, embedding)
        x2_1 = self.loc2[0](torch.cat([x2_0, self.up2[0](x3_0)], 1), embedding)
        x1_2 = self.loc2[1](torch.cat([x1_0, x1_1, self.up2[1](x2_1)], 1), embedding)
        x0_3 = self.loc2[2](torch.cat([x0_0, x0_1, x0_2, self.up2[2](x1_2)], 1), embedding)
        outputs.append(self.final_nonlin(self.seg_outputs[-3](x0_3)))

        x4_0 = self._ctx(4, x3_0, embedding)
        x3_1 = self.loc1[0](torch.cat([x3_0, self.up1[0](x4_0)], 1), embedding)
        x2_2 = self.loc1[1](torch.cat([x2_0, x2_1, self.up1[1](x3_1)], 1), embedding)
        x1_3 = self.loc1[2](torch.cat([x1_0, x1_1, x1_2, self.up1[2](x2_2)], 1), embedding)
        x0_4 = self.loc1[3](
            torch.cat([x0_0, x0_1, x0_2, x0_3, self.up1[3](x1_3)], 1), embedding
        )
        outputs.append(self.final_nonlin(self.seg_outputs[-4](x0_4)))

        x5_0 = self._ctx(5, x4_0, embedding)
        x4_1 = self.loc0[0](torch.cat([x4_0, self.up0[0](x5_0)], 1), embedding)
        x3_2 = self.loc0[1](torch.cat([x3_0, x3_1, self.up0[1](x4_1)], 1), embedding)
        x2_3 = self.loc0[2](torch.cat([x2_0, x2_1, x2_2, self.up0[2](x3_2)], 1), embedding)
        x1_4 = self.loc0[3](
            torch.cat([x1_0, x1_1, x1_2, x1_3, self.up0[3](x2_3)], 1), embedding
        )
        x0_5 = self.loc0[4](
            torch.cat([x0_0, x0_1, x0_2, x0_3, x0_4, self.up0[4](x1_4)], 1), embedding
        )
        outputs.append(self.final_nonlin(self.seg_outputs[-5](x0_5)))
        if self._deep_supervision and self.do_ds:
            return tuple(
                [outputs[-1]]
                + [
                    op(value)
                    for op, value in zip(list(self.upscale_logits_ops)[::-1], outputs[:-1][::-1])
                ]
            )
        return outputs[-1]
