"""Factory functions for the standalone UNet++ FiLM model."""

from torch import nn

from .model import Generic_UNetPlusPlus_FiLM


def create_unetplusplus_film_model(
    input_channels,
    base_num_features,
    num_classes,
    embedding_input_dims,
    embedding_dim,
    combined_embedding_dim,
    embedding_use_norm=False,
    num_pool=5,
    conv_dim=2,
    **kwargs,
):
    """Create a 2-D or 3-D Generic_UNetPlusPlus_FiLM instance.

    Extra keyword arguments are forwarded to Generic_UNetPlusPlus_FiLM.
    """
    if conv_dim == 2:
        kwargs.setdefault("conv_op", nn.Conv2d)
        kwargs.setdefault("norm_op", nn.BatchNorm2d)
        kwargs.setdefault("dropout_op", nn.Dropout2d)
    elif conv_dim == 3:
        kwargs.setdefault("conv_op", nn.Conv3d)
        kwargs.setdefault("norm_op", nn.BatchNorm3d)
        kwargs.setdefault("dropout_op", nn.Dropout3d)
    else:
        raise ValueError(f"conv_dim must be 2 or 3, got {conv_dim}")

    return Generic_UNetPlusPlus_FiLM(
        input_channels=input_channels,
        base_num_features=base_num_features,
        num_classes=num_classes,
        num_pool=num_pool,
        embedding_input_dims=embedding_input_dims,
        embedding_dim=embedding_dim,
        combined_embedding_dim=combined_embedding_dim,
        embedding_use_norm=embedding_use_norm,
        **kwargs,
    )
