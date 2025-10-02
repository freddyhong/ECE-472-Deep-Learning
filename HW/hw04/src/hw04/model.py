import jax
import jax.numpy as jnp
from flax import nnx

class Conv2d(nnx.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        *,
        strides=(1, 1),
        padding="SAME",
        use_bias=True,
        dropout=0.0,
        rngs: nnx.Rngs,
    ):
        self.conv = nnx.Conv(
            in_channels, out_channels, kernel_size,
            strides=strides, padding=padding, use_bias=use_bias, rngs=rngs
        )
        self.dropout = nnx.Dropout(dropout, rngs=rngs) if dropout > 0.0 else None

    def __call__(self, x, *, rngs: nnx.Rngs):
        x = self.conv(x)
        if self.dropout:
            x = self.dropout(x, rngs=rngs)
        return x


class ResidualBlock(nnx.Module):
    """Basic residual block."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride=(1, 1),
        dropout=0.0,
        rngs: nnx.Rngs,
    ):
        self.norm1 = nnx.GroupNorm(num_features=in_channels, num_groups=8, rngs=rngs)
        self.conv1 = Conv2d(
            in_channels, out_channels, (3, 3),
            strides=stride, dropout=dropout, rngs=rngs,
        )
        self.norm2 = nnx.GroupNorm(num_features=out_channels, num_groups=8, rngs=rngs)
        self.conv2 = Conv2d(
            out_channels, out_channels, (3, 3),
            strides=(1, 1), dropout=0.0, rngs=rngs,
        )
        self.shortcut = (
            Conv2d(
                in_channels, out_channels, (1, 1),
                strides=stride, dropout=0.0, rngs=rngs,
            )
            if in_channels != out_channels or stride != (1, 1)
            else None
        )

    def __call__(self, x, *, rngs: nnx.Rngs):
        out = self.norm1(x)
        out = nnx.relu(out)
        out = self.conv1(out, rngs=rngs)

        out = self.norm2(out)
        out = nnx.relu(out)
        out = self.conv2(out, rngs=rngs)

        identity = x if self.shortcut is None else self.shortcut(x, rngs=rngs)
        return out + identity

class ResidualStage(nnx.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        dropout: float,
        stride=(1, 1),
        *,
        rngs: nnx.Rngs,
    ):
        blocks = []
        blocks.append(
            ResidualBlock(
                in_channels,
                out_channels,
                stride=stride,
                dropout=dropout,
                rngs=rngs,
            )
        )
        # Rest keep stride = (1,1)
        for _ in range(1, num_blocks):
            blocks.append(
                ResidualBlock(
                    out_channels,
                    out_channels,
                    stride=(1, 1),
                    dropout=dropout,
                    rngs=rngs,
                )
            )
        self.blocks = nnx.List(blocks)

    def __call__(self, x, *, rngs: nnx.Rngs):
        for block in self.blocks:
            x = block(x, rngs=rngs)
        return x

class Classifier(nnx.Module):
    """Residual network classifier for MNIST."""

    def __init__(
        self,
        input_depth: int,
        layer_depths: list[int],
        blocks_per_stage,
        layer_kernel_sizes: list[tuple[int, int]],
        num_classes: int,
        *,
        rngs: nnx.Rngs,
        dropout: float = 0.1,
    ):
        assert len(layer_depths) == len(layer_kernel_sizes), \
            "layer_depths and layer_kernel_sizes must match in length"
        self.init_conv = Conv2d(input_depth, layer_depths[0],
                                kernel_size=layer_kernel_sizes[0],
                                strides=(1, 1), dropout=dropout, rngs=rngs)

        # Build stages
        stages = []
        self.blocks_per_stage = blocks_per_stage
        in_ch = layer_depths[0]
        for i, out_ch in enumerate(layer_depths):
            stride = (2, 2) if i > 0 else (1, 1)  # downsample at stage boundaries
            num_blocks = (self.blocks_per_stage[i]
                      if self.blocks_per_stage is not None
                      else 1)  
            stage = ResidualStage(
                in_channels=in_ch,
                out_channels=out_ch,
                num_blocks=num_blocks,
                stride=stride,
                dropout=dropout,
                rngs=rngs,
            )
            stages.append(stage)
            in_ch = out_ch
            
        self.stages = nnx.List(stages)

        # Final classifier head
        self.fc = nnx.Linear(in_ch, num_classes, rngs=rngs)

    def __call__(self, x, *, rngs: nnx.Rngs):
        x = self.init_conv(x, rngs=rngs)
        for stage in self.stages:
            x = stage(x, rngs=rngs)

        x = jnp.mean(x, axis=(1, 2))
       
        # Final logits
        return self.fc(x)