import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.attention.flex_attention import (
    BlockMask,
    flex_attention,
)

__all__ = ["FlexEncoderLayer", "Transformer", "compute_mask"]

def compute_mask(padding_mask: Tensor, block_size: int = 128) -> BlockMask:
    """Build the padding BlockMask analytically.

    Requires the padding to be a contiguous prefix (all real points first),
    which both the data pipeline and the generator guarantee.  This avoids
    create_block_mask, which materializes a dense (B, P, P) mask and whose
    compiled version is broken for captured-tensor mask_mods in torch 2.5.
    """
    padding_mask = padding_mask.flatten(1)
    batch_size, sequence_length = padding_mask.shape
    device = padding_mask.device
    num_points = padding_mask.sum(dim=1)
    assert bool(
        (padding_mask.to(torch.int8).diff(dim=1) <= 0).all()
    ), "padding must be a contiguous prefix of real points"

    num_blocks = (sequence_length + block_size - 1) // block_size
    active_blocks = (num_points + block_size - 1) // block_size
    block_idx = torch.arange(num_blocks, device=device)
    # rows in an active block attend to all active blocks, padded rows to none
    kv_num_blocks = torch.where(
        block_idx[None] < active_blocks[:, None], active_blocks[:, None], 0
    )
    kv_indices = block_idx.expand(batch_size, 1, num_blocks, num_blocks)

    def mask_fn(b, h, q_idx, kv_idx):
        return (q_idx < num_points[b]) & (kv_idx < num_points[b])

    return BlockMask.from_kv_blocks(
        kv_num_blocks.unsqueeze(1).to(torch.int32).contiguous(),
        kv_indices.to(torch.int32).contiguous(),
        BLOCK_SIZE=block_size,
        mask_mod=mask_fn,
    )


class FlexEncoderLayer(nn.Module):
    def __init__(
        self,
        dim_embedding: int,
        num_head: int = 4,
        dim_feedforward: int = 2048,
        activation: str | torch.nn.Module = "relu",
        dropout: float = 0.0,
    ) -> None:
        if dim_embedding % num_head != 0:
            raise ValueError(
                f"dim_embedding ({dim_embedding}) must be divisible by num_head ({num_head})."
            )
        super().__init__()

        self.num_head = num_head
        self.dim_embedding = dim_embedding
        self.dim_head = dim_embedding // num_head

        activation_classes = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "leaky_relu": nn.LeakyReLU,
        }
        if isinstance(activation, str):
            activation_module = activation_classes[activation]()
        else:
            activation_module = activation
        del activation

        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()

        self.key_query_value = nn.Linear(dim_embedding, dim_embedding * 3)
        self.feedforward = nn.Sequential(
            nn.Linear(dim_embedding, dim_feedforward),
            activation_module,
            nn.Linear(dim_feedforward, dim_embedding),
            self.dropout,
        )
        self.layer_norm1 = nn.LayerNorm(dim_embedding)
        self.layer_norm2 = nn.LayerNorm(dim_embedding)

    def multihead_attention(
        self,
        x: Tensor,
        mask: BlockMask,
    ) -> Tensor:
        key_query_value: Tensor = self.key_query_value(x)
        key_query_value = key_query_value.view(
            key_query_value.shape[0],
            key_query_value.shape[1],
            self.num_head,
            3,
            self.dim_head,
        )
        key_query_value = key_query_value.permute(3, 0, 2, 1, 4).contiguous()
        key, query, value = key_query_value

        x = flex_attention(
            query=query,
            key=key,
            value=value,
            block_mask=mask,
        )  # type: ignore

        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(x.shape[0], x.shape[1], self.dim_embedding)
        x = self.dropout(x)
        return x

    def forward(
        self,
        x: Tensor,
        mask: BlockMask,
    ) -> Tensor:
        x = self.layer_norm1(x + self.multihead_attention(x, mask=mask))
        x = self.layer_norm2(x + self.feedforward(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        dim_inputs: tuple[int, ...],
        dim_embedding: int,
        num_head: int,
        num_blocks: int,
        dim_feedforward: int = 2048,
        identity_init: bool = False,
        activation: str | torch.nn.Module = "relu",
        num_particles: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Linear(dim_inputs[0], dim_embedding)
        self.cond_embedding = nn.Linear(sum(dim_inputs[1:]), dim_embedding)
        activation_classes = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "leaky_relu": nn.LeakyReLU,
        }
        if isinstance(activation, str):
            activation_module = activation_classes[activation.lower()]()
        else:
            activation_module = activation
        del activation
        if num_particles > 1:
            self.particle_embedding = nn.Embedding(num_particles, dim_embedding)
        else:
            self.particle_embedding = None

        self.transformer_blocks = nn.ModuleList(
            [
                FlexEncoderLayer(
                    dim_embedding,
                    num_head,
                    dim_feedforward=dim_feedforward,
                    activation=activation_module,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
        )

        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()

        self.head = nn.Linear(dim_embedding, dim_inputs[0])
        if identity_init:
            with torch.no_grad():
                self.head.weight.fill_(0.0)
                self.head.bias.fill_(0.0)

    def forward(
        self,
        t: Tensor,
        x: Tensor,
        cond: Tensor,
        block_mask: BlockMask,
        label: Tensor | None = None,
    ) -> Tensor:
        x = self.embedding(x)
        cond = torch.cat([t, cond], dim=1)
        cond = self.cond_embedding(cond).unsqueeze(1)
        x += cond
        if label is not None and self.particle_embedding is not None:
            x += self.particle_embedding(label).unsqueeze(1)
        for block in self.transformer_blocks:
            x = block(x, mask=block_mask)
        return self.head(x)
