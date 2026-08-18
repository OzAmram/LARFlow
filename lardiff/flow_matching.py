import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask

from lardiff import ode_solvers
from lardiff.transformer import Transformer, compute_mask

__all__ = ["CNF"]


# partially based on https://gist.github.com/francois-rozet/fd6a820e052157f8ac6e2aa39e16c1aa
class CNF(nn.Module):
    def __init__(
        self,
        network: Transformer,
        frequencies: int = 3,
        solver: str = "heun",
    ) -> None:
        super().__init__()
        self.frequencies = nn.Buffer(
            (torch.arange(1, frequencies + 1) * torch.pi).reshape(1, -1)
        )
        self.network = network
        self.set_solver(solver)

    def set_solver(self, solver: str) -> None:
        if solver not in ode_solvers.integrators:
            raise ValueError(
                f"Solver '{solver}' is not registered. "
                f"Available solvers: {list(ode_solvers.integrators.keys())}"
            )
        self.solver = ode_solvers.integrators[solver]

    def forward(self, t: Tensor, x: Tensor, **kwargs) -> Tensor:
        t = self.frequencies * t.reshape(-1, 1)
        t = torch.cat((t.cos(), t.sin()), dim=-1)
        t = t.expand(x.shape[0], -1)

        return self.network(t, x, **kwargs)

    def __calculate_block_mask(self, kwargs: dict[str, Tensor | BlockMask]) -> None:
        if "mask" not in kwargs:
            raise ValueError(
                "The 'mask' argument must be provided in kwargs."
                "This implementation of a CNF only supports our transformers"
                "implementation as the network."
            )
        if not isinstance(kwargs["mask"], Tensor):
            raise TypeError(
                f"'mask' must be of type Tensor. Got {type(kwargs['mask'])}."
            )
        mask = kwargs["mask"]
        del kwargs["mask"]
        kwargs["block_mask"] = compute_mask(padding_mask=mask)

    def encode(self, x: Tensor, num_timesteps: int = 200, **kwargs) -> Tensor:
        self.__calculate_block_mask(kwargs)
        return self.solver(self, x, 0.0, 1.0, num_timesteps, **kwargs)

    def decode(self, z: Tensor, num_timesteps: int = 200, **kwargs) -> Tensor:
        self.__calculate_block_mask(kwargs)
        return self.solver(self, z, 1.0, 0.0, num_timesteps, **kwargs)

    def loss(
        self, x: Tensor, noise: Tensor | None, t: Tensor | None = None, **kwargs
    ) -> Tensor:
        self.__calculate_block_mask(kwargs)
        if t is None:
            t = torch.rand(
                [x.shape[0]] + [1] * (x.dim() - 1), device=x.device, dtype=x.dtype
            )
        else:
            # one time per event, broadcast over points and features
            t = t.to(device=x.device, dtype=x.dtype).reshape(
                [x.shape[0]] + [1] * (x.dim() - 1)
            )
        z = noise if noise is not None else torch.randn_like(x)
        y = (1 - t) * x + (1e-4 + (1 - 1e-4) * t) * z
        u = (1 - 1e-4) * z - x

        return (self(t.reshape(-1, 1), y, **kwargs) - u).square()

    def sample(
        self, shape: tuple[int, ...], num_timesteps: int = 200, **kwargs
    ) -> Tensor:
        z = torch.randn(
            *shape, device=self.frequencies.device, dtype=self.frequencies.dtype
        )
        return self.decode(z, num_timesteps, **kwargs)

    def __repr__(self) -> str:
        network = self.network.__repr__().replace("\n", "\n  ")
        return f"""\
{self.__class__.__name__}(
  (network): {network}
  frequencies/pi={(self.frequencies[0] / torch.pi).tolist()}
)"""
