import torch
import torch.nn as nn
from typing import Sequence, Union, List, Optional


class TorchMLP(nn.Module):
    """
    Flexible MLP for differential testing.

    Supports:
      - single hidden_dim (int) for backward compatibility
      - or multiple hidden layers via hidden_dims (Sequence[int])
      - activations: 'relu', 'tanh', 'sigmoid', 'gelu', 'elu'
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: Union[int, Sequence[int]],
        output_dim: int,
        activations: Optional[Sequence[str]] = None,
    ):
        super().__init__()

        # Normalize hidden_dim to a list
        if isinstance(hidden_dim, int):
            hidden_dims: List[int] = [hidden_dim]
        else:
            hidden_dims = list(hidden_dim)

        if activations is None:
            activations = ["relu"] * len(hidden_dims)
        else:
            activations = list(activations)

        if len(activations) != len(hidden_dims):
            raise ValueError(
                f"Number of activations ({len(activations)}) "
                f"must match number of hidden layers ({len(hidden_dims)})"
            )

        layers: List[nn.Module] = []
        prev_dim = input_dim

        for h, act in zip(hidden_dims, activations):
            layers.append(nn.Linear(prev_dim, h))

            act = act.lower()
            if act == "relu":
                layers.append(nn.ReLU())
            elif act == "tanh":
                layers.append(nn.Tanh())
            elif act == "sigmoid":
                layers.append(nn.Sigmoid())
            elif act == "gelu":
                layers.append(nn.GELU())
            elif act == "elu":
                layers.append(nn.ELU())
            else:
                raise ValueError(f"Unsupported activation for TorchMLP: {act}")

            prev_dim = h

        # Final linear output layer (no activation)
        layers.append(nn.Linear(prev_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
