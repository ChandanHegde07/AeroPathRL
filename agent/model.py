from __future__ import annotations

from typing import List, Tuple, Type

import torch
import torch.nn as nn
from torch import Tensor



def _mlp_block(
    in_dim: int,
    out_dim: int,
    activation: Type[nn.Module] = nn.Tanh,
    layer_norm: bool = False,
) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(in_dim, out_dim)]
    if layer_norm:
        layers.append(nn.LayerNorm(out_dim))
    layers.append(activation())
    return nn.Sequential(*layers)


def _build_mlp(
    dims: List[int],
    activation: Type[nn.Module] = nn.Tanh,
    layer_norm: bool = False,
) -> nn.Sequential:
    """Build a multi-layer perceptron from a list of dimensions."""
    assert len(dims) >= 2, "dims must have at least input and output size"
    layers = []
    for i in range(len(dims) - 1):
        layers.append(_mlp_block(dims[i], dims[i + 1], activation, layer_norm))
    return nn.Sequential(*layers)



class DroneActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int = 3,
        shared_arch: List[int] = None,
        actor_arch: List[int] = None,
        critic_arch: List[int] = None,
        activation: Type[nn.Module] = nn.Tanh,
        log_std_init: float = 0.0,
    ):
        super().__init__()
        shared_arch = shared_arch or [256, 256]
        actor_arch  = actor_arch  or [128]
        critic_arch = critic_arch or [128]

        encoder_dims = [obs_dim] + shared_arch
        self.encoder = _build_mlp(encoder_dims, activation, layer_norm=True)
        latent_dim = shared_arch[-1]

        actor_dims = [latent_dim] + actor_arch
        self.actor_net = _build_mlp(actor_dims, activation)
        self.actor_mean = nn.Linear(actor_arch[-1], action_dim)

        self.log_std = nn.Parameter(
            torch.ones(action_dim) * log_std_init
        )

        critic_dims = [latent_dim] + critic_arch
        self.critic_net = _build_mlp(critic_dims, activation)
        self.critic_out = nn.Linear(critic_arch[-1], 1)

        self._init_weights()


    def encode(self, obs: Tensor) -> Tensor:
        return self.encoder(obs)

    def actor_forward(self, obs: Tensor) -> Tuple[Tensor, Tensor]:
        latent = self.encode(obs)
        mean   = torch.tanh(self.actor_mean(self.actor_net(latent)))
        log_std = self.log_std.expand_as(mean)
        return mean, log_std

    def critic_forward(self, obs: Tensor) -> Tensor:
        latent = self.encode(obs)
        return self.critic_out(self.critic_net(latent))

    def forward(self, obs: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        latent  = self.encode(obs)
        mean    = torch.tanh(self.actor_mean(self.actor_net(latent)))
        log_std = self.log_std.expand_as(mean)
        value   = self.critic_out(self.critic_net(latent))
        return mean, log_std, value

    def get_action(self, obs: Tensor, deterministic: bool = False) -> Tensor:
        """Sample (or take the mean of) the action distribution."""
        mean, log_std, _ = self.forward(obs)
        if deterministic:
            return mean
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        return dist.sample().clamp(-1.0, 1.0)


    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.constant_(module.bias, 0.0)
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.orthogonal_(self.critic_out.weight, gain=1.0)


    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def export_onnx(self, path: str, obs_dim: int, device: str = "cpu"):
        self.eval()
        dummy = torch.zeros(1, obs_dim, device=device)
        torch.onnx.export(
            self,
            dummy,
            path,
            input_names=["observation"],
            output_names=["action_mean", "log_std", "value"],
            dynamic_axes={"observation": {0: "batch_size"}},
        )
        print(f"[model] ONNX exported → {path}")



def build_sb3_policy_kwargs(
    net_arch: List[int] = None,
    activation_fn_name: str = "tanh",
) -> dict:
    net_arch = net_arch or [256, 256]

    _activation_map = {
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "elu":  nn.ELU,
    }
    activation_fn = _activation_map.get(activation_fn_name.lower(), nn.Tanh)

    return {
        "net_arch":      net_arch,
        "activation_fn": activation_fn,
    }
