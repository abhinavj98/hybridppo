from typing import Any, Dict, List, Optional, Type, Union

import torch as th
from gymnasium import spaces
from torch import nn

from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
)
from stable_baselines3.sac.policies import Actor, ContinuousCritic, SACPolicy


def create_mlp_with_ln(
    input_dim: int,
    output_dim: int,
    net_arch: List[int],
    activation_fn: Type[nn.Module] = nn.ReLU,
    squash_output: bool = False,
) -> List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), with Layer Normalization after each linear layer
    (except the last one).
    """
    if len(net_arch) > 0:
        modules = []
        last_dim = input_dim
        for layer_dim in net_arch:
            modules.append(nn.Linear(last_dim, layer_dim))
            modules.append(nn.LayerNorm(layer_dim))
            modules.append(activation_fn())
            last_dim = layer_dim

        modules.append(nn.Linear(last_dim, output_dim))
    else:
        modules = [nn.Linear(input_dim, output_dim)]

    if squash_output:
        modules.append(nn.Tanh())
    return modules


class RLPDActor(Actor):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        full_std: bool = True,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        normalize_images: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            net_arch,
            features_extractor,
            features_dim,
            activation_fn,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            clip_mean,
            normalize_images,
        )

        # Rebuild latent_pi using create_mlp_with_ln
        action_dim = get_action_dim(self.action_space)

        if len(net_arch) > 0:
            latent_dim_pi = net_arch[-1]
            latent_pi_net = create_mlp_with_ln(
                self.features_dim,
                latent_dim_pi,
                net_arch[:-1],
                activation_fn,
            )
        else:
            latent_dim_pi = self.features_dim
            latent_pi_net = []

        self.latent_pi = nn.Sequential(*latent_pi_net)

        self.mu = nn.Linear(latent_dim_pi, action_dim)
        self.log_std = nn.Linear(latent_dim_pi, action_dim)


class RLPDContinuousCritic(ContinuousCritic):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        n_critics: int = 2,
        share_features_extractor: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            net_arch,
            features_extractor,
            features_dim,
            activation_fn,
            normalize_images,
            n_critics,
            share_features_extractor,
        )

        action_dim = get_action_dim(self.action_space)
        # Rebuild q_networks
        self.q_networks = []
        for idx in range(n_critics):
            q_net = create_mlp_with_ln(
                features_dim + action_dim,
                1,
                net_arch,
                activation_fn,
            )
            q_net = nn.Sequential(*q_net)
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)


class RLPDPolicy(SACPolicy):
    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Actor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return RLPDActor(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> ContinuousCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return RLPDContinuousCritic(**critic_kwargs).to(self.device)
