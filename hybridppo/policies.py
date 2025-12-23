# This file is here just to define MlpPolicy/CnnPolicy
# that work for PPO
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, MultiInputActorCriticPolicy
from typing import Dict, List, Tuple, Type, Union
import torch as th
from stable_baselines3.common.utils import get_device
from torch import nn
# class MultiInputPolicyExpert(MultiInputActorCriticPolicy):
#     def __init__(self, *args, **kwargs):
#         super(MultiInputPolicyExpert, self).__init__(*args, **kwargs)
#         self.value_net_expert = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
#     def forward_expert(self, obs, action, deterministic=False):
#         """
#         Forward pass in all the networks (actor and critic)
#
#         :param obs: Observation
#         :param deterministic: Whether to sample or use deterministic actions
#         :return: action, value and log probability of the action
#         """
#         # Preprocess the observation if needed
#         features = self.extract_features(obs)
#         if self.share_features_extractor:
#             latent_pi, latent_vf = self.mlp_extractor(features)
#         else:
#             pi_features, vf_features = features
#             latent_pi = self.mlp_extractor.forward_actor(pi_features)
#             latent_vf = self.mlp_extractor.forward_critic(vf_features)
#         # Evaluate the values for the given observations
#         values = self.value_net_expert(latent_vf.detach())
#         distribution = self._get_action_dist_from_latent(latent_pi)
#         actions = action #Replaced sampling with the expert action
#         log_prob = distribution.log_prob(actions)
#         actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]
#         return actions, values, log_prob

import copy
class MlpPolicyExpert(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(MlpPolicyExpert, self).__init__(*args, **kwargs)

    class MlpExtractor(nn.Module):
        """
        Constructs an MLP that receives the output from a previous features extractor (i.e., a CNN) or directly
        the observations (if no features extractor is applied) as an input and outputs a latent representation
        for the policy and a value network.

        The ``net_arch`` parameter allows to specify the amount and size of the hidden layers.
        It can be in either of the following forms:
        1. ``dict(vf=[<list of layer sizes>], pi=[<list of layer sizes>])``: to specify the amount and size of the layers in the
            policy and value nets individually. If it is missing any of the keys (pi or vf),
            zero layers will be considered for that key.
        2. ``[<list of layer sizes>]``: "shortcut" in case the amount and size of the layers
            in the policy and value nets are the same. Same as ``dict(vf=int_list, pi=int_list)``
            where int_list is the same for the actor and critic.

        .. note::
            If a key is not specified or an empty list is passed ``[]``, a linear network will be used.

        :param feature_dim: Dimension of the feature vector (can be the output of a CNN)
        :param net_arch: The specification of the policy and value networks.
            See above for details on its formatting.
        :param activation_fn: The activation function to use for the networks.
        :param device: PyTorch device.
        """

        def __init__(
                self,
                feature_dim: int,
                net_arch: Union[List[int], Dict[str, List[int]]],
                activation_fn: Type[nn.Module],
                device: Union[th.device, str] = "auto",
                add_layer_norm: bool = False,  # New argument to control Layer Norm
        ) -> None:
            super().__init__()
            device = get_device(device)
            policy_net: List[nn.Module] = []
            value_net: List[nn.Module] = []
            last_layer_dim_pi = feature_dim
            last_layer_dim_vf = feature_dim

            # Save dimensions of layers in policy and value nets
            if isinstance(net_arch, dict):
                # Note: if key is not specified, assume linear network
                pi_layers_dims = net_arch.get("pi", [])  # Layer sizes of the policy network
                vf_layers_dims = net_arch.get("vf", [])  # Layer sizes of the value network
            else:
                pi_layers_dims = vf_layers_dims = net_arch

            # Iterate through the policy layers and build the policy net
            for curr_layer_dim in pi_layers_dims:
                policy_net.append(nn.Linear(last_layer_dim_pi, curr_layer_dim))
                if add_layer_norm:
                    policy_net.append(nn.LayerNorm(curr_layer_dim))
                policy_net.append(activation_fn())
                last_layer_dim_pi = curr_layer_dim

            # Iterate through the value layers and build the value net
            for curr_layer_dim in vf_layers_dims:
                value_net.append(nn.Linear(last_layer_dim_vf, curr_layer_dim))
                if add_layer_norm:
                    value_net.append(nn.LayerNorm(curr_layer_dim))
                value_net.append(activation_fn())
                last_layer_dim_vf = curr_layer_dim

            # Save dim, used to create the distributions
            self.latent_dim_pi = last_layer_dim_pi
            self.latent_dim_vf = last_layer_dim_vf

            # Create networks
            # If the list of layers is empty, the network will just act as an Identity module
            self.policy_net = nn.Sequential(*policy_net).to(device)
            self.value_net = nn.Sequential(*value_net).to(device)

        def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
            """
            :return: latent_policy, latent_value of the specified network.
                If all layers are shared, then ``latent_policy == latent_value``
            """
            return self.forward_actor(features), self.forward_critic(features)

        def forward_actor(self, features: th.Tensor) -> th.Tensor:
            return self.policy_net(features)

        def forward_critic(self, features: th.Tensor) -> th.Tensor:
            return self.value_net(features)

    def forward_expert(self, obs, action, deterministic=False):
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features) #Use the expert value net for vf_features
        # Evaluate the values for the given observations
        # values = self.value_net(latent_vf)

        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = action #Replaced sampling with the expert action
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]
        return actions, values, log_prob


