# This file is here just to define MlpPolicy/CnnPolicy
# that work for PPO
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, MultiInputActorCriticPolicy
from torch import nn
class MultiInputPolicyExpert(MultiInputActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(MultiInputPolicyExpert, self).__init__(*args, **kwargs)
        self.value_net_expert = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
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
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        # Evaluate the values for the given observations
        values = self.value_net_expert(latent_vf.detach())
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = action #Replaced sampling with the expert action
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]
        return actions, values, log_prob

import copy
class MlpPolicyExpert(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(MlpPolicyExpert, self).__init__(*args, **kwargs)
        self.latent_vf_expert = copy.deepcopy(self.mlp_extractor.value_net)
        self.value_net_expert = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        self.optimizer.add_param_group({"params": list(self.latent_vf_expert.parameters()) + \
            list(self.value_net_expert.parameters())})
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
            latent_vf = self.value_net_expert(vf_features) #Use the expert value net for vf_features
        # Evaluate the values for the given observations
        # values = self.value_net(latent_vf)

        values = self.value_net_expert(latent_vf) #High is undetached
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = action #Replaced sampling with the expert action
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]
        return actions, values, log_prob

    def evaluate_actions_expert(self, obs, actions):
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation
        :param actions: Actions
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.value_net_expert(vf_features) #Use the expert value net for vf_features
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net_expert(latent_vf)
        entropy = distribution.entropy()
        return values, log_prob, entropy

class CnnPolicyExpert(ActorCriticCnnPolicy):
    def __init__(self, *args, **kwargs):
        super(CnnPolicyExpert, self).__init__(*args, **kwargs)

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
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = action #Replaced sampling with the expert action
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]
        return actions, values, log_prob



# MlpPolicy = ActorCriticPolicyExpert
# CnnPolicy = ActorCriticCnnPolicy
# MultiInputPolicy = MultiInputActorCriticPolicy