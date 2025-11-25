import rl_games.algos_torch.layers
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import rl_games.common.divergence as divergence
from rl_games.common.extensions.distributions import CategoricalMasked
from torch.distributions import Categorical
from rl_games.algos_torch.sac_helper import SquashedNormal
from rl_games.algos_torch.running_mean_std import RunningMeanStd, RunningMeanStdObs
from rl_games.algos_torch.moving_mean_std import GeneralizedMovingStats
from torch.distributions import Normal, TransformedDistribution, TanhTransform
import math

class BaseModel():
    """Base class for all custom models.
    
    This class defines the interface that all model wrappers must implement.
    It handles the creation of the underlying network using the network builder.
    """
    def __init__(self, model_class):
        # Store the type of model (e.g., 'a2c', 'sac')
        self.model_class = model_class

    def is_rnn(self):
        # Default implementation: returns False indicating this is not a Recurrent Neural Network
        return False

    def is_separate_critic(self):
        # Default implementation: returns False indicating actor and critic share weights or are not separate
        return False

    def get_value_layer(self):
        # Default implementation: returns None for the value layer
        return None

    def build(self, config):
        # Build the network model based on the configuration provided
        obs_shape = config['input_shape']
        # Check if value function normalization is enabled
        normalize_value = config.get('normalize_value', False)
        # Check if input observation normalization is enabled
        normalize_input = config.get('normalize_input', False)
        # Get the size of the value output (usually 1)
        value_size = config.get('value_size', 1)
        
        # Build the underlying network using the network builder and wrap it in the specific Network class
        return self.Network(self.network_builder.build(self.model_class, **config), obs_shape=obs_shape,
            normalize_value=normalize_value, normalize_input=normalize_input, value_size=value_size)

class BaseModelNetwork(nn.Module):
    """Base network class that wraps the actual neural network.
    
    It adds functionality for observation normalization and value function denormalization.
    """
    def __init__(self, obs_shape, normalize_value, normalize_input, value_size):
        nn.Module.__init__(self)
        self.obs_shape = obs_shape
        self.normalize_value = normalize_value
        self.normalize_input = normalize_input
        self.value_size = value_size

        # Initialize running mean and standard deviation for value normalization if enabled
        if normalize_value:
            self.value_mean_std = RunningMeanStd((self.value_size,)) 
        
        # Initialize running mean and standard deviation for input observation normalization if enabled
        if normalize_input:
            if isinstance(obs_shape, dict):
                self.running_mean_std = RunningMeanStdObs(obs_shape)
            else:
                self.running_mean_std = RunningMeanStd(obs_shape)

    def norm_obs(self, observation):
        # Normalize observations using running statistics if input normalization is enabled
        with torch.no_grad():
            return self.running_mean_std(observation) if self.normalize_input else observation

    def denorm_value(self, value):
        # Denormalize the value function output using running statistics if value normalization is enabled
        with torch.no_grad():
            return self.value_mean_std(value, denorm=True) if self.normalize_value else value
        
    def get_aux_loss(self):
        # Default implementation: returns None for auxiliary loss
        return None

class CustomModelPPO(BaseModel):
    """Custom PPO model for discrete action spaces."""
    def __init__(self, network):
        # Initialize with 'a2c' model class (PPO uses A2C-style networks)
        BaseModel.__init__(self, 'a2c')
        self.network_builder = network

    class Network(BaseModelNetwork):
        """Network implementation for CustomModelPPO."""
        def __init__(self, a2c_network, **kwargs):
            BaseModelNetwork.__init__(self,**kwargs)
            self.a2c_network = a2c_network
        
        def get_aux_loss(self):
            # Delegate auxiliary loss calculation to the underlying network
            return self.a2c_network.get_aux_loss()
        
        def is_rnn(self):
            # Check if the underlying network is an RNN
            return self.a2c_network.is_rnn()
        
        def get_default_rnn_state(self):
            # Get the default RNN state from the underlying network
            return self.a2c_network.get_default_rnn_state()            

        def get_value_layer(self):
            # Get the value layer from the underlying network
            return self.a2c_network.get_value_layer()

        def kl(self, p_dict, q_dict):
            # Calculate KL divergence for discrete distributions
            p = p_dict['logits']
            q = q_dict['logits']
            return divergence.d_kl_discrete(p, q)

        def forward(self, input_dict):
            # Forward pass of the network
            is_train = input_dict.get('is_train', True)
            action_masks = input_dict.get('action_masks', None)
            prev_actions = input_dict.get('prev_actions', None)
            
            # Normalize observations
            input_dict['obs'] = self.norm_obs(input_dict['obs'])
            
            # Get logits, value, and states from the underlying network
            logits, value, states = self.a2c_network(input_dict)

            if is_train:
                # Training mode: calculate losses and entropy
                categorical = CategoricalMasked(logits=logits, masks=action_masks)
                prev_neglogp = -categorical.log_prob(prev_actions)
                entropy = categorical.entropy()
                result = {
                    'prev_neglogp' : torch.squeeze(prev_neglogp),
                    'logits' : categorical.logits,
                    'values' : value,
                    'entropy' : entropy,
                    'rnn_states' : states
                }
                return result
            else:
                # Inference mode: sample actions
                categorical = CategoricalMasked(logits=logits, masks=action_masks)
                selected_action = categorical.sample().long()
                neglogp = -categorical.log_prob(selected_action)
                result = {
                    'neglogpacs' : torch.squeeze(neglogp),
                    'values' : self.denorm_value(value), # Denormalize value for output
                    'actions' : selected_action,
                    'logits' : categorical.logits,
                    'rnn_states' : states
                }
                return  result

class CustomModelPPOMultiDiscrete(BaseModel):
    """Custom PPO model for multi-discrete action spaces."""
    def __init__(self, network):
        BaseModel.__init__(self, 'a2c')
        self.network_builder = network

    class Network(BaseModelNetwork):
        """Network implementation for CustomModelPPOMultiDiscrete."""
        def __init__(self, a2c_network, **kwargs):
            BaseModelNetwork.__init__(self, **kwargs)
            self.a2c_network = a2c_network

        def get_aux_loss(self):
            return self.a2c_network.get_aux_loss()
        
        def is_rnn(self):
            return self.a2c_network.is_rnn()
        
        def get_default_rnn_state(self):
            return self.a2c_network.get_default_rnn_state()

        def get_value_layer(self):
            return self.a2c_network.get_value_layer()

        def kl(self, p_dict, q_dict):
            # Calculate KL divergence for a list of discrete distributions
            p = p_dict['logits']
            q = q_dict['logits']
            return divergence.d_kl_discrete_list(p, q)

        def forward(self, input_dict):
            is_train = input_dict.get('is_train', True)
            action_masks = input_dict.get('action_masks', None)
            prev_actions = input_dict.get('prev_actions', None)
            input_dict['obs'] = self.norm_obs(input_dict['obs'])
            logits, value, states = self.a2c_network(input_dict)
            
            if is_train:
                # Handle action masks for multi-discrete actions
                if action_masks is None:
                    categorical = [Categorical(logits=logit) for logit in logits]
                else:
                    # Split masks for each discrete action component
                    action_masks = np.split(action_masks,len(logits), axis=1)
                    categorical = [CategoricalMasked(logits=logit, masks=mask) for logit, mask in zip(logits, action_masks)]
                
                # Split previous actions for each component
                prev_actions = torch.split(prev_actions, 1, dim=-1)
                # Calculate negative log probability for each component
                prev_neglogp = [-c.log_prob(a.squeeze()) for c,a in zip(categorical, prev_actions)]
                # Sum negative log probabilities across components
                prev_neglogp = torch.stack(prev_neglogp, dim=-1).sum(dim=-1)
                
                # Calculate entropy for each component and sum them
                entropy = [c.entropy() for c in categorical]
                entropy = torch.stack(entropy, dim=-1).sum(dim=-1)
                
                result = {
                    'prev_neglogp' : torch.squeeze(prev_neglogp),
                    'logits' : [c.logits for c in categorical],
                    'values' : value,
                    'entropy' : torch.squeeze(entropy),
                    'rnn_states' : states
                }
                return result
            else:
                # Inference mode
                if action_masks is None:
                    categorical = [Categorical(logits=logit) for logit in logits]
                else:
                    action_masks = np.split(action_masks, len(logits), axis=1)
                    categorical = [CategoricalMasked(logits=logit, masks=mask) for logit, mask in zip(logits, action_masks)]
                
                # Sample actions for each component
                selected_action = [c.sample().long() for c in categorical]
                neglogp = [-c.log_prob(a.squeeze()) for c,a in zip(categorical, selected_action)]
                
                # Stack selected actions and sum negative log probabilities
                selected_action = torch.stack(selected_action, dim=-1)
                neglogp = torch.stack(neglogp, dim=-1).sum(dim=-1)
                
                result = {
                    'neglogpacs' : torch.squeeze(neglogp),
                    'values' : self.denorm_value(value),
                    'actions' : selected_action,
                    'logits' : [c.logits for c in categorical],
                    'rnn_states' : states
                }
                return  result

class CustomModelPPOContinuous(BaseModel):
    """Custom PPO model for continuous action spaces."""
    def __init__(self, network):
        BaseModel.__init__(self, 'a2c')
        self.network_builder = network

    class Network(BaseModelNetwork):
        """Network implementation for CustomModelPPOContinuous."""
        def __init__(self, a2c_network, **kwargs):
            BaseModelNetwork.__init__(self, **kwargs)
            self.a2c_network = a2c_network

        def get_aux_loss(self):
            return self.a2c_network.get_aux_loss()
        
        def is_rnn(self):
            return self.a2c_network.is_rnn()
            
        def get_default_rnn_state(self):
            return self.a2c_network.get_default_rnn_state()

        def get_value_layer(self):
            return self.a2c_network.get_value_layer()

        def kl(self, p_dict, q_dict):
            # Calculate KL divergence for normal distributions
            p = p_dict['mu'], p_dict['sigma']
            q = q_dict['mu'], q_dict['sigma']
            return divergence.d_kl_normal(p, q)

        def forward(self, input_dict):
            is_train = input_dict.get('is_train', True)
            prev_actions = input_dict.get('prev_actions', None)
            input_dict['obs'] = self.norm_obs(input_dict['obs'])
            
            # Get mean (mu), standard deviation (sigma), value, and states from network
            mu, sigma, value, states = self.a2c_network(input_dict)
            
            # Create a Normal distribution
            distr = torch.distributions.Normal(mu, sigma, validate_args=False)

            if is_train:
                # Calculate entropy and sum over action dimensions
                entropy = distr.entropy().sum(dim=-1)
                # Calculate negative log probability of previous actions
                prev_neglogp = -distr.log_prob(prev_actions).sum(dim=-1)
                result = {
                    'prev_neglogp' : torch.squeeze(prev_neglogp),
                    'value' : value,
                    'entropy' : entropy,
                    'rnn_states' : states,
                    'mus' : mu,
                    'sigmas' : sigma
                }
                return result
            else:
                # Sample action from distribution
                selected_action = distr.sample().squeeze()
                neglogp = -distr.log_prob(selected_action).sum(dim=-1)
                # Calculate entropy for reporting (Fixed bug: entropy was undefined here)
                entropy = distr.entropy().sum(dim=-1)
                result = {
                    'neglogpacs' : torch.squeeze(neglogp),
                    'values' : self.denorm_value(value),
                    'actions' : selected_action,
                    'entropy' : entropy,
                    'rnn_states' : states,
                    'mus' : mu,
                    'sigmas' : sigma
                }
                return  result          



class CustomModelPPOContinuousLogStd(BaseModel):
    """Custom PPO Model for Continuous Action Spaces using Log Standard Deviation.
    
    This model assumes the network outputs mean (mu) and log standard deviation (logstd)
    for the action distribution. It constructs a Normal distribution from these outputs.
    """
    def __init__(self, network):
        BaseModel.__init__(self, 'a2c')
        self.network_builder = network

    class Network(BaseModelNetwork):
        def __init__(self, a2c_network, **kwargs):
            BaseModelNetwork.__init__(self, **kwargs)
            self.a2c_network = a2c_network

        def get_aux_loss(self):
            return self.a2c_network.get_aux_loss()
        
        def is_rnn(self):
            return self.a2c_network.is_rnn()

        def get_value_layer(self):
            return self.a2c_network.get_value_layer()

        def get_default_rnn_state(self):
            return self.a2c_network.get_default_rnn_state()

        def forward(self, input_dict):
            """Forward pass for the continuous PPO model with logstd.
            
            Args:
                input_dict (dict): Inputs containing observations, previous actions, etc.
                
            Returns:
                dict: Network outputs including logits, values, and distribution info.
            """
            is_train = input_dict.get('is_train', True)
            prev_actions = input_dict.get('prev_actions', None)
            # Normalize observations before passing to the network.
            input_dict['obs'] = self.norm_obs(input_dict['obs'])
            # Get network outputs: mean, logstd, value estimate, and RNN states.
            mu, logstd, value, states = self.a2c_network(input_dict)
            sigma = torch.exp(logstd)
            distr = torch.distributions.Normal(mu, sigma, validate_args=False)
            
            if is_train:
                # During training, calculate entropy and log probability of previous actions.
                # Entropy is used to encourage exploration.
                entropy = distr.entropy().sum(dim=-1)
                # Calculate negative log probability of the previous actions (for PPO loss).
                prev_neglogp = self.neglogp(prev_actions, mu, sigma, logstd)
                result = {
                    'prev_neglogp' : torch.squeeze(prev_neglogp),
                    'values' : value,
                    'entropy' : entropy,
                    'rnn_states' : states,
                    'mus' : mu,
                    'sigmas' : sigma
                }                
                return result
            else:
                # During inference, sample an action from the distribution.
                selected_action = distr.sample()
                # Calculate negative log probability of the selected action.
                neglogp = self.neglogp(selected_action, mu, sigma, logstd)
                result = {
                    'neglogpacs' : torch.squeeze(neglogp),
                    'values' : self.denorm_value(value), # Denormalize value for reporting.
                    'actions' : selected_action,
                    'rnn_states' : states,
                    'mus' : mu,
                    'sigmas' : sigma
                }
                return result

        def neglogp(self, x, mean, std, logstd):
            """Calculate negative log probability of action x under Gaussian(mean, std).
            
            Formula: 0.5 * ((x - mu) / sigma)^2 + 0.5 * log(2 * pi) + log(sigma)
            Summed over the last dimension (action dimension).
            """
            return 0.5 * (((x - mean) / std)**2).sum(dim=-1) \
                + 0.5 * np.log(2.0 * np.pi) * x.size()[-1] \
                + logstd.sum(dim=-1)

class CustomModelPPOContinuousTanh(BaseModel):
    """Custom PPO model for continuous action spaces with Tanh squashing."""
    def __init__(self, network):
        BaseModel.__init__(self, 'a2c')
        self.network_builder = network
        
    class Network(BaseModelNetwork):
        """Network implementation for CustomModelPPOContinuousTanh."""
        def __init__(self, a2c_network, **kwargs):
            BaseModelNetwork.__init__(self, **kwargs)
            self.a2c_network = a2c_network
        def get_aux_loss(self):
            return self.a2c_network.get_aux_loss()
        
        def is_rnn(self):
            return self.a2c_network.is_rnn()

        def get_value_layer(self):
            return self.a2c_network.get_value_layer()

        def get_default_rnn_state(self):
            return self.a2c_network.get_default_rnn_state()

        def forward(self, input_dict):
            is_train = input_dict.get('is_train', True)
            prev_actions = input_dict.get('prev_actions', None)
            input_dict['obs'] = self.norm_obs(input_dict['obs'])
            
            # Get mean (mu), log standard deviation (logstd), value, and states
            mu, logstd, value, states = self.a2c_network(input_dict)
            
            # Calculate sigma using softplus to ensure positivity
            sigma = torch.nn.functional.softplus(logstd + 0.001)
            
            # Create Normal distribution followed by Tanh transform
            main_distr = NormalTanhDistribution(mu.size(-1))
            
            if is_train:
                # Calculate entropy
                entropy = main_distr.entropy(mu, logstd)
                # Calculate negative log probability of previous actions
                # Note: prev_actions need to be inverse transformed (inverse tanh) before log_prob
                prev_neglogp = -main_distr.log_prob(mu, logstd, main_distr.inverse_post_process(prev_actions))
                result = {
                    'prev_neglogp' : torch.squeeze(prev_neglogp),
                    'values' : value,
                    'entropy' : entropy,
                    'rnn_states' : states,
                    'mus' : mu,
                    'sigmas' : sigma
                }                
                return result
            else:
                # Sample raw action (pre-tanh)
                selected_action = main_distr.sample_no_postprocessing(mu, logstd)
                # Calculate negative log probability
                neglogp = -main_distr.log_prob(mu, logstd, selected_action)
                result = {
                    'neglogpacs' : torch.squeeze(neglogp),
                    'values' : self.denorm_value(value),
                    'actions' : main_distr.post_process(selected_action), # Apply Tanh to get final action
                    'rnn_states' : states,
                    'mus' : mu,
                    'sigmas' : sigma
                }
                return result


class ModelCentralValue(BaseModel):
    """Model for Central Value Function (Critic only)."""
    def __init__(self, network):
        BaseModel.__init__(self, 'a2c')
        self.network_builder = network

    class Network(BaseModelNetwork):
        """Network implementation for ModelCentralValue."""
        def __init__(self, a2c_network, **kwargs):
            BaseModelNetwork.__init__(self, **kwargs)
            self.a2c_network = a2c_network

        def get_aux_loss(self):
            return self.a2c_network.get_aux_loss()

        def is_rnn(self):
            return self.a2c_network.is_rnn()

        def get_value_layer(self):
            return self.a2c_network.get_value_layer()

        def get_default_rnn_state(self):
            return self.a2c_network.get_default_rnn_state()

        def kl(self, p_dict, q_dict):
            # KL divergence is not applicable for value-only models
            return None 

        def forward(self, input_dict):
            is_train = input_dict.get('is_train', True)
            prev_actions = input_dict.get('prev_actions', None)
            input_dict['obs'] = self.norm_obs(input_dict['obs'])
            
            # Get value and states from network (no policy output)
            value, states = self.a2c_network(input_dict)
            
            if not is_train:
                # Denormalize value during inference
                value = self.denorm_value(value)

            result = {
                'values': value,
                'rnn_states': states
            }
            return result



class ModelSACContinuous(BaseModel):
    """Custom SAC model for continuous action spaces."""
    def __init__(self, network):
        BaseModel.__init__(self, 'sac')
        self.network_builder = network
    
    class Network(BaseModelNetwork):
        """Network implementation for ModelSACContinuous."""
        def __init__(self, sac_network,**kwargs):
            BaseModelNetwork.__init__(self,**kwargs)
            self.sac_network = sac_network

        def get_aux_loss(self):
            return self.sac_network.get_aux_loss()

        def critic(self, obs, action):
            # Evaluate Q-value for given observation and action
            return self.sac_network.critic(obs, action)

        def critic_target(self, obs, action):
            # Evaluate target Q-value
            return self.sac_network.critic_target(obs, action)

        def actor(self, obs):
            # Get actor output (policy)
            return self.sac_network.actor(obs)
        
        def is_rnn(self):
            # SAC usually doesn't use RNNs in this implementation
            return False

        def forward(self, input_dict):
            # Forward pass for SAC
            is_train = input_dict.pop('is_train', True)
            # Get mean and standard deviation from network
            mu, sigma = self.sac_network(input_dict)
            # Create Squashed Normal distribution (Gaussian squashed by Tanh)
            dist = SquashedNormal(mu, sigma)
            return dist


class TanhBijector:
    """Bijector that applies the Tanh function."""

    def forward(self, x):
        # Apply tanh function
        return torch.tanh(x)

    def inverse(self, y):
        # Apply inverse tanh (arctanh)
        # Clamp input to avoid numerical instability at boundaries (-1, 1)
        y = torch.clamp(y, -0.99999997, 0.99999997)
        return 0.5 * (y.log1p() - (-y).log1p())

    def forward_log_det_jacobian(self, x):
        # Calculate the log of the absolute value of the determinant of the Jacobian
        # This is needed for change of variables in probability density
        # Formula: log(1 - tanh(x)^2) = 2 * (log(2) - x - softplus(-2x))
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))

class NormalTanhDistribution:
    """Normal distribution followed by a Tanh transform (Squashed Normal)."""

    def __init__(self, event_size, min_std=0.001, var_scale=1.0):
        """Initialize the distribution.

        Args:
            event_size (int): The size of events (i.e., actions).
            min_std (float): Minimum standard deviation for the Gaussian.
            var_scale (float): Scaling factor for the Gaussian's scale parameter.
        """
        self.param_size = event_size
        self._min_std = min_std
        self._var_scale = var_scale
        self._event_ndims = 1  # Rank of events
        self._postprocessor = TanhBijector()

    def create_dist(self, loc, scale):
        # Create the base Normal distribution
        # Ensure scale (std dev) is positive and above min_std
        scale = (F.softplus(scale) + self._min_std) * self._var_scale
        return torch.distributions.Normal(loc=loc, scale=scale)

    def sample_no_postprocessing(self, loc, scale):
        # Sample from the base Normal distribution (before Tanh)
        dist = self.create_dist(loc, scale)
        return dist.rsample() # rsample allows backpropagation through sampling

    def sample(self, loc, scale):
        """Returns a sample from the postprocessed distribution."""
        pre_tanh_sample = self.sample_no_postprocessing(loc, scale)
        return self._postprocessor.forward(pre_tanh_sample)

    def post_process(self, pre_tanh_sample):
        """Apply Tanh transform to a sample."""
        return self._postprocessor.forward(pre_tanh_sample)
    
    def inverse_post_process(self, post_tanh_sample):
        """Apply inverse Tanh transform."""
        return self._postprocessor.inverse(post_tanh_sample)
    
    def mode(self, loc, scale):
        """Returns the mode of the postprocessed distribution."""
        dist = self.create_dist(loc, scale)
        pre_tanh_mode = dist.mean  # Mode of a normal distribution is its mean
        return self._postprocessor.forward(pre_tanh_mode)

    def log_prob(self, loc, scale, actions):
        """Compute the log probability of actions."""
        # Calculate log prob under base Normal distribution
        dist = self.create_dist(loc, scale)
        log_probs = dist.log_prob(actions)
        # Adjust log prob for the Tanh transform using Jacobian
        log_probs -= self._postprocessor.forward_log_det_jacobian(actions)
        if self._event_ndims == 1:
            log_probs = log_probs.sum(dim=-1)  # Sum over action dimension
        return log_probs

    def entropy(self, loc, scale):
        """Return the entropy of the given distribution."""
        dist = self.create_dist(loc, scale)
        entropy = dist.entropy()
        # Entropy of transformed distribution involves expected log det Jacobian
        sample = dist.rsample()
        entropy += self._postprocessor.forward_log_det_jacobian(sample)
        if self._event_ndims == 1:
            entropy = entropy.sum(dim=-1)
        return entropy