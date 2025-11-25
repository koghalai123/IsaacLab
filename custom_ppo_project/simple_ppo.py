import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch import optim

from rl_games.common import a2c_common
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch import central_value
from rl_games.common import common_losses
from rl_games.common import datasets
from rl_games.common import object_factory
from rl_games.algos_torch import layers
from rl_games.algos_torch.d2rl import D2RLNet
from rl_games.algos_torch.sac_helper import SquashedNormal
from rl_games.common.layers.value import TwoHotEncodedValue, DefaultValue
from rl_games.algos_torch.spatial_softmax import SpatialSoftArgmax
from rl_games.algos_torch.running_mean_std import RunningMeanStd, RunningMeanStdObs

# --------------------------------------------------------------------------------
# Network Builder
# --------------------------------------------------------------------------------

def _create_initializer(func, **kwargs):
    return lambda v : func(v, **kwargs)

class NetworkBuilder:
    """Base class for network builders.
    
    This class provides the infrastructure for creating neural networks, including
    factories for activations and initializers, and helper methods for building
    MLPs, CNNs, and RNNs.
    """
    def __init__(self, **kwargs):
        pass

    def load(self, params):
        """Load configuration parameters."""
        pass

    def build(self, name, **kwargs):
        """Build the network with the given name and arguments."""
        pass

    def __call__(self, name, **kwargs):
        return self.build(name, **kwargs)

    class BaseNetwork(nn.Module):
        """Base class for all networks created by the builder.
        
        It initializes factories for activation functions and weight initializers.
        """
        def __init__(self, **kwargs):
            nn.Module.__init__(self, **kwargs)

            # Factory for creating activation functions from string names
            self.activations_factory = object_factory.ObjectFactory()
            self.activations_factory.register_builder('relu', lambda **kwargs : nn.ReLU(**kwargs))
            self.activations_factory.register_builder('tanh', lambda **kwargs : nn.Tanh(**kwargs))
            self.activations_factory.register_builder('sigmoid', lambda **kwargs : nn.Sigmoid(**kwargs))
            self.activations_factory.register_builder('elu', lambda  **kwargs : nn.ELU(**kwargs))
            self.activations_factory.register_builder('selu', lambda **kwargs : nn.SELU(**kwargs))
            self.activations_factory.register_builder('swish', lambda **kwargs : nn.SiLU(**kwargs))
            self.activations_factory.register_builder('gelu', lambda **kwargs: nn.GELU(**kwargs))
            self.activations_factory.register_builder('softplus', lambda **kwargs : nn.Softplus(**kwargs))
            self.activations_factory.register_builder('None', lambda **kwargs : nn.Identity())

            # Factory for creating weight initializers
            self.init_factory = object_factory.ObjectFactory()
            #self.init_factory.register_builder('normc_initializer', lambda **kwargs : normc_initializer(**kwargs))
            self.init_factory.register_builder('const_initializer', lambda **kwargs : _create_initializer(nn.init.constant_,**kwargs))
            self.init_factory.register_builder('orthogonal_initializer', lambda **kwargs : _create_initializer(nn.init.orthogonal_,**kwargs))
            self.init_factory.register_builder('glorot_normal_initializer', lambda **kwargs : _create_initializer(nn.init.xavier_normal_,**kwargs))
            self.init_factory.register_builder('glorot_uniform_initializer', lambda **kwargs : _create_initializer(nn.init.xavier_uniform_,**kwargs))
            self.init_factory.register_builder('variance_scaling_initializer', lambda **kwargs : _create_initializer(torch_ext.variance_scaling_initializer,**kwargs))
            self.init_factory.register_builder('random_uniform_initializer', lambda **kwargs : _create_initializer(nn.init.uniform_,**kwargs))
            self.init_factory.register_builder('kaiming_normal', lambda **kwargs : _create_initializer(nn.init.kaiming_normal_,**kwargs))
            self.init_factory.register_builder('orthogonal', lambda **kwargs : _create_initializer(nn.init.orthogonal_,**kwargs))
            self.init_factory.register_builder('default', lambda **kwargs : nn.Identity() )

        def is_separate_critic(self):
            # Default: Actor and Critic share weights (or there is no critic)
            return False

        def get_value_layer(self):
            # Return the layer responsible for value estimation
            return self.value

        def is_rnn(self):
            # Default: Not a Recurrent Neural Network
            return False

        def get_default_rnn_state(self):
            # Default: No RNN state
            return None

        def get_aux_loss(self):
            # Default: No auxiliary loss
            return None

        def _calc_input_size(self, input_shape,cnn_layers=None):
            """Calculate the input size for the next layer (e.g., MLP) after a CNN.
            
            It passes a dummy input through the CNN layers to determine the flattened output size.
            """
            if cnn_layers is None:
                assert(len(input_shape) == 1)
                return input_shape[0]
            else:
                return nn.Sequential(*cnn_layers)(torch.rand(1, *(input_shape))).flatten(1).data.size(1)

        def _noisy_dense(self, inputs, units):
            # Create a noisy linear layer (for exploration)
            return layers.NoisyFactorizedLinear(inputs, units)

        def _build_sequential_mlp(self, 
        input_size, 
        units, 
        activation,
        dense_func,
        norm_only_first_layer=False, 
        norm_func_name = None):
            """Build a Multi-Layer Perceptron (MLP) sequentially.
            
            Args:
                input_size (int): Size of the input feature vector.
                units (list): List of integers defining the number of units in each hidden layer.
                activation (str): Name of the activation function to use.
                dense_func (class): The class to use for dense layers (e.g., nn.Linear).
                norm_only_first_layer (bool): If True, apply normalization only to the first layer.
                norm_func_name (str): Name of the normalization function ('layer_norm', 'batch_norm').
            
            Returns:
                nn.Sequential: The constructed MLP module.
            """
            print('build mlp:', input_size)
            in_size = input_size
            layers = []
            need_norm = True
            for unit in units:
                # Add a dense (linear) layer.
                layers.append(dense_func(in_size, unit))
                # Add activation function.
                layers.append(self.activations_factory.create(activation))

                # Add normalization if requested.
                if not need_norm:
                    continue
                if norm_only_first_layer and norm_func_name is not None:
                   need_norm = False 
                if norm_func_name == 'layer_norm':
                    layers.append(torch.nn.LayerNorm(unit))
                elif norm_func_name == 'batch_norm':
                    layers.append(torch.nn.BatchNorm1d(unit))
                in_size = unit

            return nn.Sequential(*layers)

        def _build_mlp(self, 
        input_size, 
        units, 
        activation,
        dense_func, 
        norm_only_first_layer=False,
        norm_func_name = None,
        d2rl=False):
            """Wrapper to build an MLP, optionally using D2RL architecture."""
            if d2rl:
                act_layers = [self.activations_factory.create(activation) for i in range(len(units))]
                return D2RLNet(input_size, units, act_layers, norm_func_name)
            else:
                return self._build_sequential_mlp(input_size, units, activation, dense_func, norm_func_name = None,)

        def _build_conv(self, ctype, **kwargs):
            """Build a Convolutional Neural Network (CNN) based on type."""
            print('conv_name:', ctype)

            if ctype == 'conv2d':
                return self._build_cnn2d(**kwargs)
            if ctype == 'conv2d_spatial_softargmax':
                return self._build_cnn2d(add_spatial_softmax=True, **kwargs)
            if ctype == 'conv2d_flatten':
                return self._build_cnn2d(add_flatten=True, **kwargs)
            if ctype == 'coord_conv2d':
                return self._build_cnn2d(conv_func=torch_ext.CoordConv2d, **kwargs)
            if ctype == 'conv1d':
                return self._build_cnn1d(**kwargs)

        def _build_cnn2d(self, input_shape, convs, activation, conv_func=torch.nn.Conv2d, norm_func_name=None,
                         add_spatial_softmax=False, add_flatten=False):
            """Build a 2D CNN.
            
            Args:
                input_shape (tuple): Shape of the input (Channels, Height, Width).
                convs (list): List of dictionaries defining each convolution layer (filters, kernel_size, strides, padding).
                activation (str): Activation function name.
                conv_func (class): Convolution class to use.
                norm_func_name (str): Normalization type.
                add_spatial_softmax (bool): Whether to add spatial softmax at the end.
                add_flatten (bool): Whether to flatten the output.
            """
            in_channels = input_shape[0]
            layers = []
            for conv in convs:
                # Add convolution layer.
                layers.append(conv_func(in_channels=in_channels, 
                out_channels=conv['filters'], 
                kernel_size=conv['kernel_size'], 
                stride=conv['strides'], padding=conv['padding']))
                conv_func=torch.nn.Conv2d
                
                # Add activation.
                act = self.activations_factory.create(activation)
                layers.append(act)
                in_channels = conv['filters']
                
                # Add normalization.
                if norm_func_name == 'layer_norm':
                    layers.append(torch_ext.LayerNorm2d(in_channels))
                elif norm_func_name == 'batch_norm':
                    layers.append(torch.nn.BatchNorm2d(in_channels))
            
            if add_spatial_softmax:
                layers.append(SpatialSoftArgmax(normalize=True))
            if add_flatten:
                layers.append(torch.nn.Flatten())
            return nn.Sequential(*layers)

        def _build_cnn1d(self, input_shape, convs, activation, norm_func_name=None):
            print('conv1d input shape:', input_shape)
            in_channels = input_shape[0]
            layers = []
            for conv in convs:
                layers.append(torch.nn.Conv1d(in_channels, conv['filters'], conv['kernel_size'], conv['strides'], conv['padding']))
                act = self.activations_factory.create(activation)
                layers.append(act)
                in_channels = conv['filters']
                if norm_func_name == 'layer_norm':
                    layers.append(torch.nn.LayerNorm(in_channels))
                elif norm_func_name == 'batch_norm':
                    layers.append(torch.nn.BatchNorm2d(in_channels))  
            return nn.Sequential(*layers)

        def _build_value_layer(self, input_size, output_size, value_type='legacy'):
            if value_type == 'legacy':
                return torch.nn.Linear(input_size, output_size)
            if value_type == 'default':
                return DefaultValue(input_size, output_size)            
            if value_type == 'twohot_encoded':
                return TwoHotEncodedValue(input_size, output_size)

            raise ValueError('value type is not "default", "legacy" or "two_hot_encoded"')


class CustomPPOBuilder(NetworkBuilder):
    """Builder for Custom PPO Networks.
    
    This class constructs the actor and critic networks based on the provided configuration.
    It supports MLPs, CNNs, and RNNs, as well as separate or shared actor-critic architectures.
    """
    def __init__(self, **kwargs):
        NetworkBuilder.__init__(self)

    def load(self, params):
        """Load configuration parameters."""
        self.params = params

    class Network(NetworkBuilder.BaseNetwork):
        """The actual PPO Network implementation."""
        def __init__(self, params, **kwargs):
            """Initialize the Custom PPO Network.
            
            Args:
                params (dict): Configuration parameters for the network.
                kwargs (dict): Additional arguments like input_shape, actions_num, etc.
            """
            actions_num = kwargs.pop('actions_num')
            input_shape = kwargs.pop('input_shape')
            self.value_size = kwargs.pop('value_size', 1)

            NetworkBuilder.BaseNetwork.__init__(self)
            self.load(params)
            
            # Initialize containers for network components.
            self.actor_cnn = nn.Sequential()
            self.critic_cnn = nn.Sequential()
            self.actor_mlp = nn.Sequential()
            self.critic_mlp = nn.Sequential()
            
            # Build CNN layers if configured.
            if self.has_cnn:
                if self.permute_input:
                    # Ensure input is in (Channels, Width, Height) format.
                    input_shape = torch_ext.shape_whc_to_cwh(input_shape)
                cnn_args = {
                    'ctype' : self.cnn['type'], 
                    'input_shape' : input_shape, 
                    'convs' :self.cnn['convs'], 
                    'activation' : self.cnn['activation'], 
                    'norm_func_name' : self.normalization,
                }
                self.actor_cnn = self._build_conv(**cnn_args)

                if self.separate:
                    # If separate actor/critic, build a separate CNN for the critic.
                    self.critic_cnn = self._build_conv( **cnn_args)

            # Calculate output size of CNN to determine input size for MLP/RNN.
            cnn_output_size = self._calc_input_size(input_shape, self.actor_cnn)

            mlp_input_size = cnn_output_size
            if len(self.units) == 0:
                out_size = cnn_output_size
            else:
                out_size = self.units[-1]

            # Build MLP layers.
            mlp_args = {
                'input_size' : mlp_input_size,
                'units' : self.units, 
                'activation' : self.activation, 
                'norm_func_name' : self.normalization,
                'dense_func' : torch.nn.Linear,
                'd2rl' : self.is_d2rl,
                'norm_only_first_layer' : self.norm_only_first_layer
            }
            self.actor_mlp = self._build_mlp(**mlp_args)
            print(f"Building Custom MLP with units: {self.units}")
            if self.separate:
                self.critic_mlp = self._build_mlp(**mlp_args)

            # Build Value Head (outputs value estimate).
            self.value = self._build_value_layer(out_size, self.value_size)
            self.value_act = self.activations_factory.create(self.value_activation)

            # Build Action Head (Logits/Mean/Sigma).
            if self.is_discrete:
                self.logits = torch.nn.Linear(out_size, actions_num)
            '''
                for multidiscrete actions num is a tuple
            '''
            if self.is_multi_discrete:
                self.logits = torch.nn.ModuleList([torch.nn.Linear(out_size, num) for num in actions_num])
            if self.is_continuous:
                # For continuous actions, output mean (mu) and standard deviation (sigma).
                self.mu = torch.nn.Linear(out_size, actions_num)
                self.mu_act = self.activations_factory.create(self.space_config['mu_activation']) 
                mu_init = self.init_factory.create(**self.space_config['mu_init'])
                self.sigma_act = self.activations_factory.create(self.space_config['sigma_activation']) 
                sigma_init = self.init_factory.create(**self.space_config['sigma_init'])

                if self.fixed_sigma:
                    # Learnable parameter for sigma, independent of input.
                    self.sigma = nn.Parameter(torch.zeros(actions_num, requires_grad=True, dtype=torch.float32), requires_grad=True)
                else:
                    # Sigma is a function of the input.
                    self.sigma = torch.nn.Linear(out_size, actions_num)

            # Initialize weights.
            mlp_init = self.init_factory.create(**self.initializer)
            if self.has_cnn:
                cnn_init = self.init_factory.create(**self.cnn['initializer'])

            for m in self.modules():         
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                    cnn_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)
                if isinstance(m, nn.Linear):
                    mlp_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)    

            if self.is_continuous:
                mu_init(self.mu.weight)
                if self.fixed_sigma:
                    sigma_init(self.sigma)
                else:
                    sigma_init(self.sigma.weight)  

        def forward(self, obs_dict):
            """Forward pass of the network builder.
            
            This method handles the data flow through the constructed network components (CNN, MLP).
            It handles separate vs shared architectures.
            """
            obs = obs_dict['obs']
            states = None

            if self.has_cnn:
                # Preprocess input for CNN if necessary.
                # for obs shape 4
                # input expected shape (B, W, H, C)
                # convert to (B, C, W, H)
                if self.permute_input and len(obs.shape) == 4:
                    obs = obs.permute((0, 3, 1, 2))

            if self.separate:
                # Separate Actor and Critic Networks
                # In this mode, actor and critic have completely separate parameters.
                a_out = c_out = obs
                a_out = self.actor_cnn(a_out)
                a_out = a_out.contiguous().view(a_out.size(0), -1)

                c_out = self.critic_cnn(c_out)
                c_out = c_out.contiguous().view(c_out.size(0), -1)                    

                # No RNN, just pass through MLP.
                a_out = self.actor_mlp(a_out)
                c_out = self.critic_mlp(c_out)
                            
                value = self.value_act(self.value(c_out))

                if self.is_discrete:
                    logits = self.logits(a_out)
                    return logits, value, states

                if self.is_multi_discrete:
                    logits = [logit(a_out) for logit in self.logits]
                    return logits, value, states

                if self.is_continuous:
                    mu = self.mu_act(self.mu(a_out))
                    if self.fixed_sigma:
                        sigma = mu * 0.0 + self.sigma_act(self.sigma)
                    else:
                        sigma = self.sigma_act(self.sigma(a_out))

                    return mu, sigma, value, states
            else:
                # Shared Actor-Critic Network
                # In this mode, actor and critic share the initial layers (CNN/MLP).
                out = obs
                out = self.actor_cnn(out)
                out = out.flatten(1)                

                out = self.actor_mlp(out)
                
                # Value head branches off from the shared representation.
                value = self.value_act(self.value(out))

                if self.central_value:
                    return value, states

                if self.is_discrete:
                    logits = self.logits(out)
                    return logits, value, states
                if self.is_multi_discrete:
                    logits = [logit(out) for logit in self.logits]
                    return logits, value, states
                if self.is_continuous:
                    mu = self.mu_act(self.mu(out))
                    if self.fixed_sigma:
                        sigma = self.sigma_act(self.sigma)
                    else:
                        sigma = self.sigma_act(self.sigma(out))
                    return mu, mu*0 + sigma, value, states
                    
        def is_separate_critic(self):
            return self.separate

        def is_rnn(self):
            return False

        def get_default_rnn_state(self):
            return None

        def load(self, params):
            self.separate = params.get('separate', False)
            self.units = params['mlp']['units']
            self.activation = params['mlp']['activation']
            self.initializer = params['mlp']['initializer']
            self.is_d2rl = params['mlp'].get('d2rl', False)
            self.norm_only_first_layer = params['mlp'].get('norm_only_first_layer', False)
            self.value_activation = params.get('value_activation', 'None')
            self.normalization = params.get('normalization', None)
            self.has_space = 'space' in params
            self.central_value = params.get('central_value', False)
            self.joint_obs_actions_config = params.get('joint_obs_actions', None)

            if self.has_space:
                self.is_multi_discrete = 'multi_discrete'in params['space']
                self.is_discrete = 'discrete' in params['space']
                self.is_continuous = 'continuous'in params['space']
                if self.is_continuous:
                    self.space_config = params['space']['continuous']
                    self.fixed_sigma = self.space_config['fixed_sigma']
                elif self.is_discrete:
                    self.space_config = params['space']['discrete']
                elif self.is_multi_discrete:
                    self.space_config = params['space']['multi_discrete']
            else:
                self.is_discrete = False
                self.is_continuous = False
                self.is_multi_discrete = False

            if 'cnn' in params:
                self.has_cnn = True
                self.cnn = params['cnn']
                self.permute_input = self.cnn.get('permute_input', True)
            else:
                self.has_cnn = False

    def build(self, name, **kwargs):
        net = CustomPPOBuilder.Network(self.params, **kwargs)
        return net

# --------------------------------------------------------------------------------
# Custom Models
# --------------------------------------------------------------------------------

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

# --------------------------------------------------------------------------------
# Custom PPO Agent
# --------------------------------------------------------------------------------

class CustomPPOAgent(a2c_common.ContinuousA2CBase):
    """Continuous PPO Agent

    The CustomPPOAgent class inherits from the continuous asymmetric actor-critic class and makes modifications for PPO.
    It handles the training loop, gradient updates, and loss calculations for the PPO algorithm.
    """
    def __init__(self, base_name, params):
        """Initialise the algorithm with passed params

        Args:
            base_name (:obj:`str`): Name passed on to the observer and used for checkpoints etc.
            params (:obj `dict`): Algorithm parameters

        """
        # Initialize the base class (ContinuousA2CBase) with the given name and parameters.
        # This sets up the basic infrastructure for the agent, such as loading config, setting up devices, etc.
        a2c_common.ContinuousA2CBase.__init__(self, base_name, params)
        
        # Print a message to indicate that the custom PPO agent is being used.
        print("----------------------------------------------------------------")
        print("USING CUSTOM PPO AGENT")
        print("----------------------------------------------------------------")
        
        # Get the shape of the observations from the environment info.
        obs_shape = self.obs_shape
        
        # Configuration dictionary for building the neural network model.
        # This dict contains all necessary information for the network builder to construct the architecture.
        build_config = {
            'actions_num' : self.actions_num,  # Number of actions the agent can take
            'input_shape' : obs_shape,  # Shape of the input observations
            'num_seqs' : self.num_actors * self.num_agents,  # Total number of sequences (actors * agents)
            'value_size': self.env_info.get('value_size',1),  # Dimension of the value output (default is 1)
            'normalize_value' : self.normalize_value,  # Whether to normalize the value function targets
            'normalize_input': self.normalize_input,  # Whether to normalize the input observations
        }
        
        # Build the network model using the configured builder (self.network is initialized in the base class).
        self.model = self.network.build(build_config)
        
        # Move the model to the specified device (CPU or GPU).
        self.model.to(self.ppo_device)

        self.is_rnn = self.model.is_rnn()
        
        # Initialize states to None (used for RNNs).
        self.states = None
        
        # Ensure last_lr is a float.
        self.last_lr = float(self.last_lr)
        
        # Get the type of bound loss to use from the config ('regularisation' or 'bound').
        # Default is 'bound'.
        self.bound_loss_type = self.config.get('bound_loss_type', 'bound') 
        
        # Initialize the Adam optimizer with the model parameters and learning rate.
        # eps=1e-08 is a standard value for numerical stability.
        # weight_decay is used for L2 regularization.
        self.optimizer = optim.Adam(self.model.parameters(), float(self.last_lr), eps=1e-08, weight_decay=self.weight_decay)

        # Ensure seq_length and zero_rnn_on_done are defined (defaults for non-RNN)
        if not hasattr(self, 'seq_length'):
            self.seq_length = 1
        if not hasattr(self, 'zero_rnn_on_done'):
            self.zero_rnn_on_done = False
        
        # Setup central value function if configured (for asymmetric actor-critic).
        # Asymmetric Actor-Critic uses a separate value network that has access to more information (e.g. full state) than the actor.
        if self.has_central_value:
            # Configuration for the central value network.
            cv_config = {
                'state_shape' : self.state_shape,  # Shape of the full state (for critic)
                'value_size' : self.value_size,  # Dimension of the value output
                'ppo_device' : self.ppo_device,  # Device to run on
                'num_agents' : self.num_agents,  # Number of agents
                'horizon_length' : self.horizon_length,  # Length of the rollout horizon
                'num_actors' : self.num_actors,  # Number of parallel actors
                'num_actions' : self.actions_num,  # Number of actions
                'seq_length' : self.seq_length,  # Sequence length for RNNs
                'normalize_value' : self.normalize_value,  # Value normalization
                'network' : self.central_value_config['network'],  # Network architecture config for CV
                'config' : self.central_value_config,  # Full CV config
                'writter' : self.writer,  # Tensorboard writer
                'max_epochs' : self.max_epochs,  # Max epochs for training CV
                'multi_gpu' : self.multi_gpu,  # Multi-GPU flag
                'zero_rnn_on_done' : self.zero_rnn_on_done  # Whether to zero RNN states on done
            }
            # Initialize the CentralValueTrain object and move it to the device.
            self.central_value_net = central_value.CentralValueTrain(**cv_config).to(self.ppo_device)

        # Check if experimental central value usage is enabled (default True).
        self.use_experimental_cv = self.config.get('use_experimental_cv', True)
        
        # Initialize the PPO dataset handler.
        # This manages the storage and retrieval of experience batches for training.
        self.dataset = datasets.PPODataset(self.batch_size, self.minibatch_size, self.is_discrete, self.is_rnn, self.ppo_device, self.seq_length)
        
        # If value normalization is enabled, link the value_mean_std object.
        # This ensures that the normalization statistics are shared or correctly accessed.
        if self.normalize_value:
            self.value_mean_std = self.central_value_net.model.value_mean_std if self.has_central_value else self.model.value_mean_std                                                                                                      
        
        # Determine if the main model has a value loss.
        # If using experimental CV or if there is NO central value, the main model computes value loss.
        self.has_value_loss = self.use_experimental_cv or not self.has_central_value
        
        # Call the after_init method of the observer to signal initialization completion.
        self.algo_observer.after_init(self)

    def update_epoch(self):
        """Update the epoch counter.
        
        Returns:
            int: The new epoch number.
        """
        self.epoch_num += 1
        return self.epoch_num
        
    def save(self, fn):
        """Save the agent's state to a checkpoint file.
        
        Args:
            fn (str): Filename/path to save the checkpoint.
        """
        # Get the full state dictionary including weights and optimizer state.
        state = self.get_full_state_weights()
        # Save the state to the specified file using torch_ext utility.
        torch_ext.save_checkpoint(fn, state)

    def restore(self, fn, set_epoch=True):
        """Restore the agent's state from a checkpoint file.
        
        Args:
            fn (str): Filename/path of the checkpoint to load.
            set_epoch (bool): Whether to restore the epoch number from the checkpoint.
        """
        # Load the checkpoint dictionary from the file.
        checkpoint = torch_ext.load_checkpoint(fn)
        # Set the agent's state (weights, optimizer, etc.) from the checkpoint.
        self.set_full_state_weights(checkpoint, set_epoch=set_epoch)

    def restore_central_value_function(self, fn):
        """Restore the central value function weights from a checkpoint.
        
        Args:
            fn (str): Filename/path of the checkpoint.
        """
        # Load the checkpoint.
        checkpoint = torch_ext.load_checkpoint(fn)
        # Restore only the central value function weights.
        self.set_central_value_function_weights(checkpoint)

    def get_masked_action_values(self, obs, action_masks):
        """Get action values with masking.
        
        This method is not implemented and will raise an assertion error if called.
        It's likely a placeholder or not supported in this continuous PPO implementation.
        """
        assert False

    def calc_gradients(self, input_dict):
        """Compute gradients needed to step the networks of the algorithm.

        Core algo logic is defined here. It calculates the PPO loss (actor, critic, entropy, bound)
        and performs backpropagation.

        Args:
            input_dict (:obj:`dict`): Algo inputs as a dict containing batch data.

        """
        # Unpack the input dictionary containing batch data.
        # These are the values collected during the rollout phase.
        value_preds_batch = input_dict['old_values']  # Value predictions from the old policy
        old_action_log_probs_batch = input_dict['old_logp_actions']  # Log probabilities of actions from the old policy
        advantage = input_dict['advantages']  # Calculated advantages (e.g., GAE)
        old_mu_batch = input_dict['mu']  # Mean of the action distribution from the old policy
        old_sigma_batch = input_dict['sigma']  # Standard deviation of the action distribution from the old policy
        return_batch = input_dict['returns']  # Target returns (rewards + discounted future value)
        actions_batch = input_dict['actions']  # Actions taken during rollout
        obs_batch = input_dict['obs']  # Observations during rollout
        
        # Preprocess observations (e.g., normalization, casting).
        obs_batch = self._preproc_obs(obs_batch)

        # Learning rate multiplier (can be used for scheduling, currently fixed at 1.0).
        lr_mul = 1.0
        # Current epsilon for PPO clipping.
        curr_e_clip = self.e_clip

        # Prepare the batch dictionary for the model forward pass.
        batch_dict = {
            'is_train': True,  # Flag to indicate training mode
            'prev_actions': actions_batch,  # Previous actions (input to the network)
            'obs' : obs_batch,  # Observations
        }

        # Handle RNN-specific inputs if the model is recurrent.
        rnn_masks = None

        # Use mixed precision training if enabled (for faster training on GPUs).
        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            # Forward pass through the model to get new predictions.
            # The model returns a dictionary containing values, log probs, entropy, etc.
            res_dict = self.model(batch_dict)
            action_log_probs = res_dict['prev_neglogp']  # New log probabilities of the taken actions
            values = res_dict['values']  # New value predictions
            entropy = res_dict['entropy']  # Entropy of the new action distribution
            mu = res_dict['mus']  # New mean of the action distribution
            sigma = res_dict['sigmas']  # New standard deviation of the action distribution

            # Calculate Actor Loss (PPO Clip objective).
            # The PPO loss encourages the new policy to stay close to the old policy.
            # It uses a clipped surrogate objective: min(ratio * advantage, clip(ratio, 1-eps, 1+eps) * advantage)
            # This prevents the policy from changing too drastically in a single update.
            a_loss = self.actor_loss_func(old_action_log_probs_batch, action_log_probs, advantage, self.ppo, curr_e_clip)                                                                                                                   
            
            # Calculate Critic Loss (Value function loss).
            # The value loss minimizes the error between predicted values and actual returns.
            # It also uses clipping to prevent the value function from changing too much.
            # value_pred_clipped = old_value + clamp(value - old_value, -eps, eps)
            # value_loss = max((value - return)^2, (value_pred_clipped - return)^2)
            if self.has_value_loss:
                c_loss = common_losses.critic_loss(self.model,value_preds_batch, values, curr_e_clip, return_batch, self.clip_value)
            else:
                # If value loss is handled elsewhere (e.g. central value), set it to 0 here.
                c_loss = torch.zeros(1, device=self.ppo_device)
            
            # Calculate Bound Loss (to keep actions within bounds).
            # This is an auxiliary loss to encourage the mean of the action distribution to stay within a certain range.
            # It helps in preventing the policy from outputting actions that are constantly clipped by the environment.
            if self.bound_loss_type == 'regularisation':
                b_loss = self.reg_loss(mu)
            elif self.bound_loss_type == 'bound':
                b_loss = self.bound_loss(mu)
            else:
                b_loss = torch.zeros(1, device=self.ppo_device)
            
            # Apply masks for RNNs if necessary.
            # When using RNNs, we need to mask out the loss for invalid steps (e.g. padding).
            # losses, sum_mask = torch_ext.apply_masks([a_loss.unsqueeze(1), c_loss , entropy.unsqueeze(1), b_loss.unsqueeze(1)], rnn_masks)
            # a_loss, c_loss, entropy, b_loss = losses[0], losses[1], losses[2], losses[3]
            
            # Since we removed RNN masking, we need to average the losses over the batch manually.
            a_loss = a_loss.mean()
            c_loss = c_loss.mean()
            entropy = entropy.mean()
            b_loss = b_loss.mean()

            # Total Loss = Actor Loss + Critic Loss - Entropy + Bound Loss + Aux Loss
            # We minimize the actor loss and critic loss, while maximizing entropy (to encourage exploration).
            loss = a_loss + 0.5 * c_loss * self.critic_coef - entropy * self.entropy_coef + b_loss * self.bounds_loss_coef
            
            # Add any auxiliary losses from the model (e.g., from specific layers or modules).
            aux_loss = self.model.get_aux_loss()
            self.aux_loss_dict = {}
            if aux_loss is not None:
                for k, v in aux_loss.items():
                    loss += v
                    if k in self.aux_loss_dict:
                        self.aux_loss_dict[k] = v.detach()
                    else:
                        self.aux_loss_dict[k] = [v.detach()]
            
            # Zero out gradients before backpropagation.
            if self.multi_gpu:
                self.optimizer.zero_grad()
            else:
                for param in self.model.parameters():
                    param.grad = None

        # Backpropagation and optimization step.
        # Use scaler for mixed precision training to prevent underflow.
        self.scaler.scale(loss).backward()
        
        # Clip gradients to prevent exploding gradients and perform the optimizer step.
        # (Note: The comment "ugliest code of the year" refers to the implementation details inside this method in the base class)
        self.trancate_gradients_and_step()

        # Calculate KL divergence for diagnostics.
        # This measures how much the policy has changed.
        with torch.no_grad():
            reduce_kl = True
            kl_dist = torch_ext.policy_kl(mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, reduce_kl)

        # Record diagnostics for Tensorboard/logging.
        self.diagnostics.mini_batch(self,
        {
            'values' : value_preds_batch,
            'returns' : return_batch,
            'new_neglogp' : action_log_probs,
            'old_neglogp' : old_action_log_probs_batch,
            'masks' : rnn_masks
        }, curr_e_clip, 0)      

        # Store the training results to be returned.
        self.train_result = (a_loss, c_loss, entropy, \
            kl_dist, self.last_lr, lr_mul, \
            mu.detach(), sigma.detach(), b_loss)

    def train_actor_critic(self, input_dict):
        """Perform a single training step for the actor-critic algorithm.
        
        Args:
            input_dict (dict): Dictionary containing the batch of data.
            
        Returns:
            tuple: Training results (losses, entropy, KL divergence, etc.)
        """
        # Calculate gradients and perform the optimization step.
        self.calc_gradients(input_dict)
        # Return the results stored in self.train_result.
        return self.train_result

    def reg_loss(self, mu):
        """Regularization loss to penalize large action means.
        
        This simple regularization encourages the action means to stay close to 0.
        
        Args:
            mu (torch.Tensor): Mean of the action distribution.
            
        Returns:
            torch.Tensor: Regularization loss.
        """
        if self.bounds_loss_coef is not None:
            # L2 regularization on the mean: sum(mu^2)
            reg_loss = (mu*mu).sum(axis=-1)
        else:
            reg_loss = 0
        return reg_loss

    def bound_loss(self, mu):
        """Bound loss to penalize action means outside a soft range.
        
        This loss penalizes the action means if they exceed a soft bound (1.1).
        This is useful when the action space is [-1, 1] and we want to prevent the policy
        from saturating the tanh activation (if used) or simply drifting too far.
        
        Args:
            mu (torch.Tensor): Mean of the action distribution.
            
        Returns:
            torch.Tensor: Bound loss.
        """
        if self.bounds_loss_coef is not None:
            # Define a soft bound slightly larger than 1.0.
            soft_bound = 1.1
            # Calculate loss for values greater than soft_bound.
            mu_loss_high = torch.clamp_min(mu - soft_bound, 0.0)**2
            # Calculate loss for values less than -soft_bound.
            mu_loss_low = torch.clamp_max(mu + soft_bound, 0.0)**2
            # Sum the losses.
            b_loss = (mu_loss_low + mu_loss_high).sum(axis=-1)
        else:
            b_loss = 0
        return b_loss
