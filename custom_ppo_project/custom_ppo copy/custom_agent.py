from rl_games.common import a2c_common  # Base class for Actor-Critic algorithms
from rl_games.algos_torch import torch_ext  # PyTorch extensions and utilities from rl_games

from rl_games.algos_torch import central_value  # Module for Central Value function (used in asymmetric Actor-Critic)
from rl_games.common import common_losses  # Common loss functions (like critic loss)
from rl_games.common import datasets  # Dataset classes for PPO

from torch import optim  # PyTorch optimizer module
import torch  # Main PyTorch library


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
        
        # Initialize states to None (used for RNNs).
        self.states = None
        
        # Initialize RNN states from the model if it has any.
        self.init_rnn_from_model(self.model)
        
        # Ensure last_lr is a float.
        self.last_lr = float(self.last_lr)
        
        # Get the type of bound loss to use from the config ('regularisation' or 'bound').
        # Default is 'bound'.
        self.bound_loss_type = self.config.get('bound_loss_type', 'bound') 
        
        # Initialize the Adam optimizer with the model parameters and learning rate.
        # eps=1e-08 is a standard value for numerical stability.
        # weight_decay is used for L2 regularization.
        self.optimizer = optim.Adam(self.model.parameters(), float(self.last_lr), eps=1e-08, weight_decay=self.weight_decay)                                                                                                                
        
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
        if self.is_rnn:
            rnn_masks = input_dict['rnn_masks']  # Masks for valid steps in sequences
            batch_dict['rnn_states'] = input_dict['rnn_states']  # Initial RNN states for the batch
            batch_dict['seq_length'] = self.seq_length  # Sequence length

            # If configured, pass done flags to reset RNN states.
            if self.zero_rnn_on_done:
                batch_dict['dones'] = input_dict['dones']            

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
            losses, sum_mask = torch_ext.apply_masks([a_loss.unsqueeze(1), c_loss , entropy.unsqueeze(1), b_loss.unsqueeze(1)], rnn_masks)
            a_loss, c_loss, entropy, b_loss = losses[0], losses[1], losses[2], losses[3]

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
            reduce_kl = rnn_masks is None
            kl_dist = torch_ext.policy_kl(mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, reduce_kl)
            if rnn_masks is not None:
                kl_dist = (kl_dist * rnn_masks).sum() / rnn_masks.numel()  #/ sum_mask

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
