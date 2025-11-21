from .bsplines.bspline_factory import SplineFactory
import torch
from addict import Dict
from .utils import continuous_to_discrete, discrete_to_continuous, normalize_tensor, denormalize_tensor, tensor_linspace
import numpy as np
import matplotlib.pyplot as plt
import einops


class BeastTokenizer(torch.nn.Module):
    """
    B-spline based tokenizer for trajectory encoding/decoding.
    
    Converts continuous trajectories to discrete tokens and vice versa using B-splines.
    Supports continuous and discrete representations of trajectories.
    Supports sperate handling for continous action and discrete state (e.g., binarized gripper state).
    """
    
    # Class constants
    DEFAULT_DT = 0.01  # 100 Hz sampling rate
    
    def __init__(self, num_dof=1, num_basis=10, seq_len=50, vocab_size=256,
                 degree_p=4, gripper_zero_order=False, gripper_dof=1, init_cond_order=0, 
                 end_cond_order=0, enforce_init_pos=True, device="cuda"):
        super().__init__()
        
        # Store core parameters
        self.device = device
        self.seq_length = seq_len
        self.vocab_size = vocab_size
        self.num_basis = num_basis
        self.enforce_init_pos = enforce_init_pos
        self.init_cond_order = init_cond_order
        self.end_cond_order = end_cond_order
        self.dt = self.DEFAULT_DT
        self.init_pos = None
        
        # Calculate DOF distribution
        self.gripper_dof = gripper_dof if gripper_zero_order else 0
        self.joint_dof = num_dof - self.gripper_dof
        self.num_dof = self.joint_dof + self.gripper_dof
        
        # Initialize spline components
        self.bsp = self._create_bsplines(self.joint_dof, degree_p)
        self.gripper_bsp = self._create_bsplines(self.gripper_dof, 0) if gripper_zero_order else None
        
        # Setup time grid and weight bounds
        # Working with normalized time [0, 1]
        self.times = tensor_linspace(0, 1.0, seq_len).to(device)
        self._initialize_weight_bounds()
        
        self.to(self.device)
    
    def _create_bsplines(self, num_dof, degree_p):
        """Create motion primitive for joint trajectories."""
        config = Dict({
            'mp_type': 'uni_bspline',
            'device': self.device,
            'num_dof': num_dof,
            'tau': 1.0,
            'mp_args': {
                'num_basis': self.num_basis,
                'degree_p': degree_p,
                'init_condition_order': self.init_cond_order,
                'end_condition_order': self.end_cond_order,
                'dt': self.dt
            }
        })
        return SplineFactory.init_splines(**config)
    
    def _initialize_weight_bounds(self):
        """Initialize weight bounds for normalization."""
        total_params = self.num_dof * self.num_basis
        self.register_buffer("w_min", -1.0 * torch.ones(total_params))
        self.register_buffer("w_max", 1.0 * torch.ones(total_params))
    
    def _get_repeated_times(self, batch_size):
        """Get time tensor repeated for batch processing."""
        return einops.repeat(self.times, 't -> b t', b=batch_size)
    
    def _learn_trajectory_params(self, times, trajs):
        """Learn B-spline parameters from trajectories."""
        # Learn joint parameters
        joint_params = self.bsp.learn_mp_params_from_trajs(times, trajs[..., :self.joint_dof])
        
        # Learn gripper parameters if applicable
        if self.gripper_bsp is not None:
            gripper_trajs = trajs[..., -self.gripper_dof:]
            gripper_params = self.gripper_bsp.learn_mp_params_from_trajs(times, gripper_trajs)
            joint_params['params'] = torch.cat([joint_params['params'], gripper_params['params']], dim=-1)
        
        return joint_params
    
    def _reconstruct_trajectory(self, params, times):
        """Reconstruct trajectory from B-spline parameters."""
        # Reconstruct joint trajectory
        joint_params = params[..., :self.joint_dof * self.num_basis]
        self.bsp.update_inputs(times=times, params=joint_params)
        position = self.bsp.get_traj_pos()
        
        # Reconstruct gripper trajectory if applicable
        if self.gripper_bsp is not None:
            gripper_params = params[..., -self.gripper_dof * self.num_basis:]
            self.gripper_bsp.update_inputs(times=times, params=gripper_params)
            gripper_pos = self.gripper_bsp.get_traj_pos()
            position = torch.cat([position, gripper_pos], dim=-1)
        
        return position
    
    def _apply_initial_position_constraint(self, params, init_pos):
        """Apply initial position constraint to parameters."""
        if not self.init_pos or init_pos is None:
            return params
            
        # Reshape to access individual basis functions
        reshaped_params = einops.rearrange(params, "b (d t) -> b t d", t=self.num_basis, d=self.num_dof)
        
        # Set initial position for joint DOFs
        reshaped_params[:, 0, :self.joint_dof] = init_pos[:, :self.joint_dof]
        
        return einops.rearrange(reshaped_params, "b t d -> b (d t)")
    
    def compute_weights(self, demos):
        """Compute B-spline weights from demonstration trajectories."""
        times = self._get_repeated_times(demos.shape[0])
        weights = self.bsp.learn_mp_params_from_trajs(times, demos)['params']
        return weights
    
    def update_weights_bounds_per_batch(self, weights):
        """Update weight bounds based on batch statistics."""
        weights = weights.reshape(-1, self.num_dof * self.num_basis)
        batch_min = weights.min(dim=0)[0]
        batch_max = weights.max(dim=0)[0]
        
        # Update bounds with small tolerance
        tolerance = 1e-4
        smaller_mask = batch_min < (self.w_min - tolerance)
        larger_mask = batch_max > (self.w_max + tolerance)
        
        if torch.any(smaller_mask):
            self.w_min[smaller_mask] = batch_min[smaller_mask]
        if torch.any(larger_mask):
            self.w_max[larger_mask] = batch_max[larger_mask]
    
    def update_times(self, times):
        """Update time grid."""
        self.times = times
    
    @torch.no_grad() 
    def encode_discrete(self, trajs, update_bounds=False, init_p=None):
        """Encode trajectories to discrete tokens."""
        times = self._get_repeated_times(trajs.shape[0])
        params_dict = self._learn_trajectory_params(times, trajs)
        
        if update_bounds:
            self.update_weights_bounds_per_batch(params_dict['params'])
        
        # Clamp parameters to bounds
        params = torch.clamp(params_dict['params'], min=self.w_min, max=self.w_max)
        
        # Convert to discrete tokens
        tokens = continuous_to_discrete(params, min_val=self.w_min, max_val=self.w_max, num_bins=self.vocab_size)
        tokens = einops.rearrange(tokens, 'b (d t) -> b (t d)', t=self.num_basis, d=self.num_dof)
        
        return tokens
    
    @torch.no_grad()
    def decode_discrete(self, tokens, times=None, init_pos=None):
        """Decode discrete tokens to trajectories."""
        # Reshape tokens and convert to continuous parameters
        tokens = einops.rearrange(tokens, 'b (t d) -> b (d t)', t=self.num_basis, d=self.num_dof)
        params = discrete_to_continuous(tokens, min_val=self.w_min, max_val=self.w_max, num_bins=self.vocab_size)
        
        if times is None:
            times = self._get_repeated_times(params.shape[0])
        
        # Apply initial position constraint if specified
        params = self._apply_initial_position_constraint(params, init_pos)
        
        return self._reconstruct_trajectory(params, times)
    
    @torch.no_grad() 
    def encode_continuous(self, trajs, update_bounds=False):
        """Encode trajectories to continuous tokens (normalized parameters)."""
        times = self._get_repeated_times(trajs.shape[0])
        params_dict = self._learn_trajectory_params(times, trajs)
        
        if update_bounds:
            self.update_weights_bounds_per_batch(params_dict['params'])
        
        # Normalize parameters
        tokens = normalize_tensor(params_dict['params'], w_min=self.w_min, w_max=self.w_max)
        
        return tokens
    
    @torch.no_grad()
    def decode_continuous(self, params, times=None, init_pos=None):
        """Decode continuous tokens (normalized parameters) to trajectories."""
        # Denormalize parameters
        params = denormalize_tensor(params, w_min=self.w_min, w_max=self.w_max)
        
        if times is None:
            times = self._get_repeated_times(params.shape[0])
        
        # Apply initial position constraint if specified
        params = self._apply_initial_position_constraint(params, init_pos)
        
        return self._reconstruct_trajectory(params, times)
    
    def compute_reconstruction_error(self, raw_traj):
        """Compute reconstruction error for trajectory."""
        if len(raw_traj.shape) == 2:
            raw_traj = raw_traj.unsqueeze(-1)
        
        tokens, _ = self.encode_discrete(raw_traj)
        reconstructed = self.decode_discrete(tokens)
        error = torch.mean((raw_traj - reconstructed) ** 2)
        
        return error
    
    def _plot_trajectory_comparison(self, original, reconstructed, title_prefix=""):
        """Helper method to plot trajectory comparison."""
        original = original.detach().cpu().numpy()
        reconstructed = reconstructed.detach().cpu().numpy()
        x_vals = np.linspace(0, 1.0, original.shape[1])
        
        batch_size, time_steps, dof = original.shape
        
        for sample_idx in range(batch_size):
            fig, axes = plt.subplots(dof, 1, figsize=(8, 2 * dof), sharex=True)
            if dof == 1:
                axes = [axes]  # Handle single DOF case
            
            for i in range(dof):
                axes[i].plot(x_vals, reconstructed[sample_idx, :, i], 
                           marker='o', label='Reconstructed', linestyle='-', color='b')
                axes[i].plot(x_vals, original[sample_idx, :, i], 
                           marker='*', label='Ground Truth', linestyle='--', color='r')
                axes[i].set_ylabel(f"DOF {i + 1}")
                axes[i].grid(True)
                axes[i].legend(loc="best")
            
            axes[-1].set_xlabel("Time (s)")
            plt.suptitle(f"{title_prefix}Trajectory Comparison - Sample {sample_idx}")
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()
    
    def visualize_reconstruction_error_discrete(self, raw_traj):
        """Visualize reconstruction error for discrete encoding."""
        tokens = self.encode_discrete(raw_traj, update_bounds=True)
        reconstructed = self.decode_discrete(tokens)
        self._plot_trajectory_comparison(raw_traj, reconstructed, "Discrete ")
    
    def visualize_reconstruction_error_continuous(self, raw_traj):
        """Visualize reconstruction error for continuous encoding."""
        raw_traj = raw_traj.to(torch.float32)
        if len(raw_traj.shape) == 2:
            raw_traj = raw_traj.unsqueeze(0)
        
        continuous_tokens = self.encode_continuous(raw_traj, update_bounds=True)
        reconstructed = self.decode_continuous(continuous_tokens)
        self._plot_trajectory_comparison(raw_traj, reconstructed, "Continuous ")