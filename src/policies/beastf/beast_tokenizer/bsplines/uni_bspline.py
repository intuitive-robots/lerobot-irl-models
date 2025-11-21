
from typing import Union, Optional
import logging

import numpy as np
import torch

from .basis_gn import UniBSplineBasis

class UniformBSpline(torch.nn.Module):

    def __init__(self,
                 basis_gn: UniBSplineBasis,
                 num_dof: int,
                 weights_scale: float = 1.,
                 dtype: torch.dtype = torch.float32,
                 device: torch.device = 'cpu',
                 **kwargs,
                 ):
        super().__init__()

        # self.dtype = dtype
        # self.device = device
        # batch dim
        self.add_dim = list()

        self.basis_gn = basis_gn
        self.num_dof = num_dof

        # Scaling of weights
        weights_scale = \
            torch.tensor(weights_scale, dtype=self.dtype, device=self.device)
        assert weights_scale.ndim <= 1, \
            "weights_scale should be float or 1-dim vector"
        self.register_buffer("weights_scale", weights_scale, persistent=False)

        # Value caches
        # Compute values at these time points
        self.times = None

        # Learnable parameters
        self.params = None

        # Initial conditions
        self.init_pos = None
        self.init_vel = None

        # Runtime computation results, shall be reset every time when
        # inputs are reset
        self.pos = None
        self.vel = None


        #parameters bound
        # params_bound = kwargs.get("params_bound", None)
        # if not params_bound:
        #     params_bound = torch.zeros([2, self.num_params],
        #                                     dtype=self.dtype,
        #                                     device=self.device)
        #     params_bound[0, :] = -torch.inf
        #     params_bound[1, :] = torch.inf
        # else:
        #     params_bound = torch.as_tensor(self.params_bound,
        #                                         dtype=self.dtype,
        #                                         device=self.device)
        # assert list(params_bound.shape) == [2, self.num_params]
        # self.register_buffer("params_bound", params_bound, persistent=False)


        self.end_pos = None
        self.end_vel = None

        self.params_init = None
        self.params_end = None

    @property
    def device(self):
        return self.basis_gn.device

    @property
    def dtype(self):
        return self.basis_gn.dtype

    @property
    def tau(self):
        return self.basis_gn.tau

    @property
    def num_basis(self):
        return self.basis_gn.num_basis

    @property
    def num_params(self):
        return self.basis_gn.num_basis * self.num_dof

    def clear_computation_result(self):
        """
        Clear runtime computation result

        Returns:
            None
        """

        self.pos = None
        self.vel = None
        # also reset tau?

    def set_add_dim(self, add_dim: Union[list, torch.Size]):
        """
        Set additional batch dimension
        Args:
            add_dim: additional batch dimension

        Returns: None

        """
        self.add_dim = add_dim
        self.clear_computation_result()

    def set_times(self, times: Union[torch.Tensor, np.ndarray]):
        """
        Set time points
        Args:
            times: time points

        Returns:
            None
        """

        # Shape of times
        # [*add_dim, num_times]

        self.times = torch.as_tensor(times, dtype=self.dtype,
                                     device=self.device)
        tau = times.reshape(-1)[-1]
        self.basis_gn.tau.copy_(tau)
        self.clear_computation_result()

    def set_duration(self, duration: Optional[float], dt: float,):
        """

        Args:
            duration: desired duration of trajectory
            dt: control frequency
        Returns:
            None
        """

        # Shape of times
        # [*add_dim, num_times]
        dt = torch.as_tensor(dt, dtype=self.dtype, device=self.device)
        times = torch.linspace(0, duration, round(duration / dt) + 1,
                               dtype=self.dtype, device=self.device)
        times = add_expand_dim(times, list(range(len(self.add_dim))),
                                    self.add_dim)
        self.set_times(times)

    def set_params(self,
                   params: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        Set MP params
        Args:
            params: parameters

        Returns: unused parameters

        """
        # Shape of params
        # [*add_dim, num_params]

        params = torch.as_tensor(params, dtype=self.dtype, device=self.device)

        # Check number of params
        assert params.shape[-1] == self.num_params

        # Set additional batch size
        self.set_add_dim(list(params.shape[:-1]))

        self.params = params[..., :self.num_params]
        self.clear_computation_result()
        return params[..., self.num_params:]

    def update_inputs(self, times=None, params=None,
                      init_pos=None, init_vel=None, **kwargs):

        if params is not None:
            self.set_params(params)
        if times is not None:
            self.set_times(times)
        if init_pos is not None:
            self.set_initial_conditions(init_pos, init_vel, **kwargs)

        end_pos = kwargs.get('end_pos', None)
        end_vel = kwargs.get('end_vel', None)
        if any([cond is not None for cond in [end_pos, end_vel]]):
            self.set_end_condtions(end_pos, end_vel)

    def set_initial_conditions(self,
                               init_pos: Union[torch.Tensor, np.ndarray],
                               init_vel: Union[torch.Tensor, np.ndarray],
                               **kwargs):

        self.init_pos = torch.as_tensor(init_pos, dtype=self.dtype,
                                        device=self.device)
        self.init_vel = torch.as_tensor(init_vel, dtype=self.dtype,
                                   device=self.device) if init_vel is not None else None
        self.clear_computation_result()

        self.params_init = self.basis_gn.compute_init_params(self.init_pos, self.init_vel)
        if self.params_init is not None:
            self.params_init /= self.weights_scale

    def set_end_condtions(self, end_pos: Union[torch.Tensor, np.ndarray],
                          end_vel: Union[torch.Tensor, np.ndarray], **kwargs):
        self.end_pos = \
            torch.as_tensor(end_pos, device=self.device, dtype=self.dtype) \
                if end_pos is not None else None
        self.end_vel = \
            torch.as_tensor(end_vel, device=self.device, dtype=self.dtype) \
                if end_vel is not None else None

        self.params_end = self.basis_gn.compute_end_params(self.end_pos, self.end_vel)
        if self.params_end is not None:
            self.params_end /= self.weights_scale

    def get_traj_pos(self, times=None, params=None,
                     init_pos=None, init_vel=None, flat_shape=False, **kwargs):

        self.update_inputs(times, params, init_pos, init_vel, **kwargs)

        if self.pos is not None:
            pos = self.pos
        else:
            assert self.params is not None

            # Reshape params
            # [*add_dim, num_dof * num_basis] -> [*add_dim, num_dof, num_basis]
            params = self.params.reshape(*self.add_dim, self.num_dof, -1)
            # extend params with possible init and end conditions
            # shape: [*add_dim, num_dof, num_ctrlp]
            if self.params_init is not None:
                params = torch.cat((self.params_init, params), dim=-1)
            if self.params_end is not None:
                params = torch.cat((params, self.params_end), dim=-1)

            # Get basis
            # Shape: [*add_dim, num_times, num_ctrlp]
            basis_single_dof = \
                self.basis_gn.basis(self.times) * self.weights_scale

            # Einsum shape: [*add_dim, num_times, num_ctrlp],
            #               [*add_dim, num_dof, num_ctrlp]
            #            -> [*add_dim, num_times, num_dof]
            pos = torch.einsum('...ik,...jk->...ij', basis_single_dof, params)

            self.pos = pos

        if flat_shape:
            # Switch axes to [*add_dim, num_dof, num_times]
            pos = torch.einsum('...ji->...ij', pos)

            # Reshape to [*add_dim, num_dof * num_times]
            pos = pos.reshape(*self.add_dim, -1)

        return pos

    def get_traj_vel(self, times=None, params=None,
                     init_pos=None, init_vel=None, flat_shape=False, **kwargs):

        self.update_inputs(times, params, init_pos, init_vel,
                           **kwargs)

        if self.vel is not None:
            vel = self.vel
        else:
            assert self.params is not None

            # Reshape params
            # [*add_dim, num_dof * num_basis] -> [*add_dim, num_dof, num_basis]
            params = self.params.reshape(*self.add_dim, self.num_dof, -1)
            # extend params with possible init and end conditions
            # shape: [*add_dim, num_dof, num_ctrlp]
            if self.params_init is not None:
                params = torch.cat((self.params_init, params), dim=-1)
            if self.params_end is not None:
                params = torch.cat((params, self.params_end), dim=-1)

            # velocity control points shape: [*add_dim, num_dof, num_ctrlp-1]
            vel_ctrlp = self.basis_gn.velocity_control_points(params)
            vel_ctrlp = torch.einsum("...ij,...->...ij", vel_ctrlp,
                                     1 / self.tau)

            # vel_basis shape: [*add_dim, num_times, num_ctrlp-1]
            vel_basis = self.basis_gn.vel_basis(self.times) * self.weights_scale

            # Einsum shape: [*add_dim, num_times, num_ctrlp-1],
            #               [*add_dim, num_dof, num_ctrlp-1]
            #            -> [*add_dim, num_times, num_dof]
            vel = torch.einsum('...ik,...jk->...ij', vel_basis, vel_ctrlp)

            self.vel = vel

        if flat_shape:
            # Switch axes to [*add_dim, num_dof, num_times]
            vel = torch.einsum('...ji->...ij', vel)

            # Reshape to [*add_dim, num_dof * num_times]
            vel = vel.reshape(*self.add_dim, -1)

        return vel

    def learn_mp_params_from_trajs(self, times: torch.Tensor,
                                   trajs: torch.Tensor, reg=1e-5, **kwargs):

        # only works for learn_tau=False, learn_delay=False. And delay=0 (or you
        # need to give the initial condition by yourself)

        # Shape of times
        # [*add_dim, num_times]
        #
        # Shape of trajs:
        # [*add_dim, num_times, num_dof]
        #
        # Shape of params:
        # [*add_dim, num_dof * num_basis]

        assert trajs.shape[:-1] == times.shape
        assert trajs.shape[-1] == self.num_dof

        times = torch.as_tensor(times, dtype=self.dtype, device=self.device)
        trajs = torch.as_tensor(trajs, dtype=self.dtype, device=self.device)

        # Setup stuff
        self.set_add_dim(list(trajs.shape[:-2]))
        self.set_times(times)
        dummy_params = torch.zeros(*self.add_dim, self.num_dof, self.num_basis,
                                   device=self.device, dtype=self.dtype)

        # Get initial conditions
        if self.basis_gn.init_cond_order != 0:
            if any([key in kwargs.keys()
                    for key in [ "init_pos", "init_vel"]]):
                logging.warning("uses the given initial conditions")
                init_pos = kwargs.get("init_pos")
                init_vel = kwargs.get("init_vel")
            else:
                init_pos = trajs[..., 0, :]
                dt = (times[..., 1] - times[..., 0])
                init_vel = torch.einsum("...i,...->...i",
                                        torch.diff(trajs, dim=-2)[..., 0, :],
                                        1/dt)
            self.set_initial_conditions(init_pos, init_vel)
            if self.params_init is not None:
                dummy_params = torch.cat([self.params_init, dummy_params],
                                         dim=-1)

        if self.basis_gn.end_cond_order != 0:
            if any([key in kwargs.keys()
                    for key in ["end_pos", "end_vel"]]):
                logging.warning("uses the given end conditions")
                end_pos = kwargs.get("end_pos")
                end_vel = kwargs.get("end_vel")
            else:
                end_pos = trajs[..., -1, :]
                dt = (times[..., 1] - times[..., 0])
                end_vel = torch.einsum("...i,...->...i",
                                       torch.diff(trajs, dim=-2)[..., -1, :],
                                       1/dt)
            self.set_end_condtions(end_pos, end_vel)
            if self.params_end is not None:
                dummy_params = torch.cat([dummy_params, self.params_end],
                                         dim=-1)

        basis_single_dof = self.basis_gn.basis(times) * self.weights_scale
        # shape: [*add_dim, num_time, num_ctrlp]
        #        [*add_dim, num_dof, num_ctrlp]
        #        [*add_dim, num_times, num_dof]
        pos_det = torch.einsum('...ik,...jk->...ij', basis_single_dof, dummy_params)
        # swtich axes to [*add_dim, num_dof, num_times]
        pos_det = torch.einsum('...ij->...ji', pos_det)
        pos_det = pos_det.reshape(*self.add_dim, -1)

        basis_multi_dofs = self.basis_gn.basis_multi_dofs(self.times, self.num_dof) * self.weights_scale
        # Solve this: Aw = B -> w = A^{-1} B
        # Einsum_shape: [*add_dim, num_dof * num_times, num_dof * num_basis]
        #               [*add_dim, num_dof * num_times, num_dof * num_basis]
        #            -> [*add_dim, num_dof * num_basis, num_dof * num_basis]
        A = torch.einsum('...ki,...kj->...ij', basis_multi_dofs,
                         basis_multi_dofs)
        A += torch.eye(self.num_params,
                       dtype=self.dtype,
                       device=self.device) * reg

        # Swap axis and reshape: [*add_dim, num_times, num_dof]
        #                     -> [*add_dim, num_dof, num_times]
        trajs = torch.einsum("...ij->...ji", trajs)
        # Reshape [*add_dim, num_dof, num_times]
        #      -> [*add_dim, num_dof * num_times]
        trajs = trajs.reshape([*self.add_dim, -1])

        # Position minus initial condition terms,
        pos_w = trajs - pos_det

        # Einsum_shape: [*add_dim, num_dof * num_times, num_dof * num_basis]
        #               [*add_dim, num_dof * num_times]
        #            -> [*add_dim, num_dof * num_basis]
        B = torch.einsum('...ki,...k->...i', basis_multi_dofs, pos_w)

        # Shape of weights: [*add_dim, num_dof * num_basis]
        params = torch.linalg.solve(A, B)

        self.set_params(params)

        return {"params": params,
                "init_pos": self.init_pos,
                "init_vel": self.init_vel,
                "end_pos": self.end_pos,
                "end_vel": self.end_vel,
                }


def add_expand_dim(data: Union[torch.Tensor, np.ndarray],
                   add_dim_indices: [int],
                   add_dim_sizes: [int]) -> Union[torch.Tensor, np.ndarray]:
    """
    Add additional dimensions to tensor and expand accordingly
    Args:
        data: tensor to be operated. Torch.Tensor or numpy.ndarray
        add_dim_indices: the indices of added dimensions in the result tensor
        add_dim_sizes: the expanding size of the additional dimensions

    Returns:
        result: result tensor after adding and expanding
    """
    num_data_dim = data.ndim
    num_dim_to_add = len(add_dim_indices)

    add_dim_reverse_indices = [num_data_dim + num_dim_to_add + idx for idx in
                               add_dim_indices]

    str_add_dim = ""
    str_expand = ""
    add_dim_index = 0
    for dim in range(num_data_dim + num_dim_to_add):
        if dim in add_dim_indices or dim in add_dim_reverse_indices:
            str_add_dim += "None, "
            str_expand += str(add_dim_sizes[add_dim_index]) + ", "
            add_dim_index += 1
        else:
            str_add_dim += ":, "
            if type(data) == torch.Tensor:
                str_expand += "-1, "
            elif type(data) == np.ndarray:
                str_expand += "1, "
            else:
                raise NotImplementedError

    str_add_dime_eval = "data[" + str_add_dim + "]"
    if type(data) == torch.Tensor:
        return eval("eval(str_add_dime_eval).expand(" + str_expand + ")")
    else:
        return eval("np.tile(eval(str_add_dime_eval),[" + str_expand + "])")