import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any, Generic, TypeAlias, TypeVar, cast, no_type_check

import numpy as np
import torch
from torch import nn

import wandb
import snntorch as snn
from snntorch import surrogate
from alif import ScheduledSigmoidFunction as scheduled_sigmoid
from tianshou.utils.net.common import (
    MLP,
    BaseActor,
    Net,
    TActionShape,
    TLinearLayer,
    get_output_dim,
    NetBase,
    ModuleType,
    ArgsType
)

SIGMA_MIN = -20
SIGMA_MAX = 2

T = TypeVar("T")

TRecurrentState = TypeVar("TRecurrentState", bound=Any)

class IntegratorSpiker(torch.nn.Module):
    def __init__(self, layer_size = 64, integrator_ratio=0.5):
        n_integrators = int(layer_size * integrator_ratio)
        n_spikers = layer_size - n_integrators

        self.spike_grad = snn.surrogate.fast_sigmoid(10)

        self.betas = torch.nn.Parameter(torch.rand(n_spikers))
        self.thresholds = torch.nn.Parameter(torch.rand(n_spikers))
        self.lif1 = snn.Leaky(beta=self.betas, learn_beta=True,
                            threshold=self.thresholds, learn_threshold=True,
                            spikegrad=self.spike_grad)
        
        
        self.thresholds_integrators = torch.nn.Parameter(torch.rand(n_integrators))
        self.lif_integrators = snn.Leaky(beta=1, learn_beta=False,
                            threshold=self.thresholds_integrators, learn_threshold=True,
                            spikegrad=self.spike_grad)
        
        self.reset()
        
    def set_slope(self, slope):
        self.spike_grad = snn.surrogate.fast_sigmoid(slope)
        self.lif1.spikegrad = self.spike_grad
        self.lif_integrators.spikegrad = self.spike_grad

    def reset(self):
        self.cur_1 = self.lif1.init_leaky()
        self.cur_int = self.lif_integrators.init_leaky()

    def forward(self, x, hiddens):
        if hiddens is not None:
            self.cur_1 = hiddens[0]
            self.cur_int = hiddens[1]

        x_1, self.cur_1 = self.lif1(x, self.cur_1)

        x_int, self.cur_int = self.lif_integrators(x, self.cur_int)

        x = torch.cat((x_1, x_int), dim=1)

        return x, [self.cur_1, self.cur_int]


class SMLP(nn.Module):
    """
    A simple spiking multi-layer perceptron (MLP) network.

    :param input_dim
    :param output_dim
    :param hidden_sizes
    :param norm_layer
    :param norm_args
    :param activation
    :param act_args
    :param device
    :param linear_layer
    """
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_sizes: Sequence[int],
                 activation: ModuleType | Sequence[ModuleType] | None = snn.Leaky,
                 device: str | int | torch.device = "cpu",
                 ) -> None:
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_sizes = hidden_sizes

        # Initialize surrogate gradient
        self._slope = 10
        self._n_backwards = 0
        self.spike_grad1 = surrogate.fast_sigmoid(self._slope)  # passes default parameters from a closure
        # spike_grad1 = scheduled_sigmoid(10)

        # create layers and spiking layers
        self.layer_in = nn.Linear(input_dim, hidden_sizes[0], device=self.device)

        
        betas_in = torch.rand(hidden_sizes[0])
        thresh_in = torch.rand(hidden_sizes[0])
        self.lif_in   = snn.Leaky(beta=betas_in, learn_beta=True, 
                                  threshold=thresh_in, learn_threshold=True, 
                                  spike_grad=self.spike_grad1).to(self.device)

        # velocity and orientation prediction and injection layers
        self.vel_orient_layer = nn.Linear(hidden_sizes[0], 6, device=self.device)
        self.vel_orient_injection = torch.cat(torch.eye(6,requires_grad=False), torch.zeros(6,hidden_sizes[0]-6)).to(self.device)

        self.hidden_layers = []
        for i in range(len(hidden_sizes) - 1):
            self.hidden_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1], device=self.device))

            betas = torch.rand(hidden_sizes[i + 1])
            thresh = torch.rand(hidden_sizes[i + 1])
            self.hidden_layers.append(snn.Leaky(beta=betas, learn_beta=True,
                                                threshold=thresh, learn_threshold=True,
                                                spike_grad=self.spike_grad1).to(self.device))
            
        self.layer_out = nn.Linear(hidden_sizes[-1], output_dim, device=self.device)
        betas_out = torch.rand(output_dim)
        thresh_out = torch.rand(output_dim)
        self.lif_out = snn.Leaky(beta=betas_out, learn_beta=True,
                                    threshold=thresh_out, learn_threshold=True,
                                    spike_grad=self.spike_grad1).to(self.device)
        
        if wandb.run is None:
            print("wandb.run is None")
            self.run = wandb.init(project="spikingActorProb", reinit=True)
        else:
            self.run = wandb.run

        self.reset()
        self._update_slope()

    # def _register_backward_passes(self,module, grad_input, grad_output):
    #     self.backwards = []
    # def _register_nr_backward_hooks(self):
    #     """
    #     Registers the number of time each weight is updated.
    #     """
    #     for m in self.modules():
    #         if isinstance(m, nn.Linear):
    #             m.register_full_backward_hook(self._register_backward_passes)
    def update_slope(self, slope):
        self._slope = slope
        self.spike_grad1 = surrogate.fast_sigmoid(self._slope)
        self.lif_in.spike_grad = self.spike_grad1
        self.lif_out.spike_grad = self.spike_grad1
        for i in range(int(len(self.hidden_layers)/2)):
            self.hidden_layers[2*i+1].spike_grad = self.spike_grad1

    def reset(self):
        '''
        Reset the network's internal state
        '''
        self.cur_in = self.lif_in.init_leaky()
        self.cur_lst = [self.cur_in]
        for i in range(int(len(self.hidden_layers)/2)):
            self.cur_lst.append(self.hidden_layers[2*i+1].init_leaky())
        self.hidden_states = self.cur_lst
        self.cur_out = self.lif_out.init_leaky()

    def forward(self, x: torch.Tensor, hidden_states: list=None) -> torch.Tensor:
        '''
        Forward pass through the network
        '''
        if hidden_states is not None:
            self.cur_in = hidden_states[0]
            self.cur_lst = hidden_states[1:-1]
            self.cur_out = hidden_states[-1]

        x = self.layer_in(x)
        x, self.cur_in = self.lif_in(x, self.cur_in)
        # self.cur_in = x
        for i in range(int(len(self.hidden_layers)/2)):
            x = self.hidden_layers[2*i](x)
            if i == 0:
                vel_orient = self.vel_orient_layer(x)
                x = x + torch.matmul(vel_orient, self.vel_orient_injection)
            x, self.cur_lst[i] = self.hidden_layers[2*i+1](x, self.cur_lst[i])
            self.cur_lst[i] = x
        x = self.layer_out(x)
        x, self.cur_out = self.lif_out(x, self.cur_out)
        # self.cur_out = x
        self.hidden_states = [self.cur_in] + self.cur_lst + [self.cur_out]
        return x, self.hidden_states, vel_orient

    def __call__(self, *args: Any) -> Any:
        return self.forward(*args)
    


class SpikingNet(NetBase[Any]):
    """A spiking network for DRL usage.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.

    :param state_shape: int or a sequence of int of the shape of state.
    :param action_shape: int or a sequence of int of the shape of action.
    :param hidden_sizes: shape of MLP passed in as a list.
    :param norm_layer: use which normalization before activation, e.g.,
        ``nn.LayerNorm`` and ``nn.BatchNorm1d``. Default to no normalization.
        You can also pass a list of normalization modules with the same length
        of hidden_sizes, to use different normalization module in different
        layers. Default to no normalization.
    :param activation: which activation to use after each layer, can be both
        the same activation for all layers if passed in nn.Module, or different
        activation for different Modules if passed in a list. Default to
        nn.ReLU.
    :param device: specify the device when the network actually runs. Default
        to "cpu".
    :param softmax: whether to apply a softmax layer over the last layer's
        output.
    :param concat: whether the input shape is concatenated by state_shape
        and action_shape. If it is True, ``action_shape`` is not the output
        shape, but affects the input shape only.
    :param num_atoms: in order to expand to the net of distributional RL.
        Default to 1 (not use).
    :param dueling_param: whether to use dueling network to calculate Q
        values (for Dueling DQN). If you want to use dueling option, you should
        pass a tuple of two dict (first for Q and second for V) stating
        self-defined arguments as stated in
        class:`~tianshou.utils.net.common.MLP`. Default to None.
    :param linear_layer: use this module constructor, which takes the input
        and output dimension as input, as linear layer. Default to nn.Linear.
    :param reset_in_call: whether to reset the hidden states in the forward, useful for if running in realtime on sequences
    :param repeat: the number of times to repeat the network per given input. Default to 4.

    .. seealso::

        Please refer to :class:`~tianshou.utils.net.common.MLP` for more
        detailed explanation on the usage of activation, norm_layer, etc.

        You can also refer to :class:`~tianshou.utils.net.continuous.Actor`,
        :class:`~tianshou.utils.net.continuous.Critic`, etc, to see how it's
        suggested be used.
    """

    def __init__(
        self,
        state_shape: int | Sequence[int],
        action_shape: TActionShape = 0,
        hidden_sizes: Sequence[int] = (),
        norm_layer: ModuleType | Sequence[ModuleType] | None = None,
        norm_args: ArgsType | None = None,
        activation: ModuleType | Sequence[ModuleType] | None = snn.Leaky,
        act_args: ArgsType | None = None,
        device: str | int | torch.device = "cpu",
        softmax: bool = False,
        concat: bool = False,
        num_atoms: int = 1,
        dueling_param: tuple[dict[str, Any], dict[str, Any]] | None = None,
        linear_layer: TLinearLayer = nn.Linear,
        reset_in_call: bool = True,
        repeat: int = 4,
        scheduled_surrogate: bool = False,

    ) -> None:
        super().__init__()
        self.device = device
        self.softmax = softmax
        self.num_atoms = num_atoms

        input_dim = int(np.prod(state_shape))
        action_dim = int(np.prod(action_shape)) * num_atoms
        if action_dim == 0:
            raise UserWarning("Action Dimension set to 0.")
        if concat:
            input_dim += action_dim
        self.use_dueling = dueling_param is not None
        output_dim = action_dim if not self.use_dueling and not concat else 0
        
        self.output_dim = output_dim
        print("output_dim: ", output_dim)
        self.model = SMLP(
            input_dim,
            output_dim,
            hidden_sizes,
            activation,
            device,
        )

        self.repeat = repeat
        self.reset_in_call = reset_in_call
        # self.model = MLP(
        #     input_dim,
        #     output_dim,
        #     hidden_sizes,
        #     norm_layer,
        #     norm_args,
        #     activation,
        #     act_args,
        #     device,
        #     linear_layer,
        # )
        if self.use_dueling:  # dueling DQN
            raise NotImplementedError("Dueling DQN is not supported in spiking networks.")
        else:
            self.output_dim = self.model.output_dim

        
        self.model.reset()
        self._n_backwards = 0
        self._n_reset = 0
        self.scheduled = False
        if scheduled_surrogate:
            self._slope = 10
            self.scheduled = True
            # self._update_slope()
        

    def _update_slope(self):
        """
        Update the slope of the surrogate gradient.
        """
        def inner(module, grad_input, grad_output):
            self._n_backwards += 1
            
        self.model.layer_out.register_full_backward_hook(inner)

    def forward(
        self,
        obs: np.ndarray | torch.Tensor,
        state: Any = None,
        info: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, Any]:
        """Mapping: obs -> flatten (inside MLP)-> logits.

        :param obs:
        :param state: unused and returned as is
        :param info: unused
        """
        if len(obs.shape) == 1:
            obs = obs.unsqueeze(0)
        assert len(obs.shape) == 2 # (batch size, obs size) AKA not a sequence
        if self.reset_in_call:
            self.model.reset()
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, device=self.device, dtype=torch.float32)
        logits = torch.zeros(obs.shape[0], self.output_dim, device=self.device)

        hidden_state = state
        for _ in range(self.repeat):
            last_logits, hidden_state = self.model(obs, hidden_state)

            logits += last_logits
        # logits = torch.sum(logits, dim=1


        if self.softmax:
            logits = torch.softmax(logits, dim=-1)
        return logits, state

    def reset(self):
        if self.scheduled:
            self._n_reset += 1
            # print(self._n_reset)
            # schedule first 20 epochs nothing happens
            # after 20 epochs start making the surrogate steeper every 3*60e3 steps +1 to the slope until 30
            # n_reset is 10e3 per epoch
            steps_per_epoch = 10e3
            start_resets = steps_per_epoch*60    
            if self._n_reset > start_resets: # after 20 epochs start making the surrogate steeper
                
                update_interval = steps_per_epoch*10
                if self._n_reset % update_interval == 0 and self._slope<30: # each epoch is 20e4 steps -> every 2 epochs # every 100 steps is 400 backwards -> 5e3 steps is 20e3 backwards every 9 epochs would be 18e4 backwards
                    if wandb.run is not None:
                        # print('logging')
                        wandb.run.log({"surrogate fast sigmoid slope": self._slope})
                    # save self.model.state_dict()
                    # torch.save(self.model.state_dict(), "smlp.pth")
                    # print("saving model")
                    # for i in self.model.state_dict():
                    #     print(i)
                    #     # compute average and std of the weights
                    #     print(torch.mean(self.model.state_dict()[i]), torch.std(self.model.state_dict()[i]))

                    # self._slope +=1
                    self._slope = min(10+(self._n_reset - start_resets)/update_interval, 30)
                    # print("updating model, current slope: ", self._slope)
                    # create model with new slope
                    self.model.update_slope(self._slope)
                    # load the saved state dict
                    # self.model.load_state_dict(torch.load("smlp.pth"))
                    # self._update_slope()
                    print("updating slope: ", self.model._slope)
                    
        self.model.reset()


class POMDPDActorProb(BaseActor):
    """
    Spiking Actor for POMDPs.
    Utilizes an extra output for the velocity and orientation predictions.
    A loss function based on these outputs can be added to the RL algorithm.
    
    Used primarily in SAC, PPO and variants thereof. For deterministic policies, see :class:`~Actor`.

    :param preprocess_net: a self-defined preprocess_net, see usage.
        Typically, an instance of :class:`~tianshou.utils.net.common.Net`.
    :param action_shape: a sequence of int for the shape of action.
    :param hidden_sizes: a sequence of int for constructing the MLP after
        preprocess_net.
    :param max_action: the scale for the final action logits.
    :param unbounded: whether to apply tanh activation on final logits.
    :param conditioned_sigma: True when sigma is calculated from the
        input, False when sigma is an independent parameter.
    :param preprocess_net_output_dim: the output dimension of
        `preprocess_net`. Only used when `preprocess_net` does not have the attribute `output_dim`.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    # TODO: force kwargs, adjust downstream code
    def __init__(
        self,
        preprocess_net: nn.Module | Net,
        action_shape: TActionShape,
        hidden_sizes: Sequence[int] = (),
        max_action: float = 1.0,
        device: str | int | torch.device = "cpu",
        unbounded: bool = False,
        conditioned_sigma: bool = False,
        preprocess_net_output_dim: int | None = None,
        state_pred_size: int = 6,
    ) -> None:
        super().__init__()
        if unbounded and not np.isclose(max_action, 1.0):
            warnings.warn("Note that max_action input will be discarded when unbounded is True.")
            max_action = 1.0
        self.preprocess = preprocess_net
        self.device = device
        self.output_dim = int(np.prod(action_shape))
        input_dim = get_output_dim(preprocess_net, preprocess_net_output_dim)

        if len(hidden_sizes) >= 1:
            warnings.warn("Hidden sizes larger than one are now ANN rather than SNN.")

        self.mu = MLP(input_dim, self.output_dim, hidden_sizes, device=self.device)
        self._c_sigma = conditioned_sigma
        if conditioned_sigma:
            self.sigma = MLP(
                input_dim,
                self.output_dim,
                hidden_sizes,
                device=self.device,
            )
        else:
            warnings.warn("Fixed sigma is not tested for SNNs, could lead to bad performance.")
            self.sigma_param = nn.Parameter(torch.zeros(self.output_dim, 1))

        # also have head for the additional output
        self.vel_orient_head = MLP(input_dim, state_pred_size, hidden_sizes=hidden_sizes, device=self.device)

        
        self.max_action = max_action
        self._unbounded = unbounded

    def get_preprocess_net(self) -> nn.Module:
        return self.preprocess

    def get_output_dim(self) -> int:
        return self.output_dim

    def forward(
        self,
        obs: np.ndarray | torch.Tensor,
        # vel_orient: torch.Tensor,
        state: Any = None,
        info: dict[str, Any] | None = None,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], Any]:
        """Mapping: obs -> logits -> (mu, sigma)."""
        if info is None:
            info = {}
        
        # obs = torch.cat((obs, vel_orient), dim=1)
        logits, hidden = self.preprocess(obs, state)
        mu = self.mu(logits)
        vel_orient = self.vel_orient_head(logits)
        if not self._unbounded:
            mu = self.max_action * torch.tanh(mu)
        if self._c_sigma:
            sigma = torch.clamp(self.sigma(logits), min=SIGMA_MIN, max=SIGMA_MAX).exp()
        else:
            shape = [1] * len(mu.shape)
            shape[1] = -1
            sigma = (self.sigma_param.view(shape) + torch.zeros_like(mu)).exp()
        
        return (mu, sigma), state, vel_orient



class ActorProb(BaseActor):
    """Simple actor network that outputs `mu` and `sigma` to be used as input for a `dist_fn` (typically, a Gaussian).

    Used primarily in SAC, PPO and variants thereof. For deterministic policies, see :class:`~Actor`.

    :param preprocess_net: a self-defined preprocess_net, see usage.
        Typically, an instance of :class:`~tianshou.utils.net.common.Net`.
    :param action_shape: a sequence of int for the shape of action.
    :param hidden_sizes: a sequence of int for constructing the MLP after
        preprocess_net.
    :param max_action: the scale for the final action logits.
    :param unbounded: whether to apply tanh activation on final logits.
    :param conditioned_sigma: True when sigma is calculated from the
        input, False when sigma is an independent parameter.
    :param preprocess_net_output_dim: the output dimension of
        `preprocess_net`. Only used when `preprocess_net` does not have the attribute `output_dim`.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    # TODO: force kwargs, adjust downstream code
    def __init__(
        self,
        preprocess_net: nn.Module | Net,
        action_shape: TActionShape,
        hidden_sizes: Sequence[int] = (),
        max_action: float = 1.0,
        device: str | int | torch.device = "cpu",
        unbounded: bool = False,
        conditioned_sigma: bool = False,
        preprocess_net_output_dim: int | None = None,
    ) -> None:
        super().__init__()
        if unbounded and not np.isclose(max_action, 1.0):
            warnings.warn("Note that max_action input will be discarded when unbounded is True.")
            max_action = 1.0
        self.preprocess = preprocess_net
        self.device = device
        self.output_dim = int(np.prod(action_shape))
        input_dim = get_output_dim(preprocess_net, preprocess_net_output_dim)
        self.mu = MLP(input_dim, self.output_dim, hidden_sizes, device=self.device)
        self._c_sigma = conditioned_sigma
        if conditioned_sigma:
            self.sigma = MLP(
                input_dim,
                self.output_dim,
                hidden_sizes,
                device=self.device,
            )
        else:
            self.sigma_param = nn.Parameter(torch.zeros(self.output_dim, 1))
        self.max_action = max_action
        self._unbounded = unbounded

    def get_preprocess_net(self) -> nn.Module:
        return self.preprocess

    def get_output_dim(self) -> int:
        return self.output_dim

    def forward(
        self,
        obs: np.ndarray | torch.Tensor,
        state: Any = None,
        info: dict[str, Any] | None = None,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], Any]:
        """Mapping: obs -> logits -> (mu, sigma)."""
        if info is None:
            info = {}
        logits, hidden = self.preprocess(obs, state)
        mu = self.mu(logits)
        if not self._unbounded:
            mu = self.max_action * torch.tanh(mu)
        if self._c_sigma:
            sigma = torch.clamp(self.sigma(logits), min=SIGMA_MIN, max=SIGMA_MAX).exp()
        else:
            shape = [1] * len(mu.shape)
            shape[1] = -1
            sigma = (self.sigma_param.view(shape) + torch.zeros_like(mu)).exp()
        
        return (mu, sigma), state
