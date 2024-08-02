from tianshou.utils.net.common import NetBase, MLP

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any, Generic, TypeAlias, TypeVar, cast, no_type_check

import numpy as np
import torch
from torch import nn

from tianshou.data.batch import Batch
from tianshou.data.types import RecurrentStateBatch

ModuleType = type[nn.Module]
ArgsType = tuple[Any, ...] | dict[Any, Any] | Sequence[tuple[Any, ...]] | Sequence[dict[Any, Any]]
TActionShape: TypeAlias = Sequence[int] | int | np.int64
TLinearLayer: TypeAlias = Callable[[int, int], nn.Module]
T = TypeVar("T")


def create_mask_weights(mask: np.ndarray) -> nn.Parameter:
    """Create the mask weights for the masked network.

    :param mask: np.ndarray of the same shape as state_shape, with 1 for the states being used, 0 for masked states.
    returns: np.ndarray of the same shape as state_shape, with 1 for the states being used, 0 for masked states.
    """
    n = np.count_nonzero(mask)  # Number of 1s
    m = len(mask)   # Number of 0s and 1s

    positions = np.where(mask == 1)[0]  # Indices of 1s
    
    # Initialize the result matrix with zeros
    result = torch.zeros((n , m), dtype=torch.float32)
    
    # Place 1s in the appropriate positions
    for i, pos in enumerate(positions):
        result[ i,pos] = 1
    
    # weights
    result = nn.Parameter(result, requires_grad=False)
    return result

class MaskedNet(NetBase[Any]):
    """Wrapper of MLP to support more specific DRL usage.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.

    :param state_shape: int or a sequence of int of the shape of state.
    :param mask: np.ndarray of the same shape as state_shape, with 1 for the states being used, 0 for masked states.
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
        mask: np.ndarray,
        action_shape: TActionShape = 0,
        hidden_sizes: Sequence[int] = (),
        norm_layer: ModuleType | Sequence[ModuleType] | None = None,
        norm_args: ArgsType | None = None,
        activation: ModuleType | Sequence[ModuleType] | None = nn.ReLU,
        act_args: ArgsType | None = None,
        device: str | int | torch.device = "cpu",
        softmax: bool = False,
        concat: bool = False,
        num_atoms: int = 1,
        dueling_param: tuple[dict[str, Any], dict[str, Any]] | None = None,
        linear_layer: TLinearLayer = nn.Linear,
    ) -> None:
        super().__init__()
        self.device = device
        self.softmax = softmax
        self.num_atoms = num_atoms
        self.Q: MLP | None = None
        self.V: MLP | None = None


        input_dim_mask = int(np.prod(state_shape))

        input_dim = int(np.sum(mask))

        self.mask = nn.Linear(input_dim_mask, input_dim, bias=False,)
        self.mask.weight = create_mask_weights(mask)

        action_dim = int(np.prod(action_shape)) * num_atoms
        if concat:
            input_dim += action_dim
        self.use_dueling = dueling_param is not None
        output_dim = action_dim if not self.use_dueling and not concat else 0
        self.model = MLP(
            input_dim,
            output_dim,
            hidden_sizes,
            norm_layer,
            norm_args,
            activation,
            act_args,
            device,
            linear_layer,
        )
        if self.use_dueling:  # dueling DQN
            assert dueling_param is not None
            kwargs_update = {
                "input_dim": self.model.output_dim,
                "device": self.device,
            }
            # Important: don't change the original dict (e.g., don't use .update())
            q_kwargs = {**dueling_param[0], **kwargs_update}
            v_kwargs = {**dueling_param[1], **kwargs_update}

            q_kwargs["output_dim"] = 0 if concat else action_dim
            v_kwargs["output_dim"] = 0 if concat else num_atoms
            self.Q, self.V = MLP(**q_kwargs), MLP(**v_kwargs)
            self.output_dim = self.Q.output_dim
        else:
            self.output_dim = self.model.output_dim

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
        # pass through mask
        obs = self.mask(obs)

        logits = self.model(obs)
        batch_size = logits.shape[0]
        if self.use_dueling:  # Dueling DQN
            assert self.Q is not None
            assert self.V is not None
            q, v = self.Q(logits), self.V(logits)
            if self.num_atoms > 1:
                q = q.view(batch_size, -1, self.num_atoms)
                v = v.view(batch_size, -1, self.num_atoms)
            logits = q - q.mean(dim=1, keepdim=True) + v
        elif self.num_atoms > 1:
            logits = logits.view(batch_size, -1, self.num_atoms)
        if self.softmax:
            logits = torch.softmax(logits, dim=-1)
        return logits, state

if __name__ == "__main__":
    mask = np.array([0,0,0,1,0,1])

    mask_weights = create_mask_weights(mask)
    print(mask_weights)
    linear = nn.Linear(6,2,bias=False)
    linear.weight = mask_weights

    # test forward pass of linear with random input
    input = torch.rand(1,6)
    output = linear(input)
    print(output)