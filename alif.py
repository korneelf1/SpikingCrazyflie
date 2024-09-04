from snntorch._neurons import LIF
import torch
from torch import nn

import torch
import math
import numpy as np

import wandb

# Spike-gradient functions

# slope = 25
# """``snntorch.surrogate.slope``
# parameterizes the transition rate of the surrogate gradients."""

# class scheduled_sigmoid:
#     def __init__(self, slope_init=10):
#         self.slope =slope_init
#         self.counter = 0
#         self.step_size = 1
#         self.gamma = 1.1
#         self.slope_back_max = 50

#     def update_slope(self):
#         self.counter += 1
#         self.slope = self.slope
#         if self.counter % 1e4 == 0 and (self.slope < 50):
#             self.slope = min(self.slope * 1.1,50)

#         return self.slope

#     def __call__(self, x):
#         self.slope = self.update_slope()
#         def inner(x):
#             return ScheduledSigmoid.apply(x, self.slope)

#         return inner

class ScheduledSigmoid(torch.autograd.Function):
    """
    Surrogate gradient of the Heaviside step function.

    **Forward pass:** Sigmoid function.

        .. math::

            S = \\frac{1}{1 + e^{-kU}}*2

    **Backward pass:** Gradient of fast sigmoid function.

        .. math::

                S&≈\\frac{2*e^{-kU}}{1 + k|U|} \\\\
                \\frac{∂S}{∂U}&=\\frac{2*e^{-kU}}{(1+e^{-kU})^2}

    :math:`k` defaults to 25, and can be modified by calling \
        ``surrogate.fast_sigmoid(slope=25)``.
    """

    @staticmethod
    def forward(ctx, input_, slope):
        ctx.save_for_backward(input_)
        ctx.slope = slope
        fac = torch.exp(-ctx.slope * input_)
        if slope > 50:
            out = (input_ > 0).float()
        else:
            out = (2 / (1 + fac)).float()
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input_, = ctx.saved_tensors
        fac = torch.exp(-ctx.slope * input_)
        grad_input = 2 * fac / (1 + fac)**2 * grad_output
        print("max grad_input: ", grad_input.max())
        print("max grad_output: ", grad_output.min())
        return grad_input, None

class ScheduledSigmoidFunction:
    def __init__(self, slope_init=10, step_size=1, gamma=1.1, slope_back_max=50):
        self.slope = slope_init
        self.counter = 0
        self.step_size = step_size
        self.gamma = gamma
        self.slope_back_max = slope_back_max

    def update_slope(self):
        self.counter += 1
        if self.counter % int(1e4) == 0 and self.slope < self.slope_back_max:

            self.slope = min(self.slope * self.gamma, self.slope_back_max)
            if wandb.run is not None:
                wandb.log({"Sigmoid Slope": self.slope})
        return self.slope

    def __call__(self, x):
        current_slope = self.update_slope()
        return ScheduledSigmoid.apply(x, current_slope)

class CustomSigmoid(nn.Module):
    def __init__(self, b=.2):
        super(CustomSigmoid, self).__init__()
        self.b = b  # The parameter controlling the steepness of the sigmoid in the backward pass
        self._n_backwards = 0

    def forward(self, x):
        # Forward pass with the standard sigmoid (b=0)
        self.input = x  # Store input for use in backward pass
        self.sigmoid_output = torch.sigmoid(x)
        return self.sigmoid_output
    
    def backward(self, grad_output):
        # Backward pass using the derivative of the sigmoid with respect to bx (b non-zero)
        self._n_backwards += 1
        if self._n_backwards % 10e4 == 0:
                self.b = 1-1/max(5,min(self._n_backwards/30e4))
        sigmoid_b_output = torch.sigmoid(self.b * self.input)
        sigmoid_derivative = sigmoid_b_output * (1 - sigmoid_b_output)
        grad_input = grad_output * sigmoid_derivative
        return grad_input