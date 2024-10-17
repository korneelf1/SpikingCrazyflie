'''
This file does the same as surrogate_scheduling_analysis.py, but for the ANN model.
We create a custom sigmoid with an altered backward pass.'''
"""
This file creates an SMLP model and performs a one gradient update with it,
the gradient is tracked for different settings of surrogate gradient slope,
and the statistical properties of the gradients across layers are plotted.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from spikingActorProb import SMLP
import seaborn as sb
from tianshou.utils.net.common import MLP
from torch.nn.modules import Module
INPUT_SIZE = 63
OUTPUT_SIZE = 64
HIDDEN_SIZE = 64
N_LAYERS = 10
HIDDEN_LAYER_LST = [HIDDEN_SIZE]*(N_LAYERS - 1)

BIAS = False
N_SAMPLES = 256

import torch


class Sigmoid(torch.autograd.Function):
    """
    Surrogate gradient of the Heaviside step function.

    **Forward pass:** Heaviside step function shifted.

        .. math::

            S=\\begin{cases} 1 & \\text{if U ≥ U$_{\\rm thr}$} \\\\
            0 & \\text{if U < U$_{\\rm thr}$}
            \\end{cases}

    **Backward pass:** Gradient of sigmoid function.

        .. math::

                S&≈\\frac{1}{1 + {\\rm exp}(-kU)} \\\\
                \\frac{∂S}{∂U}&=\\frac{k
                {\\rm exp}(-kU)}{[{\\rm exp}(-kU)+1]^2}

    :math:`k` defaults to 25, and can be modified by calling \
        ``surrogate.sigmoid(slope=25)``.


    Adapted from:

    *F. Zenke, S. Ganguli (2018) SuperSpike: Supervised Learning
    in Multilayer Spiking
    Neural Networks. Neural Computation, pp. 1514-1541.*"""
    @staticmethod
    def forward(ctx, input_, slope):
        ctx.save_for_backward(input_)
        ctx.slope = slope
        out = torch.sigmoid(input_)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input_,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = (
            grad_input
            * ctx.slope
            * torch.exp(-ctx.slope * input_)
            / ((torch.exp(-ctx.slope * input_) + 1) ** 2)
        )
        return grad, None


def sigmoid_fn(slope=25):
    """Sigmoid surrogate gradient enclosed with a parameterized slope."""
    slope = 1/slope

    def inner(x):
        return Sigmoid.apply(x, slope)

    return inner

class sigmoid_2(Module):
    def __init__(self, slope=20):
        super(sigmoid_2, self).__init__()
        self.slope = slope

    def forward(self, x):
        return Sigmoid.apply(x, self.slope)

class sigmoid_7(Module):
    def __init__(self, slope=15):
        super(sigmoid_7, self).__init__()
        self.slope = slope

    def forward(self, x):
        return Sigmoid.apply(x, self.slope)
class sigmoid_12(Module):
    def __init__(self, slope=10):
        super(sigmoid_12, self).__init__()
        self.slope = slope

    def forward(self, x):
        return Sigmoid.apply(x, self.slope)
class sigmoid_17(Module):
    def __init__(self, slope=5):
        super(sigmoid_17, self).__init__()
        self.slope = slope

    def forward(self, x):
        return Sigmoid.apply(x, self.slope)
# class customSigmoid_2(customSigmoid):
#     def __init__(self, b=1/2):
#         super(customSigmoid_2, self).__init__(b)
# class customSigmoid_7(customSigmoid):
#     def __init__(self, b=1/7):
#         super(customSigmoid_7, self).__init__(b)
# class customSigmoid_12(customSigmoid):
#     def __init__(self, b=1/12):
#         super(customSigmoid_12, self).__init__(b)
# class customSigmoid_17(customSigmoid):
#     def __init__(self, b=1/17):
#         super(customSigmoid_17, self).__init__(b)
class NoBiasLinear(nn.Linear):
    def __init__(self, in_features, out_features, **kwargs):
        super(NoBiasLinear, self).__init__(in_features, out_features, bias=False, **kwargs)
class Wrapper(nn.Module):
    '''
    Wraps SMLP model with one nn.Linear output of size 4,2 to be able to use nn.MSELoss
    '''
    def __init__(self, model):
        super(Wrapper, self).__init__()
        self.model = model
        self.output = nn.Linear(HIDDEN_SIZE,OUTPUT_SIZE)
    def forward(self, x):
        x = self.model(x)
        output = x
        return self.output(output)
    
# create a simple model with 3 hidden layers of size 4 each
# create 5 models with slopes 2,7,12,17,true
# by having action size 0 and 3 hidden layers, we enforce an activation layer before the output!
model_2 = Wrapper(MLP(INPUT_SIZE,0, HIDDEN_LAYER_LST,activation=sigmoid_2,flatten_input=False,linear_layer=NoBiasLinear))
model_7 = Wrapper(MLP(INPUT_SIZE,0, HIDDEN_LAYER_LST, activation=sigmoid_7,flatten_input=False,linear_layer=NoBiasLinear))
model_12 = Wrapper(MLP(INPUT_SIZE,0, HIDDEN_LAYER_LST, activation=sigmoid_12,flatten_input=False,linear_layer=NoBiasLinear))
model_17 = Wrapper(MLP(INPUT_SIZE,0, HIDDEN_LAYER_LST, activation=sigmoid_17,flatten_input=False,linear_layer=NoBiasLinear))
model_true = Wrapper(MLP(INPUT_SIZE,0, HIDDEN_LAYER_LST,flatten_input=False,linear_layer=NoBiasLinear))
model_label_noise = Wrapper(MLP(INPUT_SIZE,0, HIDDEN_LAYER_LST,flatten_input=False,linear_layer=NoBiasLinear))
model_noisy = Wrapper(MLP(INPUT_SIZE,0, HIDDEN_LAYER_LST,flatten_input=False,linear_layer=NoBiasLinear))
print("MAKE SURE TO CHANGE DEFAULT ACTIVATION TO SIGMOID IN MLP")
print(model_2)
# make the weights and biases the same for all models
model_2_sate_dict = model_2.state_dict()
model_7.load_state_dict(model_2_sate_dict)
model_12.load_state_dict(model_2_sate_dict)
model_17.load_state_dict(model_2_sate_dict)
model_true.load_state_dict(model_2_sate_dict)
model_noisy.load_state_dict(model_2_sate_dict)

# create a simple optimizer
optimizer_2 = optim.Adam(model_2.parameters(), lr=1)
optimizer_7 = optim.Adam(model_7.parameters(), lr=1)
optimizer_12 = optim.Adam(model_12.parameters(), lr=1)
optimizer_17 = optim.Adam(model_17.parameters(), lr=1)
optimizer_true = optim.Adam(model_true.parameters(), lr=1)
optimizer_noisy = optim.Adam(model_noisy.parameters(), lr=1)
# create a loss function
loss_fn = nn.MSELoss()

# create a sac loss function which is a sum of the mean squared error and the entropy of the output
def sac_loss(output, target, entropy_weight=0.01):
    mse = loss_fn(output, target)
    entropy = -torch.mean(torch.sum(output*torch.log(output),dim=1))
    return mse - entropy_weight*entropy
# create a list to store the gradients
grads_2 = []
grads_7 = []
grads_12 = []
grads_17 = []
grads_noisy = []
grads_true = []

# create random inputs of INPUT_SIZE and labels of OUTPUT_SIZE
inputs = torch.randn(N_SAMPLES,INPUT_SIZE)
labels = torch.randn(N_SAMPLES,OUTPUT_SIZE)   

def add_param_noise(model, std=0.01):
    for param in model.parameters():
        param.data += torch.randn_like(param.data)*std


for i in range(N_SAMPLES):
    input = inputs[i]
    label = labels[i]
    # zero the gradients
    optimizer_2.zero_grad()
    optimizer_7.zero_grad()
    optimizer_12.zero_grad()
    optimizer_17.zero_grad()
    optimizer_noisy.zero_grad()
    optimizer_true.zero_grad()

    # add noise to noisy model parameters
    model_noisy.load_state_dict(model_2_sate_dict)
    add_param_noise(model_noisy, std=0.05)

    # forward pass
    output_2 = model_2(input)
    output_7 = model_7(input)
    output_12 = model_12(input)
    output_17 = model_17(input)
    output_noisy = model_noisy(input)
    output_true = model_true(input)

    # assure all outputs are same
    assert torch.all(torch.eq(output_2, output_7))
    assert torch.all(torch.eq(output_2, output_12))
    assert torch.all(torch.eq(output_2, output_17))
    assert torch.all(torch.eq(output_2, output_true))

    # calculate the loss
    loss_2 = loss_fn(output_2, label)
    loss_7 = loss_fn(output_7, label)
    loss_12 = loss_fn(output_12, label)
    loss_17 = loss_fn(output_17, label)
    loss_noisy = loss_fn(output_noisy, label)
    loss_true = loss_fn(output_true, label)

    # backward pass
    loss_2.backward()
    loss_7.backward()
    loss_12.backward()
    loss_17.backward()
    loss_noisy.backward()
    loss_true.backward()

    # store the gradients
    # for param in model_2.named_parameters():
    #     print(param)
    #     if 'weight' in param[0]:
    #         grads_2.append(param.grad)
    grads_2.append([param[1].grad for param in model_2.named_parameters() if 'weight' in param[0]])
    grads_7.append([param[1].grad for param in model_7.named_parameters() if 'weight' in param[0]])
    grads_12.append([param[1].grad for param in model_12.named_parameters() if 'weight' in param[0]])
    grads_17.append([param[1].grad for param in model_17.named_parameters() if 'weight' in param[0]])
    grads_noisy.append([param[1].grad for param in model_noisy.named_parameters() if 'weight' in param[0]])
    grads_true.append([param[1].grad for param in model_true.named_parameters() if 'weight' in param[0]])

grads_2_stacked = [torch.stack(layer) for layer in zip(*grads_2)]
grads_7_stacked = [torch.stack(layer) for layer in zip(*grads_7)]
grads_12_stacked = [torch.stack(layer) for layer in zip(*grads_12)]
grads_17_stacked = [torch.stack(layer) for layer in zip(*grads_17)]
grads_noisy_stacked = [torch.stack(layer) for layer in zip(*grads_noisy)]
grads_true_stacked = [torch.stack(layer) for layer in zip(*grads_true)]

all_grads = [grads_2_stacked, 
            grads_7_stacked, 
            grads_12_stacked, 
            grads_17_stacked, 
            grads_noisy_stacked,
            grads_true_stacked]


def gradient_heatmap():
    vmin = -0.01
    vmax = 0.01

    fig, axs = plt.subplots(nrows=len(all_grads), ncols=N_LAYERS)
    for i, grads in enumerate(all_grads):
        for j, grad in enumerate(grads):
            grad = grad.detach().numpy().sum(axis=0)/N_SAMPLES
            
            heatmap = sb.heatmap(grad, cmap='viridis', ax=axs[i,j], vmin=vmin, vmax=vmax)
            # make box plot of the gradients

            # set x-axis label
            heatmap.set_xlabel(f'layer {j} gradient')
            # draw this in subplot on ax[i], column j
        heatmap.set_ylabel(f'Model with slope {2+5*i}')

def gradient_boxplot():
    # create a boxplot of the gradients for each layer
    fig, axs = plt.subplots(nrows=1, ncols=N_LAYERS)
    for j in range(N_LAYERS):
        data = [grads_2_stacked[j].flatten().numpy(), 
                grads_7_stacked[j].flatten().numpy(), 
                grads_12_stacked[j].flatten().numpy(), 
                grads_17_stacked[j].flatten().numpy(), 
                grads_noisy_stacked[j].flatten().numpy(),
                grads_true_stacked[j].flatten().numpy()]
        axs[j].violinplot(data, showmeans=True, showextrema=True, showmedians=True)
        axs[j].set_title(f'Layer {j} Gradients')
        axs[j].set_xticklabels(['2','7','12','17','noise on Parameters','true'])
        # print the bias and variance of the gradients compared to ground truth (grads_true_stacked)
        print(f'Layer {j} bias: {np.mean(data, axis=1)-grads_true_stacked[j].flatten().numpy()}')
        print(f'Layer {j} variance: {np.var(data, axis=1)}')
        # make y axis of all subplots the same
        axs[j].set_ylim(-1,1)

def calculate_sign_reversal_accross_batch():
    # calculate sign reversal
    fig, axs = plt.subplots(nrows=len(all_grads), ncols=N_LAYERS)
    fig_boxplot, axs_boxplot = plt.subplots(nrows=1, ncols=N_LAYERS) # for boxplot

    for i, grads in enumerate(all_grads):
        for j, grad in enumerate(grads):
            grad = grad.detach().numpy()
            grad_high_slope = all_grads[-1][j].detach().numpy()

            signs = np.sign(grad-grad_high_slope)
            # accross the (batchsize, layer_size) dimension, calculate the probability of the sign of the gradient being the same as the high slope model
            # this is a measure of the similarity of the gradient to the high slope model
            # if the gradient is the same as the high slope model, the value is 1
            # if the gradient is the opposite of the high slope model, the value is -1
            # if the gradient is orthogonal to the high slope model, the value is 0
            # this is a measure of the similarity of the gradient to the high slope model
            signs[signs == 1] = 0
            signs[signs == -1] = 1 # if signed reversed 1

            data_accross_layer = signs.sum(axis=0)/N_SAMPLES # probability of sign reversal for each weight   
            # sum from Batch, layer size to batch size 
            # data_accross_batch = signs.sum(axis=1).sum(axis=2).flatten()/N_SAMPLES # probability of sign reversal for each weight
            # # add to boxplot
            # axs_boxplot[j].boxplot(data_accross_batch)
            # axs_boxplot[j].set_title(f'Layer {j} Sign Reversal Probability')

            heatmap = sb.heatmap(data_accross_layer, cmap='viridis', ax=axs[i,j], vmin=0, vmax=1)
            # make box plot of the gradients

            # set x-axis label
            heatmap.set_xlabel(f'layer {j} sign reversal probability')
            # draw this in subplot on ax[i], column j
        heatmap.set_ylabel(f'Model with slope {2+5*i}')

        # axs[i].imshow(grad, cmap='viridis')
        # axs[i].set_title(f'Layer {j+1} Gradients for model with slope {2+5*i}')
    # calculate the cosine similarity between the gradients of the high slope model and the other models

def calculate_sign_reversal_accross_layer():
    fig_boxplot, axs_boxplot = plt.subplots(nrows=1, ncols=N_LAYERS) # for boxplot
    data_models = []
    for i, grads in enumerate(all_grads):
        data = []
        for j, grad in enumerate(grads):
            grad = grad.detach().numpy()
            grad_high_slope = all_grads[-1][j].detach().numpy()

            signs = np.sign(grad-grad_high_slope)
            # accross the (batchsize, layer_size) dimension, calculate the probability of the sign of the gradient being the same as the high slope model
            # this is a measure of the similarity of the gradient to the high slope model
            # if the gradient is the same as the high slope model, the value is 1
            # if the gradient is the opposite of the high slope model, the value is -1
            # if the gradient is orthogonal to the high slope model, the value is 0
            # this is a measure of the similarity of the gradient to the high slope model
            signs[signs == 1] = 0
            signs[signs == -1] = 1 # if signed reversed 1
            # sum from Batch, layer size to batch size 
            N_ELS = grad.shape[-1]*grad.shape[-2]
            data_accross_batch = signs.reshape(N_SAMPLES,-1).sum(axis=-1)/(N_ELS) # probability of sign reversal for each weight
            data.append(data_accross_batch)
        data_models.append(data)
    for j in range(N_LAYERS):
        # gather data for layer j for each model
        data_accross_batch = [model[j] for model in data_models]
        # # add to boxplot
        axs_boxplot[j].violinplot(data_accross_batch,
                showmeans=False,
                showmedians=True)
        axs_boxplot[j].set_title(f'Layer {j} Sign Reversal Probability')

        plt.title("Sign Reversal Probability For Each Layer (1-N_LAYERS)")
 
def calculate_cosine_similarity():
    fig, axs = plt.subplots(nrows=1, ncols=3)
    data_models = []
    for i, grads in enumerate(all_grads):
        data = []
        for j, grad in enumerate(grads):
            grad = grad.detach().numpy().reshape(N_SAMPLES,-1)
            grad_high_slope = all_grads[-1][j].detach().numpy().reshape(N_SAMPLES,-1)
            # calculate the cosine similarity between the gradients of the high slope model and the other models
            cos_sim = np.sum(grad * grad_high_slope, axis=1)/(np.linalg.norm(grad,axis=1)*np.linalg.norm(grad_high_slope, axis=1))
            data.append(cos_sim)
        data_models.append(data)
    for j in range(3):
        # gather data for layer j for each model
        data_accross_batch = [model[j] for model in data_models]
        # # add to boxplot
        # color code eacht model
        # axs[j].violinplot(data_accross_batch,
        #         showmeans=False,
        #         showmedians=True)
        axs[j].violinplot(data_accross_batch,
                showmeans=False,
                showmedians=True)
        # axs[j].set_xticklabels(['2','7','10','20','noise on Parameters','true'])
        axs[j].set_title(f'Layer {j} Cosine Similarity')
        # share y axis
        axs[j].set_ylim(-1,1)
        # plt.title("Cosine Similarity For Each Layer (1-N_LAYERS)")


            # draw this in subplot on ax[i], column j

        # axs[i].imshow(grad, cmap='viridis')
        # axs[i].set_title(f'Layer {j+1} Gradients for model with slope {2+5*i}')
    # calculate the cosine similarity between the gradients of the high slope model and the other models

def calculate_l2_distance():
    fig, axs = plt.subplots(nrows=1, ncols=N_LAYERS)
    data_models = []
    for i, grads in enumerate(all_grads):
        data = []
        for j, grad in enumerate(grads):
            grad = grad.detach().numpy().reshape(N_SAMPLES,-1)
            grad_high_slope = all_grads[-1][j].detach().numpy().reshape(N_SAMPLES,-1)
            # calculate the cosine similarity between the gradients of the high slope model and the other models
            l2_dist = np.linalg.norm(grad - grad_high_slope, axis=1)
            data.append(l2_dist)
        data_models.append(data)
def calculate_gradient_distribution():
    '''Calculate the distribution of the gradients for each layer, for each model use a different colour when plotting'''
    fig, axs = plt.subplots(nrows=1, ncols=N_LAYERS)
    for j in range(N_LAYERS):
        data = [grads_2_stacked[j].flatten().numpy()-grads_true_stacked[j].flatten().numpy(), 
                grads_7_stacked[j].flatten().numpy()-grads_true_stacked[j].flatten().numpy(), 
                grads_12_stacked[j].flatten().numpy()-grads_true_stacked[j].flatten().numpy(), 
                grads_17_stacked[j].flatten().numpy()-grads_true_stacked[j].flatten().numpy(), 
                grads_noisy_stacked[j].flatten().numpy()-grads_true_stacked[j].flatten().numpy(),
                grads_true_stacked[j].flatten().numpy()-grads_true_stacked[j].flatten().numpy()]
        sb.kdeplot(data, ax=axs[j], common_norm=True)
        # draw vertical line at x = 0
        axs[j].axvline(x=0, color='black', linestyle='--')
# calculate_sign_reversal_accross_layer()
# gradient_boxplot()
calculate_cosine_similarity()
# calculate_gradient_distribution()

plt.show()