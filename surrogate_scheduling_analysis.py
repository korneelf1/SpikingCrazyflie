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

INPUT_SIZE = 4
OUTPUT_SIZE = 4
HIDDEN_SIZE = 64

N_SAMPLES = 100


class Wrapper(nn.Module):
    '''
    Wraps SMLP model with one nn.Linear output of size 4,2 to be able to use nn.MSELoss
    '''
    def __init__(self, model):
        super(Wrapper, self).__init__()
        self.model = model
        self.output = nn.Linear(HIDDEN_SIZE,OUTPUT_SIZE)
    def forward(self, x):
        inp = x
        x, hidden = self.model(inp, None)
        output = x
        for i in range(50):
            x, hidden = self.model(inp, hidden)
            output += x
            
        return self.output(output)
# create a simple model with 3 hidden layers of size 4 each
# create 5 models with slopes 2,7,12,17,22
model_2 = Wrapper(SMLP(INPUT_SIZE,HIDDEN_SIZE, [HIDDEN_SIZE, HIDDEN_SIZE], slope=2))
model_7 = Wrapper(SMLP(INPUT_SIZE,HIDDEN_SIZE, [HIDDEN_SIZE, HIDDEN_SIZE], slope=7))
model_12 = Wrapper(SMLP(INPUT_SIZE,HIDDEN_SIZE, [HIDDEN_SIZE, HIDDEN_SIZE], slope=12))
model_17 = Wrapper(SMLP(INPUT_SIZE,HIDDEN_SIZE, [HIDDEN_SIZE, HIDDEN_SIZE], slope=17))
model_22 = Wrapper(SMLP(INPUT_SIZE,HIDDEN_SIZE, [HIDDEN_SIZE, HIDDEN_SIZE], slope=22))

# make the weights and biases the same for all models
model_2_sate_dict = model_2.state_dict()
model_7.load_state_dict(model_2_sate_dict)
model_12.load_state_dict(model_2_sate_dict)
model_17.load_state_dict(model_2_sate_dict)
model_22.load_state_dict(model_2_sate_dict)

# create a simple optimizer
optimizer_2 = optim.Adam(model_2.parameters(), lr=1)
optimizer_7 = optim.Adam(model_7.parameters(), lr=1)
optimizer_12 = optim.Adam(model_12.parameters(), lr=1)
optimizer_17 = optim.Adam(model_17.parameters(), lr=1)
optimizer_22 = optim.Adam(model_22.parameters(), lr=1)

# create a loss function
loss_fn = nn.MSELoss()

# create a list to store the gradients
grads_2 = []
grads_7 = []
grads_12 = []
grads_17 = []
grads_22 = []

# create random inputs of INPUT_SIZE and labels of OUTPUT_SIZE
inputs = torch.randn(N_SAMPLES,INPUT_SIZE)
labels = torch.randn(N_SAMPLES,OUTPUT_SIZE)   


for i in range(N_SAMPLES):
    model_2.model.reset()
    model_7.model.reset()
    model_12.model.reset()
    model_17.model.reset()
    model_22.model.reset()

    input = inputs[i]
    label = labels[i]
    # zero the gradients
    optimizer_2.zero_grad()
    optimizer_7.zero_grad()
    optimizer_12.zero_grad()
    optimizer_17.zero_grad()
    optimizer_22.zero_grad()

    # forward pass
    output_2 = model_2(input)
    output_7 = model_7(input)
    output_12 = model_12(input)
    output_17 = model_17(input)
    output_22 = model_22(input)

    # assure all outputs are same
    assert torch.all(torch.eq(output_2, output_7))
    assert torch.all(torch.eq(output_2, output_12))
    assert torch.all(torch.eq(output_2, output_17))
    assert torch.all(torch.eq(output_2, output_22))

    # calculate the loss
    loss_2 = loss_fn(output_2, label)
    loss_7 = loss_fn(output_7, label)
    loss_12 = loss_fn(output_12, label)
    loss_17 = loss_fn(output_17, label)
    loss_22 = loss_fn(output_22, label)

    # backward pass
    loss_2.backward()
    loss_7.backward()
    loss_12.backward()
    loss_17.backward()
    loss_22.backward()

    # store the gradients
    # for param in model_2.named_parameters():
    #     print(param)
    #     if 'weight' in param[0]:
    #         grads_2.append(param.grad)
    grads_2.append([param[1].grad for param in model_2.named_parameters() if 'weight' in param[0]])
    grads_7.append([param[1].grad for param in model_7.named_parameters() if 'weight' in param[0]])
    grads_12.append([param[1].grad for param in model_12.named_parameters() if 'weight' in param[0]])
    grads_17.append([param[1].grad for param in model_17.named_parameters() if 'weight' in param[0]])
    grads_22.append([param[1].grad for param in model_22.named_parameters() if 'weight' in param[0]])

grads_2_stacked = [torch.stack(layer) for layer in zip(*grads_2)]
grads_7_stacked = [torch.stack(layer) for layer in zip(*grads_7)]
grads_12_stacked = [torch.stack(layer) for layer in zip(*grads_12)]
grads_17_stacked = [torch.stack(layer) for layer in zip(*grads_17)]
grads_22_stacked = [torch.stack(layer) for layer in zip(*grads_22)]

all_grads = [grads_2_stacked, 
            grads_7_stacked, 
            grads_12_stacked, 
            grads_17_stacked, 
            grads_22_stacked]


def gradient_heatmap():
    vmin = -0.01
    vmax = 0.01

    fig, axs = plt.subplots(nrows=len(all_grads), ncols=4)
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
    fig, axs = plt.subplots(nrows=1, ncols=4)
    for j in range(4):
        data = [grads_2_stacked[j].flatten().numpy(), 
                grads_7_stacked[j].flatten().numpy(), 
                grads_12_stacked[j].flatten().numpy(), 
                grads_17_stacked[j].flatten().numpy(), 
                grads_22_stacked[j].flatten().numpy()]
        axs[j].boxplot(data)
        axs[j].set_title(f'Layer {j} Gradients')
        axs[j].set_xticklabels(['2','7','12','17','22'])
        # make y axis of all subplots the same
        axs[j].set_ylim(-1,1)

def calculate_sign_reversal_accross_batch():
    # calculate sign reversal
    fig, axs = plt.subplots(nrows=len(all_grads), ncols=4)
    fig_boxplot, axs_boxplot = plt.subplots(nrows=1, ncols=4) # for boxplot

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
    fig_boxplot, axs_boxplot = plt.subplots(nrows=1, ncols=4) # for boxplot
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
    for j in range(4):
        # gather data for layer j for each model
        data_accross_batch = [model[j] for model in data_models]
        # # add to boxplot
        axs_boxplot[j].violinplot(data_accross_batch,
                showmeans=False,
                showmedians=True)
        axs_boxplot[j].set_title(f'Layer {j} Sign Reversal Probability')

        plt.title("Sign Reversal Probability For Each Layer (1-4)")
 
def calculate_cosine_similarity():
    fig, axs = plt.subplots(nrows=1, ncols=4)
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
    for j in range(4):
        # gather data for layer j for each model
        data_accross_batch = [model[j] for model in data_models]
        # # add to boxplot
        axs[j].violinplot(data_accross_batch,
                showmeans=False,
                showmedians=True)
        axs[j].set_title(f'Layer {j} Cosine Similarity')
        # share y axis
        axs[j].set_ylim(-1,1)
        plt.title("Cosine Similarity For Each Layer (1-4)")


            # draw this in subplot on ax[i], column j

        # axs[i].imshow(grad, cmap='viridis')
        # axs[i].set_title(f'Layer {j+1} Gradients for model with slope {2+5*i}')
    # calculate the cosine similarity between the gradients of the high slope model and the other models

# calculate_sign_reversal_accross_layer()
calculate_cosine_similarity()
plt.show()