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

INPUT_SIZE = 63
OUTPUT_SIZE = 64
HIDDEN_SIZE = 64
N_LAYERS = 4
HIDDEN_LAYER_LST = [HIDDEN_SIZE]*(N_LAYERS - 1)

BIAS = False
N_SAMPLES = 256


class Wrapper(nn.Module):
    '''
    Wraps SMLP model with one nn.Linear output of size N_LAYERS,2 to be able to use nn.MSELoss
    '''
    def __init__(self, model):
        super(Wrapper, self).__init__()
        self.model = model
        self.output = nn.Linear(HIDDEN_SIZE,OUTPUT_SIZE)
    def forward(self, x):
        inp = x
        x, hidden = self.model(inp, None)
        output = x
        for i in range(1):
            x, hidden = self.model(inp, hidden)
            output += x
            
        return self.output(output)
# create a simple model with 3 hidden layers of size N_LAYERS each
# create 5 models with slopes 2,7,12,17,22
model_1 = Wrapper(SMLP(INPUT_SIZE,HIDDEN_SIZE, HIDDEN_LAYER_LST, slope=1))
model_10 = Wrapper(SMLP(INPUT_SIZE,HIDDEN_SIZE, HIDDEN_LAYER_LST, slope=10))
model_25 = Wrapper(SMLP(INPUT_SIZE,HIDDEN_SIZE, HIDDEN_LAYER_LST, slope=25))
model_50 = Wrapper(SMLP(INPUT_SIZE,HIDDEN_SIZE, HIDDEN_LAYER_LST, slope=50))
model_100 = Wrapper(SMLP(INPUT_SIZE,HIDDEN_SIZE, HIDDEN_LAYER_LST, slope=100))
print(model_1)
def plot_surrogate_gradients():
    # plot the surrogate gradients
    def surr_grad(x, slope):
        """
        1 / (ctx.slope * torch.abs(input_) + 1.0) ** 2"""
        return 1/(slope*torch.abs(x)+1)**2
    slopes = [1,10,25,50,100]
    fig, axs = plt.subplots(nrows=1, ncols=1)
    x = np.linspace(-1.5,.5,1000)
    # plot a RED dirac delta as well:
    
    for slope in slopes:
        y = surr_grad(torch.tensor(x), slope)
        plt.plot(x,y, label=f'slope {slope}', linestyle='--')
    plt.plot([0,0],[0,1], label='dirac delta', linewidth=4, color='red')
    plt.legend()
    plt.title('Surrogate Gradient')
    plt.xlabel('Membrane potential, U')
    plt.ylabel('Gradient')
    plt.show()

plot_surrogate_gradients()

# make the weights and biases the same for all models
model_1_sate_dict = model_1.state_dict()
model_10.load_state_dict(model_1_sate_dict)
model_25.load_state_dict(model_1_sate_dict)
model_50.load_state_dict(model_1_sate_dict)
model_100.load_state_dict(model_1_sate_dict)

# create a simple optimizer
optimizer_1 = optim.Adam(model_1.parameters(), lr=1)
optimizer_10 = optim.Adam(model_10.parameters(), lr=1)
optimizer_25 = optim.Adam(model_25.parameters(), lr=1)
optimizer_50 = optim.Adam(model_50.parameters(), lr=1)
optimizer_100 = optim.Adam(model_100.parameters(), lr=1)

# create a loss function
loss_fn = nn.MSELoss()

# create a list to store the gradients
grads_1 = []
grads_10 = []
grads_25 = []
grads_50 = []
grads_100 = []

# create random inputs of INPUT_SIZE and labels of OUTPUT_SIZE
inputs = torch.randn(N_SAMPLES,INPUT_SIZE)
labels = torch.randn(N_SAMPLES,OUTPUT_SIZE)   


for i in range(N_SAMPLES):
    model_1.model.reset()
    model_10.model.reset()
    model_25.model.reset()
    model_50.model.reset()
    model_100.model.reset()

    input = inputs[i]
    label = labels[i]
    # zero the gradients
    optimizer_1.zero_grad()
    optimizer_10.zero_grad()
    optimizer_25.zero_grad()
    optimizer_50.zero_grad()
    optimizer_100.zero_grad()

    # forward pass
    output_1 = model_1(input)
    output_10 = model_10(input)
    output_25 = model_25(input)
    output_50 = model_50(input)
    output_100 = model_100(input)

    # assure all outputs are same
    assert torch.all(torch.eq(output_1, output_10))
    assert torch.all(torch.eq(output_1, output_25))
    assert torch.all(torch.eq(output_1, output_50))
    assert torch.all(torch.eq(output_1, output_100))

    # calculate the loss
    loss_1 = loss_fn(output_1, label)
    loss_10 = loss_fn(output_10, label)
    loss_25 = loss_fn(output_25, label)
    loss_50 = loss_fn(output_50, label)
    loss_100 = loss_fn(output_100, label)

    # backward pass
    loss_1.backward()
    loss_10.backward()
    loss_25.backward()
    loss_50.backward()
    loss_100.backward()

    # store the gradients
    # for param in model_1.named_parameters():
    #     if 'weight' in param[0]:
    #         print(param[0])
            # grads_1.append(param[1].grad)
    grads_1.append([param[1].grad for param in model_1.named_parameters() if 'weight' in param[0]])
    grads_10.append([param[1].grad for param in model_10.named_parameters() if 'weight' in param[0]])
    grads_25.append([param[1].grad for param in model_25.named_parameters() if 'weight' in param[0]])
    grads_50.append([param[1].grad for param in model_50.named_parameters() if 'weight' in param[0]])
    grads_100.append([param[1].grad for param in model_100.named_parameters() if 'weight' in param[0]])

grads_1_stacked = [torch.stack(layer) for layer in zip(*grads_1)]
grads_10_stacked = [torch.stack(layer) for layer in zip(*grads_10)]
grads_25_stacked = [torch.stack(layer) for layer in zip(*grads_25)]
grads_50_stacked = [torch.stack(layer) for layer in zip(*grads_50)]
grads_100_stacked = [torch.stack(layer) for layer in zip(*grads_100)]

all_grads = [grads_1_stacked, 
            grads_10_stacked, 
            grads_25_stacked, 
            grads_50_stacked, 
            grads_100_stacked]


def gradient_heatmap():
    # vmin = -0.01
    # vmax = 0.01

    fig, axs = plt.subplots(nrows=len(all_grads), ncols=N_LAYERS+1)
    for i, grads in enumerate(all_grads):
        for j, grad in enumerate(grads):
            grad = grad.detach().numpy().sum(axis=0)/N_SAMPLES
            # take log of the gradients
            grad = np.log(np.abs(grad)+1e-15)
            heatmap = sb.heatmap(grad, cmap='viridis', ax=axs[i,j])
            # make box plot of the gradients

            # set x-axis label
            heatmap.set_xlabel(f'layer {j} gradient')
            # draw this in subplot on ax[i], column j
        heatmap.set_ylabel(f'Model with slope {2+5*i}')



def gradient_boxplot():
    # create a boxplot of the gradients for each layer
    fig, axs = plt.subplots(nrows=1, ncols=N_LAYERS)
    for j in range(N_LAYERS):
        data = [grads_1_stacked[j].flatten().numpy(), 
                grads_10_stacked[j].flatten().numpy(), 
                grads_25_stacked[j].flatten().numpy(), 
                grads_50_stacked[j].flatten().numpy(), 
                grads_100_stacked[j].flatten().numpy()]
        axs[j].boxplot(data)
        axs[j].set_title(f'Layer {j} Gradients')
        axs[j].set_xticklabels(['2','7','12','17','22'])
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
 
def calculate_cosine_similarity(display='violin'):
    '''
    Calculate the cosine similarity between the gradients of the high slope model and the other models
    display the results in a violin plot or in one line plot with error bars'''
    
    data_models = []
    # for i, grads in enumerate(all_grads):
    #     data = []
    #     for j, grad in enumerate(grads):
    #         # grad = grad.detach().numpy().reshape(N_SAMPLES,-1)
    #         # grad_high_slope = all_grads[-1][j].detach().numpy().reshape(N_SAMPLES,-1)
    #         # # calculate the cosine similarity between the gradients of the high slope model and the other models
    #         # cos_sim = np.sum(grad * grad_high_slope, axis=1)/(np.linalg.norm(grad,axis=1)*np.linalg.norm(grad_high_slope, axis=1))
    #         # data.append(cos_sim)
    #         data_models = []
    for i, grads in enumerate(all_grads):
        data = []
        for j, grad in enumerate(grads):
            # Reshape the gradients
            grad = grad.detach().numpy().reshape(N_SAMPLES, -1)
            grad_high_slope = all_grads[-1][j].detach().numpy().reshape(N_SAMPLES, -1)
            
            # Compute the norms of the gradients
            norm_grad = np.linalg.norm(grad, axis=1)
            norm_grad_high_slope = np.linalg.norm(grad_high_slope, axis=1)
            
            # Avoid division by zero by setting cosine similarity to 0 for zero-norm cases
            zero_norm_mask = (norm_grad == 0) | (norm_grad_high_slope == 0)  # Check where norms are zero
            
            # Compute cosine similarity only for non-zero norm entries
            cos_sim = np.sum(grad * grad_high_slope, axis=1) / (norm_grad * norm_grad_high_slope)
            
            # Replace NaN or Inf values with 0 or another default value
            cos_sim[zero_norm_mask] = 0
            
            data.append(cos_sim)
        data_models.append(data)

    if display == 'violin':
        fig, axs = plt.subplots(nrows=1, ncols=N_LAYERS)
        for j in range(5):
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
    elif display == 'line':
        fig, axs = plt.subplots(nrows=1, ncols=1)
        model_slopes = [1,10,25,50,100]
        for i, model_data in enumerate(data_models):
            means = [np.mean(layer_data) for layer_data in model_data]
            stds = [np.std(layer_data) for layer_data in model_data]
            assert len(means) == N_LAYERS+1
            print(means)
            plt.plot(range(N_LAYERS+1), means, label=f'Model {model_slopes[i]}')
            # plt.fill_between(x, y-error, y+error)
            # plt.fill_between(range(N_LAYERS+1), np.array(means)-np.array(stds), np.array(means)+np.array(stds), alpha=0.5)
            # axs.errorbar(range(N_LAYERS), means, yerr=stds, label=f'Model {2 + 5 * i}')
        axs.legend()
        axs.xaxis.set_ticks(range(N_LAYERS+1))
        axs.set_ylabel('Cosine Similarity')
        axs.set_xlabel('Layer')
        fig.suptitle('Cosine Similarity for Surrogate Gradient Slopes 1 to 100')

def count_non_zero_gradient():
    fig, axs = plt.subplots(nrows=1, ncols=1)
    data_models = []
    for i, grads in enumerate(all_grads):
        data = []
        for j, grad in enumerate(grads):
            grad = grad.detach().numpy().reshape(N_SAMPLES,-1)
            # calculate the cosine similarity between the gradients of the high slope model and the other models
            # all gradients with magnituede < 1e-8 are considered zero
            grad[grad < 1e-18] = 0
            non_zero = np.count_nonzero(grad, axis=1)
            # normalize by number of weights in layer
            non_zero = non_zero/(grad.shape[1])
            data.append(non_zero)
        data_models.append(data)
    
    # calculate ranges for each layer
    # create plot with on x-axis layer, on y-axis non zero gradients
    # one line for each model, color coded
    # represent ranges as a std around the line
    model_slopes = [1,10,25,50,100]
    for i, model_data in enumerate(data_models):
        means = [np.mean(layer_data) for layer_data in model_data]
        stds = [np.std(layer_data) for layer_data in model_data]

        plt.plot(range(N_LAYERS+1), means, label=f'Model {model_slopes[i]}')
        # plt.fill_between(x, y-error, y+error)
        plt.fill_between(range(N_LAYERS+1), np.array(means)-np.array(stds), np.array(means)+np.array(stds), alpha=0.5)
        # axs.errorbar(range(N_LAYERS+1), means, yerr=stds, label=f'Model {2 + 5 * i}')
    axs.legend()
    axs.xaxis.set_ticks(range(N_LAYERS+1))
    axs.set_ylabel('Fraction of Non Zero Gradients')
    axs.set_xlabel('Layer')
    fig.suptitle('Fraction of Non Zero Gradients for Each Layer for Shallow to Steep Surrogate Gradient Slopes')
    # axs.set_xticklabels(['Input Layer'] + [f'Layer {i}' for i in range(N_LAYERS)])



    # for j in range(N_LAYERS):
    #     # gather data for layer j for each model
    #     data_accross_batch = [model[j] for model in data_models]
    #     # # add to boxplot
    #     axs[j].violinplot(data_accross_batch,
    #             showmeans=False,
    #             showmedians=True)
    #     axs[j].set_title(f'Layer {j} Non Zero Gradients')
    #     axs[j].set_ylim(0,1)
    # # make y axis same for every model
    
    
        # plt.title("Cosine Similarity For Each Layer (1-N_LAYERS)")

def calculate_avg_grad_mag():
    fig, axs = plt.subplots(nrows=1, ncols=1)
    data_models = []
    for i, grads in enumerate(all_grads):
        data = []
        for j, grad in enumerate(grads):
            grad = grad.detach().numpy().reshape(N_SAMPLES,-1)
            # calculate the cosine similarity between the gradients of the high slope model and the other models
            # all gradients with magnituede < 1e-8 are considered zero
            grad[grad < 1e-18] = 0
            avg_mag = np.mean(np.abs(grad), axis=1)
            data.append(avg_mag)
        data_models.append(data)
        model_slopes = [1,10,25,50,100]
    for i, model_data in enumerate(data_models):
        means = [np.mean(layer_data) for layer_data in model_data]
        stds = [np.std(layer_data) for layer_data in model_data]

        plt.plot(range(N_LAYERS+1), means, label=f'Model {model_slopes[i]}')
        # plt.fill_between(x, y-error, y+error)
        # plt.fill_between(range(N_LAYERS+1), np.array(means)-np.array(stds), np.array(means)+np.array(stds), alpha=0.5)
        # axs.errorbar(range(N_LAYERS+1), means, yerr=stds, label=f'Model {2 + 5 * i}')
    axs.legend()
    # set y to logscale
    axs.set_yscale('log')
    axs.xaxis.set_ticks(range(N_LAYERS+1))
    axs.set_ylabel('Average Gradient Magnitude')
    axs.set_xlabel('Layer')
    fig.suptitle('Average Gradient Magnitude')
    
    
    # calculate_sign_reversal_accross_layer()
# calculate_cosine_similarity(display='line')
# count_non_zero_gradient()
# calculate_avg_grad_mag()
# gradient_heatmap()
plt.show()