import torch
from torch.utils.data import DataLoader

import snntorch as snn

from torch import nn
from snntorch import surrogate
from l2f_gym import Learning2Fly

from neurobench.models.snntorch_models import SNNTorchAgent
from neurobench.models.torch_model import TorchAgent
from neurobench.benchmarks import Benchmark_Closed_Loop
from neurobench.examples.cartpole.processors import discrete_Actor_Critic

from neurobench.examples.cartpole.agents import ActorCriticSNN_LIF_Smallest, ActorCritic_ANN_Smallest, ActorCriticSNN_LIF_Small, ActorCriticSNN_LIF_Smallest_pruned, ActorCritic_ANN

import numpy as np


env = Learning2Fly()

from spikingActorProb import ActorProb, SpikingNet, SMLP

    
state_shape = env.observation_space.shape or env.observation_space.n
action_shape = env.action_space.shape or env.action_space.n
max_action = env.action_space.high[0]

class StochtoDeterm(nn.Module):
    def __init__(self, model):
        super(StochtoDeterm, self).__init__()
        self.model = model
    def forward(self, x):
        return np.array(self.model(x)[0][0].detach().clone().cpu())
# model
hidden_sizes = [256, 256]
net_a = SpikingNet(state_shape=state_shape, hidden_sizes=hidden_sizes, action_shape=256, repeat=4)
actor = ActorProb(
    net_a,
    action_shape,
    unbounded=True,
    conditioned_sigma=True,
)
print(actor.state_dict().keys())
# state dict to state_dict only actor. keys are included, actor. part is removed
dict_policy = torch.load('stabilize/sac/policy_snn_actor_1.pth')
dict_actor = {}
for key in list(dict_policy.keys()):
    if key.startswith('actor.'):
        dict_actor[key[6:]] = dict_policy[key]

actor.load_state_dict(dict_actor)
# postprocessors
postprocessors = [] # goes from probalities to actions

static_metrics = ["model_size", "connection_sparsity"]
data_metrics = ["activation_sparsity", 'reward_score','synaptic_operations']

model_snn = SNNTorchAgent(StochtoDeterm(actor))

benchmark = Benchmark_Closed_Loop(model_snn, env, [], postprocessors, [static_metrics, data_metrics])
results = benchmark.run(nr_interactions=10, max_length=500) # for risk, now min 20 interactions as risk is lowest 5 percentile
print(results)




# SNN =  ActorCriticSNN_LIF_Smallest_pruned(nr_ins,env.action_space, hidden_size=11,
#                                     inp_min = torch.tensor([-4.8, -10,-0.418,-2]), 
#                                     inp_max=  torch.tensor([4.8, 10,0.418,2]), 
#                                     nr_passes = 1)
# SNN.load_state_dict(torch.load('neurobench/examples/cartpole/model_data/SNN_in248out_25e3_0gain_pruned_full.pt'))

# model_snn = SNNTorchAgent(SNN)

# benchmark = Benchmark_Closed_Loop(model_snn, env, [], postprocessors, [static_metrics, data_metrics])
# results = benchmark.run(nr_interactions=1000, max_length=10000)
# print(results)