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
from pprint import pprint

env = Learning2Fly(ez_reset=True)

from spikingActorProb import ActorProb, SpikingNet, SMLP
from activity_based_pruning import ActivityBasedPruning

import wandb
import os
# Replace with your W&B project and entity details
PROJECT_NAME = "l2f_bc"
CONFIG_COLUMN = "hidden_sizes"  # The config key you're filtering on
CONFIG_VALUE = [128,128]        # The value you're looking for in the config
ARTIFACT_NAME = "policy_streaming"  # Name of the artifact to download (e.g., 'model')

# Initialize wandb API
api = wandb.Api()
# run = wandb.init()
def download_artifacts():
    # Fetch all runs in the project
    runs = api.runs(f"{PROJECT_NAME}")

    # Iterate over the runs and filter based on the config column
    for run in runs:
        config_value = run.config.get(CONFIG_COLUMN)
        
        if config_value == CONFIG_VALUE:
            print(f"Run {run.name} matches criteria. Searching for the latest artifact...")

            # Get the latest artifact associated with the run
            artifacts = run.logged_artifacts()
            if artifacts:
                latest_artifact = sorted(artifacts, key=lambda a: a.updated_at, reverse=True)[0]  # Sort artifacts by date, pick the latest
                print(f"Found latest artifact {latest_artifact.name}. Downloading...")
                # Download the artifact
                artifact_dir = latest_artifact.download()
                algo_name = run.config.get("Algo")
                slope = run.config.get("slope")
                # Rename the directory to the format: run_name_configvalue
                new_dir_name = f"artifacts/{algo_name}/{run.name}_{str(config_value)}_{str(slope)}"
                os.rename(artifact_dir, new_dir_name)
                print(f"Artifact downloaded and renamed to: {new_dir_name}")
            else:
                print(f"No artifacts found for run {run.name}.")
        else:
            print(f"Run {run.name} does not match criteria.")
# download_artifacts()
#


state_shape = env.observation_space.shape or env.observation_space.n
action_shape = env.action_space.shape or env.action_space.n
max_action = env.action_space.high[0]

class StochtoDeterm(nn.Module):
    def __init__(self, model):
        super(StochtoDeterm, self).__init__()
        self.model = model
    def forward(self, x):
        return np.array(self.model(x)[0][0].detach().clone().cpu())




# print(actor.state_dict().keys())

# gather all file names in artifacts/BC/... to the pth file
def get_file_names(algo='BC'):
    file_names = []
    for root, dirs, files in os.walk(f'artifacts/{algo}/'):
        for file in files:
            if file.endswith('.pth'):
                file_names.append(os.path.join(root, file))
                print("Found: ", os.path.join(root, file))
    return file_names

def benchmark_file(filename):
    print(f'\n\n\n Benchmarking {filename} \n\n\n')
    # state dict to state_dict only actor. keys are included, actor. part is removed
    # dict_policy = torch.load(str(file))
    # dict_actor = {}
    # for key in list(dict_policy.keys()):
    #     if key.startswith('actor.'):
    #         dict_actor[key[6:]] = dict_policy[key]

    # model
    hidden_sizes = [128, 128]
    net_a = SpikingNet(state_shape=18, hidden_sizes=hidden_sizes[:-1], action_shape=hidden_sizes[-1], repeat=1, reset_in_call=False)
    class Wrapper(nn.Module):
        def __init__(self,model , stoch=False, warmup=50):
            super().__init__()
            self.preprocess = model
            self.mu = nn.Linear(128,4)
            self.sigma = nn.Linear(128,4)
            self.t = 0
            self.warmup = warmup
            if stoch:
                self.dist = torch.distributions.Normal
        def forward(self, x):
            if x.shape[1]>18:
                x = x[:,:18]
            while self.t<self.warmup:
                self.t+=1
                _ = self.preprocess(x)
            x = self.preprocess(x)
            if isinstance(x, tuple):
                x = x[0]    #spiking
            if hasattr(self, 'dist'):
                return nn.Tanh()(self.dist(self.mu(x), self.sigma(x)))
            return np.array(nn.Tanh()(self.mu(x)).detach().clone().cpu())
        def reset(self):
            self.t = 0
            if hasattr(self.preprocess, 'reset'):
                self.preprocess.reset()
        # def to_cuda(self):
        #     return self.preprocess.to(device)

    actor = Wrapper(net_a, stoch=False)
    actor.load_state_dict(torch.load(str(filename), map_location='cpu'))
    # postprocessors
    postprocessors = [] # goes from probalities to actions

    static_metrics = ["model_size", "connection_sparsity"]
    data_metrics = ["activation_sparsity", 'reward_score','synaptic_operations']

    model_snn = SNNTorchAgent(actor)

    pruner = ActivityBasedPruning()

    benchmark = Benchmark_Closed_Loop(model_snn, env, [], postprocessors, [static_metrics, data_metrics])
    results = benchmark.run(nr_interactions=10, max_length=500) # for risk, now min 20 interactions as risk is lowest 5 percentile

    pprint(results)

    ####### Test pruned model
    print('\n\n\n After pruning \n\n\n')
    # model
    hidden_sizes = [128, 128]
    net_a = SpikingNet(state_shape=18, hidden_sizes=hidden_sizes[:-1], action_shape=hidden_sizes[-1], repeat=1,reset_in_call=False)
    actor = Wrapper(net_a, stoch=False)
    actor.load_state_dict(torch.load(str(filename), map_location='cpu'))
    # actor = ActorProb(
    #     net_a,
    #     action_shape,
    #     unbounded=True,
    #     conditioned_sigma=True,
    # )

    # # state dict to state_dict only actor. keys are included, actor. part is removed
    # dict_policy = torch.load('stabilize/sac/policy_snn_actor_1.pth')
    # dict_actor = {}
    # for key in list(dict_policy.keys()):
    #     if key.startswith('actor.'):
    #         dict_actor[key[6:]] = dict_policy[key]

    # actor.load_state_dict(dict_actor)
    # postprocessors
    postprocessors = [] # goes from probalities to actions

    static_metrics = ["model_size",]
    data_metrics = ['reward_score']

    model_snn = SNNTorchAgent(actor)
    pruned_model = pruner.prune(model_snn, test_pruning=False, threshold=0.01, n_runs=100, create_histogram=True)
    model_snn = SNNTorchAgent(StochtoDeterm(pruned_model))
    benchmark = Benchmark_Closed_Loop(model_snn, env, [], postprocessors, [static_metrics, data_metrics])
    results = benchmark.run(nr_interactions=20, max_length=500) # for risk, now min 20 interactions as risk is lowest 5 percentile

    pprint(results)

benchmark_file('artifacts/BC/royal-disco-223_[128, 128]_25/model_bc.pth')
# for file in get_file_names(algo='BC'):
    # benchmark_file(file)