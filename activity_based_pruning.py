import snntorch as snn
import torch
from spikingActorProb import ActorProb, SpikingNet, SMLP
from l2f_gym import Learning2Fly, create_learning2fly_env
import neurobench
from neurobench.benchmarks import data_metrics
from tqdm import tqdm
import numpy as np
from torch import nn

class ActivityBasedPruning():
    def __init__(self, threshold=0.1, env_create_fn = create_learning2fly_env, env_kwargs:dict = {}, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.env = env_create_fn(**env_kwargs)

    def prune(self, model, n_runs = 10, min_run_len = 100, threshold=0.01, create_histogram:bool = False, test_pruning:bool = False):
        """
        Prune the model based on the activity of the neurons

        Parameters
        ----------
        model : model to be pruned
        n_runs : number of runs to compute activity
        min_run_len : minimum length of a run
        threshold : activity threshold to prune
        create_histogram : create histogram of the activity
        test_pruning : test the pruning, will return the model with the neurons deactivated for neurobench testing
        """
        data_metrics.detect_activations_connections(model)
        for run in tqdm(range(n_runs),
                        desc="Pruning",):
            obs = self.env.reset()[0]

            done = False
            t = 0
            while not done:
                obs = obs.reshape(1,1,-1)
                action = model(obs)
                obs, reward, done, info,_ = self.env.step(action)
                t += 1
                if t<min_run_len and done:
                    done=False
                    obs = self.env.reset()[0]
                    t = 0

        total_spike_num_layer = []
        for hook in model.activation_hooks:
            total_spike_num = np.zeros_like(hook.activation_outputs[0].detach().numpy())
            n_steps = len(hook.activation_outputs)
            for spikes in hook.activation_outputs:  # do we need a function rather than a member
                # assume each activarion output is a tensor of size (batch, neurons)
                assert len(spikes.shape) == 2
                total_spike_num += torch.sum(spikes, dim=0).cpu().detach().numpy()
            total_spike_num /= n_steps
            total_spike_num_layer.append(total_spike_num*100)
        
        if create_histogram:
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(len(total_spike_num_layer))
            
            for i, layer in enumerate(total_spike_num_layer):
                axs[i].hist(layer.squeeze(0), bins=50, range=(0, 100), cumulative=False, density=True)
                # calculate number of activations lower than threshold*100
                lower_then_threshold = np.sum(layer.squeeze(0)<threshold*100)
                # zeros = np.sum(layer.squeeze(0)==0)
                axs[i].set_title(f'Layer {i}, with {lower_then_threshold} neurons with activity below {threshold*100}')
            plt.suptitle('Histogram of the activity of the neurons')
            plt.show()

        
        if test_pruning:
            one_hot_lst = [] 
            for layer in total_spike_num_layer:
                one_hot = np.zeros_like(layer)
                one_hot[layer<threshold*100] = 1
                one_hot_lst.append(one_hot)
            model = self._deactivate_neurons(model,one_hot_lst)
        return model

    def _deactivate_neurons(self, model, neuron_layer_idx:list):
        """
        Deactivate neurons from the model, for testing

        Parameters
        ----------
        model : model to be pruned
        neuron_layer_idx : list of len n_layers with neurons to deactivate indexes (one_hot)
        """
        counter = 0
        activation_layers = model.activation_hooks
        
        for i, layer in enumerate(activation_layers):
            thresh_curr = layer.layer.threshold 
            # set thresholds where neuron_layer_idx[i] is 1 to int(1e8) to deactivate spiking
            above = np.where(neuron_layer_idx[i], np.array(1e8), thresh_curr.detach().numpy())
            layer.layer.threshold = nn.Parameter(torch.tensor(above))
            print(f'Layer {i} deactivated {np.sum(neuron_layer_idx[i])} neurons')

        
        return model
    def _prune_neurons(self, model, neuron_layer_idx:list):
        """
        Prune neurons from the model

        Parameters
        ----------
        model : model to be pruned
        neuron_layer_idx : list of len n_layers with neurons to remove indexes (one_hot)
        """
        pass
if __name__=='__main__':
    env = Learning2Fly()

    from spikingActorProb import ActorProb, SpikingNet, SMLP
    from neurobench.models.snntorch_models import SNNTorchAgent
        
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

    model_snn = SNNTorchAgent(StochtoDeterm(actor))

    pruner = ActivityBasedPruning(env_create_fn=create_learning2fly_env, env_kwargs={'rpm':False})
    pruner.prune(model_snn, n_runs=1, create_histogram=True, threshold=0.05, test_pruning=True)