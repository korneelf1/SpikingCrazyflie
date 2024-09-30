import snntorch as snn
import torch
from spikingActorProb import ActorProb, SpikingNet, SMLP
from l2f_gym import Learning2Fly, create_learning2fly_env

class ActivityBasedPruning():
    def __init__(self, threshold=0.1, env_create_fn = create_learning2fly_env, env_kwargs:dict | None = None, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.env = env_create_fn(**env_kwargs)

    def prune(self, model, n_runs = 10, min_run_len = 100, threshold=0.1):
        """
        Prune the model based on the activity of the neurons

        Parameters
        ----------
        model : model to be pruned
        n_runs : number of runs to compute activity
        min_run_len : minimum length of a run
        threshold : activity threshold to prune
        """
        for run in range(n_runs):
            obs = self.env.reset()

            t = 0
            

       
        return model