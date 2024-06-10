import tianshou as ts
from custom_collector import ParallelCollector
import gym_sim as g
import torch, numpy as np
from torch import nn
import numba as nb

# debug mode
jitter = lambda signature: nb.jit(signature, nopython=True, fastmath=False)
kerneller = lambda signature, map: nb.guvectorize(signature, map, target='parallel', nopython=True, fastmath=False)
GRAVITY = 9.80665


env = g.Drone_Sim(N_cpu=10)
class Net(nn.Module):
    def __init__(self, state_shape, action_shape):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(np.prod(state_shape), 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Linear(128, np.prod(action_shape)),
        )

    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float)
        batch = obs.shape[0]
        logits = self.model(obs.view(batch, -1))
        return logits, state
state_shape = env.observation_space.shape or env.observation_space.n
action_shape = env.action_space.shape or env.action_space.n
net = Net(state_shape, action_shape)

optim = torch.optim.Adam(net.parameters(), lr=1e-3)
policy = ts.policy.DQNPolicy(
    model=net,
    optim=optim,
    action_space=env.action_space,
    discount_factor=0.9,
    estimation_step=3,
    target_update_freq=320
)

train_collector = ParallelCollector(policy, env, ts.data.ReplayBuffer(size=20000))

train_collector.collect(n_step=100, random=False)