from l2f_gym import Learning2Fly
from spikingActorProb import SpikingNet
import torch
from torch import nn
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt


# Wrapper class from TD3BC_Online.py
class Wrapper(nn.Module):
    def __init__(self, model, size=128, action_size=4):
        super().__init__()
        self.preprocess = model
        self.mu = nn.Linear(size, action_size)
        self.sigma = nn.Linear(size, action_size)
    
    def forward(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        x = self.preprocess(x)
        if isinstance(x, tuple):
            x = x[0]  # spiking networks return tuple
        return nn.Tanh()(self.mu(x))


# L2F Agent model from l2f_agent.py
class ConvertedModel(nn.Module):
    def __init__(self):
        super(ConvertedModel, self).__init__()
        self.layer0 = nn.Linear(146, 64)
        self.layer1 = nn.Linear(64, 64)
        self.layer2 = nn.Linear(64, 4)

    def forward(self, x):
        x = torch.tanh(self.layer0(x))
        x = torch.tanh(self.layer1(x))
        x = torch.tanh(self.layer2(x))
        return x


# Setup
device = 'cpu'
hidden_sizes = [256, 128]

def create_actor(device='cpu', hidden_sizes=[256, 128]):
    """Create the TD3BC_Online actor architecture"""
    spiking_module = SpikingNet(
        state_shape=18, 
        action_shape=hidden_sizes[-1],
        hidden_sizes=hidden_sizes[:-1],
        device=device,
        reset_in_call=False,
        repeat=1,
        slope=10.0,
        schedule='fixed',
        verbose=False
    ).to(device)
    
    model = Wrapper(spiking_module, size=hidden_sizes[-1]).to(device)
    return model

# Create environment
print("Creating environment...")
env = Learning2Fly()

# Function to run rollouts for a policy
def run_rollouts(actor, env, n_rollouts=25, use_full_obs=False):
    """Run multiple rollouts and return actions and observations
    
    Args:
        actor: The policy model
        env: The environment
        n_rollouts: Number of rollouts to run
        use_full_obs: If True, use full 146-dim privileged obs; if False, use 18-dim actor obs
    """
    actions_lst = []
    observations_lst = []
    
    for i in range(n_rollouts):
        obs = env.reset()[0]
        done = False
        actions = []
        observations = []
        while not done:
            # Select observation dimensions based on model type
            if use_full_obs:
                obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
            else:
                obs_tensor = torch.tensor(obs[:18], dtype=torch.float32).to(device)
            
            observations.append(obs.copy())  # Store full observation as numpy
            
            with torch.no_grad():
                action = actor(obs_tensor).squeeze().cpu().numpy()
            
            obs, reward, done, _, info = env.step(action)
            actions.append(action)
        
        print(f"  Rollout {i+1}/{n_rollouts} completed with {len(actions)} timesteps")
        actions_lst.append(actions)
        observations_lst.append(observations)
    
    return actions_lst, observations_lst


# Load and evaluate first policy (stable)
print("\n=== Policy 1: TD3BC_Online_stable ===")
actor1 = create_actor(device=device, hidden_sizes=hidden_sizes)
actor1.load_state_dict(torch.load('from_wandb_policy/TD3BC_Online_stable.pth', map_location=device))
actor1.eval()

print("Running rollouts for stable policy...")
actions_lst_1, observations_lst_1 = run_rollouts(actor1, env, n_rollouts=25)

# Load and evaluate second policy (random)
print("\n=== Policy 2: TD3BC_Online_random ===")
actor2 = create_actor(device=device, hidden_sizes=hidden_sizes)
actor2.load_state_dict(torch.load('from_wandb_policy/TD3BC_Online_random.pth', map_location=device))
actor2.eval()

print("Running rollouts for random policy...")
actions_lst_2, observations_lst_2 = run_rollouts(actor2, env, n_rollouts=25)

# Load and evaluate third policy (l2f_agent - the original controller)
print("\n=== Policy 3: L2F_Agent (Original Controller) ===")
actor3 = ConvertedModel()
actor3.load_state_dict(torch.load('l2f_agent.pth', map_location=device))
actor3.eval()

print("Running rollouts for L2F agent...")
actions_lst_3, observations_lst_3 = run_rollouts(actor3, env, n_rollouts=25, use_full_obs=True)

# Find longest rollout for each policy
longest_idx_1 = np.argmax([len(obs) for obs in observations_lst_1])
longest_idx_2 = np.argmax([len(obs) for obs in observations_lst_2])
longest_idx_3 = np.argmax([len(obs) for obs in observations_lst_3])

rollout_states_1 = np.array(observations_lst_1[longest_idx_1])
rollout_actions_1 = np.array(actions_lst_1[longest_idx_1])

rollout_states_2 = np.array(observations_lst_2[longest_idx_2])
rollout_actions_2 = np.array(actions_lst_2[longest_idx_2])

rollout_states_3 = np.array(observations_lst_3[longest_idx_3])
rollout_actions_3 = np.array(actions_lst_3[longest_idx_3])

print(f"\nPolicy 1 (Stable) longest: {len(rollout_states_1)} timesteps (avg: {np.mean([len(a) for a in actions_lst_1]):.1f})")
print(f"Policy 2 (Random) longest: {len(rollout_states_2)} timesteps (avg: {np.mean([len(a) for a in actions_lst_2]):.1f})")
print(f"Policy 3 (L2F Agent) longest: {len(rollout_states_3)} timesteps (avg: {np.mean([len(a) for a in actions_lst_3]):.1f})")

# Plot actions comparison
fig, axes = plt.subplots(4, 1, figsize=(14, 10))
fig.suptitle('Agent Actions Comparison (3 Policies)', fontsize=14)
for i in range(4):
    axes[i].plot(rollout_actions_1[:, i], label='TD3BC Stable', alpha=0.8, linewidth=1.5, color='blue')
    # axes[i].plot(rollout_actions_2[:, i], label='TD3BC Random', alpha=0.8, linewidth=1.5, linestyle='--', color='orange')
    axes[i].plot(rollout_actions_3[:, i], label='L2F Agent (Original)', alpha=0.8, linewidth=1.5, linestyle='-.', color='green')
    axes[i].set_ylabel(f'Motor {i+1}')
    axes[i].legend(loc='upper right', fontsize=8)
    axes[i].grid(True, alpha=0.3)
axes[-1].set_xlabel('Timestep')
plt.tight_layout()
plt.savefig('comparison_actions.png', dpi=150, bbox_inches='tight')
print("Saved comparison_actions.png")
plt.close()

# Plot states comparison
# Observation structure (from l2f_thesis/l2f/interface.cpp):
# - Position (3): indices 0-2
# - Orientation Rotation Matrix (9): indices 3-11 (3x3 matrix flattened)
# - Linear Velocity (3): indices 12-14
# - Angular Velocity (3): indices 15-17
# Total: 18 elements (actor_obs_size)
fig, axs = plt.subplots(3, 1, figsize=(14, 12))
fig.suptitle('Agent State Trajectories Comparison (3 Policies)', fontsize=14)

# Position (indices 0:3)
axs[0].plot(rollout_states_1[:, 0], label='TD3BC Stable: x', alpha=0.7, linewidth=1.5, color='blue')
axs[0].plot(rollout_states_1[:, 1], label='TD3BC Stable: y', alpha=0.7, linewidth=1.5, color='blue', linestyle=':')
axs[0].plot(rollout_states_1[:, 2], label='TD3BC Stable: z', alpha=0.7, linewidth=1.5, color='blue', linestyle='-.')
# axs[0].plot(rollout_states_2[:, 0], label='TD3BC Random: x', alpha=0.7, linewidth=1.5, linestyle='--', color='orange')
# axs[0].plot(rollout_states_2[:, 1], label='TD3BC Random: y', alpha=0.7, linewidth=1.5, linestyle=':', color='orange')
# axs[0].plot(rollout_states_2[:, 2], label='TD3BC Random: z', alpha=0.7, linewidth=1.5, linestyle='-.', color='orange')
axs[0].plot(rollout_states_3[:, 0], label='L2F Agent: x', alpha=0.7, linewidth=1.8, color='green')
axs[0].plot(rollout_states_3[:, 1], label='L2F Agent: y', alpha=0.7, linewidth=1.8, linestyle=':', color='green')
axs[0].plot(rollout_states_3[:, 2], label='L2F Agent: z', alpha=0.7, linewidth=1.8, linestyle='-.', color='green')
axs[0].set_ylabel('Position (m)')
axs[0].legend(ncol=3, fontsize=7)
axs[0].grid(True, alpha=0.3)

# Linear Velocity (indices 12:15)
axs[1].plot(rollout_states_1[:, 12], label='TD3BC Stable: vx', alpha=0.7, linewidth=1.5, color='blue')
axs[1].plot(rollout_states_1[:, 13], label='TD3BC Stable: vy', alpha=0.7, linewidth=1.5, color='blue', linestyle=':')
axs[1].plot(rollout_states_1[:, 14], label='TD3BC Stable: vz', alpha=0.7, linewidth=1.5, color='blue', linestyle='-.')
# axs[1].plot(rollout_states_2[:, 12], label='TD3BC Random: vx', alpha=0.7, linewidth=1.5, linestyle='--', color='orange')
# axs[1].plot(rollout_states_2[:, 13], label='TD3BC Random: vy', alpha=0.7, linewidth=1.5, linestyle=':', color='orange')
# axs[1].plot(rollout_states_2[:, 14], label='TD3BC Random: vz', alpha=0.7, linewidth=1.5, linestyle='-.', color='orange')
axs[1].plot(rollout_states_3[:, 12], label='L2F Agent: vx', alpha=0.7, linewidth=1.8, color='green')
axs[1].plot(rollout_states_3[:, 13], label='L2F Agent: vy', alpha=0.7, linewidth=1.8, linestyle=':', color='green')
axs[1].plot(rollout_states_3[:, 14], label='L2F Agent: vz', alpha=0.7, linewidth=1.8, linestyle='-.', color='green')
axs[1].set_ylabel('Velocity (m/s)')
axs[1].legend(ncol=3, fontsize=7)
axs[1].grid(True, alpha=0.3)

# Angular velocities (indices 15:18)
axs[2].plot(rollout_states_1[:, 15], label='TD3BC Stable: ωx', alpha=0.7, linewidth=1.5, color='blue')
axs[2].plot(rollout_states_1[:, 16], label='TD3BC Stable: ωy', alpha=0.7, linewidth=1.5, color='blue', linestyle=':')
axs[2].plot(rollout_states_1[:, 17], label='TD3BC Stable: ωz', alpha=0.7, linewidth=1.5, color='blue', linestyle='-.')
# axs[2].plot(rollout_states_2[:, 15], label='TD3BC Random: ωx', alpha=0.7, linewidth=1.5, linestyle='--', color='orange')
# axs[2].plot(rollout_states_2[:, 16], label='TD3BC Random: ωy', alpha=0.7, linewidth=1.5, linestyle=':', color='orange')
# axs[2].plot(rollout_states_2[:, 17], label='TD3BC Random: ωz', alpha=0.7, linewidth=1.5, linestyle='-.', color='orange')
axs[2].plot(rollout_states_3[:, 15], label='L2F Agent: ωx', alpha=0.7, linewidth=1.8, color='green')
axs[2].plot(rollout_states_3[:, 16], label='L2F Agent: ωy', alpha=0.7, linewidth=1.8, linestyle=':', color='green')
axs[2].plot(rollout_states_3[:, 17], label='L2F Agent: ωz', alpha=0.7, linewidth=1.8, linestyle='-.', color='green')
axs[2].set_ylabel('Angular velocity (rad/s)')
axs[2].set_xlabel('Timestep')
axs[2].legend(ncol=3, fontsize=7)
axs[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('observation_vis/comparison_states.png', dpi=150, bbox_inches='tight')
print("Saved comparison_states.png")
plt.close()

print(f"\n=== Summary ===")
print(f"TD3BC Stable policy - Avg length: {np.mean([len(a) for a in actions_lst_1]):.1f} steps, Max: {len(rollout_states_1)}")
print(f"TD3BC Random policy - Avg length: {np.mean([len(a) for a in actions_lst_2]):.1f} steps, Max: {len(rollout_states_2)}")
print(f"L2F Agent (Original) - Avg length: {np.mean([len(a) for a in actions_lst_3]):.1f} steps, Max: {len(rollout_states_3)}")
print("\nVisualization complete! Generated:")
print("  - comparison_actions.png")
print("  - observation_vis/comparison_states.png")
