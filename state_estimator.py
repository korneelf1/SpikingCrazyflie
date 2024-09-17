from l2f_gym import Learning2Fly
from imu import IMU
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import stable_baselines3 as sb3
from spikingActorProb import SMLP
from tianshou.utils.net.common import MLP
import snntorch as snn


def gather_dataset(num_samples=1000, T_max=100):
    '''Uses a trained controller (ann_actor) to collect data from the environment. The data saved consists of the states.
    These states are then appended with IMU data and saved in a numpy array.
    The labels consist of the velocities and orientations of the drone.'''
    env = Learning2Fly(imu=False,t_history=1,euler=False)
    # actor = torch.load("ann_actor.pth")
    actor = sb3.SAC.load("sac_l2f.zip")
    print(actor)
    # Initialize the IMU
    imu = IMU()

    # Initialize the state and label arrays
    states = torch.empty((num_samples, T_max, 17))

    labels = torch.empty((num_samples, T_max, 9))

    t = 0
    for i in tqdm(range(num_samples), desc="Gathering data"):
        state = env.reset()[0]
        imu.reset()
        # print(state.shape)
        state = torch.tensor(state, dtype=torch.float32)
        states[i, t] = state
        while t<T_max-1:
            t += 1
            # Get the action from the actor
            # print(state.shape)
            action = actor.predict(state,deterministic=True)[0]
            # print(action)
            state, reward, done, _ ,_= env.step(action)
            state = torch.tensor(state, dtype=torch.float32)
            states[i, t] = state

            # Get the IMU data
            imu_data = imu.simulate(state)
            # pos_imu = torch.cat((state[0:3], torch.from_numpy(imu_data)), dim=0)
            # Get the labels
            labels[i, t] = torch.from_numpy(imu_data)

            if done:
                i -= 1 # discard the episode
                # print("Discarded episode")
                continue
        
    dataset = TensorDataset(states, labels)

    
    return dataset

class IntegratorSpiker(torch.nn.Module):
    def __init__(self, layer_size = 64, integrator_ratio=0.5):
        n_integrators = int(layer_size * integrator_ratio)
        n_spikers = layer_size - n_integrators

        self.spike_grad = snn.surrogate.fast_sigmoid(10)

        self.betas = torch.nn.Parameter(torch.rand(n_spikers))
        self.thresholds = torch.nn.Parameter(torch.rand(n_spikers))
        self.lif1 = snn.Leaky(beta=self.betas, learn_beta=True,
                            threshold=self.thresholds, learn_threshold=True,
                            spikegrad=self.spike_grad)
        
        
        self.thresholds_integrators = torch.nn.Parameter(torch.rand(n_integrators))
        self.lif_integrators = snn.Leaky(beta=1, learn_beta=False,
                            threshold=self.thresholds_integrators, learn_threshold=True,
                            spikegrad=self.spike_grad)
        
        self.reset()
        
    def set_slope(self, slope):
        self.spike_grad = snn.surrogate.fast_sigmoid(slope)
        self.lif1.spikegrad = self.spike_grad
        self.lif_integrators.spikegrad = self.spike_grad

    def reset(self):
        self.cur_1 = self.lif1.init_leaky()
        self.cur_int = self.lif_integrators.init_leaky()

    def forward(self, x, hiddens):
        if hiddens is not None:
            self.cur_1 = hiddens[0]
            self.cur_int = hiddens[1]

        x_1, self.cur_1 = self.lif1(x, self.cur_1)

        x_int, self.cur_int = self.lif_integrators(x, self.cur_int)

        x = torch.cat((x_1, x_int), dim=1)

        return x, [self.cur_1, self.cur_int]
    
class StateEstimator(torch.nn.Module):
    def __init__(self, spiking=True):
        super(StateEstimator, self).__init__()
        self.spike_grad = snn.surrogate.fast_sigmoid(10)

        self.lin_1 = torch.nn.Linear(9, 32)
        
        self.leaky = IntegratorSpiker(layer_size=64, integrator_ratio=0.5)
        self.lin_2 = torch.nn.Linear(64, 9)

        self.reset()

    def reset(self):
        self.leaky.reset()

    def forward_single(self, x, hidden=None):
        x = self.lin_1(x)
        x, hidden = self.leaky(x, hidden)
        x = self.lin_2(x)
        return x, hidden
    
    def forward(self, x):
        '''
        x: [batch_size, T, 9]
        '''
        if self.spiking:
            self.model.reset()
        hidden = None
        for t in range(x.shape[1]):
            x_t = x[:, t,:]
            x_t, hidden = self.forward_single(x_t, hidden)
            if t == 0:
                x_out = x_t
            else:
                x_out = torch.cat((x_out, x_t), dim=1)
        return x
    
    def train(self, dataset, epochs=10):
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        if self.spiking:
            self.model.train()
        else:
            self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = torch.nn.MSELoss()

        for epoch in range(epochs):
            for true_states, imu in dataloader:
                optimizer.zero_grad()
                y_pred = self.forward(imu)
                # extract intial states from true_states and substract from predicted states
                y_init = true_states[:, 0, :]
                loss = criterion(y_pred, true_states)

                loss.backward()
                optimizer.step()
                print(f"Epoch {epoch}, Loss: {loss.item()}")
        torch.save(self.model, "state_estimator.pth")
    
if __name__ == "__main__":
    # dataset = gather_dataset(num_samples=10000, T_max=100)
    # torch.save(dataset, "dataset.pth")
    dataset = torch.load("dataset.pth")
    # print(data.tensors)
    # dataset = TensorDataset(torch.load("dataset.pth"))

    stateestimator = StateEstimator(spiking=True)
    stateestimator.train(dataset, epochs=10)    