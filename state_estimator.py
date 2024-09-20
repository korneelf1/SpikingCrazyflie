from l2f_gym import Learning2Fly
from imu import IMU
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import stable_baselines3 as sb3
from spikingActorProb import SMLP
from tianshou.utils.net.common import MLP
import snntorch as snn


def gather_dataset(num_samples=1000, T_max=250):
    '''Uses a trained controller (ann_actor) to collect data from the environment. The data saved consists of the states.
    These states are then appended with IMU data and saved in a numpy array.
    The labels consist of the velocities and orientations of the drone.'''
    env = Learning2Fly(imu=False,t_history=1,euler=True) # if euler states are pos, vel_body, orientation_euler, angular velocity
    print(env.observation_space)
    # actor = torch.load("ann_actor.pth")
    actor = sb3.SAC.load("SAC_l2f_euler.zip")
    print(actor)
    # Initialize the IMU
    imu = IMU(euler_in=True )

    # state is pos, vel, orientation, angular velocity
    # Initialize the state and label arrays
    states = torch.empty((num_samples, T_max, 13))

    labels = torch.empty((num_samples, T_max, 9+7))

    t = 0
    for i in tqdm(range(num_samples), desc="Gathering data"):
        filled = False
        while not filled:
            t = 0
            early_stop = False
            state = env.reset()[0]
            imu.reset()
            # print(state.shape)
            state = torch.tensor(state, dtype=torch.float32)
            states[i, t] = state[0:13]
            while t<T_max-1 and not early_stop:
                t += 1
                # Get the action from the actor
                # print(state.shape)
                action = actor.predict(state,deterministic=True)[0]
                # print(action)
                state, reward, done, _ ,_= env.step(action)
                state = torch.tensor(state, dtype=torch.float32)
                states[i, t] = state[0:13]

                # Get the IMU data
                imu_data = imu.simulate(state)
                # pos_imu = torch.cat((state[0:3], torch.from_numpy(imu_data)), dim=0)
                # Get the labels
                labels[i, t] = torch.cat((torch.from_numpy(imu_data), state[3:10]),dim=0)

                if done:
                    t= 0 # discard the episode
                    # print("Discarded episode")
                    early_stop = True
                if t == T_max-1:
                    filled = True
    print(states.shape)
    print(labels.shape)
    dataset = TensorDataset(labels, states)

    
    return dataset

class IntegratorSpiker(torch.nn.Module):
    def __init__(self, layer_size = 64, integrator_ratio=0.5):
        super().__init__()
        # n_integrators = int(layer_size * integrator_ratio)
        # n_spikers = layer_size - n_integrators
        n_integrators = layer_size
        n_spikers = layer_size


        self.spike_grad = snn.surrogate.fast_sigmoid(10)

        self.betas = torch.nn.Parameter(torch.rand(n_spikers))
        self.thresholds = torch.nn.Parameter(torch.rand(n_spikers))
        self.lif1 = snn.Leaky(beta=self.betas, learn_beta=True,
                            threshold=self.thresholds, learn_threshold=True,
                            spike_grad=self.spike_grad)
        
        
        self.thresholds_integrators = torch.nn.Parameter(torch.rand(n_integrators))
        self.lif_integrators = snn.Leaky(beta=1, learn_beta=False,
                            threshold=self.thresholds_integrators, learn_threshold=True,
                            spike_grad=self.spike_grad)
        
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
        self.spiking = spiking

        self.spike_grad = snn.surrogate.fast_sigmoid(10)

        self.lin_1 = torch.nn.Linear(16, 32)
        
        self.leaky = IntegratorSpiker(layer_size=32, integrator_ratio=0.5)
        self.lin_2 = torch.nn.Linear(64, 6)

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
            self.reset()
        hidden = None
        x_out = torch.empty((x.shape[0], x.shape[1], 6))
        for t in range(x.shape[1]):
            x_t, hidden = self.forward_single(x[:,t,:], hidden)
            x_out[:, t] = x_t
        return x_out
    
    def train(self, dataset, epochs=10, warmup=0):
        dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
        if self.spiking:
            super().train()
        # else:
        #     self.model.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        criterion = torch.nn.MSELoss()

        for epoch in range(epochs):
            losses = []
            for imu,true_states in dataloader:
                optimizer.zero_grad()
                y_pred = self.forward(imu)
                # print(y_pred.shape, true_states.shape)
                # extract intial states from true_states and substract from predicted states
                # gotta compare to t+1 true state shape B, T, 13
                loss = criterion(y_pred[:,warmup:-1], true_states[:,warmup+1:,3:9])
                losses.append(loss.item())
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch}, Loss: {torch.mean(torch.tensor(losses))}")
        
        # create one plot with the last model
        self.plot_difference_prediction_true(y_pred[1,:], true_states[1,:,3:9])
        torch.save(self, "state_estimator.pth")
    def plot_difference_prediction_true(self, prediction, true):
        # print(prediction, true)
        import matplotlib.pyplot as plt
        # tensor to array
        prediction = prediction.detach().numpy()
        true = true.detach().numpy()

        fig, ax = plt.subplots(2, 3)
        # in each subplot plot one of prediction[:,:, i] and true[:,:, i]
        subfigs = ["x", "y", "z", "vx", "vy", "vz"]
        for i in range(6):
            ax[i//3, i%3].plot(prediction[:,i], c='r')
            ax[i//3, i%3].plot(true[:,i], c='g')
            ax[i//3, i%3].set_title(subfigs[i])
        # ax[0].plot(prediction)
        # ax[0].set_title("Prediction")
        # ax[1].plot(true)
        # ax[1].set_title("True")
        plt.show()
        
if __name__ == "__main__":
    print("Gathering data")
    # dataset = gather_dataset(num_samples=10000, T_max=250)
    # torch.save(dataset, "dataset.pth")
    dataset = torch.load("dataset.pth")
    # print(data.tensors)
    # dataset = TensorDataset(torch.load("dataset.pth"))

    stateestimator = StateEstimator(spiking=True)
    print(stateestimator)
    stateestimator.train(dataset=dataset, epochs=100, warmup = 20)    