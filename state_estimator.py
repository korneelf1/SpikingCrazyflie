import torch.utils
import torch.utils.data
from l2f_gym import Learning2Fly
from imu import IMU
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import stable_baselines3 as sb3
from spikingActorProb import SMLP
from tianshou.utils.net.common import MLP
import snntorch as snn
import wandb


import matplotlib.pyplot as plt
# plt.ion()

import torch
'''
This manual implementation is required to be able to pickle 
the model for parallelization'''
def fast_sigmoid_forward(ctx, input_, slope):
    ctx.save_for_backward(input_)
    ctx.slope = slope
    out = (input_ > 0).float()
    return out

def fast_sigmoid_backward(ctx, grad_output):
    (input_,) = ctx.saved_tensors
    grad_input = grad_output.clone()
    grad = grad_input / (ctx.slope * torch.abs(input_) + 1.0) ** 2
    return grad, None

class FastSigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, slope=25):
        return fast_sigmoid_forward(ctx, input_, slope)

    @staticmethod
    def backward(ctx, grad_output):
        return fast_sigmoid_backward(ctx, grad_output)

class FastSigmoidWrapper:
    def __init__(self, slope=25):
        self.slope = slope

    def __call__(self, x):
        return FastSigmoid.apply(x, self.slope)


def fast_sigmoid(slope=25):
    """Returns a callable object for the FastSigmoid function with a specific slope."""
    return FastSigmoidWrapper(slope)



def gather_dataset(num_samples=1000, T_max=250):
    '''Uses a trained controller (ann_actor) to collect data from the environment. The data saved consists of the states.
    These states are then appended with IMU data and saved in a numpy array.
    The labels consist of the velocities and orientations of the drone.'''
    env = Learning2Fly(imu=False,t_history=1,euler=True) # if euler states are pos, vel_body, orientation_euler, angular velocity
    print(env.observation_space)
    # actor = torch.load("ann_actor.pth")
    actor = sb3.SAC.load("SAC_l2f_IMU_vel_penalty_further")
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


        self.spike_grad = fast_sigmoid(10)

        self.betas = torch.nn.Parameter(torch.rand(n_spikers))
        self.alphas = torch.nn.Parameter(torch.rand(n_spikers))
        self.thresholds = torch.nn.Parameter(torch.rand(n_spikers))
        self.lif1 = snn.Leaky(beta=self.betas, learn_beta=True,
                            threshold=self.thresholds, learn_threshold=True,
                            spike_grad=self.spike_grad)

        self.syn1 = snn.Synaptic(beta= self.betas, learn_beta=True,
                                 alpha= self.alphas, learn_alpha=True,
                                 threshold=self.thresholds, learn_threshold=True,
                                 spike_grad=self.spike_grad)
        
        
        self.thresholds_integrators = torch.nn.Parameter(torch.rand(n_integrators))
        self.lif_integrators = snn.Leaky(beta=1, learn_beta=False,
                            threshold=self.thresholds_integrators, learn_threshold=True,
                            spike_grad=self.spike_grad)
        
        self.reset()
        
    def set_slope(self, slope):
        self.spike_grad = fast_sigmoid(slope)
        self.lif1.spikegrad = self.spike_grad
        self.lif_integrators.spikegrad = self.spike_grad

    def reset(self):
        self.cur_1 = self.lif1.init_leaky()
        self.cur_syn, self.I_syn = self.syn1.init_synaptic()
        self.cur_int = self.lif_integrators.init_leaky()

    def forward(self, x, hiddens):
        if hiddens is not None:
            self.cur_1, self.I_syn = hiddens[0]
            self.cur_int = hiddens[1]

        # x_1, self.cur_1 = self.lif1(x, self.cur_1)
        x_1, self.cur_syn, self.I_syn = self.syn1(x, self.cur_1, self.I_syn)
        x_int, self.cur_int = self.lif_integrators(x, self.cur_int)

        x = torch.cat((x_1, x_int), dim=1)

        return x, [(self.cur_1, self.I_syn), self.cur_int]
    
class StateEstimator(torch.nn.Module):
    def __init__(self, spiking=True):
        super(StateEstimator, self).__init__()
        self.spiking = spiking

        self.spike_grad = fast_sigmoid(10)

        self.lin_1 = torch.nn.Linear(16, 32*2)
        
        self.leaky = IntegratorSpiker(layer_size=32*2, integrator_ratio=0.5)

        self.lin_2 = torch.nn.Linear(64*2, 6)


        self.reset()
        self.out = torch.zeros(6)
        self.alpha = .999

    def reset(self):
        self.leaky.reset()
        self.out = torch.zeros(6)

    def forward_single(self, x_in: torch.Tensor, hidden=None):
        x = x_in.clone()
        x[:,-6:] = (self.alpha) * self.out + (1-self.alpha) * x_in[:, -6:]
        x = self.lin_1(x)
        x, hidden = self.leaky(x, hidden)
        x = self.lin_2(x)
        self.out = x.detach().clone()
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
        
        run = wandb.init(project='StateEstimator')
        dataset_train, dataset_test = torch.utils.data.random_split(dataset,[0.75,0.25])
        dataloader = DataLoader(dataset_train, batch_size=1028, shuffle=True)
        dataloader_test = DataLoader(dataset_test, batch_size=128, shuffle=True)
        if self.spiking:
            super().train()
        # else:
        #     self.model.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        criterion = torch.nn.MSELoss()

        for epoch in tqdm(range(epochs), desc="Training"):
            losses = []
            if epoch % 10==0:
                # do test
                for imu,true_states in dataloader_test:
                    y_pred = self.forward(imu)
                    # print(y_pred.shape, true_states.shape)
                    # extract intial states from true_states and substract from predicted states
                    # gotta compare to t+1 true state shape B, T, 13
                    loss = criterion(y_pred[:,warmup:-1], true_states[:,warmup+1:,3:9])
                    losses.append(loss.item())
                wandb.log({'loss test': torch.mean(torch.tensor(losses))})
                if epoch/epochs < 0.95:
                
                    self.alpha = 1*((epoch/(epochs*0.95)))
                else:
                    self.alpha = 1
                wandb.log({'alpha': self.alpha})
            
                
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
            wandb.log({'loss train': torch.mean(torch.tensor(losses))})
            print(f"Epoch {epoch}, Loss: {torch.mean(torch.tensor(losses))}")
            if epoch%25==0 or epoch == epochs-1:
                # create one plot with the last model
                self.plot_difference_prediction_true(y_pred[1,:], true_states[1,:,:9], epoch = epoch)
        torch.save(self, "state_estimator.pth")

    def plot_difference_prediction_true(self, prediction, true, epoch=0):
        # print(prediction, true)
        
        # tensor to array

        prediction = prediction.detach().numpy()
        true = true.detach().numpy()
        true_pos = true[:,:3]
        true = true[:,3:]
        fig, ax = plt.subplots(3, 3)
        # in each subplot plot one of prediction[:,:, i] and true[:,:, i]
        ax[0,0].plot(true_pos[:,0], c='g')
        ax[0,0].set_title("x")
        ax[0,1].plot(true_pos[:,1], c='g')
        ax[0,1].set_title("y")
        ax[0,2].plot(true_pos[:,2], c='g')
        ax[0,2].set_title("z")

        subfigs = ["vx", "vy", "vz", "vvx", "vvy", "vvz"]
        for i in range(6):
            ax[i//3+1, i%3].plot(prediction[:,i], c='r')
            ax[i//3+1, i%3].plot(true[:,i], c='g')
            ax[i//3+1, i%3].set_title(subfigs[i])
        # ax[0].plot(prediction)
        # ax[0].set_title("Prediction")
        # ax[1].plot(true)
        # ax[1].set_title("True")
        # Redraw the plot
        # plt.draw()
 
        # # Pause for a short duration to allow visualization
        # plt.pause(0.001)
        # Log the image
        wandb.log({"img": [wandb.Image(fig, caption=f"State Estimator Performance epoch {epoch}")]})
        # plt.savefig('VelocityOrientationTracking.png')
        
if __name__ == "__main__":
    print("Gathering data")
    # dataset = gather_dataset(num_samples=10000, T_max=250)
    # torch.save(dataset, "dataset.pth")
    dataset = torch.load("dataset.pth")
    # print(data.tensors)
    # dataset = TensorDataset(torch.load("dataset.pth"))

    stateestimator = StateEstimator(spiking=True)
    print(stateestimator)
    stateestimator.train(dataset=dataset, epochs=500, warmup = 35)    

    # # train ANN:
    # stateestimator = StateEstimator(spiking=False)
    # print(stateestimator)
    # stateestimator.train(dataset=dataset, epochs=50, warmup = 20)