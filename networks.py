import torch


class Actor_ANN(torch.nn.Module):
    def __init__(self, state_dim, action_dim, max_action=1):
        super(Actor_ANN, self).__init__()

        self.layer_1 = torch.nn.Linear(state_dim, 400)
        self.layer_2 = torch.nn.Linear(400, 300)
        self.layer_3 = torch.nn.Linear(300, action_dim)

        self.max_action = max_action

    def forward(self, state):
        state =  torch.from_numpy(state).float()
        a = torch.nn.ReLU()(self.layer_1(state))
        a = torch.nn.ReLU()(self.layer_2(a))
        return self.max_action * torch.nn.Sigmoid()(self.layer_3(a))
    
class Critic_ANN(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic_ANN, self).__init__()

        # Q1 architecture
        self.layer_1 = torch.nn.Linear(state_dim + action_dim, 400)
        self.layer_2 = torch.nn.Linear(400, 300)
        self.layer_3 = torch.nn.Linear(300, 1)

        # Q2 architecture
        self.layer_4 = torch.nn.Linear(state_dim + action_dim, 400)
        self.layer_5 = torch.nn.Linear(400, 300)
        self.layer_6 = torch.nn.Linear(300, 1)