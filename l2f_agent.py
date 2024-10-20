import torch
import torch.nn as nn
import struct
import numpy as np
# Example byte array
byte_array = [196, 144, 96, 61, 46, 25, 238, 61, 128, 237, 203, 59, 144, 63, 23, 61]

def convert_to_float(byte_array):
    # Convert to bytes (bytearray)
    byte_data = bytearray(byte_array)

    # Iterate over each 4-byte chunk and convert to float
    float_values = [struct.unpack('f', byte_data[i:i+4])[0] for i in range(0, len(byte_data), 4)]
    return float_values

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
        return x.detach().numpy()

# Create the model
model = ConvertedModel()

# Load weights and biases from the actor.txt file where the naming convention is weights0, bias0, weights1, bias1, etc.
# Layer 0
from actor import weights0, bias0, weights1, bias1, weights2, bias2

# Convert byte array to float
weights0 = convert_to_float(weights0)
bias0 = convert_to_float(bias0)
weights1 = convert_to_float(weights1)
bias1 = convert_to_float(bias1)
weights2 = convert_to_float(weights2)
bias2 = convert_to_float(bias2)

# Set the weights and biases
model.layer0.weight.data = torch.tensor(weights0).reshape(64, 146)
model.layer0.bias.data = torch.tensor(bias0).reshape(64)
model.layer1.weight.data = torch.tensor(weights1).reshape(64, 64)
model.layer1.bias.data = torch.tensor(bias1)
model.layer2.weight.data = torch.tensor(weights2).reshape(4, 64)
model.layer2.bias.data = torch.tensor(bias2)

torch.save(model.state_dict(), "l2f_agent.pth")