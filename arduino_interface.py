import numpy as np
# from network_controller import init_network, reset_network, py_forward_network, set_network_input
from network_controller import NetworkController_Korneel

# Initialize the network
net = NetworkController_Korneel(in_size=9, hid1_size=256, hid2_size=128, out_size=4)
net.load_network_from_header()
# load from header
net.init_network()