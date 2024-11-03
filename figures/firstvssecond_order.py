import snntorch as snn
import matplotlib.pyplot as plt
import numpy as np
import torch

from snntorch.spikeplot import traces
# compare a Leaky and Synaptic neuron on membrane potential and outgoing spike
def compare_neurons():
    # Create a Leaky neuron
    leaky = snn.Leaky(beta=0.95, threshold=1.0)
    # Create a Synaptic neuron
    synaptic = snn.Synaptic(alpha=0.85, beta=0.95, threshold=1.0)

    # Create a time vector
    time = torch.arange(0, 20, 0.1)
    # Create a step function
    step = torch.zeros_like(time)
    step[time % 5 == 0] = .4
    cur_leaky = leaky.init_leaky()
    cur_synaptic, mem_syn = synaptic.init_synaptic()
    v_leaky = torch.zeros_like(time)
    leaky_spike = torch.zeros_like(time)
    v_synaptic = torch.zeros_like(time)
    synaptic_spike = torch.zeros_like(time)
    for t,i  in enumerate(step):
        out, cur_leaky = leaky(i, cur_leaky)
        v_leaky[t]= cur_leaky
        leaky_spike[t] = out
        out, cur_synaptic, mem_syn = synaptic(i, cur_synaptic,mem_syn)
        v_synaptic[t] = mem_syn
        synaptic_spike[t] = out

    # traces(v_leaky.reshape(-1,1), leaky_spike.reshape(-1,1))
    v_leaky = v_leaky.detach().numpy()
    leaky_spike = leaky_spike.detach().numpy()
    v_synaptic = v_synaptic.detach().numpy()
    synaptic_spike = synaptic_spike.detach().numpy()
    
    # plt.show()
    # Plot the membrane potential and spike for the Leaky neuron
    plt.figure()
    plt.plot(time, v_leaky, label="Leaky Membrane Potential")
    # draw a vertical line where leaky_spike is 1
    for t in range(len(leaky_spike)):
        if leaky_spike[t] == 1:
            plt.axvline(x=time[t], color='r', linestyle='--', label="Leaky Spike")
    
    # plt.scatter(time, leaky_spike, label="Leaky Spike")
    plt.legend()
    plt.title("Leaky Neuron")
    plt.xlabel("Time (ms)")
    plt.ylabel("Membrane Potential")

    # Plot the membrane potential and spike for the Synaptic neuron
    plt.figure()
    plt.plot(time, v_synaptic, label="Synaptic Membrane Potential")
    # plt.plot(time, synaptic_spike, label="Synaptic Spike")
    for t in range(len(synaptic_spike)):
        if synaptic_spike[t] == 1:
            plt.axvline(x=time[t], color='r', linestyle='--', label="Output Spike")
    plt.plot(time, step, label="Input Current")
    plt.legend()
    plt.title("Synaptic Neuron")
    plt.xlabel("Time (ms)")
    plt.ylabel("Membrane Potential")

    plt.show()

compare_neurons()