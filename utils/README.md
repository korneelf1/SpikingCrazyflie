# Utilities for SpikingCrazyflie

This folder contains utility scripts for the SpikingCrazyflie project.

## Buffer Collector

The `buffer_collector.py` script provides utilities to collect and save replay buffers using the l2f_actor. These buffers can be used for offline reinforcement learning algorithms like BC or TD3BC.

### Usage

#### Command Line Interface

You can use the buffer collector from the command line:

```bash
python buffer_collector.py --size 5000 --rollout-len 501 --sequence-length 100 --sequence-stride 50 --min-valid-length 100 --filename my_buffer.hdf5
```

Command line arguments:

- `--size`: Size of the buffer to collect (default: 1000)
- `--rollout-len`: Length of each rollout (default: 501)
- `--warmup`: Number of warmup steps before using the controller (default: 50)
- `--sequence-length`: Length of each sequence to add to the buffer (default: 100)
- `--sequence-stride`: Stride between consecutive sequences (default: 50)
- `--min-valid-length`: Minimum length of a valid rollout (default: 100)
- `--filename`: Filename to save the buffer to (default: timestamp-based name)
- `--device`: Device to use for tensor operations (default: "cpu")
- `--use-wandb`: Whether to log data to wandb (flag)
- `--project-name`: Name of the wandb project (default: "buffer_collection")

#### Python API

You can also use the buffer collector from Python:

```python
from buffer_collector import BufferCollector

# Create buffer collector
collector = BufferCollector(
    device="cuda",  # or "cpu" or "mps"
    use_wandb=False  # Set to True if you want to log to wandb
)

# Collect buffer
buffer = collector.gather_buffer(
    size=5000,
    rollout_len=501,
    sequence_length=100,
    sequence_stride=50,
    min_valid_length=100,
    verbose=True
)

# Save buffer
filename = collector.save_buffer(buffer, "my_buffer.hdf5")

# Or collect and save in one operation
filename = collector.collect_and_save(
    size=5000,
    rollout_len=501,
    sequence_length=100,
    sequence_stride=50,
    min_valid_length=100,
    filename="my_buffer.hdf5",
    verbose=True
)
```

### Using the Buffer with BC or TD3BC

Once you have collected a buffer, you can use it with BC or TD3BC:

```python
from tianshou.data import ReplayBuffer
from BC import BehavioralCloning  # Import your BC implementation

# Load the buffer
buffer = ReplayBuffer.load_hdf5("my_buffer.hdf5")

# Use with BC
bc = BehavioralCloning(buffer=buffer, ...)
bc.learn()

# Or use with TD3BC
from TD3BC import TD3BC  # Import your TD3BC implementation
td3bc = TD3BC(buffer=buffer, ...)
td3bc.learn()
```

### Example

See `collect_buffer_example.py` for a complete example of how to use the buffer collector.

## Other Utilities

(Add descriptions of other utilities as they are added to this folder) 