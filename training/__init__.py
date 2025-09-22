# Training module for SpikingCrazyflie
from .bc import BC
from .td3bc import TD3BC
from .td3bc_online import TD3BC_Online

__all__ = ['BC', 'TD3BC', 'TD3BC_Online']
