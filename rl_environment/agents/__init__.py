"""Research agent extension point. Original GNN module stays import-stable.

Historical curriculum agents remain registered in registry.py. Hardware-aware
GNN and audited PPO/REINFORCE implementations live in hardware_agent.py and
hardware_algorithms.py, with their own source-defined method factory. No
untrusted dynamic imports or pretrained checkpoint loading.
"""
from .registry import create_agent, available_agents, register_agent

__all__ = ["create_agent", "available_agents", "register_agent"]
