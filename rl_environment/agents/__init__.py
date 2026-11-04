"""Research agent extension point. Original GNN module stays import-stable.

New algorithms live here. Register them explicitly in registry.py; no
untrusted dynamic imports and no automatic pretrained checkpoint loads.
"""
from .registry import create_agent, available_agents, register_agent

__all__ = ["create_agent", "available_agents", "register_agent"]
