# src/leadership/__init__.py
# Leadership module for the Love‑Swarm Nullification Enhanced framework.
# Exposes LeaderSelector and the emergence experiment.
from .leader import LeaderSelector
from .leader_emergence import run_leader_emergence_experiment

__all__ = ["LeaderSelector", "run_leader_emergence_experiment"]