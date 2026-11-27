from .train_dqn import main as train_dqn_main
from .train_ddqn import main as train_ddqn_main
from .train import main as train_q_main
from .plot_results import main as plot_results_main

__all__ = ["train_dqn_main", "train_ddqn_main", "train_q_main", "plot_results_main"]

