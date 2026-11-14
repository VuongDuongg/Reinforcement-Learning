"""Shim tương thích ngược: chuyển tiếp đến highway_project.training.plot_results."""
from .training.plot_results import main, plot_dqn_metrics

if __name__ == "__main__":
    main()
