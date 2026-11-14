"""Shim tương thích ngược: chuyển tiếp đến highway_project.training.train_dqn."""
from .training.train_dqn import main, evaluate, train, warmup_replay_buffer

if __name__ == "__main__":
    main()
