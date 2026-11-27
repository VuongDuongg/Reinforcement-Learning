"""Shim tương thích ngược: chuyển tiếp đến highway_project.training.train_ddqn."""
from .training.train_ddqn import main, evaluate, warmup_replay_buffer

if __name__ == "__main__":
    main()
