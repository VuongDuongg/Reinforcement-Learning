"""Shim tương thích ngược: chuyển tiếp đến highway_project.training.train."""
from .training.train import main, evaluate

if __name__ == "__main__":
    main()
