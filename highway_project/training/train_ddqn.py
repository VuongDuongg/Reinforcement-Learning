"""Huấn luyện Double Dueling DQN (PyTorch) — Thuật toán 2 (Ngô Tuấn Anh)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ..algorithms.ddqn_agent import DEFAULT_MODEL, DoubleDuelingDQNAgent
from ..algorithms.q_agent import QLearningAgent
from ..environment.environment import make_env


def warmup_replay_buffer(agent: DoubleDuelingDQNAgent, env, num_episodes: int = 35) -> None:
    """Nạp các kinh nghiệm né tránh thành công từ Q-learning vào Replay Buffer của DDQN."""
    try:
        teacher = QLearningAgent.load()
    except FileNotFoundError:
        return

    teacher.epsilon = 0.05
    rng = np.random.default_rng(1234)
    print(f"Khởi động Replay Buffer cho DDQN với {num_episodes} episodes kinh nghiệm...", flush=True)

    for ep in range(num_episodes):
        lane_id = ep % 3
        traffic = int(rng.choice([0, 3, 6]))
        config = {
            "vehicles_count": traffic,
            "obstacle_count": 1,
            "obstacle_distance": float(rng.uniform(65, 120)),
            "initial_lane_id": lane_id,
            "route_length": 260.0,
            "duration": 25.0,
            "random_obstacles": False,
        }
        env.unwrapped.configure(config)
        obs, info = env.reset(seed=int(rng.integers(0, 1_000_000)))
        teacher.reset(seed=42)

        while True:
            action = teacher.act(obs, info)
            next_obs, reward, terminated, truncated, next_info = env.step(action)
            agent.observe(obs, action, reward, next_obs, terminated, truncated, info, next_info)
            obs, info = next_obs, next_info
            if terminated or truncated:
                break


def evaluate(agent: DoubleDuelingDQNAgent, *, traffic: int, seeds: list[int], config: dict | None = None) -> dict:
    """Đánh giá hiệu năng của Double Dueling DQN trên tập seeds xác định."""
    env = make_env(config={"vehicles_count": traffic, **(config or {})})
    episodes = []
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0

    try:
        for seed in seeds:
            obs, info = env.reset(seed=seed)
            agent.reset(seed=seed)
            total_reward, speeds, lane_changes = 0.0, [], 0
            while True:
                action = agent.act(obs, info)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                speeds.append(info["speed"] * 3.6)
                lane_changes += int(action in (0, 2))
                if terminated or truncated:
                    break
            episodes.append({
                "seed": seed,
                "success": bool(info["is_success"]),
                "crashed": bool(info["crashed"]),
                "distance_m": float(info["distance_m"]),
                "reward": float(total_reward),
                "mean_speed_kmh": float(np.mean(speeds)) if speeds else 0.0,
                "lane_change_requests": int(lane_changes),
                "end_reason": str(info["end_reason"]),
            })
    finally:
        env.close()
        agent.epsilon = old_epsilon

    return {
        "traffic": traffic,
        "episodes": len(episodes),
        "success_rate": float(sum(row["success"] for row in episodes) / len(episodes)),
        "collision_rate": float(sum(row["crashed"] for row in episodes) / len(episodes)),
        "mean_reward": float(np.mean([row["reward"] for row in episodes])),
        "mean_speed_kmh": float(np.mean([row["mean_speed_kmh"] for row in episodes])),
        "details": episodes,
    }


def main():
    parser = argparse.ArgumentParser(description="Huấn luyện Double Dueling DQN (PyTorch - Ngô Tuấn Anh).")
    parser.add_argument("--episodes", type=int, default=160, help="Số episode huấn luyện")
    parser.add_argument("--seed", type=int, default=2026, help="Hạt giống ngẫu nhiên")
    parser.add_argument("--eval-episodes", type=int, default=10, help="Số episode đánh giá mỗi mức traffic")
    parser.add_argument("--tau", type=float, default=0.005, help="Hệ số cập nhật mềm Soft Update Target Network")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate cho Adam optimizer")
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL, help="Nơi lưu mô hình .pth")
    parser.add_argument("--evaluate-only", action="store_true", help="Chỉ đánh giá mô hình đã có")
    parser.add_argument("--device", type=str, default="cpu", help="Thiết bị tính toán ('cpu' hoặc 'cuda')")
    args = parser.parse_args()

    if args.episodes < 1 or args.eval_episodes < 1:
        parser.error("Số episode phải lớn hơn 0")

    start_time = time.perf_counter()
    logs = []

    if args.evaluate_only:
        agent = DoubleDuelingDQNAgent.load(args.output, device=args.device)
        print(f"Nạp mô hình Double Dueling DQN từ {args.output} để đánh giá...", flush=True)
    else:
        agent = DoubleDuelingDQNAgent(seed=args.seed, lr=args.lr, tau=args.tau, device=args.device)
        rng = np.random.default_rng(args.seed)
        env = make_env()

        warmup_replay_buffer(agent, env, num_episodes=40)

        print(
            f"Bắt đầu huấn luyện Double Dueling DQN ({args.episodes} episodes, tau={args.tau}) trên {agent.device}...",
            flush=True,
        )

        try:
            for episode in range(args.episodes):
                progress = episode / max(1, args.episodes - 1)
                traffic = 0 if progress < 0.25 else int(rng.choice([3, 6, 12]))
                initial_lane = int(episode % 3)
                config = {
                    "vehicles_count": traffic,
                    "obstacle_count": 1,
                    "obstacle_distance": float(rng.uniform(65, 140)),
                    "initial_lane_id": initial_lane,
                    "route_length": 260.0,
                    "duration": 25.0,
                    "random_obstacles": False,
                }
                env.unwrapped.configure(config)

                seed = int(rng.integers(0, 1_000_000))
                obs, info = env.reset(seed=seed)
                agent.reset(seed=seed)

                agent.epsilon = max(0.05, 0.8 * (1.0 - progress) ** 1.8)
                total_reward = 0.0

                while True:
                    action = agent.act(obs, info)
                    next_obs, reward, terminated, truncated, next_info = env.step(action)
                    agent.observe(obs, action, reward, next_obs, terminated, truncated, info, next_info)

                    obs, info = next_obs, next_info
                    total_reward += reward

                    if terminated or truncated:
                        break

                logs.append({
                    "episode": episode + 1,
                    "reward": float(total_reward),
                    "success": bool(info["is_success"]),
                    "traffic": traffic,
                    "epsilon": float(agent.epsilon),
                    "seed": seed,
                })

                if (episode + 1) % 10 == 0 or episode == 0:
                    recent_success = float(np.mean([row["success"] for row in logs[-10:]]))
                    recent_reward = float(np.mean([row["reward"] for row in logs[-10:]]))
                    print(
                        json.dumps({
                            "episode": episode + 1,
                            "steps": agent.training_steps,
                            "recent_success": round(recent_success, 3),
                            "recent_reward": round(recent_reward, 2),
                            "epsilon": round(agent.epsilon, 3),
                            "elapsed_s": round(time.perf_counter() - start_time, 1),
                        }),
                        flush=True,
                    )

            agent.save(
                args.output,
                metadata={
                    "episodes": args.episodes,
                    "seed": args.seed,
                    "training_config": "Curriculum: 3 lanes, 0->12 traffic, 1 obstacle, route 260m",
                },
            )
            print(f"Đã lưu mô hình Double Dueling DQN tại: {args.output}", flush=True)

        finally:
            env.close()

    agent.epsilon = 0.0
    untrained = DoubleDuelingDQNAgent(seed=999, device=args.device, epsilon=0.0)
    eval_seeds = list(range(1_000_000, 1_000_000 + args.eval_episodes))

    report = {
        "algorithm": "double_dueling_dqn_pytorch",
        "author": "Ngo Tuan Anh (2351170570)",
        "model": str(args.output),
        "training_steps": agent.training_steps,
        "buffer_size": len(agent.buffer),
        "tau": agent.tau,
        "training_history": logs,
        "evaluation": [],
    }

    print("\n--- ĐÁNH GIÁ DOUBLE DUELING DQN TRÊN CÁC MỨC GIAO THÔNG ---", flush=True)
    for traffic in (0, 6, 12):
        before = evaluate(untrained, traffic=traffic, seeds=eval_seeds)
        after = evaluate(agent, traffic=traffic, seeds=eval_seeds)
        report["evaluation"].append({"traffic": traffic, "before": before, "after": after})
        print(
            json.dumps({
                "traffic": traffic,
                "before_success": before["success_rate"],
                "after_success": after["success_rate"],
                "after_collision": after["collision_rate"],
                "after_mean_speed": round(after["mean_speed_kmh"], 1),
            }),
            flush=True,
        )

    report["elapsed_s"] = time.perf_counter() - start_time
    report_path = Path(__file__).resolve().parent.parent / "results" / "ddqn_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nBáo cáo chi tiết đã lưu tại: {report_path}", flush=True)


if __name__ == "__main__":
    main()
