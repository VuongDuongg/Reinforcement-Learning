"""Huấn luyện Q-learning trên CPU và đánh giá tách seed khỏi tập huấn luyện."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .environment import make_env
from .q_agent import DEFAULT_MODEL, QLearningAgent


def evaluate(agent, *, traffic, seeds, config=None):
    env = make_env(config={"vehicles_count": traffic, **(config or {})})
    episodes = []
    try:
        for seed in seeds:
            obs, info = env.reset(seed=seed)
            agent.reset(seed=seed)
            total, speeds, changes = 0.0, [], 0
            while True:
                action = agent.act(obs, info)
                obs, reward, terminated, truncated, info = env.step(action)
                total += reward
                speeds.append(info["speed"] * 3.6)
                changes += action in (0, 2)
                if terminated or truncated:
                    break
            episodes.append({"seed": seed, "success": info["is_success"], "crashed": bool(info["crashed"]),
                             "distance_m": info["distance_m"], "reward": total,
                             "mean_speed_kmh": float(np.mean(speeds)), "lane_change_requests": changes,
                             "end_reason": info["end_reason"]})
    finally:
        env.close()
    return {"traffic": traffic, "episodes": len(episodes),
            "success_rate": sum(row["success"] for row in episodes) / len(episodes),
            "collision_rate": sum(row["crashed"] for row in episodes) / len(episodes),
            "mean_reward": float(np.mean([row["reward"] for row in episodes])),
            "mean_speed_kmh": float(np.mean([row["mean_speed_kmh"] for row in episodes])),
            "details": episodes}


def main():
    parser = argparse.ArgumentParser(description="Q-learning nhanh, không cần GPU.")
    parser.add_argument("--episodes", type=int, default=180)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    if args.episodes < 1 or args.eval_episodes < 1:
        parser.error("Số episode phải lớn hơn 0")
    start = time.perf_counter()
    agent = QLearningAgent(seed=args.seed)
    logs = []
    if args.evaluate_only:
        agent = QLearningAgent.load(args.output)
    else:
        rng = np.random.default_rng(args.seed)
        env = make_env()
        try:
            for episode in range(args.episodes):
                progress = episode / max(1, args.episodes - 1)
                traffic = 0 if progress < 0.25 else int(rng.choice([3, 6, 12]))
                config = {"vehicles_count": traffic, "obstacle_count": 1,
                          "obstacle_distance": float(rng.uniform(65, 145)),
                          "initial_lane_id": int(rng.integers(0, 3)),
                          "route_length": 260.0, "duration": 25.0}
                env.unwrapped.configure(config)
                # Tất cả seed học nhỏ hơn 1 triệu; seed kiểm tra bắt đầu từ 1 triệu.
                seed = int(rng.integers(0, 1_000_000))
                obs, info = env.reset(seed=seed)
                agent.reset(seed=seed)
                agent.epsilon = max(0.05, 0.8 * (1 - progress) ** 2)
                total = 0.0
                while True:
                    action = agent.act(obs, info)
                    next_obs, reward, terminated, truncated, next_info = env.step(action)
                    agent.observe(obs, action, reward, next_obs, terminated, truncated, next_info)
                    obs, info = next_obs, next_info
                    total += reward
                    if terminated or truncated:
                        break
                logs.append({"episode": episode + 1, "reward": total, "success": info["is_success"],
                             "traffic": traffic, "epsilon": agent.epsilon, "seed": seed})
                if (episode + 1) % 10 == 0 or episode == 0:
                    print(json.dumps({"episode": episode + 1, "steps": agent.training_steps,
                                      "recent_success": float(np.mean([row["success"] for row in logs[-10:]])),
                                      "elapsed_s": round(time.perf_counter() - start, 1)}), flush=True)
            agent.save(args.output, metadata={"episodes": args.episodes, "seed": args.seed,
                                             "training_config": "one obstacle, 260 m, traffic 0/3/6/12"})
        finally:
            env.close()

    agent.epsilon = 0.0
    # Giữ tốc là chính sách của bản trước và cũng là chính sách Q=0 trước khi học.
    untrained = QLearningAgent()
    seeds = list(range(1_000_000, 1_000_000 + args.eval_episodes))
    report = {"algorithm": "tabular_q_learning", "model": str(args.output),
              "training_steps": agent.training_steps,
              "visited_states": int(np.count_nonzero(agent.visits)), "state_count": int(agent.visits.size),
              "training_history": logs, "evaluation": []}
    for traffic in (0, 6, 12):
        before = evaluate(untrained, traffic=traffic, seeds=seeds)
        after = evaluate(agent, traffic=traffic, seeds=seeds)
        report["evaluation"].append({"before": before, "after": after})
        print(json.dumps({"traffic": traffic, "before_success": before["success_rate"],
                          "after_success": after["success_rate"], "after_collision": after["collision_rate"]}), flush=True)
    report["elapsed_s"] = time.perf_counter() - start
    report_path = Path(__file__).parent / "results" / "q_learning_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Model: {args.output}\nReport: {report_path}", flush=True)


if __name__ == "__main__":
    main()
