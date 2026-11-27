
"""Huấn luyện Deep Q-Network (DQN 2013 - PyTorch) với Replay Warmup, Curriculum Learning và Best Checkpoint."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ..algorithms.dqn_agent import DEFAULT_MODEL, DQNAgent
from ..algorithms.q_agent import QLearningAgent
from ..environment.environment import make_env


def warmup_replay_buffer(agent: DQNAgent, env, num_episodes: int = 35) -> int:
    """Chỉ nạp kinh nghiệm từ Q-learning vào Replay Buffer (không gọi learn sớm).

    Trả về số mẫu đã nạp (0 nếu chưa có file q_learning.npz).
    """
    try:
        teacher = QLearningAgent.load()
    except FileNotFoundError:
        print(
            "Không tìm thấy checkpoint Q-learning để warmup; bỏ qua bước warmup teacher.",
            flush=True,
        )
        return 0

    teacher.epsilon = 0.05
    rng = np.random.default_rng(1234)
    print(f"Khởi động Replay Buffer với {num_episodes} episodes kinh nghiệm (chỉ nạp, chưa train)...", flush=True)

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
            agent.push_experience(
                obs,
                action,
                reward,
                next_obs,
                terminated,
                truncated,
                info=info,
                next_info=next_info,
            )
            obs, info = next_obs, next_info
            if terminated or truncated:
                break

    print(f"Đã nạp {len(agent.buffer)} mẫu vào Replay Buffer.", flush=True)
    return len(agent.buffer)


def evaluate(agent: DQNAgent, *, traffic: int, seeds: list[int], config: dict | None = None) -> dict:
    """Đánh giá hiệu năng của agent trên tập seeds xác định."""
    env = make_env(config={"vehicles_count": traffic, **(config or {})})
    episodes = []
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0  # Hoàn toàn khai thác khi đánh giá

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
            episodes.append(
                {
                    "seed": seed,
                    "success": bool(info["is_success"]),
                    "crashed": bool(info["crashed"]),
                    "distance_m": float(info["distance_m"]),
                    "reward": float(total_reward),
                    "mean_speed_kmh": float(np.mean(speeds)) if speeds else 0.0,
                    "lane_change_requests": int(lane_changes),
                    "end_reason": str(info["end_reason"]),
                }
            )
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
    parser = argparse.ArgumentParser(description="Huấn luyện Deep Q-Network (DQN 2013 - PyTorch).")
    parser.add_argument("--episodes", type=int, default=200, help="Số episode huấn luyện")
    parser.add_argument("--seed", type=int, default=2026, help="Hạt giống ngẫu nhiên")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate cho Adam")
    parser.add_argument("--learn-freq", type=int, default=2, help="Số bước môi trường giữa 2 lần gọi learn()")
    parser.add_argument("--min-buffer-size", type=int, default=1000, help="Số mẫu tối thiểu trước khi bắt đầu học")
    parser.add_argument(
        "--reward-clip",
        type=float,
        default=0.0,
        help="Ngưỡng cắt phần thưởng [-r, r] (mặc định 0 = tắt để giữ nguyên tín hiệu phạt va chạm)",
    )
    parser.add_argument(
        "--no-safety-shield",
        action="store_true",
        help="Tắt lớp an toàn trong act() để huấn luyện & đánh giá thuần DQN 2013",
    )
    parser.add_argument("--eval-freq", type=int, default=10, help="Chu kỳ (episode) đánh giá validation và lưu best model")
    parser.add_argument("--eval-episodes", type=int, default=10, help="Số episode đánh giá mỗi mức traffic")
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL, help="Nơi lưu mô hình .pth")
    parser.add_argument("--evaluate-only", action="store_true", help="Chỉ đánh giá mô hình đã có")
    parser.add_argument("--device", type=str, default="cpu", help="Thiết bị tính toán ('cpu' hoặc 'cuda')")
    args = parser.parse_args()

    if args.episodes < 1 or args.eval_episodes < 1:
        parser.error("Số episode phải lớn hơn 0")

    use_safety_shield = not args.no_safety_shield
    reward_clip = args.reward_clip if args.reward_clip > 0 else None
    start_time = time.perf_counter()
    logs = []
    best_episode = None
    best_score = -float("inf")
    best_val_success = 0.0

    if args.evaluate_only:
        agent = DQNAgent.load(args.output, device=args.device, use_safety_shield=use_safety_shield)
        print(f"Nạp mô hình từ {args.output} để đánh giá (safety_shield={use_safety_shield})...", flush=True)
    else:
        agent = DQNAgent(
            seed=args.seed,
            device=args.device,
            lr=args.lr,
            learn_freq=args.learn_freq,
            min_buffer_size=args.min_buffer_size,
            reward_clip=reward_clip,
            use_safety_shield=use_safety_shield,
        )
        rng = np.random.default_rng(args.seed)
        env = make_env()

        # Tập validation cố định cho cả 3 mức traffic (0, 6, 12), tách biệt với eval_seeds
        val_seeds = list(range(500_000, 500_004))

        # Nạp kinh nghiệm ban đầu vào buffer mà chưa cập nhật trọng số
        warmed_samples = warmup_replay_buffer(agent, env, num_episodes=40)
        if warmed_samples == 0 and agent.min_buffer_size > 256:
            agent.min_buffer_size = max(agent.batch_size, 256)
            print(
                f"Không có dữ liệu warmup -> tự giảm min_buffer_size xuống {agent.min_buffer_size} để học sớm hơn.",
                flush=True,
            )

        print(
            f"Bắt đầu huấn luyện DQN 2013 ({args.episodes} episodes, lr={args.lr}, "
            f"learn_freq={args.learn_freq}, min_buffer={agent.min_buffer_size}, "
            f"reward_clip={reward_clip}, safety_shield={use_safety_shield}) trên {agent.device}...",
            flush=True,
        )

        try:
            for episode in range(args.episodes):
                progress = episode / max(1, args.episodes - 1)
                # Curriculum: 25% đầu chạy traffic=0, sau đó giữ lại cả 0 trong [0, 3, 6, 12] để tránh quên đường vắng
                traffic = 0 if progress < 0.25 else int(rng.choice([0, 3, 6, 12]))
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

                # Epsilon decay
                agent.epsilon = max(0.05, 0.8 * (1.0 - progress) ** 1.8)

                total_reward = 0.0
                ep_losses: list[float] = []

                while True:
                    action = agent.act(obs, info)
                    next_obs, reward, terminated, truncated, next_info = env.step(action)
                    loss = agent.observe(
                        obs,
                        action,
                        reward,
                        next_obs,
                        terminated,
                        truncated,
                        info=info,
                        next_info=next_info,
                    )
                    if loss is not None:
                        ep_losses.append(loss)

                    obs, info = next_obs, next_info
                    total_reward += reward

                    if terminated or truncated:
                        break

                mean_loss = float(np.mean(ep_losses)) if ep_losses else 0.0
                logs.append(
                    {
                        "episode": episode + 1,
                        "reward": float(total_reward),
                        "success": bool(info["is_success"]),
                        "traffic": traffic,
                        "epsilon": float(agent.epsilon),
                        "mean_loss": mean_loss,
                        "mean_q": float(agent.last_q_mean),
                        "seed": seed,
                    }
                )

                should_eval = ((episode + 1) % args.eval_freq == 0) or (episode + 1 == args.episodes)
                if should_eval:
                    # Chấm validation đều trên cả 3 mức giao thông (0, 6, 12)
                    val_results = [evaluate(agent, traffic=t, seeds=val_seeds) for t in (0, 6, 12)]
                    val_success = float(np.mean([r["success_rate"] for r in val_results]))
                    val_reward = float(np.mean([r["mean_reward"] for r in val_results]))
                    val_by_traffic = [round(r["success_rate"], 2) for r in val_results]
                    val_score = val_success * 100.0 + val_reward

                    is_best = val_score > best_score
                    if is_best:
                        best_score = val_score
                        best_episode = episode + 1
                        best_val_success = val_success
                        agent.save(
                            args.output,
                            metadata={
                                "episodes": args.episodes,
                                "best_episode": best_episode,
                                "val_success": val_success,
                                "val_reward": val_reward,
                                "val_by_traffic": {
                                    "0": val_results[0]["success_rate"],
                                    "6": val_results[1]["success_rate"],
                                    "12": val_results[2]["success_rate"],
                                },
                                "seed": args.seed,
                                "training_config": "Curriculum: 3 lanes, traffic [0,3,6,12], 1 obstacle, route 260m",
                            },
                        )

                    recent_success = float(np.mean([row["success"] for row in logs[-10:]]))
                    recent_reward = float(np.mean([row["reward"] for row in logs[-10:]]))
                    recent_loss = float(np.mean([row["mean_loss"] for row in logs[-10:]]))
                    print(
                        json.dumps(
                            {
                                "episode": episode + 1,
                                "steps": agent.training_steps,
                                "recent_success": round(recent_success, 3),
                                "recent_reward": round(recent_reward, 2),
                                "recent_loss": round(recent_loss, 4),
                                "mean_q": round(agent.last_q_mean, 3),
                                "val_success": round(val_success, 3),
                                "val_0_6_12": val_by_traffic,
                                "val_reward": round(val_reward, 2),
                                "best_saved": is_best,
                                "epsilon": round(agent.epsilon, 3),
                                "elapsed_s": round(time.perf_counter() - start_time, 1),
                            }
                        ),
                        flush=True,
                    )
                elif episode == 0:
                    print(
                        json.dumps(
                            {
                                "episode": 1,
                                "steps": agent.training_steps,
                                "recent_success": float(info["is_success"]),
                                "recent_reward": round(total_reward, 2),
                                "epsilon": round(agent.epsilon, 3),
                                "elapsed_s": round(time.perf_counter() - start_time, 1),
                            }
                        ),
                        flush=True,
                    )

            print(
                f"Hoàn tất huấn luyện. Best checkpoint ở episode {best_episode} "
                f"(val_success={best_val_success:.3f}) đã lưu tại: {args.output}",
                flush=True,
            )

        finally:
            env.close()

        # Nạp lại đúng Best Checkpoint để đánh giá công bằng
        agent = DQNAgent.load(args.output, device=args.device, use_safety_shield=use_safety_shield)

    # Đánh giá sau huấn luyện (cùng cấu hình safety_shield giữa before và after)
    agent.epsilon = 0.0
    untrained = DQNAgent(
        seed=999,
        device=args.device,
        epsilon=0.0,
        use_safety_shield=use_safety_shield,
    )
    eval_seeds = list(range(1_000_000, 1_000_000 + args.eval_episodes))

    report = {
        "algorithm": "dqn_2013_pytorch",
        "model": str(args.output),
        "best_episode": best_episode,
        "use_safety_shield": use_safety_shield,
        "training_steps": agent.training_steps,
        "buffer_size": len(agent.buffer),
        "training_history": logs,
        "evaluation": [],
    }

    print("\n--- ĐÁNH GIÁ MÔ HÌNH TRÊN CÁC MỨC GIAO THÔNG ---", flush=True)
    for traffic in (0, 6, 12):
        before = evaluate(untrained, traffic=traffic, seeds=eval_seeds)
        after = evaluate(agent, traffic=traffic, seeds=eval_seeds)
        report["evaluation"].append({"traffic": traffic, "before": before, "after": after})
        print(
            json.dumps(
                {
                    "traffic": traffic,
                    "before_success": before["success_rate"],
                    "after_success": after["success_rate"],
                    "after_collision": after["collision_rate"],
                    "after_mean_speed": round(after["mean_speed_kmh"], 1),
                }
            ),
            flush=True,
        )

    report["elapsed_s"] = time.perf_counter() - start_time
    report_path = Path(__file__).resolve().parent.parent / "results" / "dqn_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nBáo cáo chi tiết đã lưu tại: {report_path}", flush=True)


if __name__ == "__main__":
    main()
