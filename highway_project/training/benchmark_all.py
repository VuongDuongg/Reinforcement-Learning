"""Script Đánh giá Đối sánh Toàn diện (Benchmarking Suite) cho các thuật toán.

Đo lường 4 tiêu chuẩn định lượng cốt lõi:
1. Tỷ lệ va chạm (Collision Rate - %)
2. Vận tốc trung bình (Average Speed - km/h)
3. Phần thưởng tích lũy (Mean Cumulative Reward)
4. Tỷ lệ hoàn thành hành trình / né vật cản thành công (Success Rate - %)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np

# Đảm bảo UTF-8
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_agent(agent_name: str, agent, env, episodes: int = 20, seed_start: int = 100) -> dict:
    """Chạy đánh giá một tác nhân qua N episodes và tính toán các chỉ số tiêu chuẩn."""
    total_rewards = []
    speeds_all_steps = []
    crashes = 0
    successes = 0
    steps_list = []

    for ep in range(episodes):
        ep_seed = seed_start + ep
        obs, info = env.reset(seed=ep_seed)
        agent.reset(seed=ep_seed)

        ep_reward = 0.0
        ep_steps = 0
        crashed = False
        success = False

        while True:
            action = agent.act(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_steps += 1

            speed_kmh = info.get("sensors", {}).get("speed_kmh", env.unwrapped.vehicle.speed * 3.6)
            speeds_all_steps.append(speed_kmh)

            if info.get("crashed", False) or env.unwrapped.vehicle.crashed:
                crashed = True
            if info.get("is_success", False) or (env.unwrapped.vehicle.position[0] >= env.unwrapped.goal_x and not crashed):
                success = True

            if terminated or truncated:
                break

        total_rewards.append(ep_reward)
        steps_list.append(ep_steps)
        if crashed:
            crashes += 1
        if success:
            successes += 1

    collision_rate = (crashes / episodes) * 100.0
    success_rate = (successes / episodes) * 100.0
    avg_speed = float(np.mean(speeds_all_steps)) if speeds_all_steps else 0.0
    mean_reward = float(np.mean(total_rewards))
    std_reward = float(np.std(total_rewards))
    avg_steps = float(np.mean(steps_list))

    return {
        "agent": agent_name,
        "episodes": episodes,
        "collision_rate_percent": round(collision_rate, 2),
        "success_rate_percent": round(success_rate, 2),
        "avg_speed_kmh": round(avg_speed, 2),
        "mean_reward": round(mean_reward, 2),
        "std_reward": round(std_reward, 2),
        "avg_steps": round(avg_steps, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Đánh giá đối sánh Baseline và các thuật toán RL")
    parser.add_argument("--episodes", type=int, default=20, help="Số episodes đánh giá cho mỗi thuật toán (mặc định 20)")
    parser.add_argument("--lanes", type=int, default=3, help="Số làn đường (mặc định 3)")
    parser.add_argument("--traffic", default="normal", help="Mật độ xe (sparse, normal, dense, none)")
    parser.add_argument("--obstacles", type=int, default=1, help="Số vật cản xe hỏng (mặc định 1)")
    parser.add_argument("--algorithms", nargs="+", default=["baseline", "dqn", "ddqn", "q_learning"],
                        help="Danh sách thuật toán cần đánh giá")
    args = parser.parse_args()

    os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ["SDL_AUDIODRIVER"] = "dummy"

    from highway_project.environment.environment import make_env
    from highway_project.algorithms.baseline_agent import RuleBasedAgent
    from highway_project.algorithms.dqn_agent import DQNAgent, DEFAULT_MODEL as DEFAULT_DQN_MODEL
    from highway_project.algorithms.ddqn_agent import DoubleDuelingDQNAgent, DEFAULT_MODEL as DEFAULT_DDQN_MODEL
    from highway_project.algorithms.q_agent import QLearningAgent, DEFAULT_MODEL as DEFAULT_Q_MODEL

    env = make_env(
        seed=42,
        config={
            "lanes_count": args.lanes,
            "vehicles_count": args.traffic,
            "obstacle_count": args.obstacles,
        },
    )

    print("=" * 86)
    print("      HỆ THỐNG TIÊU CHUẨN ĐÁNH GIÁ ĐỐI SÁNH (BENCHMARKING & EVALUATION)")
    print(f"      Số episode test: {args.episodes} | Số làn: {args.lanes} | Mật độ xe: {args.traffic} | Vật cản: {args.obstacles}")
    print("=" * 86)

    results = {}

    for algo in args.algorithms:
        algo_lower = algo.lower()
        print(f"\n[+] Đang chạy kiểm thử đối sánh: {algo_lower.upper()}...")
        try:
            if algo_lower in ("baseline", "rule", "rule_based"):
                agent = RuleBasedAgent()
                display_name = "Rule-based Baseline"
            elif algo_lower == "dqn":
                if DEFAULT_DQN_MODEL.exists():
                    agent = DQNAgent.load(DEFAULT_DQN_MODEL)
                    agent.epsilon = 0.0
                    display_name = "DQN (Đã huấn luyện)"
                else:
                    agent = DQNAgent(epsilon=0.0)
                    display_name = "DQN (Chưa huấn luyện)"
            elif algo_lower == "ddqn":
                if DEFAULT_DDQN_MODEL.exists():
                    agent = DoubleDuelingDQNAgent.load(DEFAULT_DDQN_MODEL)
                    agent.epsilon = 0.0
                    display_name = "Double Dueling DQN (Đã train)"
                else:
                    agent = DoubleDuelingDQNAgent(epsilon=0.0)
                    display_name = "Double Dueling DQN (Chưa train)"
            elif algo_lower in ("q_learning", "q"):
                if DEFAULT_Q_MODEL.exists():
                    agent = QLearningAgent.load(DEFAULT_Q_MODEL)
                    agent.epsilon = 0.0
                    display_name = "Q-Learning (Đã train)"
                else:
                    agent = QLearningAgent(epsilon=0.0)
                    display_name = "Q-Learning (Chưa train)"
            else:
                print(f"Bỏ qua thuật toán không xác định: {algo}")
                continue

            metrics = evaluate_agent(display_name, agent, env, episodes=args.episodes)
            results[algo_lower] = metrics
            print(f"    ✓ Tỷ lệ va chạm: {metrics['collision_rate_percent']}% | Tốc độ TB: {metrics['avg_speed_kmh']} km/h | Thưởng TB: {metrics['mean_reward']}")

        except Exception as e:
            print(f"    ✗ Lỗi khi đánh giá {algo}: {e}")

    env.close()

    # BẢNG TỔNG HỢP SO SÁNH TIÊU CHUẨN
    print("\n" + "=" * 86)
    print("                    BẢNG ĐỐI SÁNH HIỆU NĂNG CÁC THUẬT TOÁN")
    print("=" * 86)
    header = f"{'Thuật toán':<28} | {'Tỉ lệ va chạm':<14} | {'Tốc độ TB':<12} | {'Thành công':<12} | {'Phần thưởng TB':<14}"
    print(header)
    print("-" * 86)

    for key, data in results.items():
        row = (
            f"{data['agent']:<28} | "
            f"{str(data['collision_rate_percent']) + '%':<14} | "
            f"{str(data['avg_speed_kmh']) + ' km/h':<12} | "
            f"{str(data['success_rate_percent']) + '%':<12} | "
            f"{str(data['mean_reward']) + ' (±' + str(data['std_reward']) + ')':<14}"
        )
        print(row)
    print("=" * 86)

    # Lưu file kết quả JSON
    comparison_file = RESULTS_DIR / "benchmark_comparison.json"
    comparison_file.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[✓] Đã lưu kết quả đối sánh tại: {comparison_file}")

    if "baseline" in results:
        baseline_file = RESULTS_DIR / "baseline_report.json"
        baseline_file.write_text(json.dumps(results["baseline"], indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[✓] Đã lưu báo cáo Baseline riêng tại: {baseline_file}")


if __name__ == "__main__":
    main()
