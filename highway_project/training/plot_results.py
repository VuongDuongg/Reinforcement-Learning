"""Vẽ đồ thị theo dõi quá trình huấn luyện và đánh giá mô hình DQN."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
DQN_REPORT_PATH = RESULTS_DIR / "dqn_report.json"
DDQN_REPORT_PATH = RESULTS_DIR / "ddqn_report.json"
Q_REPORT_PATH = RESULTS_DIR / "q_learning_report.json"
OUTPUT_PLOT = RESULTS_DIR / "dqn_performance.png"


def plot_dqn_metrics(
    dqn_report_path: Path = DQN_REPORT_PATH,
    q_report_path: Path = Q_REPORT_PATH,
    output_path: Path = OUTPUT_PLOT,
    ddqn_report_path: Path = DDQN_REPORT_PATH,
):
    if not dqn_report_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file báo cáo tại: {dqn_report_path}")

    dqn_data = json.loads(dqn_report_path.read_text(encoding="utf-8"))
    ddqn_data = json.loads(ddqn_report_path.read_text(encoding="utf-8")) if ddqn_report_path.exists() else None
    q_data = json.loads(q_report_path.read_text(encoding="utf-8")) if q_report_path.exists() else None

    history = dqn_data.get("training_history", [])
    dqn_eval = dqn_data.get("evaluation", [])

    if not history:
        print("Lịch sử huấn luyện trống!")
        return

    episodes = [row["episode"] for row in history]
    rewards = [row["reward"] for row in history]
    successes = [row["success"] for row in history]
    epsilons = [row["epsilon"] for row in history]

    # Tính trung bình trượt (rolling mean) window=10
    window = min(10, len(rewards))
    kernel = np.ones(window) / window
    rolling_rewards = np.convolve(rewards, kernel, mode="valid")
    rolling_success = np.convolve(successes, kernel, mode="valid")
    rolling_episodes = episodes[window - 1 :]

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig = plt.figure(figsize=(15, 11))

    # 1. Đồ thị Phần thưởng (Reward)
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(episodes, rewards, color="#94a3b8", alpha=0.35, label="Reward từng Episode")
    ax1.plot(rolling_episodes, rolling_rewards, color="#0284c7", linewidth=2.5, label="Trung bình trượt (10 eps)")
    ax1.axhline(0, color="#ef4444", linestyle="--", alpha=0.5)
    ax1.set_title("1. Tiến trình Phần thưởng DQN (Episode Reward)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Tổng Reward")
    ax1.legend(loc="lower right")

    # 2. Đồ thị Tỷ lệ thành công (Success Rate) & Epsilon
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(rolling_episodes, rolling_success * 100, color="#10b981", linewidth=2.5, label="Tỷ lệ thành công (%)")
    ax2.set_title("2. Tỷ lệ Đến đích & Hệ số Khám phá (Epsilon)", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Tỷ lệ thành công (%)", color="#10b981")
    ax2.set_ylim(-5, 105)

    ax2_twin = ax2.twinx()
    ax2_twin.plot(episodes, epsilons, color="#f59e0b", linestyle=":", linewidth=2, label="Epsilon (khám phá)")
    ax2_twin.set_ylabel("Epsilon", color="#f59e0b")
    ax2_twin.set_ylim(0, 1.05)
    ax2_twin.grid(False)

    lines_1, labels_1 = ax2.get_legend_handles_labels()
    lines_2, labels_2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc="center left")

    # 3. So sánh Benchmark giữa Q-Learning dạng bảng và DQN
    if dqn_eval:
        ax3 = fig.add_subplot(2, 1, 2)
        traffics = [f"Traffic {item['traffic']} xe" for item in dqn_eval]
        dqn_success = [item["after"]["success_rate"] * 100 for item in dqn_eval]
        dqn_collision = [item["after"]["collision_rate"] * 100 for item in dqn_eval]

        x = np.arange(len(traffics))

        if q_data and "evaluation" in q_data and ddqn_data and "evaluation" in ddqn_data:
            q_eval = q_data["evaluation"]
            ddqn_eval = ddqn_data["evaluation"]
            q_success = [row["after"]["success_rate"] * 100 for row in q_eval]
            ddqn_success = [row["after"]["success_rate"] * 100 for row in ddqn_eval]
            width = 0.18

            rects_q = ax3.bar(x - 1.5 * width, q_success, width, label="Q-learning Dạng Bảng (%)", color="#f59e0b")
            rects_dqn = ax3.bar(x - 0.5 * width, dqn_success, width, label="Deep Q-Network (DQN) (%)", color="#10b981")
            rects_ddqn = ax3.bar(x + 0.5 * width, ddqn_success, width, label="Double Dueling DQN (%)", color="#3b82f6")
            rects_col = ax3.bar(x + 1.5 * width, dqn_collision, width, label="Tỷ lệ va chạm DQN (%)", color="#ef4444")

            for rect in rects_q:
                h = rect.get_height()
                ax3.annotate(f"{h:.0f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                             textcoords="offset points", ha="center", va="bottom", fontsize=9)
            for rect in rects_ddqn:
                h = rect.get_height()
                ax3.annotate(f"{h:.0f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                             textcoords="offset points", ha="center", va="bottom", fontweight="bold", color="#1d4ed8", fontsize=10)
        elif q_data and "evaluation" in q_data:
            q_eval = q_data["evaluation"]
            q_success = [row["after"]["success_rate"] * 100 for row in q_eval]
            width = 0.22

            rects_q = ax3.bar(x - width, q_success, width, label="Q-learning Dạng Bảng (%)", color="#f59e0b")
            rects_dqn = ax3.bar(x, dqn_success, width, label="Deep Q-Network (DQN) (%)", color="#10b981")
            rects_col = ax3.bar(x + width, dqn_collision, width, label="Tỷ lệ va chạm DQN (%)", color="#ef4444")

            for rect in rects_q:
                h = rect.get_height()
                ax3.annotate(f"{h:.0f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                             textcoords="offset points", ha="center", va="bottom", fontsize=10)
        else:
            width = 0.28
            rects_dqn = ax3.bar(x - width / 2, dqn_success, width, label="Tỷ lệ thành công DQN (%)", color="#10b981")
            rects_col = ax3.bar(x + width / 2, dqn_collision, width, label="Tỷ lệ va chạm DQN (%)", color="#ef4444")

        ax3.set_ylabel("Tỷ lệ (%)", fontsize=11)
        ax3.set_title("3. So sánh Hiệu năng: Q-Learning Dạng Bảng vs Deep Q-Network (10 episodes độc lập)", fontsize=13, fontweight="bold")
        ax3.set_xticks(x)
        ax3.set_xticklabels(traffics, fontsize=11)
        ax3.set_ylim(0, 115)
        ax3.legend(loc="upper right")

        for rect in rects_dqn:
            h = rect.get_height()
            ax3.annotate(f"{h:.0f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                         textcoords="offset points", ha="center", va="bottom", fontweight="bold", color="#065f46", fontsize=11)
        for rect in rects_col:
            h = rect.get_height()
            ax3.annotate(f"{h:.0f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                         textcoords="offset points", ha="center", va="bottom", color="#b91c1c", fontsize=10)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Đã cập nhật đồ thị so sánh tại: {output_path}")



main = plot_dqn_metrics


if __name__ == "__main__":
    main()

