"""Chạy bằng: python -m highway_project.run"""

from __future__ import annotations

import argparse
import json
import os


def parse_traffic(value: str) -> str:
    val = value.strip().lower()
    mapping = {
        "sparse": "sparse", "thua": "sparse", "thua_thot": "sparse",
        "normal": "normal", "vua": "normal", "binh_thuong": "normal",
        "dense": "dense", "day": "dense", "day_dac": "dense",
        "none": "none", "khong": "none",
    }
    if val in mapping:
        return mapping[val]
    try:
        num = int(val)
        if 0 <= num <= 60:
            return str(num)
    except ValueError:
        pass
    raise argparse.ArgumentTypeError(
        f"Mật độ giao thông không hợp lệ: '{value}'. Chọn: sparse (thưa), normal (bình thường), dense (dày đặc), none hoặc số 0..60."
    )


def main():
    parser = argparse.ArgumentParser(description="Môi trường xe né vật cản và Q-learning / Deep Q-Network.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lanes", type=int, choices=range(2, 7), default=3, metavar="2..6", help="Số làn đường cao tốc (2..6, mặc định 3)")
    parser.add_argument(
        "--traffic",
        type=parse_traffic,
        default="normal",
        help="Mật độ xe giao thông: sparse (thưa thớt: 8 xe), normal (bình thường: 20 xe), dense (dày đặc: 38 xe), none hoặc số 0..60",
    )
    parser.add_argument("--speed", type=float, default=72.0, help="Vận tốc xuất phát xe của bạn theo km/h (mặc định 72 km/h)")
    parser.add_argument("--max-speed", type=float, default=108.0, help="Vận tốc tối đa cho phép theo km/h (mặc định 108 km/h)")
    parser.add_argument(
        "--traffic-speed",
        choices=["slow", "normal", "fast"],
        default="normal",
        help="Dải tốc độ dòng xe giao thông (slow: 36-65 km/h, normal: 50-95 km/h, fast: 75-115 km/h)",
    )
    parser.add_argument("--obstacles", type=int, default=2, choices=range(0, 4), help="Số xe hỏng chắn đường (mặc định 2)")
    parser.add_argument("--fixed-obstacles", action="store_true", help="Cố định vị trí xe hỏng thay vì sinh ngẫu nhiên")
    parser.add_argument("--route-length", type=float, default=1000.0, help="Độ dài quãng đường tới đích theo mét (mặc định 1000m)")
    parser.add_argument("--predictable", action="store_true", help="Tắt tính năng xe khó đoán (trở về mặc định cũ)")
    parser.add_argument("--mode", choices=["keyboard", "agent"], default="keyboard")
    parser.add_argument("--paused", action="store_true")
    parser.add_argument("--headless", action="store_true", help="Chạy kiểm tra không mở cửa sổ")
    parser.add_argument("--steps", type=int, default=100, help="Số bước tối đa cho chế độ headless")
    parser.add_argument("--screenshot", help="Lưu ảnh PNG giao diện")
    parser.add_argument("--algo", choices=["auto", "dqn", "q_learning"], default="auto", help="Thuật toán sử dụng")
    parser.add_argument("--model", help="Đường dẫn mô hình (bảng Q hoặc checkpoint DQN .pth)")
    parser.add_argument("--auto-reset", action="store_true", help="Agent tự chuyển episode sau 2 giây")
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps phải lớn hơn 0")
    if args.speed <= 0:
        parser.error("--speed phải lớn hơn 0")
    if args.max_speed < args.speed:
        parser.error("--max-speed phải lớn hơn hoặc bằng --speed")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    import tempfile
    os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
    if args.headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        os.environ["SDL_AUDIODRIVER"] = "dummy"

    import pygame
    from pathlib import Path
    from .environment.environment import ACTIONS, make_env
    from .gui.motion import MotionBuffer
    from .algorithms.q_agent import DEFAULT_MODEL as DEFAULT_Q_MODEL, QLearningAgent
    from .algorithms.dqn_agent import DEFAULT_MODEL as DEFAULT_DQN_MODEL, DQNAgent
    from .gui.view import HighwayView

    traffic_ranges = {
        "slow": (10.0, 18.0),
        "normal": (14.0, 26.5),
        "fast": (21.0, 32.0),
    }
    t_min, t_max = traffic_ranges.get(args.traffic_speed, (14.0, 26.5))

    env = make_env(
        seed=args.seed,
        config={
            "lanes_count": args.lanes,
            "vehicles_count": args.traffic,
            "initial_speed": args.speed / 3.6,
            "max_speed": args.max_speed / 3.6,
            "traffic_speed_min": t_min,
            "traffic_speed_max": t_max,
            "obstacle_count": args.obstacles,
            "route_length": args.route_length,
            "duration": max(75.0, args.route_length / 10.0),
            "unpredictable_traffic": not args.predictable,
            "random_obstacles": not args.fixed_obstacles,
        },
    )

    algo = args.algo
    model_path = Path(args.model) if args.model else None

    # Tự động nhận diện thuật toán nếu chọn auto
    if algo == "auto":
        if model_path:
            algo = "dqn" if model_path.suffix in (".pth", ".pt") else "q_learning"
        elif DEFAULT_DQN_MODEL.exists():
            algo = "dqn"
            model_path = DEFAULT_DQN_MODEL
        elif DEFAULT_Q_MODEL.exists():
            algo = "q_learning"
            model_path = DEFAULT_Q_MODEL
        else:
            algo = "dqn"
            model_path = DEFAULT_DQN_MODEL

    if algo == "dqn":
        target_path = model_path or DEFAULT_DQN_MODEL
        try:
            agent = DQNAgent.load(target_path)
            agent.epsilon = 0.0
        except FileNotFoundError:
            if args.mode == "agent":
                parser.error(f"Chưa có mô hình DQN tại {target_path}. Chạy python -m highway_project.train_dqn trước.")
            agent = DQNAgent()
            agent.label = "DQN (PyTorch) chưa huấn luyện"
    else:
        target_path = model_path or DEFAULT_Q_MODEL
        try:
            agent = QLearningAgent.load(target_path)
        except FileNotFoundError:
            if args.mode == "agent":
                parser.error(f"Chưa có bảng Q tại {target_path}. Chạy python -m highway_project.train trước.")
            agent = QLearningAgent()
            agent.label = "Q-LEARNING chưa huấn luyện (giữ tốc khi bằng điểm)"

    motion = MotionBuffer()
    seed, episode, mode = args.seed, 0, args.mode
    view = None
    try:
        view = HighwayView(headless=args.headless)
        clock = pygame.time.Clock()
        running, paused = True, args.paused
        accumulator = 0.0
        finished_at = None
        step_dt = 1 / env.unwrapped.config["policy_frequency"]

        traffic_cycle = ["sparse", "normal", "dense", "none"]
        speed_cycle = [54.0, 72.0, 90.0, 108.0]
        current_lanes = args.lanes
        current_traffic = args.traffic
        current_speed = args.speed
        current_traffic_speed = args.traffic_speed

        def rebuild_env(*, lanes=None, traffic=None, speed=None, traffic_speed=None):
            nonlocal env, current_lanes, current_traffic, current_speed, current_traffic_speed, step_dt
            if lanes is not None:
                current_lanes = lanes
            if traffic is not None:
                current_traffic = traffic
            if speed is not None:
                current_speed = speed
            if traffic_speed is not None:
                current_traffic_speed = traffic_speed

            t_min, t_max = traffic_ranges.get(current_traffic_speed, (14.0, 26.5))
            env.close()
            env = make_env(
                seed=seed,
                config={
                    "lanes_count": current_lanes,
                    "vehicles_count": current_traffic,
                    "initial_speed": current_speed / 3.6,
                    "max_speed": max(args.max_speed, current_speed) / 3.6,
                    "traffic_speed_min": t_min,
                    "traffic_speed_max": t_max,
                    "obstacle_count": args.obstacles,
                    "route_length": args.route_length,
                    "duration": max(75.0, args.route_length / 10.0),
                    "unpredictable_traffic": not args.predictable,
                    "random_obstacles": not args.fixed_obstacles,
                },
            )
            step_dt = 1 / env.unwrapped.config["policy_frequency"]
            reset()

        def reset():
            nonlocal episode, obs, info, done, total_reward, last_action, accumulator, pending_action, finished_at
            obs, info = env.reset(seed=seed)
            agent.reset(seed=seed)
            episode += 1
            done, total_reward, last_action = False, 0.0, "IDLE"
            accumulator, pending_action = 0.0, 1
            finished_at = None
            motion.reset(env.unwrapped)

        def advance(action):
            nonlocal obs, info, done, total_reward, last_action, finished_at
            if not env.action_space.contains(action):
                raise ValueError(f"Agent phải trả về số nguyên 0..4, nhận được {action!r}")
            next_obs, reward, terminated, truncated, next_info = env.step(action)
            # Chạy demo chỉ suy luận bảng Q; học được thực hiện riêng trong train.py.
            obs, info = next_obs, next_info
            done = terminated or truncated
            total_reward += reward
            last_action = ACTIONS[int(action)]
            motion.capture(env.unwrapped)
            if done:
                finished_at = pygame.time.get_ticks()

        obs = info = None
        done, total_reward, last_action, pending_action = False, 0.0, "IDLE", 1
        reset()
        if args.headless:
            steps = 0
            for _ in range(args.steps):
                advance(agent.act(obs, info) if mode == "agent" else 1)
                steps += 1
                if done:
                    break
            view.draw(env.unwrapped, info, paused=False, mode=mode, last_action=last_action, total_reward=total_reward, episode=episode, agent_label=agent.label)
            if args.screenshot:
                view.save(args.screenshot)
            print(json.dumps({"steps": steps, "reward": total_reward, **info}, ensure_ascii=True, default=float))
            return

        while running:
            dt = min(clock.tick(60) / 1000, 0.25)
            for event in pygame.event.get():
                command = None
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    command = next((key for key, rect in view.buttons.items() if rect.collidepoint(event.pos)), None)
                elif event.type == pygame.KEYDOWN:
                    command = {
                        pygame.K_SPACE: "pause",
                        pygame.K_r: "reset",
                        pygame.K_n: "new",
                        pygame.K_TAB: "mode",
                        pygame.K_l: "lanes",
                        pygame.K_t: "traffic",
                        pygame.K_s: "speed",
                        pygame.K_1: "preset_1",
                        pygame.K_2: "preset_2",
                        pygame.K_3: "preset_3",
                        pygame.K_4: "preset_4",
                    }.get(event.key)
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    if mode == "keyboard" and not paused and not done:
                        pending_action = {pygame.K_LEFT: 0, pygame.K_RIGHT: 2, pygame.K_UP: 3, pygame.K_DOWN: 4}.get(event.key, pending_action)
                if command == "pause":
                    paused = not paused
                    accumulator, pending_action = 0.0, 1
                elif command in ("reset", "new"):
                    if command == "new":
                        seed += 1
                    reset()
                elif command == "mode":
                    mode = "agent" if mode == "keyboard" else "keyboard"
                    agent.reset(seed=seed)
                    pending_action = 1
                elif command == "lanes":
                    next_lanes = 2 if current_lanes >= 6 else current_lanes + 1
                    rebuild_env(lanes=next_lanes)
                elif command == "traffic":
                    idx = traffic_cycle.index(current_traffic) if current_traffic in traffic_cycle else 1
                    next_traffic = traffic_cycle[(idx + 1) % len(traffic_cycle)]
                    rebuild_env(traffic=next_traffic)
                elif command == "speed":
                    idx = min(range(len(speed_cycle)), key=lambda i: abs(speed_cycle[i] - current_speed))
                    next_speed = speed_cycle[(idx + 1) % len(speed_cycle)]
                    rebuild_env(speed=next_speed)
                elif command == "preset_1":
                    rebuild_env(lanes=2, traffic="sparse", speed=90.0, traffic_speed="normal")
                elif command == "preset_2":
                    rebuild_env(lanes=3, traffic="normal", speed=72.0, traffic_speed="normal")
                elif command == "preset_3":
                    rebuild_env(lanes=5, traffic="dense", speed=80.0, traffic_speed="fast")
                elif command == "preset_4":
                    rebuild_env(lanes=4, traffic="none", speed=90.0, traffic_speed="normal")
            if not running:
                break
            if not paused and not done:
                accumulator += dt
                while accumulator >= step_dt and not done:
                    action = agent.act(obs, info) if mode == "agent" else pending_action
                    advance(action)
                    pending_action = 1
                    accumulator -= step_dt
            else:
                accumulator = 0.0
            if args.auto_reset and mode == "agent" and done and not paused and pygame.time.get_ticks() - finished_at >= 2000:
                seed += 1
                reset()
            view.draw(env.unwrapped, info, paused=paused, mode=mode, last_action=last_action, total_reward=total_reward, episode=episode,
                      motion=motion, alpha=1.0 if paused or done else accumulator / step_dt, agent_label=agent.label)
            pygame.display.flip()
        if args.screenshot:
            view.save(args.screenshot)
    finally:
        env.close()
        pygame.quit()


if __name__ == "__main__":
    main()
