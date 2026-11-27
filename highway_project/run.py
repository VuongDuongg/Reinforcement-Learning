"""Chạy bằng: python -m highway_project.run"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any


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
    parser.add_argument("--algo", choices=["auto", "dqn", "ddqn", "q_learning", "baseline"], default="auto", help="Thuật toán sử dụng (dqn, ddqn = Double Dueling DQN, q_learning, baseline)")
    parser.add_argument("--model", help="Đường dẫn mô hình (bảng Q hoặc checkpoint DQN/DDQN .pth)")
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
    import numpy as np
    from highway_env.road.road import RoadNetwork
    from highway_env.vehicle.behavior import IDMVehicle
    from .environment.environment import ACTIONS, make_env, TRAFFIC_PRESETS
    from .gui.motion import MotionBuffer
    from .algorithms.q_agent import DEFAULT_MODEL as DEFAULT_Q_MODEL, QLearningAgent
    from .algorithms.dqn_agent import DEFAULT_MODEL as DEFAULT_DQN_MODEL, DQNAgent
    from .algorithms.ddqn_agent import DEFAULT_MODEL as DEFAULT_DDQN_MODEL, DoubleDuelingDQNAgent
    from .algorithms.baseline_agent import RuleBasedAgent
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
            stem_lower = model_path.stem.lower()
            if "ddqn" in stem_lower or "dueling" in stem_lower:
                algo = "ddqn"
            else:
                algo = "dqn" if model_path.suffix in (".pth", ".pt") else "q_learning"
        elif DEFAULT_DQN_MODEL.exists():
            algo = "dqn"
            model_path = DEFAULT_DQN_MODEL
        elif DEFAULT_DDQN_MODEL.exists():
            algo = "ddqn"
            model_path = DEFAULT_DDQN_MODEL
        elif DEFAULT_Q_MODEL.exists():
            algo = "q_learning"
            model_path = DEFAULT_Q_MODEL
        else:
            algo = "dqn"
            model_path = DEFAULT_DQN_MODEL

    agent_cache: dict[str, Any] = {}

    def build_agent(target_algo: str, custom_path: Path | None = None, *, strict: bool = False):
        if custom_path is None and target_algo in agent_cache:
            return agent_cache[target_algo]

        if target_algo == "baseline":
            loaded = RuleBasedAgent()
            loaded.label = "Rule-based Baseline (Tập luật Heuristic)"
        elif target_algo == "ddqn":
            target_path = custom_path or DEFAULT_DDQN_MODEL
            try:
                loaded = DoubleDuelingDQNAgent.load(target_path)
                loaded.epsilon = 0.0
            except FileNotFoundError:
                if strict:
                    parser.error(f"Chưa có mô hình Double Dueling DQN tại {target_path}. Chạy python -m highway_project.train_ddqn trước.")
                loaded = DoubleDuelingDQNAgent(epsilon=0.0)
                loaded.label = "Double Dueling DQN (Ngô Tuấn Anh · chưa huấn luyện)"
        elif target_algo == "dqn":
            target_path = custom_path or DEFAULT_DQN_MODEL
            try:
                loaded = DQNAgent.load(target_path)
                loaded.epsilon = 0.0
            except FileNotFoundError:
                if strict:
                    parser.error(f"Chưa có mô hình DQN tại {target_path}. Chạy python -m highway_project.train_dqn trước.")
                loaded = DQNAgent(epsilon=0.0)
                loaded.label = "DQN (PyTorch) chưa huấn luyện"
        else:
            target_path = custom_path or DEFAULT_Q_MODEL
            try:
                loaded = QLearningAgent.load(target_path)
            except FileNotFoundError:
                if strict:
                    parser.error(f"Chưa có bảng Q tại {target_path}. Chạy python -m highway_project.train trước.")
                loaded = QLearningAgent()
                loaded.label = "Q-LEARNING chưa huấn luyện (giữ tốc khi bằng điểm)"

        if custom_path is None:
            agent_cache[target_algo] = loaded
        return loaded

    agent = build_agent(algo, model_path, strict=(args.mode == "agent" and args.headless))

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
        traffic_speed_cycle = ["slow", "normal", "fast"]
        current_lanes = args.lanes
        current_traffic = args.traffic
        current_speed = args.speed
        current_traffic_speed = args.traffic_speed

        def live_reconfigure(*, lanes=None, traffic=None, speed=None, traffic_speed=None, toast=None):
            nonlocal current_lanes, current_traffic, current_speed, current_traffic_speed, obs, info
            unwrapped = env.unwrapped
            road = unwrapped.road

            # 1. Cập nhật làn đường trực tiếp trên mạng lưới road.network (KHÔNG RESET XE)
            if lanes is not None and lanes != current_lanes:
                current_lanes = int(lanes)
                unwrapped.config["lanes_count"] = current_lanes
                route_len = float(unwrapped.config.get("route_length", 1000.0))
                road_length = max(route_len + 2500.0, float(unwrapped.goal_x + 1500.0))
                road.network = RoadNetwork.straight_road_network(current_lanes, length=road_length)

                for v in [*road.vehicles, *road.objects]:
                    v.road = road
                    curr_lane_id = int(v.lane_index[2]) if hasattr(v, "lane_index") else 0
                    clamped_lane = min(curr_lane_id, current_lanes - 1)
                    v.lane_index = ("0", "1", clamped_lane)
                    if hasattr(v, "target_lane_index"):
                        v.target_lane_index = ("0", "1", clamped_lane)
                    if hasattr(v, "lane"):
                        v.lane = road.network.get_lane(v.lane_index)

                    if v is unwrapped.vehicle or v in unwrapped.road.objects:
                        if curr_lane_id >= current_lanes:
                            v.position[1] = clamped_lane * 4.0
                    else:
                        if v.position[1] > (current_lanes - 1) * 4.0 + 2.0:
                            v.position[1] = clamped_lane * 4.0

            # 2. Cập nhật tốc độ xe của bạn (KHÔNG RESET XE)
            if speed is not None:
                current_speed = float(speed)
                new_speed_ms = current_speed / 3.6
                unwrapped.config["initial_speed"] = new_speed_ms
                unwrapped.vehicle.target_speed = new_speed_ms
                unwrapped.vehicle.speed = new_speed_ms

            # 3. Cập nhật tốc độ dòng xe xung quanh
            if traffic_speed is not None:
                current_traffic_speed = traffic_speed
                spd_min, spd_max = traffic_ranges.get(current_traffic_speed, (14.0, 26.5))
                unwrapped.config["traffic_speed_min"] = spd_min
                unwrapped.config["traffic_speed_max"] = spd_max
                for v in road.vehicles:
                    if v is not unwrapped.vehicle:
                        v.target_speed = float(np.random.uniform(spd_min, max(spd_min + 1.0, spd_max)))

            # 4. Cập nhật mật độ xe giao thông trực tiếp (KHÔNG RESET XE)
            if traffic is not None:
                current_traffic = traffic
                target_count = TRAFFIC_PRESETS.get(current_traffic, 20)
                unwrapped.config["vehicles_count"] = target_count
                unwrapped.traffic_density = current_traffic

                traffic_vehicles = [v for v in road.vehicles if v is not unwrapped.vehicle]
                if target_count == 0:
                    road.vehicles = [unwrapped.vehicle]
                elif target_count < len(traffic_vehicles):
                    ego_x = unwrapped.vehicle.position[0]
                    traffic_vehicles.sort(key=lambda v: abs(v.position[0] - ego_x))
                    road.vehicles = [unwrapped.vehicle, *traffic_vehicles[:target_count]]
                elif target_count > len(traffic_vehicles):
                    needed = target_count - len(traffic_vehicles)
                    spd_min, spd_max = traffic_ranges.get(current_traffic_speed, (14.0, 26.5))
                    ego_x = unwrapped.vehicle.position[0]
                    for _ in range(needed):
                        lane_id = int(np.random.randint(0, current_lanes))
                        side = np.random.choice([1, -1], p=[0.7, 0.3])
                        if side == 1:
                            spawn_x = ego_x + float(np.random.uniform(35.0, 220.0))
                        else:
                            spawn_x = max(unwrapped.start_x, ego_x - float(np.random.uniform(35.0, 140.0)))

                        all_objs = [*road.vehicles, *road.objects]
                        if any(abs(obj.position[0] - spawn_x) < 22.0 and abs(obj.position[1] - lane_id * 4.0) < 2.0 for obj in all_objs):
                            continue

                        v_spd = float(np.random.uniform(spd_min, spd_max))
                        new_v = IDMVehicle.make_on_lane(
                            road, ("0", "1", lane_id), spawn_x, speed=v_spd
                        )
                        new_v.target_speed = float(np.random.uniform(spd_min, max(spd_min + 1.0, spd_max)))
                        new_v.randomize_behavior()
                        new_v.enable_lane_change = True
                        road.vehicles.append(new_v)

            # 5. Cập nhật tức thời trạng thái quan sát và cảm biến (Xe tiếp tục tiến trình)
            obs = unwrapped.observation_type.observe()
            info = unwrapped._info(obs)
            motion.reset(unwrapped)
            if toast and view:
                view.show_toast(toast)

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
                    clicked = next((key for key, rect in view.buttons.items() if rect.collidepoint(event.pos)), None)
                    if view.menu_open:
                        if clicked is None and not view.modal_rect.collidepoint(event.pos):
                            view.menu_open = False
                        elif clicked in ("menu_close", "menu_close_btn"):
                            view.menu_open = False
                        elif clicked and clicked.startswith("menu_lane_"):
                            target_lane = int(clicked.replace("menu_lane_", ""))
                            live_reconfigure(lanes=target_lane, toast=f"Đã chuyển sang {target_lane} làn (Xe tiếp tục chạy)")
                        elif clicked and clicked.startswith("menu_traffic_"):
                            target_traffic = clicked.replace("menu_traffic_", "")
                            d_vi = {"sparse": "Thưa", "normal": "Vừa", "dense": "Dày", "none": "Không có xe"}.get(target_traffic, target_traffic)
                            live_reconfigure(traffic=target_traffic, toast=f"Mật độ xe: {d_vi} (Xe tiếp tục chạy)")
                        elif clicked and clicked.startswith("menu_spd_"):
                            target_spd = float(clicked.replace("menu_spd_", ""))
                            live_reconfigure(speed=target_spd, toast=f"Vận tốc xe: {int(target_spd)} km/h (Xe tiếp tục chạy)")
                        elif clicked and clicked.startswith("menu_flow_"):
                            target_flow = clicked.replace("menu_flow_", "")
                            f_vi = {"slow": "Chậm", "normal": "Vừa", "fast": "Nhanh"}.get(target_flow, target_flow)
                            live_reconfigure(traffic_speed=target_flow, toast=f"Dòng xe: {f_vi} (Xe tiếp tục chạy)")
                        elif clicked == "menu_preset_1":
                            live_reconfigure(lanes=2, traffic="sparse", speed=90.0, traffic_speed="normal", toast="Kịch bản 1: 2 làn · Thưa (8 xe) · 90 km/h (Xe tiếp tục chạy)")
                        elif clicked == "menu_preset_2":
                            live_reconfigure(lanes=3, traffic="normal", speed=72.0, traffic_speed="normal", toast="Kịch bản 2: 3 làn · Chuẩn (20 xe) · 72 km/h (Xe tiếp tục chạy)")
                        elif clicked == "menu_preset_3":
                            live_reconfigure(lanes=5, traffic="dense", speed=80.0, traffic_speed="fast", toast="Kịch bản 3: 5 làn · Dày (38 xe) · Nhanh (Xe tiếp tục chạy)")
                        elif clicked == "menu_preset_4":
                            live_reconfigure(lanes=4, traffic="none", speed=90.0, traffic_speed="normal", toast="Kịch bản 4: 4 làn · Đường vắng (Xe tiếp tục chạy)")
                        elif clicked == "menu_mode_agent":
                            algo = "dqn"
                            agent = build_agent("dqn")
                            mode = "agent"
                            agent.reset(seed=seed)
                            pending_action = 1
                            view.show_toast("Chế độ: AI tự lái (DQN)")
                        elif clicked == "menu_mode_ddqn":
                            algo = "ddqn"
                            agent = build_agent("ddqn")
                            mode = "agent"
                            agent.reset(seed=seed)
                            pending_action = 1
                            view.show_toast("Chế độ: AI tự lái (Double Dueling DQN)")
                        elif clicked == "menu_mode_baseline":
                            algo = "baseline"
                            agent = build_agent("baseline")
                            mode = "agent"
                            agent.reset(seed=seed)
                            pending_action = 1
                            view.show_toast("Chế độ: AI tự lái (Rule-based Baseline)")
                        elif clicked == "menu_mode_keyboard":
                            mode = "keyboard"
                            agent.reset(seed=seed)
                            pending_action = 1
                            view.show_toast("Chế độ: Người tự lái (Phím)")
                    else:
                        if clicked == "menu_toggle":
                            view.toggle_menu()
                        elif clicked in ("pause", "mode", "reset"):
                            command = clicked

                elif event.type == pygame.KEYDOWN:
                    scancode = getattr(event, "scancode", None)
                    ch = getattr(event, "unicode", "").lower()
                    if event.key == pygame.K_m or scancode == pygame.KSCAN_M or ch == "m":
                        view.toggle_menu()
                    elif event.key == pygame.K_ESCAPE or scancode == pygame.KSCAN_ESCAPE:
                        if view.menu_open:
                            view.menu_open = False
                        else:
                            running = False
                    else:
                        key_map = {
                            pygame.K_SPACE: "pause",
                            pygame.K_r: "reset",
                            pygame.K_n: "new",
                            pygame.K_TAB: "mode",
                            pygame.K_l: "lanes",
                            pygame.K_t: "traffic",
                            pygame.K_s: "speed",
                            pygame.K_f: "traffic_speed",
                            pygame.K_1: "preset_1",
                            pygame.K_2: "preset_2",
                            pygame.K_3: "preset_3",
                            pygame.K_4: "preset_4",
                        }
                        scan_map = {
                            pygame.KSCAN_SPACE: "pause",
                            pygame.KSCAN_R: "reset",
                            pygame.KSCAN_N: "new",
                            pygame.KSCAN_TAB: "mode",
                            pygame.KSCAN_L: "lanes",
                            pygame.KSCAN_T: "traffic",
                            pygame.KSCAN_S: "speed",
                            pygame.KSCAN_F: "traffic_speed",
                            pygame.KSCAN_1: "preset_1",
                            pygame.KSCAN_2: "preset_2",
                            pygame.KSCAN_3: "preset_3",
                            pygame.KSCAN_4: "preset_4",
                        }
                        char_map = {
                            "r": "reset",
                            "n": "new",
                            "l": "lanes",
                            "t": "traffic",
                            "s": "speed",
                            "f": "traffic_speed",
                            "1": "preset_1",
                            "2": "preset_2",
                            "3": "preset_3",
                            "4": "preset_4",
                        }
                        command = key_map.get(event.key) or scan_map.get(scancode) or char_map.get(ch)
                        if mode == "keyboard" and not paused and not done:
                            pending_action = {
                                pygame.K_LEFT: 0,
                                pygame.K_RIGHT: 2,
                                pygame.K_UP: 3,
                                pygame.K_DOWN: 4,
                            }.get(event.key, pending_action)
                if command == "pause":
                    paused = not paused
                    accumulator, pending_action = 0.0, 1
                    if view:
                        view.show_toast("Đã tạm dừng" if paused else "Đang chạy tiếp", color=(255, 204, 0) if paused else (46, 213, 162))
                elif command in ("reset", "new"):
                    if command == "new":
                        seed += 1
                        if view:
                            view.show_toast(f"Đã tạo kịch bản mới (Seed {seed})", color=(255, 204, 0))
                    elif view:
                        view.show_toast("Đã khởi động lại từ đầu", color=(255, 204, 0))
                    reset()
                elif command == "mode":
                    mode = "agent" if mode == "keyboard" else "keyboard"
                    agent.reset(seed=seed)
                    pending_action = 1
                    if view:
                        if algo == "baseline":
                            ai_name = "Rule-based Baseline"
                        elif algo == "ddqn":
                            ai_name = "Double Dueling DQN"
                        elif algo == "dqn":
                            ai_name = "DQN"
                        else:
                            ai_name = "Q-Learning"
                        m_str = f"AI tự lái ({ai_name})" if mode == "agent" else "Người tự lái (Phím)"
                        view.show_toast(f"Chế độ: {m_str}")
                elif command == "lanes":
                    next_lanes = 2 if current_lanes >= 6 else current_lanes + 1
                    live_reconfigure(lanes=next_lanes, toast=f"Đã chuyển sang {next_lanes} làn (Xe tiếp tục chạy)")
                elif command == "traffic":
                    idx = traffic_cycle.index(current_traffic) if current_traffic in traffic_cycle else 1
                    next_traffic = traffic_cycle[(idx + 1) % len(traffic_cycle)]
                    d_vi = {"sparse": "Thưa", "normal": "Vừa", "dense": "Dày", "none": "Không có xe"}.get(next_traffic, next_traffic)
                    live_reconfigure(traffic=next_traffic, toast=f"Mật độ xe: {d_vi} (Xe tiếp tục chạy)")
                elif command == "speed":
                    idx = min(range(len(speed_cycle)), key=lambda i: abs(speed_cycle[i] - current_speed))
                    next_speed = speed_cycle[(idx + 1) % len(speed_cycle)]
                    live_reconfigure(speed=next_speed, toast=f"Vận tốc xe: {int(next_speed)} km/h (Xe tiếp tục chạy)")
                elif command == "traffic_speed":
                    idx = traffic_speed_cycle.index(current_traffic_speed) if current_traffic_speed in traffic_speed_cycle else 1
                    next_flow = traffic_speed_cycle[(idx + 1) % len(traffic_speed_cycle)]
                    f_vi = {"slow": "Chậm", "normal": "Vừa", "fast": "Nhanh"}.get(next_flow, next_flow)
                    live_reconfigure(traffic_speed=next_flow, toast=f"Dòng xe: {f_vi} (Xe tiếp tục chạy)")
                elif command == "preset_1":
                    live_reconfigure(lanes=2, traffic="sparse", speed=90.0, traffic_speed="normal", toast="Kịch bản 1: 2 làn · Thưa (8 xe) · 90 km/h (Xe tiếp tục chạy)")
                elif command == "preset_2":
                    live_reconfigure(lanes=3, traffic="normal", speed=72.0, traffic_speed="normal", toast="Kịch bản 2: 3 làn · Chuẩn (20 xe) · 72 km/h (Xe tiếp tục chạy)")
                elif command == "preset_3":
                    live_reconfigure(lanes=5, traffic="dense", speed=80.0, traffic_speed="fast", toast="Kịch bản 3: 5 làn · Dày (38 xe) · Nhanh (Xe tiếp tục chạy)")
                elif command == "preset_4":
                    live_reconfigure(lanes=4, traffic="none", speed=90.0, traffic_speed="normal", toast="Kịch bản 4: 4 làn · Đường vắng (Xe tiếp tục chạy)")
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
