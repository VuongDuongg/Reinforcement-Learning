"""Chạy bằng: python -m highway_project.run"""

from __future__ import annotations

import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser(description="Môi trường xe né vật cản và Q-learning.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--traffic", type=int, default=12, choices=range(0, 41), metavar="0..40")
    parser.add_argument("--obstacles", type=int, default=1, choices=range(0, 4))
    parser.add_argument("--mode", choices=["keyboard", "agent"], default="keyboard")
    parser.add_argument("--paused", action="store_true")
    parser.add_argument("--headless", action="store_true", help="Chạy kiểm tra không mở cửa sổ")
    parser.add_argument("--steps", type=int, default=100, help="Số bước tối đa cho chế độ headless")
    parser.add_argument("--screenshot", help="Lưu ảnh PNG giao diện")
    parser.add_argument("--model", help="Đường dẫn bảng Q đã huấn luyện")
    parser.add_argument("--auto-reset", action="store_true", help="Agent tự chuyển episode sau 2 giây")
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps phải lớn hơn 0")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    if args.headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        os.environ["SDL_AUDIODRIVER"] = "dummy"

    import pygame
    from .environment import ACTIONS, make_env
    from .motion import MotionBuffer
    from .q_agent import DEFAULT_MODEL, QLearningAgent
    from .view import HighwayView

    env = make_env(seed=args.seed, config={"vehicles_count": args.traffic, "obstacle_count": args.obstacles})
    model_path = args.model or DEFAULT_MODEL
    try:
        agent = QLearningAgent.load(model_path)
    except FileNotFoundError:
        if args.mode == "agent":
            parser.error("Chưa có bảng Q. Chạy python -m highway_project.train trước.")
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
                    command = {pygame.K_SPACE: "pause", pygame.K_r: "reset", pygame.K_n: "new", pygame.K_TAB: "mode"}.get(event.key)
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
