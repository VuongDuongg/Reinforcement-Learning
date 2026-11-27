"""Trình khởi chạy tương tác đa kịch bản cho highway_project."""

from __future__ import annotations

import sys

# Đảm bảo Windows Terminal / cmd.exe xuất nhập UTF-8 không bao giờ bị UnicodeEncodeError
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stdin, "reconfigure"):
            sys.stdin.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def safe_print(*args, **kwargs):
    try:
        print(*args, **kwargs)
    except Exception:
        try:
            text = " ".join(str(a) for a in args)
            clean = text.encode("ascii", errors="replace").decode("ascii")
            print(clean, **kwargs)
        except Exception:
            pass


def interactive_menu() -> list[str]:
    # Nếu chạy từ non-interactive (IDE, runner, file redirect), tự động chạy ngay kịch bản mặc định
    if not sys.stdin.isatty():
        return ["--algo", "dqn", "--mode", "agent", "--auto-reset"]

    safe_print("=" * 78)
    safe_print("        MÔ PHỎNG XE TỰ HÀNH NÉ VẬT CẢN (REINFORCEMENT LEARNING)")
    safe_print("=" * 78)
    safe_print()
    safe_print("  Chọn kịch bản muốn chạy:")
    safe_print()
    safe_print("   [1] Mặc định: Cao tốc 3 làn · Mật độ Bình thường (20 xe) · Tốc độ 72 km/h (DQN)")
    safe_print("   [2] Thuật toán 2: Double Dueling DQN (Ngô Tuấn Anh) · Cao tốc 3 làn · 72 km/h")
    safe_print("   [3] Kịch bản 1: Cao tốc hẹp 2 làn · Mật độ Thưa thớt (8 xe) · Tốc độ cao 90 km/h")
    safe_print("   [4] Kịch bản 2: Cao tốc mở rộng 5 làn · Mật độ Dày đặc (38 xe) · Dòng xe nhanh")
    safe_print("   [5] Kịch bản 3: Đường vắng không có xe (4 làn, 90 km/h, chỉ né xe hỏng)")
    safe_print("   [6] Kịch bản 4: Tự lái bằng bàn phím (Người điều khiển)")
    safe_print("   [7] TỰ TÙY BIẾN CHI TIẾT (Chọn thuật toán, số làn, mật độ xe, tốc độ theo ý bạn)")
    safe_print("   [0] Thoát")
    safe_print()
    safe_print("=" * 78)

    try:
        choice = input("Nhập lựa chọn của bạn [1-7] (Mặc định: 1): ").strip()
    except EOFError:
        return ["--algo", "dqn", "--mode", "agent", "--auto-reset"]
    except KeyboardInterrupt:
        sys.exit(0)

    if not choice or choice == "1":
        return ["--algo", "dqn", "--mode", "agent", "--auto-reset"]
    elif choice == "2":
        return ["--algo", "ddqn", "--mode", "agent", "--auto-reset"]
    elif choice == "3":
        return ["--lanes", "2", "--traffic", "sparse", "--speed", "90", "--max-speed", "120", "--algo", "dqn", "--mode", "agent", "--auto-reset"]
    elif choice == "4":
        return ["--lanes", "5", "--traffic", "dense", "--speed", "80", "--max-speed", "110", "--traffic-speed", "fast", "--algo", "dqn", "--mode", "agent", "--auto-reset"]
    elif choice == "5":
        return ["--lanes", "4", "--traffic", "none", "--speed", "90", "--algo", "dqn", "--mode", "agent", "--auto-reset"]
    elif choice == "6":
        return ["--mode", "keyboard"]
    elif choice == "0":
        sys.exit(0)
    elif choice == "7":
        safe_print()
        safe_print("=" * 78)
        safe_print("                      CẤU HÌNH TÙY BIẾN KỊCH BẢN")
        safe_print("=" * 78)
        safe_print()

        # 1. Số làn
        try:
            lanes_in = input("1. Số làn đường cao tốc [2..6] (Mặc định: 3): ").strip()
        except EOFError:
            lanes_in = "3"
        lanes = lanes_in if lanes_in in ("2", "3", "4", "5", "6") else "3"

        # 2. Mật độ xe
        safe_print()
        safe_print("  2. Chọn mật độ xe giao thông:")
        safe_print("   [1] Thưa thớt (sparse: 8 xe)")
        safe_print("   [2] Bình thường (normal: 20 xe - Mặc định)")
        safe_print("   [3] Dày đặc (dense: 38 xe)")
        safe_print("   [4] Không có xe (none: 0 xe)")
        try:
            dens_in = input("Nhập lựa chọn [1-4] (Mặc định: 2): ").strip()
        except EOFError:
            dens_in = "2"
        dens_map = {"1": "sparse", "2": "normal", "3": "dense", "4": "none"}
        traffic = dens_map.get(dens_in, "normal")

        # 3. Vận tốc
        safe_print()
        try:
            speed_in = input("3. Vận tốc xuất phát xe của bạn theo km/h (Mặc định: 72): ").strip()
        except EOFError:
            speed_in = "72"
        try:
            speed_val = float(speed_in) if speed_in else 72.0
            if speed_val <= 0:
                speed_val = 72.0
        except ValueError:
            speed_val = 72.0
        speed = str(speed_val)

        # 4. Tốc độ dòng xe
        safe_print()
        safe_print("  4. Chọn tốc độ dòng xe xung quanh:")
        safe_print("   [1] Chậm (slow: 36-65 km/h)")
        safe_print("   [2] Bình thường (normal: 50-95 km/h - Mặc định)")
        safe_print("   [3] Nhanh (fast: 75-115 km/h)")
        try:
            tspd_in = input("Nhập lựa chọn [1-3] (Mặc định: 2): ").strip()
        except EOFError:
            tspd_in = "2"
        tspd_map = {"1": "slow", "2": "normal", "3": "fast"}
        traffic_speed = tspd_map.get(tspd_in, "normal")

        # 5. Chế độ điều khiển & Thuật toán
        safe_print()
        safe_print("  5. Chọn chế độ điều khiển / thuật toán:")
        safe_print("   [1] AI tự lái bằng Deep Q-Network (DQN - Mặc định)")
        safe_print("   [2] AI tự lái bằng Double Dueling DQN (Thuật toán 2 - Ngô Tuấn Anh)")
        safe_print("   [3] Bạn tự lái bằng bàn phím")
        try:
            mode_in = input("Nhập lựa chọn [1-3] (Mặc định: 1): ").strip()
        except EOFError:
            mode_in = "1"
        if mode_in == "2":
            algo_choice, run_mode = "ddqn", "agent"
        elif mode_in == "3":
            algo_choice, run_mode = "dqn", "keyboard"
        else:
            algo_choice, run_mode = "dqn", "agent"

        safe_print()
        safe_print("=" * 78)
        safe_print("Đang khởi động mô phỏng với cấu hình:")
        safe_print(f"  - Thuật toán: {algo_choice}")
        safe_print(f"  - Số làn đường: {lanes} làn")
        safe_print(f"  - Mật độ xe: {traffic}")
        safe_print(f"  - Vận tốc xuất phát: {speed} km/h")
        safe_print(f"  - Dòng xe: {traffic_speed}")
        safe_print(f"  - Chế độ: {run_mode}")
        safe_print("=" * 78)
        safe_print()

        return [
            "--lanes", lanes,
            "--traffic", traffic,
            "--speed", speed,
            "--traffic-speed", traffic_speed,
            "--algo", algo_choice,
            "--mode", run_mode,
            "--auto-reset",
        ]
    else:
        safe_print("Lựa chọn không hợp lệ, dùng cấu hình mặc định.")
        return ["--algo", "dqn", "--mode", "agent", "--auto-reset"]


def main():
    args = sys.argv[1:]
    if not args:
        args = interactive_menu()

    from .run import main as run_main
    sys.argv = [sys.argv[0], *args]
    run_main()


if __name__ == "__main__":
    main()

