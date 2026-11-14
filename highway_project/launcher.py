"""Trình khởi chạy tương tác đa kịch bản cho highway_project."""

from __future__ import annotations

import sys


def interactive_menu() -> list[str]:
    print("=" * 78)
    print("        MÔ PHỎNG XE TỰ HÀNH NÉ VẬT CẢN (REINFORCEMENT LEARNING)")
    print("=" * 78)
    print()
    print("  Chọn kịch bản muốn chạy:")
    print()
    print("   [1] Mặc định: Cao tốc 3 làn · Mật độ Bình thường (20 xe) · Tốc độ 72 km/h (DQN)")
    print("   [2] Kịch bản 1: Cao tốc hẹp 2 làn · Mật độ Thưa thớt (8 xe) · Tốc độ cao 90 km/h")
    print("   [3] Kịch bản 2: Cao tốc mở rộng 5 làn · Mật độ Dày đặc (38 xe) · Dòng xe nhanh")
    print("   [4] Kịch bản 3: Đường vắng không có xe (4 làn, 90 km/h, chỉ né xe hỏng)")
    print("   [5] Kịch bản 4: Tự lái bằng bàn phím (Người điều khiển)")
    print("   [6] TỰ TÙY BIẾN CHI TIẾT (Chọn số làn, mật độ xe, tốc độ theo ý bạn)")
    print("   [0] Thoát")
    print()
    print("=" * 78)

    try:
        choice = input("Nhập lựa chọn của bạn [1-6] (Mặc định: 1): ").strip()
    except (KeyboardInterrupt, EOFError):
        sys.exit(0)

    if not choice or choice == "1":
        return ["--algo", "dqn", "--mode", "agent", "--auto-reset"]
    elif choice == "2":
        return ["--lanes", "2", "--traffic", "sparse", "--speed", "90", "--max-speed", "120", "--algo", "dqn", "--mode", "agent", "--auto-reset"]
    elif choice == "3":
        return ["--lanes", "5", "--traffic", "dense", "--speed", "80", "--max-speed", "110", "--traffic-speed", "fast", "--algo", "dqn", "--mode", "agent", "--auto-reset"]
    elif choice == "4":
        return ["--lanes", "4", "--traffic", "none", "--speed", "90", "--algo", "dqn", "--mode", "agent", "--auto-reset"]
    elif choice == "5":
        return ["--mode", "keyboard"]
    elif choice == "0":
        sys.exit(0)
    elif choice == "6":
        print()
        print("=" * 78)
        print("                      CẤU HÌNH TÙY BIẾN KỊCH BẢN")
        print("=" * 78)
        print()

        # 1. Số làn
        lanes_in = input("1. Số làn đường cao tốc [2..6] (Mặc định: 3): ").strip()
        lanes = lanes_in if lanes_in in ("2", "3", "4", "5", "6") else "3"

        # 2. Mật độ xe
        print()
        print("  2. Chọn mật độ xe giao thông:")
        print("   [1] Thưa thớt (sparse: 8 xe)")
        print("   [2] Bình thường (normal: 20 xe - Mặc định)")
        print("   [3] Dày đặc (dense: 38 xe)")
        print("   [4] Không có xe (none: 0 xe)")
        dens_in = input("Nhập lựa chọn [1-4] (Mặc định: 2): ").strip()
        dens_map = {"1": "sparse", "2": "normal", "3": "dense", "4": "none"}
        traffic = dens_map.get(dens_in, "normal")

        # 3. Vận tốc
        print()
        speed_in = input("3. Vận tốc xuất phát xe của bạn theo km/h (Mặc định: 72): ").strip()
        try:
            speed_val = float(speed_in) if speed_in else 72.0
            if speed_val <= 0:
                speed_val = 72.0
        except ValueError:
            speed_val = 72.0
        speed = str(speed_val)

        # 4. Tốc độ dòng xe
        print()
        print("  4. Chọn tốc độ dòng xe xung quanh:")
        print("   [1] Chậm (slow: 36-65 km/h)")
        print("   [2] Bình thường (normal: 50-95 km/h - Mặc định)")
        print("   [3] Nhanh (fast: 75-115 km/h)")
        tspd_in = input("Nhập lựa chọn [1-3] (Mặc định: 2): ").strip()
        tspd_map = {"1": "slow", "2": "normal", "3": "fast"}
        traffic_speed = tspd_map.get(tspd_in, "normal")

        # 5. Chế độ điều khiển
        print()
        print("  5. Chọn chế độ điều khiển:")
        print("   [1] AI tự lái bằng Deep Q-Network (DQN - Mặc định)")
        print("   [2] Bạn tự lái bằng bàn phím")
        mode_in = input("Nhập lựa chọn [1-2] (Mặc định: 1): ").strip()
        run_mode = "keyboard" if mode_in == "2" else "agent"

        print()
        print("=" * 78)
        print("Đang khởi động mô phỏng với cấu hình:")
        print(f"  - Số làn đường: {lanes} làn")
        print(f"  - Mật độ xe: {traffic}")
        print(f"  - Vận tốc xuất phát: {speed} km/h")
        print(f"  - Dòng xe: {traffic_speed}")
        print(f"  - Chế độ: {run_mode}")
        print("=" * 78)
        print()

        return [
            "--lanes", lanes,
            "--traffic", traffic,
            "--speed", speed,
            "--traffic-speed", traffic_speed,
            "--algo", "dqn",
            "--mode", run_mode,
            "--auto-reset",
        ]
    else:
        print("Lựa chọn không hợp lệ, dùng cấu hình mặc định.")
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
