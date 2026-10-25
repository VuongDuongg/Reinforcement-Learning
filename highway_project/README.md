# Môi trường xe né vật cản và Q-learning

Đã triển khai đường **một chiều, 3 làn**, một xe điều khiển, 12 xe giao thông và một xe hỏng đứng yên. Vật lý, giao thông và va chạm dùng HighwayEnv; giao diện Pygame nội suy vị trí và hướng xe giữa các bước mô phỏng để vẽ ở 60 FPS. Quyết định vẫn ở 10 Hz; nội suy chỉ làm mượt hình, không sửa vật lý và có độ trễ tối đa một bước (0,1 giây).

## Xem xe tự chạy bằng chính sách đã học

Nhấp đúp **`CHAY_XE_TU_HOC.cmd`** ở thư mục gốc. Xe dùng bảng Q đã huấn luyện, tự chuyển episode sau 2 giây khi đến đích/va chạm/hết giờ. **Tab** chuyển sang điều khiển bàn phím; **Space** tạm dừng, **R** chạy lại, **N** đổi seed. Script chỉ huấn luyện nếu chưa có model; khi có model sẽ tải để suy luận.

Đây là **Q-learning dạng bảng**, dùng NumPy trên CPU, không cần cài TensorFlow/PyTorch. MDP là mô hình toán của bài toán chứ không phải thuật toán học. Bảng Q học bằng cập nhật Bellman từ các bước tương tác, không có luật cài sẵn để chọn làn tránh va chạm. Có giới hạn lệnh chuyển làn mới khi lệnh trước chưa hoàn tất để tránh đổi mục tiêu liên tục.

Chính sách này dùng `info["sensors"]`: cảm biến ba làn trong phạm vi 180 m (khoảng trống trước/sau, tốc độ khép khoảng cách), tốc độ xe, làn hiện tại và trạng thái chuyển làn. Các giá trị được chia nhóm thành **648 trạng thái**, mỗi trạng thái có 5 giá trị Q. Bản này mở rộng đầu vào ngoài ma trận quan sát (10, 5); thông tin đích hoặc seed không dùng để chọn hành động. Do gộp trạng thái và thiếu đầy đủ thông tin giao thông, đây là xấp xỉ bài toán quan sát một phần, không phải bằng chứng hội tụ MDP chính xác.

Huấn luyện thực hiện riêng, không làm nặng vòng vẽ giao diện:

```powershell
.\.venv-highway\Scripts\python.exe -m highway_project.train --episodes 180 --eval-episodes 10
.\.venv-highway\Scripts\python.exe -m highway_project.run --mode agent --auto-reset
```

Model: `models/q_learning.npz`. Báo cáo JSON và lịch sử học: `results/q_learning_report.json`. Seed học nhỏ hơn một triệu; seed kiểm tra từ một triệu trở lên. Kiểm tra so sánh với chính sách giữ tốc (bảng Q=0, trước học) trên cùng seed, đường 650 m, một vật cản, với 0/6/12 xe giao thông. Khi đánh giá, epsilon bằng 0 và bảng Q không cập nhật. Tỷ lệ qua bài kiểm tra nhỏ không bảo đảm tránh mọi va chạm khi đổi kịch bản.

Các tệp thuật toán chính: `q_agent.py` (trạng thái, chọn hành động, cập nhật Bellman, lưu/tải bảng Q), `train.py` (huấn luyện/đánh giá), `motion.py` (nội suy chuyển động). `agent.py` giữ mẫu giao diện cho thuật toán khác; launcher hiện dùng `QLearningAgent`, nên cần đổi phần khởi tạo agent trong `run.py` khi tích hợp thuật toán mới.

## Mở ngay trên Windows

Ở thư mục gốc `RL`, nhấp đúp **`CHAY_MOI_TRUONG.cmd`**. Cửa sổ mở ở trạng thái tạm dừng để bạn làm quen. Nhấn **Space** hoặc nút **Chạy / Dừng** để xe bắt đầu chạy.

Nếu máy khác chưa có thư viện: cài Python (bản đã kiểm tra ở máy hiện tại: 3.13.7), chạy `CAI_DAT_MOI_TRUONG.cmd` trước. Script tạo môi trường ảo riêng `.venv-highway` và cài thư viện từ Internet. Không cần NetLogo hoặc TensorFlow để chạy mô phỏng này.

| Phím | Tác dụng |
|---|---|
| ← / → | Chuyển sang làn trái / phải theo hướng lái; trên màn hình tương ứng làn trên / dưới |
| ↑ / ↓ | Tăng / giảm một mức tốc độ mục tiêu: 0, 5, 10, 15, 20, 25, 30 m/s |
| Space | Chạy / tạm dừng |
| R | Khởi tạo lại cùng seed, giữ trạng thái chạy/dừng |
| N | Đổi seed để tạo vị trí xe giao thông mới; giữ trạng thái chạy/dừng |
| Tab | Đổi giữa bàn phím và Q-learning |
| Esc | Thoát |

Xe màu xanh ngọc là xe của bạn, màu xanh lam là xe giao thông, màu đỏ có dấu X là xe hỏng. Xe xuất phát ở làn giữa với vận tốc 20 m/s (72 km/h). Xe hỏng đầu tiên cách điểm xuất phát 160 m. Trong chế độ bàn phím, bạn cần chuyển làn hoặc giảm tốc trước khi tới đó; chế độ agent dùng bảng Q để chọn hành động. Điều khiển tăng/giảm tốc là thay tốc độ mục tiêu, không phải phanh dừng tức thời.

Va chạm, ra khỏi đường hoặc tới đích sẽ kết thúc episode và giữ hình cho bạn quan sát; nhấn R hoặc N để thử tiếp. Đích cách điểm xuất phát 650 m; hết 60 giây mà chưa tới đích là hết thời gian, không tính thành công.

## Tùy chỉnh số xe và vật cản

Chạy PowerShell tại thư mục gốc dự án:

```powershell
.\.venv-highway\Scripts\python.exe -m highway_project.run
.\.venv-highway\Scripts\python.exe -m highway_project.run --traffic 5 --obstacles 2 --seed 7
.\.venv-highway\Scripts\python.exe -m highway_project.run --traffic 0 --obstacles 1
```

Không có `--paused` thì xe chạy ngay. Cho phép 0–40 xe giao thông và 0–3 vật cản. Vật cản đầu luôn ở làn xuất phát để tạo tình huống né; các vật cản tiếp theo cách nhau 180 m và có làn phụ thuộc seed. Chưa có làn khẩn cấp hoặc học đa tác nhân trong bản này.

Các tham số khác nằm trong `LaneObstacleEnv.default_config()` ở `environment.py`: thời gian episode, khoảng cách vật cản, độ dài hành trình, tần suất quyết định và trọng số thưởng.

## Sau này điền thuật toán ở đâu?

Sửa **`agent.py` → `CustomAgent`**:

1. `act(observation, info)`: trả về hành động nguyên từ 0 đến 4.
2. `observe(...)`: nhận kinh nghiệm sau mỗi bước, có riêng `terminated` và `truncated`.
3. `reset(seed=...)`: xóa bộ nhớ episode nếu cần; không mặc định xóa trọng số đã học.

Mẫu `CustomAgent` trong `agent.py` vẫn chỉ giữ tốc; để dùng mẫu đó cần đổi phần khởi tạo trong `run.py`. Chế độ `--mode agent` hiện tải **QLearningAgent** trong `q_agent.py`. Giao diện không gọi `observe()` để huấn luyện; học được thực hiện riêng trong `train.py`.

| ID | Hành động |
|---|---|
| 0 | Chuyển làn trái |
| 1 | Giữ tốc độ mục tiêu |
| 2 | Chuyển làn phải |
| 3 | Tăng tốc độ mục tiêu |
| 4 | Giảm tốc độ mục tiêu |

Đầu vào là `float32`, shape **(10, 5)**. Mỗi dòng có `presence, x, y, vx, vy`; dòng đầu là xe điều khiển, các dòng tiếp theo là các đối tượng gần nhất trong tầm cảm nhận (gồm xe hỏng), dòng thiếu được đệm 0. Dữ liệu được chuẩn hóa theo HighwayEnv, tọa độ các đối tượng xung quanh là tương đối với xe điều khiển. Có giới hạn tầm nhìn và số đối tượng; không phải trạng thái đầy đủ của toàn bộ thế giới. Nếu mạng cần vector, dùng `observation.flatten()` để có 50 phần tử, không dùng mặc định 5/8 chiều của mã NetLogo cũ.

`info` có `is_success`, `crashed`, `distance_m`, `remaining_m`, `obstacles_passed`, `elapsed_s`, `available_actions`, `end_reason` và cảm biến `sensors`. Q-learning dùng cảm biến; các trường tiến độ/đích chỉ dành cho giám sát và đánh giá.

Có thể huấn luyện không cần giao diện:

```python
from highway_project.environment import make_env

env = make_env(config={"vehicles_count": 12, "obstacle_count": 1})
try:
    obs, info = env.reset(seed=42)
    # Thay bằng vòng lặp thuật toán sau này:
    next_obs, reward, terminated, truncated, info = env.step(1)
finally:
    env.close()
```

Hàm thưởng: va chạm -20; ra khỏi đường -10; đi tới đích +10; mỗi bước có `0.03 * speed / 30`, phạt -0.02 khi yêu cầu đổi làn, phạt -0.15 nếu khoảng trống trước <10 m hoặc thời gian tới va chạm <2 giây khi đang tiến gần. Thưởng theo bước phụ thuộc tần suất quyết định; cần giữ tần suất cố định khi so sánh.

## Kiểm tra và chụp ảnh

```powershell
.\.venv-highway\Scripts\python.exe -m unittest highway_project.test_environment highway_project.test_learning -v
.\.venv-highway\Scripts\python.exe -m highway_project.run --headless --steps 30 --screenshot highway_project/results/preview.png
```

Kiểm tra gồm hợp đồng Gymnasium, tái lập seed, vật cản xuất hiện trong quan sát và đứng yên, va chạm kết thúc episode, chuyển làn qua vật cản rồi đến đích, phân biệt `terminated/truncated`, cập nhật Bellman, lưu/tải bảng Q và nội suy vị trí/hướng mà không sửa trạng thái vật lý.

`requirements.txt` cố định phiên bản hai thư viện chính; `requirements-lock.txt` ghi đầy đủ phiên bản được cài trên Windows/Python 3.13 cho khả năng tái lập cùng nền tảng. Số liệu thực nghiệm được ghi từ lần chạy thật trong báo cáo JSON.

Nguồn API: [HighwayEnv](https://highway-env.farama.org/), [hành động](https://highway-env.farama.org/actions/), [tùy biến môi trường](https://highway-env.farama.org/make_your_own/).
