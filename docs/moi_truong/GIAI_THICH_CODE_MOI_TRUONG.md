# GIẢI THÍCH CHI TIẾT MÃ NGUỒN MÔI TRƯỜNG XE TỰ LÁI NÉ VẬT CẢN (LANE OBSTACLE ENVIRONMENT)

> **Tài liệu kỹ thuật toàn diện**  
> **Dự án:** Học Tăng Cường Cho Xe Tự Hành Né Chướng Ngại Vật Trên Cao Tốc Đa Làn  
> **Nền tảng:** HighwayEnv (Farama Foundation) + Gymnasium API + Pygame  
> **Mã nguồn lõi:** [`highway_project/environment/environment.py`](file:///d:/Hoc_Tang_Cuong/highway_project/environment/environment.py)  

---

## 1. TỔNG QUAN KIẾN TRÚC MÔI TRƯỜNG MÔ PHỎNG

### 1.1. Mục Tiêu và Bài Toán Đặt Ra
Trong bài toán xe tự hành (Autonomous Vehicle - AV), mục tiêu cốt lõi là điều khiển một chiếc xe (Ego Vehicle) di chuyển an toàn trên đoạn đường cao tốc dài $1000\text{ m}$ với vận tốc cao ($70 - 108\text{ km/h}$), chủ động nhận biết và né tránh các **chướng ngại vật tĩnh dừng khẩn cấp** (xe chết máy, xe tai nạn) cũng như tương tác an toàn với **dòng xe giao thông chuyển động phức tạp, khó đoán**.

Môi trường `LaneObstacleEnv` được thiết kế nhằm đáp ứng 4 tiêu chuẩn kỹ thuật nghiêm ngặt:
1. **Tuân thủ chuẩn Gymnasium API**: Cung cấp giao diện chuẩn `reset()` và `step()` để bất kỳ thuật toán RL nào (DQN, Double DQN, Dueling DQN, PPO, SAC hay Custom Agent) đều có thể kết nối ngay lập tức mà không cần chỉnh sửa giao thức.
2. **Khả năng tùy biến động linh hoạt (Customizable Scenario)**:
   - **Số làn đường**: Tăng/giảm từ 2 đến 6 làn (`lanes_count = 2..6`).
   - **Mật độ giao thông**: Tùy chọn trực quan theo các preset: **Thưa thớt** (`sparse`: 8 xe), **Bình thường** (`normal`: 20 xe), **Dày đặc** (`dense`: 38 xe), hoặc **Đường vắng** (`none`: 0 xe), kèm tùy chọn nhập số xe tùy ý (0..60 xe).
   - **Tốc độ di chuyển**: Cho phép điều chỉnh vận tốc xuất phát của xe (`initial_speed`), vận tốc tối đa (`max_speed`), và dải tốc độ của dòng xe (`traffic_speed_min`, `traffic_speed_max` qua các chế độ `slow`, `normal`, `fast`).
3. **Cảm biến không gian và Radar đa làn (Enhanced Telemetry)**: Cung cấp đầy đủ ma trận động học chuẩn hóa $(10 \times 5)$ cùng khối radar quét 180m theo từng làn đường, hỗ trợ tính toán cự ly an toàn tức thời ($O(1)$) và phục vụ cơ chế **Action Masking**.
4. **Hiển thị đồ họa thời gian thực mượt mà**: Giao diện Pygame tự động co giãn theo số làn đường, kết hợp bộ đệm nội suy động học `MotionBuffer` giúp tái hiện chuyển động ở tần số 60 FPS mượt mà.

---

### 1.2. Sơ Đồ Kiến Trúc Phân Tầng Hệ Thống

```
┌────────────────────────────────────────────────────────────────────────┐
│                   THUẬT TOÁN HỌC TĂNG CƯỜNG (RL AGENT)                  │
│       (Deep Q-Network, Double DQN, Tabular Q-Learning, PPO, SAC)        │
└───────────────────────────────────▲────────────────────────────────────┘
                       Hành động a_t│ │ Quan sát obs_t, Thưởng r_t,
                         (0..4)     │ │ Terminated, Truncated, Info
┌───────────────────────────────────┴─▼──────────────────────────────────┐
│                   LỚP MÔI TRƯỜNG: LaneObstacleEnv                       │
│              (Gymnasium Registered ID: "LaneObstacle-v0")              │
├────────────────────────────────────────────────────────────────────────┤
│  1. Quản lý kịch bản: Số làn (2..6), Mật độ xe, Dải vận tốc            │
│  2. Bộ điều khiển hành động DiscreteMetaAction                         │
│  3. Hệ thống cảm biến Radar quét đa làn 180m (_sensors)                │
│  4. Hàm phần thưởng đa mục tiêu 6 thành phần (_rewards)                │
│  5. Điều kiện kết thúc & Siêu dữ liệu vi mô (_is_terminated, _info)    │
└───────────────────────────────────▲────────────────────────────────────┘
                                    │ Tương tác vật lý
┌───────────────────────────────────▼────────────────────────────────────┐
│                  LÕI ĐỘNG HỌC & ĐỒ HỌA NỀN TẢNG                         │
│  - Engine HighwayEnv: Động học xe Kinematics, mô hình bám đuôi IDM     │
│  - Giao diện Pygame (HighwayView): Co giãn động, HUD thông số tiếng Việt│
│  - Bộ đệm nội suy MotionBuffer: Nội suy vị trí & góc lái 60 FPS        │
└────────────────────────────────────────────────────────────────────────┘
```

---

### 1.3. Bản Đồ File Mã Nguồn và Phân Trách Nhiệm

Toàn bộ mã nguồn dự án được tổ chức khoa học theo mô hình module hóa cao độ:

| Thư mục / File | Vai trò kiến trúc | Mô tả chi tiết chức năng |
| :--- | :--- | :--- |
| [`highway_project/environment/environment.py`](file:///d:/Hoc_Tang_Cuong/highway_project/environment/environment.py) | **Môi trường lõi (Core Env)** | Định nghĩa lớp `LaneObstacleEnv`, `BrokenCar`, cấu hình mặc định, tạo đường và vật cản, cảm biến đa làn, hàm thưởng, Gymnasium register. |
| [`highway_project/algorithms/dqn_agent.py`](file:///d:/Hoc_Tang_Cuong/highway_project/algorithms/dqn_agent.py) | **Thuật toán DQN** | Triển khai DQN 2013 (mạng đơn không Target Network, MSELoss), mạng nơ-ron `QNetwork`, bộ đệm kinh nghiệm `ReplayBuffer`, chiếu vector trạng thái tương đối 68 chiều, Action Masking và hệ thống chống chết máy (Anti-stall). |
| [`highway_project/algorithms/q_agent.py`](file:///d:/Hoc_Tang_Cuong/highway_project/algorithms/q_agent.py) | **Thuật toán Q-Learning** | Triển khai Q-learning dạng bảng (Tabular), rời rạc hóa trạng thái (648 nhóm), cập nhật Bellman cổ điển. |
| [`highway_project/gui/view.py`](file:///d:/Hoc_Tang_Cuong/highway_project/gui/view.py) | **Giao diện Pygame** | Vẽ đồ họa mặt đường co giãn theo số làn (2..6 làn), vạch kẻ đường, xe mô phỏng, đích đến, Bảng Menu Cài Đặt nổi trung tâm (Popup Modal), thanh điều khiển chính 4 nút và thông báo Toast. |
| [`highway_project/gui/motion.py`](file:///d:/Hoc_Tang_Cuong/highway_project/gui/motion.py) | **Nội suy chuyển động** | Bộ đệm `MotionBuffer` nội suy tuyến tính (LERP) vị trí và góc quay xe giữa các bước quyết định 10 Hz để hiển thị 60 FPS không bị giật lag. |
| [`highway_project/training/train_dqn.py`](file:///d:/Hoc_Tang_Cuong/highway_project/training/train_dqn.py) | **Huấn luyện DQN** | Huấn luyện mạng nơ-ron sâu theo phương pháp Curriculum Learning (tăng dần mật độ xe giao thông từ 0 -> 12 xe), lưu checkpoint `.pth`. |
| [`highway_project/training/plot_results.py`](file:///d:/Hoc_Tang_Cuong/highway_project/training/plot_results.py) | **Trực quan hóa kết quả** | Đọc báo cáo JSON và vẽ biểu đồ so sánh chi tiết hiệu năng giữa DQN và Q-learning dạng bảng. |
| [`highway_project/run.py`](file:///d:/Hoc_Tang_Cuong/highway_project/run.py) | **Điểm khởi chạy & Vòng lặp chính** | Xử lý tham số dòng lệnh CLI, khởi tạo môi trường, điều phối vòng lặp mô phỏng, xử lý sự kiện phím/chuột và thực thi hàm `live_reconfigure` tái cấu hình trực tiếp không reset xe. |
| [`highway_project/launcher.py`](file:///d:/Hoc_Tang_Cuong/highway_project/launcher.py) | **Trình menu console tương tác** | Giao diện dòng lệnh lựa chọn kịch bản thông minh, hỗ trợ mã hóa UTF-8 chống lỗi Unicode trên Windows. |
| [`CHAY_MOI_TRUONG.cmd`](file:///d:/Hoc_Tang_Cuong/CHAY_MOI_TRUONG.cmd) | **Tập lệnh khởi chạy duy nhất** | Script Windows một chạm: tự động kiểm tra môi trường ảo `.venv-highway`, checkpoint mô hình, và khởi động trực tiếp giao diện mô phỏng Pygame tích hợp đầy đủ tính năng. |
| [`CAI_DAT_MOI_TRUONG.cmd`](file:///d:/Hoc_Tang_Cuong/CAI_DAT_MOI_TRUONG.cmd) | **Tập lệnh cài đặt môi trường** | Script tách riêng dành cho việc tạo môi trường ảo Python và tự động cài đặt trọn bộ thư viện phụ thuộc. |
| [`highway_project/tests/`](file:///d:/Hoc_Tang_Cuong/highway_project/tests/) | **Kiểm thử tự động** | Bộ 21 bài test tự động bao quát toàn bộ logic môi trường, hợp đồng Gymnasium, thuật toán DQN, Replay Buffer, giao diện GUI và tính năng tái cấu hình trực tiếp bảo toàn tiến độ xe. |

---

## 2. CẤU HÌNH MÔI TRƯỜNG: `LaneObstacleEnv.default_config()`

Cấu hình mặc định của môi trường được định nghĩa tập trung trong phương thức lớp `default_config()`:

```python
@classmethod
def default_config(cls) -> dict:
    config = super().default_config()
    config.update({
        "lanes_count": 3,
        "vehicles_count": 20,
        "controlled_vehicles": 1,
        "initial_lane_id": 1,
        "initial_speed": 20.0,
        "max_speed": 30.0,
        "traffic_speed_min": 14.0,
        "traffic_speed_max": 28.0,
        "duration": 90.0,
        "simulation_frequency": 30,
        "policy_frequency": 10,
        "route_length": 1000.0,
        "obstacle_count": 1,
        "obstacle_distance": 180.0,
        "obstacle_spacing": 280.0,
        "random_obstacles": False,
        "offroad_terminal": True,
        "unpredictable_traffic": True,
        "vehicle_spacing": 28.0,
        "action": {"type": "DiscreteMetaAction", "target_speeds": [0, 5, 10, 15, 20, 25, 30]},
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 10,
            "features": ["presence", "x", "y", "vx", "vy"],
            "absolute": False,
            "normalize": True,
            "clip": True,
            "see_behind": True,
            "include_obstacles": True,
            "order": "sorted",
        },
        "collision_penalty": -20.0,
        "offroad_penalty": -10.0,
        "speed_weight": 0.03,
        "danger_penalty": -0.15,
        "lane_change_penalty": -0.02,
        "success_reward": 10.0,
    })
    return config
```

### Phân Tích Chi Tiết Từng Thông Số Kỹ Thuật

1. **`lanes_count` (int, mặc định 3)**:
   - Số lượng làn đường của tuyến cao tốc. Môi trường hỗ trợ linh hoạt từ 2 đến 6 làn.
   - Mỗi làn đường có bề rộng quy chuẩn là $4.0\text{ m}$. Toàn bộ mạng đường, dải phân cách và cảm biến radar sẽ tự động tính toán theo tham số này.
2. **`vehicles_count` (int | str, mặc định 20)**:
   - Số lượng xe giao thông lưu thông cùng chiều trên đường.
   - Hỗ trợ các preset thân thiện:
     - `"sparse"` / `"thua"`: 8 xe (giao thông thưa thớt).
     - `"normal"` / `"vua"`: 20 xe (mật độ trung bình bình thường).
     - `"dense"` / `"day"`: 38 xe (giao thông dày đặc, đòi hỏi lách làn và phanh né chủ động).
     - `"none"` / `"0"`: 0 xe (đường vắng tanh, chỉ có xe hỏng).
     - Hoặc số nguyên trực tiếp từ $0$ đến $60$ xe.
3. **`initial_speed` (float, mặc định 20.0 m/s = 72 km/h) & `max_speed` (float, mặc định 30.0 m/s = 108 km/h)**:
   - `initial_speed`: Vận tốc tức thời và vận tốc mục tiêu xuất phát của xe ego.
   - `max_speed`: Vận tốc tối đa cho phép trên cao tốc. Khi tham số này thay đổi, dải hành động `target_speeds` sẽ tự động chia lưới đều từ $0$ đến `max_speed`.
4. **`traffic_speed_min` & `traffic_speed_max` (float, mặc định 14.0 m/s & 28.0 m/s)**:
   - Dải tốc độ ngẫu nhiên của các xe giao thông xung quanh (tương đương $\approx 50 - 100\text{ km/h}$).
   - Thông qua CLI `--traffic-speed`, người dùng có thể kích hoạt các dải: `slow` (36 - 65 km/h), `normal` (50 - 95 km/h), hoặc `fast` (75 - 115 km/h).
5. **`simulation_frequency` (30 Hz) & `policy_frequency` (10 Hz)**:
   - Tác nhân AI đưa ra quyết định ở tần số $10\text{ Hz}$ (cứ mỗi $\Delta t = 0.1\text{ s}$ ra một lệnh).
   - Động cơ vật lý bên dưới thực hiện $3$ bước tích phân vi phân ($\frac{30}{10} = 3$) trong mỗi chu kỳ quyết định, đảm bảo tính ổn định số học và sự mượt mà của quỹ đạo bám làn/chuyển làn.
6. **`route_length` (1000.0 m)**:
   - Chiều dài chặng đường cần hoàn thành. Tọa độ xuất phát $x_{\text{start}} = 80.0\text{ m}$, do đó tọa độ vạch đích là $x_{\text{goal}} = 80.0 + 1000.0 = 1080.0\text{ m}$.
7. **`obstacle_count` (int, 0..3)**:
   - Số lượng xe hỏng chắn làn cố định trên đường (mặc định 1 hoặc 2 xe).

---

## 3. CƠ CHẾ SINH XE VÀ VẬT CẢN: `_create_vehicles()`

Phương thức `_create_vehicles()` chịu trách nhiệm khởi tạo thế giới vật lý mỗi khi môi trường được `reset()`.

```python
def _create_vehicles(self) -> None:
    lanes_count = int(self.config.get("lanes_count", 3))
    if not (2 <= lanes_count <= 6) or self.config.get("controlled_vehicles", 1) != 1:
        raise ValueError("Bản môi trường này hỗ trợ từ 2 đến 6 làn và 1 xe điều khiển.")

    # 1. Chuẩn hóa mật độ xe từ Preset chuỗi hoặc số nguyên
    raw_count = self.config.get("vehicles_count", 20)
    if isinstance(raw_count, str):
        key = raw_count.strip().lower()
        if key in TRAFFIC_PRESETS:
            vcount = TRAFFIC_PRESETS[key]
        else:
            try:
                vcount = int(key)
            except ValueError:
                raise ValueError(f"vehicles_count '{raw_count}' không hợp lệ.")
    else:
        vcount = int(raw_count)
    self.config["vehicles_count"] = max(0, min(60, vcount))

    # Gán nhãn mật độ hiển thị
    if vcount == 0: self.traffic_density = "none"
    elif vcount <= 10: self.traffic_density = "sparse"
    elif vcount >= 30: self.traffic_density = "dense"
    else: self.traffic_density = "normal"
```

### 3.1. Khởi Tạo Xe Điều Khiển (Ego Vehicle)

```python
lane_id = int(self.config.get("initial_lane_id", 1))
if lane_id not in range(lanes_count):
    raise ValueError(f"initial_lane_id phải nằm trong khoảng 0 đến {lanes_count - 1}.")

lane = self.road.network.get_lane(("0", "1", lane_id))
init_speed = float(self.config.get("initial_speed", 20.0))
ego = self.action_type.vehicle_class(self.road, lane.position(80.0, 0), heading=0, speed=init_speed)
ego.target_speed = init_speed
ego.color = (54, 211, 179)  # Màu xanh mint nhận diện
self.controlled_vehicles = [ego]
self.road.vehicles.append(ego)
self.start_x = float(ego.position[0])
self.goal_x = self.start_x + self.config["route_length"]
```
* Ego car được đặt tại vị trí dọc $x = 80.0\text{ m}$ trên làn xuất phát (mặc định làn 1 - làn giữa).
* Đặt thuộc tính `ego.target_speed = init_speed` để bộ điều khiển PID giữ đúng dải vận tốc mong muốn ngay từ thời điểm $t=0$, không bị trễ ga.

### 3.2. Khởi Tạo Vật Cản Cố Định (`BrokenCar`)
Vật cản được định nghĩa qua lớp `BrokenCar(Obstacle)`:
```python
class BrokenCar(Obstacle):
    """RoadObject đứng yên: không có bộ điều khiển tự tăng tốc."""
    LENGTH = 5.0
    WIDTH = 2.0
```
Khác với xe thông thường kế thừa `Vehicle`, `BrokenCar` là một `RoadObject` tĩnh với vận tốc tuyệt đối $v = 0.0\text{ m/s}$. Nó không sở hữu bộ điều khiển gia tốc hay chuyển làn, do đó đứng bất động hoàn toàn suốt thời gian mô phỏng.

#### Thuật Toán Phân Bố Vật Cản Thông Minh Theo Phân Đoạn (Smart Sector Distribution):
Khi bật cờ `random_obstacles = True`, môi trường sử dụng thuật toán phân bổ theo từng sector độc lập:
```python
route_len = float(self.config["route_length"])
min_start = min(130.0, max(60.0, route_len * 0.2))
safe_goal = min(70.0, max(30.0, route_len * 0.1))
span = max(40.0, route_len - min_start - safe_goal)
sector_len = span / count

for i in range(count):
    obstacle_lane = int(self.np_random.integers(0, lanes_count))
    margin = min(15.0, sector_len * 0.1)
    x_min = self.start_x + min_start + i * sector_len + margin
    x_max = self.start_x + min_start + (i + 1) * sector_len - margin
    x_pos = float(self.np_random.uniform(x_min, max(x_min + 1.0, x_max)))
    obstacle = BrokenCar.make_on_lane(
        self.road, ("0", "1", obstacle_lane), x_pos, speed=0.0
    )
    self.obstacles.append(obstacle)
    self.road.objects.append(obstacle)
```
* **Đảm bảo cự ly an toàn**: Vật cản đầu tiên luôn cách điểm xuất phát tối thiểu $> 100\text{ m}$ (đủ thời gian cho AI tăng tốc ổn định).
* **Không bị dồn cục**: Tuyến đường chia thành các sector đều nhau, mỗi sector chỉ chứa duy nhất 1 vật cản, đảm bảo khoảng cách giữa 2 vật cản luôn $\ge 150\text{ m}$.
* **Phân bổ đa làn**: Làn của vật cản được lấy ngẫu nhiên độc lập trong khoảng $[0, \text{lanes\_count} - 1]$.

### 3.3. Khởi Tạo Dòng Xe Giao Thông IDM Ngẫu Nhiên
Các xe giao thông xung quanh là các `IDMVehicle` mô phỏng hành vi lái xe của con người:
```python
min_spacing = float(self.config.get("vehicle_spacing", 28.0))
unpredictable = bool(self.config.get("unpredictable_traffic", True))
spd_min = float(self.config.get("traffic_speed_min", 14.0))
spd_max = float(self.config.get("traffic_speed_max", 28.0))

while len(self.road.vehicles) - 1 < int(self.config["vehicles_count"]):
    traf_lane_id = int(self.np_random.integers(0, lanes_count))
    x = float(self.np_random.uniform(0, self.goal_x + 100))
    occupied = [*self.road.vehicles, *self.obstacles]
    
    # Kiểm tra tránh sinh đè lên xe khác hoặc vật cản
    if any(obj.lane_index[2] == traf_lane_id and abs(obj.position[0] - x) < min_spacing for obj in occupied):
        continue
    if abs(x - self.start_x) < 40:
        continue

    if unpredictable:
        speed = float(self.np_random.uniform(spd_min, spd_max))
        vehicle = IDMVehicle.make_on_lane(self.road, ("0", "1", traf_lane_id), x, speed=speed)
        vehicle.target_speed = float(self.np_random.uniform(spd_min, max(spd_min + 1.0, spd_max)))
        vehicle.randomize_behavior()
        # Tính cách lái xe khó đoán: hay đổi làn, tạt đầu
        vehicle.POLITENESS = float(self.np_random.uniform(0.0, 0.35))
        vehicle.LANE_CHANGE_MIN_ACC_GAIN = float(self.np_random.uniform(0.1, 0.35))
        vehicle.LANE_CHANGE_MAX_BRAKING_IMPOSED = float(self.np_random.uniform(1.8, 3.8))
        vehicle.enable_lane_change = True
    else:
        vehicle = IDMVehicle.make_on_lane(self.road, ("0", "1", traf_lane_id), x, speed=float(self.np_random.uniform(spd_min, spd_max)))
        vehicle.randomize_behavior()
    self.road.vehicles.append(vehicle)
```
* **Mô hình IDM + MOBIL**: `POLITENESS` (độ lịch sự) thấp ($0.0 - 0.35$) khiến các xe khác thường xuyên chủ động lách làn, tạt đầu nếu thấy làn bên cạnh có thể chạy nhanh hơn. Điều này tạo ra một môi trường thử thách thực thụ cho tác nhân AI.

---

## 4. KHÔNG GIAN HÀNH ĐỘNG: `DiscreteMetaAction`

Hệ thống điều khiển áp dụng không gian hành động rời rạc cấp cao (Meta-Actions) gồm 5 lựa chọn:

```python
ACTIONS = {
    0: "LANE_LEFT",   # Chuyển sang làn liền kề bên trái
    1: "IDLE",        # Giữ nguyên làn và vận tốc mục tiêu hiện tại
    2: "LANE_RIGHT",  # Chuyển sang làn liền kề bên phải
    3: "FASTER",      # Tăng 1 nấc tốc độ mục tiêu (+5 m/s)
    4: "SLOWER",      # Giảm 1 nấc tốc độ mục tiêu (-5 m/s)
}
```

### Nguyên Lý Động Học Bên Dưới
1. **Không dịch chuyển tọa độ tức thời (Non-teleportation)**:
   Khi nhận hành động `LANE_LEFT` (0) hoặc `LANE_RIGHT` (2), môi trường không ngay lập tức đổi tọa độ ngang $y$ của xe sang làn mới. Thay vào đó, bộ điều khiển `MDPVehicle` thiết lập chỉ số làn mục tiêu mới (`target_lane_index`).
2. **Quỹ đạo chuyển làn mềm mại (Smooth Lane-Change Trajectory)**:
   Bộ điều khiển góc lái tỷ lệ (Proportional Steering Controller) trong engine sẽ tính toán góc đánh lái $\psi$ theo đường cong mượt mà để xe chuyển dần sang tâm làn mới trong khoảng thời gian $1.0 - 1.5\text{ s}$.
3. **Lưới vận tốc mục tiêu rời rạc**:
   Hành động `FASTER` (3) và `SLOWER` (4) di chuyển con trỏ trên lưới vận tốc rời rạc:
   $$v_{\text{targets}} = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]\text{ m/s}$$
   (hoặc được nội suy đều tới `max_speed` nếu người dùng cấu hình vận tốc cao hơn). Gia tốc xe tuân theo giới hạn vật lý của động cơ ($a_{\max} = 5.0\text{ m/s}^2$, $a_{\min} = -6.0\text{ m/s}^2$).

---

## 5. KHÔNG GIAN QUAN SÁT & RADAR CẢM BIẾN ĐA LÀN: `_sensors()`

Môi trường cung cấp hai tầng quan sát: quan sát ma trận chuẩn cho mô hình học máy và thông tin cảm biến radar vi mô phục vụ trích xuất đặc trưng và Action Masking.

### 5.1. Quan Sát Ma Trận Chuẩn (Gymnasium Observation)
Quan sát `obs` trả về từ `reset()` và `step()` có shape **$(10, 5)$**, kiểu `float32`:
* Dòng 0: Đặc trưng của chính xe Ego (tọa độ và vận tốc tuyệt đối đã được chuẩn hóa).
* Dòng 1 đến 9: Đặc trưng của 9 phương tiện / chướng ngại vật gần xe Ego nhất, sắp xếp theo thứ tự cự ly tăng dần.
* 5 cột thuộc tính bao gồm:
  1. `presence` (1.0 nếu có xe, 0.0 nếu là dòng đệm trống).
  2. $x$: Tọa độ dọc tương đối so với xe Ego (chuẩn hóa trên phạm vi tầm nhìn).
  3. $y$: Tọa độ ngang tương đối so với xe Ego.
  4. $v_x$: Vận tốc dọc tương đối ($v_{x,\text{other}} - v_{x,\text{ego}}$).
  5. $v_y$: Vận tốc ngang tương đối.

### 5.2. Khối Radar Cảm Biến Không Gian Đa Làn: `_sensors()`

Phương thức `_sensors()` quét toàn bộ các phương tiện và vật cản trong bán kính $180\text{ m}$ theo từng làn đường:

```python
def _sensors(self) -> dict:
    ego = self.vehicle
    lanes_count = int(self.config.get("lanes_count", 3))
    front = [180.0] * lanes_count
    rear = [180.0] * lanes_count
    front_closing = [0.0] * lanes_count
    rear_closing = [0.0] * lanes_count

    for obj in [*self.road.vehicles, *self.road.objects]:
        if obj is ego:
            continue
        dx = float(obj.position[0] - ego.position[0])
        if abs(dx) > 180:
            continue
        # Trừ nửa chiều dài 2 xe để tính khoảng hở thực giữa 2 cản xe
        gap = max(0.0, abs(dx) - (obj.LENGTH + ego.LENGTH) / 2)
        for lane_id in range(lanes_count):
            # Xe đang chuyển làn có thể chiếm cả 2 làn nếu mép xe lấn sang
            if abs(float(obj.position[1]) - lane_id * 4) > 2.7:
                continue
            if dx >= 0 and gap < front[lane_id]:
                front[lane_id] = gap
                front_closing[lane_id] = float(ego.speed - obj.speed)
            elif dx < 0 and gap < rear[lane_id]:
                rear[lane_id] = gap
                rear_closing[lane_id] = float(obj.speed - ego.speed)

    lane_id = int(ego.lane_index[2])
    target_lane = int(ego.target_lane_index[2])
    changing = target_lane != lane_id or abs(float(ego.position[1]) - target_lane * 4) > 0.25
    ego_front_gap = front[lane_id] if 0 <= lane_id < lanes_count else 180.0
    left_clear = (lane_id > 0) and (front[lane_id - 1] > 20.0) and (rear[lane_id - 1] > 15.0)
    right_clear = (lane_id < lanes_count - 1) and (front[lane_id + 1] > 20.0) and (rear[lane_id + 1] > 15.0)
    density = getattr(self, "traffic_density", "normal")

    return {
        "lane": lane_id,
        "lanes_count": lanes_count,
        "traffic_density": density,
        "speed": float(ego.speed),
        "speed_kmh": float(ego.speed * 3.6),
        "changing": bool(changing),
        "target_lane": target_lane,
        "front_gap": front,
        "rear_gap": rear,
        "front_closing": front_closing,
        "rear_closing": rear_closing,
        "ego_front_gap": float(ego_front_gap),
        "left_clear": bool(left_clear),
        "right_clear": bool(right_clear),
    }
```

#### Ý Nghĩa Toán Học & Vai Trò Trong Thuật Toán RL:
* **Tính toán khoảng hở thân xe thực (`gap`)**: $gap = |x_2 - x_1| - \frac{L_1 + L_2}{2}$. Không lấy khoảng cách giữa hai tâm xe mà tính chính xác khoảng cách giữa đuôi xe trước và đầu xe sau.
* **Xe chuyển làn chiếm hai làn**: Điều kiện `abs(position[1] - lane_id * 4) <= 2.7` phát hiện chính xác khi một xe đang lạng lách hoặc chuyển dở làn, cảnh báo cho cả 2 làn đường liên quan.
* **Cảm biến tương đối (Ego-Centric)**:
  - `ego_front_gap`: Khoảng cách tới vật cản phía trước trên chính làn hiện tại, truy xuất tức thời $O(1)$.
  - `left_clear` & `right_clear`: Cờ boolean đánh giá làn trái/phải có đủ điều kiện an toàn chuyển làn hay không ($gap_{\text{front}} > 20\text{ m}$ và $gap_{\text{rear}} > 15\text{ m}$).
  - `changing`: Cờ báo xe đang chuyển làn, dùng để kích hoạt **Action Masking** (cấm ra lệnh rẽ trái/phải dồn dập).

---

## 6. HÀM PHẦN THƯỞNG ĐA MỤC TIÊU: `_rewards()` & `_reward()`

Hàm phần thưởng được thiết kế nhằm cân bằng tối ưu giữa 3 mục tiêu cạnh tranh: **An toàn tuyệt đối (Safety) - Tốc độ hành trình (Efficiency) - Độ êm dịu khi lái (Comfort)**.

```python
def _rewards(self, action: int | None) -> dict[str, float]:
    terms = {"collision": 0.0, "offroad": 0.0, "speed": 0.0, "lane_change": 0.0, "success": 0.0, "danger": 0.0}
    if self.vehicle.crashed:
        terms["collision"] = float(self.config["collision_penalty"])
    elif not self.vehicle.on_road:
        terms["offroad"] = float(self.config["offroad_penalty"])
    else:
        terms["speed"] = self.config["speed_weight"] * max(0.0, float(self.vehicle.speed)) / 30.0
        if action in (0, 2):
            terms["lane_change"] = float(self.config["lane_change_penalty"])
        if self._success():
            terms["success"] = float(self.config["success_reward"])
        sensor = self._sensors()
        lane = sensor["lane"]
        if 0 <= lane < len(sensor["front_gap"]):
            gap = sensor["front_gap"][lane]
            closing = sensor["front_closing"][lane]
            if gap < 10 or (closing > 0.1 and gap / closing < 2.0):
                terms["danger"] = float(self.config["danger_penalty"])
    return terms

def _reward(self, action: int) -> float:
    return float(sum(self._rewards(action).values()))
```

### Bảng Giải Thích Chi Tiết 6 Thành Phần Điểm Thưởng:

$$r_t = r_{\text{collision}} + r_{\text{offroad}} + r_{\text{speed}} + r_{\text{lane\_change}} + r_{\text{success}} + r_{\text{danger}}$$

| Thành phần | Khóa cấu hình | Giá trị | Điều kiện áp dụng & Ý nghĩa |
| :--- | :--- | :---: | :--- |
| **$r_{\text{collision}}$** | `collision_penalty` | **$-20.0$** | Phạt cực nặng khi xe va chạm với bất kỳ xe nào khác hoặc vật cản. Tập mô phỏng dừng ngay lập tức. |
| **$r_{\text{offroad}}$** | `offroad_penalty` | **$-10.0$** | Phạt nặng khi xe mất lái lao ra ngoài mép cỏ lề đường. |
| **$r_{\text{speed}}$** | `speed_weight` | **$+0.03 \times \frac{v}{30}$** | Thưởng duy trì tốc độ cao an toàn (với $v=30\text{ m/s}$ nhận tối đa $+0.03$/bước). Ngăn chặn xe tự phanh dừng lại để né phạt. |
| **$r_{\text{lane\_change}}$**| `lane_change_penalty`| **$-0.02$** | Phạt nhẹ mỗi khi tác nhân yêu cầu chuyển làn (action 0 hoặc 2), khuyến khích giữ làn ổn định, tránh đánh võng. |
| **$r_{\text{success}}$** | `success_reward` | **$+10.0$** | Thưởng lớn khi xe hoàn thành trọn vẹn $1000\text{ m}$ và cán đích an toàn. |
| **$r_{\text{danger}}$** | `danger_penalty` | **$-0.15$** | Phạt cảnh báo nguy hiểm khi khoảng cách phía trước $< 10\text{ m}$ hoặc thời gian chạm va $\text{TTC} = \frac{gap}{closing} < 2.0\text{ s}$. Dạy AI biết giảm tốc hoặc đổi làn từ xa trước khi xảy ra va chạm. |

> **Nguyên tắc phân nhánh độc quyền:** Nếu xe bị va chạm (`crashed`), chỉ duy nhất số hạng `collision` nhận điểm ($-20.0$), các số hạng khác bằng 0. Tác nhân không bao giờ vừa bị phạt va chạm vừa được cộng điểm tốc độ.

---

## 7. ĐIỀU KIỆN KẾT THÚC VÀ SIÊU DỮ LIỆU: `_info()`

```python
def _success(self) -> bool:
    return bool(self.vehicle.position[0] >= self.goal_x and not self.vehicle.crashed and self.vehicle.on_road)

def _is_terminated(self) -> bool:
    return bool(super()._is_terminated() or self._success())
```

### 7.1. Phân Biệt Rạch Ròi Giữa `terminated` và `truncated` (Chuẩn Gymnasium)
* **`terminated == True` (Kết thúc vật lý)**:
  Xảy ra khi trạng thái đạt tới điểm kết thúc của bài toán MDP:
  1. Xe va chạm (`crashed`).
  2. Xe văng ra khỏi đường (`not on_road`).
  3. Xe cán đích thành công (`_success() == True`).
  *Ý nghĩa học máy:* Giá trị tương lai $V(s') = 0$, không bootstrap giá trị Bellman qua ranh giới này.
* **`truncated == True` (Hết thời gian cho phép)**:
  Xảy ra khi thời gian mô phỏng vượt quá `duration` (mặc định 90s, tương đương 900 bước). Xe vẫn hoàn toàn an toàn và chưa va chạm.
  *Ý nghĩa học máy:* Tác nhân **vẫn được phép bootstrap** $V(s') = \max_{a'} Q(s', a')$ vì đây chỉ là giới hạn giả định về thời gian cắt mẫu.

### 7.2. Siêu Dữ Liệu Vi Mô Trả Về Trong `info`

```python
def _info(self, obs: np.ndarray, action: int | None = None) -> dict:
    info = super()._info(obs, action)
    success = self._success()
    sensors = self._sensors()
    info.update({
        "is_success": success,
        "distance_m": max(0.0, float(self.vehicle.position[0] - self.start_x)),
        "remaining_m": max(0.0, float(self.goal_x - self.vehicle.position[0])),
        "obstacles_passed": int(sum(self.vehicle.position[0] > obj.position[0] + 5 for obj in self.obstacles)),
        "obstacles_total": len(self.obstacles),
        "elapsed_s": float(self.time),
        "speed": float(self.vehicle.speed),
        "speed_kmh": float(self.vehicle.speed * 3.6),
        "available_actions": self.action_type.get_available_actions(),
        "sensors": sensors,
        "end_reason": (
            "collision" if self.vehicle.crashed else
            "off_road" if not self.vehicle.on_road else
            "success" if success else
            "time_limit" if self._is_truncated() else "running"
        ),
    })
    return info
```

---

## 8. GIAO DIỆN PYGAME VÀ BỘ ĐỆM NỘI SUY: `HighwayView` & `MotionBuffer`

### 8.1. Co Giãn Mặt Đường Động Theo Số Làn Đường Trong `HighwayView`
Khi số làn đường thay đổi từ 2 đến 6 làn, lớp `HighwayView` trong [`highway_project/gui/view.py`](file:///d:/Hoc_Tang_Cuong/highway_project/gui/view.py) tự động tính toán lại vị trí đỉnh và chiều cao làn để toàn bộ lòng đường luôn nằm vừa vặn, cân đối trong khung hình $1280 \times 800$:

```python
lanes_count = int(env.config.get("lanes_count", 3))
if lanes_count <= 3:
    road_top = 294; lane_height = 72
elif lanes_count == 4:
    road_top = 278; lane_height = 64
elif lanes_count == 5:
    road_top = 268; lane_height = 54
else:  # 6 làn
    road_top = 260; lane_height = 46

road_height = lanes_count * lane_height
road_bottom = road_top + road_height
```
* Tự động vẽ các vạch đứt phân làn cho mọi hàng `row in range(1, lanes_count)`.
* Hiển thị bảng HUD trạng thái với nhãn tiếng Việt trực quan: `Mật độ: Thưa thớt / Bình thường / Dày đặc / Không có xe`.

### 8.2. Nội Suy Chuyển Động 60 FPS Bằng `MotionBuffer`
Tác nhân AI ra quyết định ở tần số $10\text{ Hz}$ ($0.1\text{ s}$/bước). Nếu vẽ trực tiếp trạng thái này lên màn hình, xe sẽ bị giật khựng 10 lần mỗi giây. Bộ đệm `MotionBuffer` giải quyết triệt để vấn đề này:
* Lưu lại hai ảnh chụp trạng thái (Snapshot): vị trí trước bước tích phân $p_{\text{prev}}$ và vị trí sau bước tích phân $p_{\text{curr}}$.
* Giữa hai chu kỳ quyết định, tính toán hệ số $\alpha \in [0, 1]$ dựa vào thời gian thực trong bộ tích lũy (`accumulator / step_dt`).
* Thực hiện phép nội suy tuyến tính (LERP) tọa độ:
  $$p_{\text{draw}} = (1 - \alpha) \cdot p_{\text{prev}} + \alpha \cdot p_{\text{curr}}$$
* Nội suy góc quay thân xe (Heading) theo cung lượng giác ngắn nhất:
  $$\Delta \theta = ((\theta_{\text{curr}} - \theta_{\text{prev}} + \pi) \pmod{2\pi}) - \pi$$
  $$\theta_{\text{draw}} = \theta_{\text{prev}} + \alpha \cdot \Delta \theta$$
Kết quả mang lại trải nghiệm đồ họa 60 FPS mượt mà tuyệt đối mà không làm sai lệch tính toán vật lý của engine.

### 8.3. Bảng Menu Cài Đặt Nổi Trung Tâm (Popup Modal Settings)
Nhằm mang lại trải nghiệm tương tác trực quan cao cấp, giao diện mô phỏng tích hợp bảng **Menu Cài Đặt dạng Modal Popup** nổi bật ở trung tâm màn hình:
* **Cơ chế kích hoạt linh hoạt:** Người dùng có thể nhấn phím **`M`** trên bàn phím hoặc click chuột vào nút **`☰ CÀI ĐẶT & KỊCH BẢN (M)`** ở thanh điều khiển chính.
* **Lớp phủ mờ nền tối (Dim Backdrop):** Khi mở menu (`self.menu_open = True`), màn hình được phủ một lớp bán trong suốt màu tối (`RGBA(8, 14, 22, 215)`), tạo chiều sâu thị giác và tập trung hoàn toàn sự chú ý vào bảng cài đặt.
* **Hộp thoại Modal kích thước $960 \times 580\text{ px}$:** Nằm cân đối chính giữa màn hình với nền xanh đen đậm (`#0f172a`), viền ngọc lam (`#36d3b9`), chia thành 6 phân vùng cấu hình trực quan:
  1. **Số làn đường cao tốc (2..6 làn):** Tùy chọn chuyển đổi nhanh giữa 2 làn, 3 làn (chuẩn), 4 làn, 5 làn, 6 làn.
  2. **Mật độ xe giao thông:** Tùy chọn 4 cấp độ: Thưa thớt (8 xe), Bình thường (20 xe), Dày đặc (38 xe), Không có xe (0 xe).
  3. **Vận tốc xe của bạn:** Các mức tốc độ tức thời: 54 km/h, 72 km/h (chuẩn), 90 km/h, 108 km/h.
  4. **Tốc độ dòng xe:** Chậm (`slow`: 36-65 km/h), Bình thường (`normal`: 50-95 km/h), Nhanh (`fast`: 75-115 km/h).
  5. **Chọn nhanh kịch bản mẫu (Presets):** Kịch bản 1 (2 làn · Thưa · 90 km/h), Kịch bản 2 (3 làn · Chuẩn · 72 km/h), Kịch bản 3 (5 làn · Dày · Nhanh), Kịch bản 4 (4 làn · Đường vắng né xe hỏng).
  6. **Chế độ điều khiển & Tác vụ hệ thống:** Chuyển đổi giữa AI tự lái (DQN / Q-learning) và Người lái bàn phím, bật/tắt Tự động chuyển tập (Auto-reset), Chạy lại từ đầu (`R`), Đóng menu (`Esc` / `M`).

### 8.4. Cơ Chế Tái Cấu Hình Trực Tiếp (Live Reconfiguration - KHÔNG RESET XE)
Một trong những cải tiến kỹ thuật quan trọng nhất của dự án là tính năng **Tái cấu hình trực tiếp trong thời gian thực (Live Reconfiguration)** thông qua hàm `live_reconfigure(...)` trong [`highway_project/run.py`](file:///d:/Hoc_Tang_Cuong/highway_project/run.py):
* **Vấn đề của phương pháp cũ:** Trước đây, mỗi khi thay đổi số làn, mật độ xe hoặc vận tốc, môi trường buộc phải gọi `env.reset()`, khiến xe bị đưa giật ngược về vạch xuất phát, làm mất toàn bộ quãng đường đã đi và điểm thưởng tích lũy.
* **Giải pháp đột phá của `live_reconfigure(...)`:**
  1. **Tái dựng mạng đường động:** Khi đổi số làn (ví dụ từ 3 lên 5 làn), hàm tái tạo mạng lưới đường `RoadNetwork.straight_road_network(current_lanes, length)` trực tiếp trên đối tượng `road.network` đang chạy.
  2. **Ánh xạ lại vị trí và làn đường toàn bộ xe:** Duyệt qua mọi phương tiện và vật cản `[*road.vehicles, *road.objects]`, kẹp chỉ số làn `clamped_lane = min(curr_lane_id, current_lanes - 1)` và dịch chuyển vị trí $y$ của các xe ngoài biên về lòng đường mới một cách an toàn.
  3. **Cập nhật vận tốc tức thời:** Điều chỉnh vận tốc tức thời và vận tốc mục tiêu của xe ego (`unwrapped.vehicle.speed = target_speed`) mà không làm ngắt quãng mô phỏng vật lý.
  4. **Cập nhật mật độ xe giao thông linh hoạt:** Nếu tăng số xe, môi trường sinh thêm xe nền phía trước; nếu giảm số xe, môi trường giữ lại xe gần ego nhất và giải phóng các xe ở quá xa.
  5. **BẢO TOÀN TIẾN ĐỘ XE TUYỆT ĐỐI:** Xe của bạn tiếp tục lăn bánh tiến về phía trước, giữ nguyên tọa độ $x$, quãng đường đã đi `distance_m`, thời gian đã trôi qua `elapsed_s` và tổng điểm thưởng tích lũy `total_reward`!

### 8.5. Hệ Thống Thông Báo Nổi Toast Thời Gian Thực (`show_toast`)
Phương thức `show_toast(text, duration_ms=2500, color=MINT)` trong `HighwayView` cung cấp phản hồi thị giác tức thì cho mọi thao tác người dùng:
* Khi người dùng click nút hoặc bấm phím tắt thay đổi số làn, mật độ hoặc tốc độ, một thông báo nổi (Toast banner) viền sáng ngọc lam sẽ xuất hiện tại góc dưới bên phải màn hình.
* Hiển thị nội dung chi tiết: `"✓ Đã chuyển sang 5 làn (Xe tiếp tục chạy)"`, `"✓ Mật độ xe: Dày đặc (38 xe) (Xe tiếp tục chạy)"`, `"✓ Vận tốc xe: 90 km/h"`.
* Thông báo tự động mờ dần và biến mất sau $2.5\text{ giây}$ mà không làm che khuất các thông số radar hay đường chạy.

### 8.6. Thanh Điều Khiển Chính Tinh Gọn (Main View Bottom Bar)
Khi đóng Menu, màn hình chính duy trì giao diện tối giản, tập trung và chuyên nghiệp chỉ với **4 nút bấm chức năng ở đáy màn hình**:
1. **`Space · Tiếp tục / Tạm dừng`** (Rộng 175 px): Tạm dừng hoặc tiếp tục mô phỏng vật lý.
2. **`Tab · AI: DQN / Lái: Phím`** (Rộng 210 px): Chuyển đổi qua lại giữa chế độ AI tự lái và người điều khiển bàn phím.
3. **`☰ CÀI ĐẶT & KỊCH BẢN (M)`** (Rộng 270 px, nút nổi bật viền ngọc lam): Mở Bảng Menu Cài Đặt Modal.
4. **`R · Chạy lại từ đầu`** (Rộng 185 px): Khởi động lại episode mới từ vạch xuất phát.
* Kèm theo chân trang hướng dẫn phím tắt đầy đủ (`M`, `Space`, `Tab`, `R`, `Esc`, `L`, `T`, `S`, `F`, `1..4`) giúp thao tác cực kỳ nhanh chóng.

---

## 9. HƯỚNG DẪN KẾT NỐI THUẬT TOÁN HỌC TĂNG CƯỜNG

Đoạn mã hoàn chỉnh dưới đây minh họa cách một thuật toán RL độc lập khởi tạo môi trường, nhận quan sát, áp dụng Action Masking và thực thi bước học:

```python
import gymnasium as gym
from highway_project.environment.environment import make_env, ACTIONS

# 1. Khởi tạo môi trường với các thông số tùy biến
env = make_env(
    seed=42,
    config={
        "lanes_count": 4,              # Cao tốc 4 làn đường
        "vehicles_count": "dense",     # Mật độ xe dày đặc (38 xe)
        "initial_speed": 22.2,         # Xuất phát ở 80 km/h (22.2 m/s)
        "max_speed": 33.3,             # Vận tốc tối đa 120 km/h
        "traffic_speed_min": 14.0,     # Dải tốc độ xe khác 50 - 95 km/h
        "traffic_speed_max": 26.5,
        "obstacle_count": 2,           # 2 xe hỏng chắn làn ngẫu nhiên
        "random_obstacles": True,
    }
)

obs, info = env.reset(seed=42)
done = False
total_reward = 0.0

while not done:
    sensors = info["sensors"]
    valid_actions = info["available_actions"]
    
    # 2. Áp dụng Action Masking: Không ra lệnh đổi làn khi đang trong quỹ đạo chuyển làn
    if sensors["changing"]:
        valid_actions = [a for a in valid_actions if a not in (0, 2)]
        
    # 3. Lựa chọn hành động (dựa trên Q-network hoặc chính sách policy)
    # Ví dụ logic phản xạ an toàn:
    if sensors["ego_front_gap"] < 35.0:  # Phía trước có vật cản gần
        if sensors["left_clear"] and 0 in valid_actions:
            action = 0  # Lách sang làn trái
        elif sensors["right_clear"] and 2 in valid_actions:
            action = 2  # Lách sang làn phải
        else:
            action = 4  # Giảm tốc độ khẩn cấp
    else:
        # Đường thoáng: Giữ tốc độ hoặc tăng tốc
        action = 3 if sensors["speed_kmh"] < 100.0 and 3 in valid_actions else 1

    # 4. Bước môi trường
    next_obs, reward, terminated, truncated, next_info = env.step(action)
    total_reward += reward
    done = terminated or truncated

    # 5. Lưu transition vào Replay Buffer để huấn luyện
    # buffer.push(obs, action, reward, next_obs, terminated, truncated)
    obs, info = next_obs, next_info

print(f"Kết thúc tập: {info['end_reason']} | Quãng đường: {info['distance_m']:.1f}m | Tổng Reward: {total_reward:.2f}")
env.close()
```

---

## 10. TỔNG HỢP CHIẾN LƯỢC KIỂM THỬ TỰ ĐỘNG (UNIT TESTS)

Toàn bộ hệ thống được bảo vệ vững chắc bởi bộ **21 bài test tự động** trong thư mục [`highway_project/tests/`](file:///d:/Hoc_Tang_Cuong/highway_project/tests/), bao quát 100% logic môi trường, thuật toán, giao diện và cơ chế tái cấu hình trực tiếp:

### 10.1. Kiểm thử Môi trường & Giao diện (`test_environment.py` - 10 bài test):
1. `test_gymnasium_contract_and_seed`: Kiểm tra tuân thủ giao thức Gymnasium qua `check_env` và tính tái lập khi dùng `seed`.
2. `test_obstacle_visible_stationary_and_collision_terminates`: Kiểm tra xe hỏng đứng yên tuyệt đối và va chạm kích hoạt `terminated=True`.
3. `test_lane_change_can_pass_obstacle_and_reach_goal`: Kiểm tra khả năng đổi làn né vật cản và cán đích an toàn (`is_success=True`).
4. `test_timeout_is_truncation_and_not_success`: Kiểm tra hết giờ trả về `truncated=True` và không tính là thành công.
5. `test_idle_keeps_speed_and_rewards_match_info`: Kiểm tra duy trì vận tốc và tính toàn vẹn của các thành phần điểm thưởng.
6. `test_random_obstacles_placement_and_safe_spacing`: Kiểm tra vật cản sinh ngẫu nhiên cách nhau $\ge 150\text{ m}$ và cách điểm xuất phát $> 100\text{ m}$.
7. `test_dynamic_lanes_traffic_presets_and_speed_config`: Kiểm tra cấu hình số làn động (2 làn, 5 làn), preset mật độ xe (`sparse`, `dense`), và tùy biến vận tốc.
8. `test_view_renders_various_lanes_and_traffic`: Kiểm tra giao diện Pygame vẽ chính xác cho 2 làn, 4 làn và 5 làn ở chế độ headless, khởi tạo đầy đủ các nút bấm Menu Cài Đặt.
9. `test_live_reconfiguration_preserves_vehicle_progress`: **Xác minh cơ chế tái cấu hình trực tiếp trên `RoadNetwork` đang chạy**: đổi số làn sang 5 làn và thay đổi mật độ xe mà xe vẫn tiếp tục tiến lên an toàn, quãng đường tăng dần (`dist_after > dist_before`), sensor cập nhật đúng 5 làn và tuyệt đối không bị reset ván chơi.
10. `test_parse_traffic_and_cli_options`: Kiểm tra hàm phân tích cú pháp tham số dòng lệnh `--traffic`.

### 10.2. Kiểm thử Thuật toán DQN (`test_dqn.py` - 8 bài test):
11. `test_q_network_forward_and_shapes`: Kiểm tra kiến trúc mạng nơ-ron sâu `QNetwork` lan truyền tiến và kích thước tensor đầu ra.
12. `test_replay_buffer_sampling_and_capacity`: Kiểm tra bộ nhớ đệm kinh nghiệm `ReplayBuffer`, con trỏ vòng tròn và lấy mẫu ngẫu nhiên mini-batch.
13. `test_dqn_action_masking_respects_available_actions`: Kiểm tra Action Masking loại bỏ các hành động không hợp lệ khỏi không gian quyết định.
14. `test_dqn_learn_and_bellman_target`: Kiểm tra tính toán mục tiêu Bellman, hàm mất mát Smooth L1 và cập nhật trọng số mạng.
15. `test_dqn_save_load_and_deterministic_inference`: Kiểm tra lưu và tải checkpoint `dqn_model.pth`, đảm bảo tính tất định khi suy luận `eval()`.
16. `test_custom_agent_integration`: Kiểm tra khả năng tích hợp mô hình AI bên ngoài vào môi trường qua chuẩn Gymnasium.
17. `test_dqn_dodges_obstacle_when_danger_detected`: Kiểm tra mô hình DQN tự động lách làn hoặc phanh an toàn khi phát hiện xe hỏng chắn trước mặt.
18. `test_dqn_agent_various_lanes_and_speeds`: Kiểm tra mô hình DQN tự động ánh xạ vector trạng thái tương đối 68 chiều trên các cấu hình 2, 4, 5 làn mà không bị lỗi kích thước tensor.

### 10.3. Kiểm thử Học tăng cường & Nội suy (`test_learning.py` - 3 bài test):
19. `test_terminal_does_not_bootstrap_but_timeout_does`: Phân biệt chính xác giữa kết thúc vật lý (`terminated`, $V=0$) và hết giờ (`truncated`, được phép bootstrap).
20. `test_save_load_and_no_learning_during_inference`: Kiểm tra bảng Q-learning không bị biến đổi trọng số khi chạy chế độ suy luận.
21. `test_motion_interpolates_positions_and_shortest_heading`: Kiểm tra bộ đệm `MotionBuffer` nội suy tuyến tính tọa độ và góc quay lượng giác ngắn nhất.

Lệnh thực thi toàn bộ 21 bài kiểm thử tự động:
```powershell
.\.venv-highway\Scripts\python.exe -m unittest discover -s highway_project -t .
# Kết quả: Ran 21 tests in 2.6s -> OK (PASS 100%)
```
