# Kế Hoạch Triển Khai: Tùy Biến Số Làn, Mật Độ Xe (Thưa Thớt / Bình Thường / Dày Đặc), Tốc Độ Chạy & Chuẩn Hóa Giao Diện Trả Về

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Mở rộng toàn diện môi trường `LaneObstacleEnv`, giao diện Pygame và CLI `run.py` để người dùng có thể linh hoạt tùy biến:
1. **Số lượng làn đường**: Tăng/giảm từ 2 đến 6 làn (`--lanes 2..6`).
2. **Mật độ xe giao thông**: Tùy chọn trực quan theo 3 cấp độ: **`thưa thớt`** (sparse: 8 xe), **`bình thường`** (normal: 20 xe - mặc định), **`dày đặc`** (dense: 38 xe) (hỗ trợ nhập số nguyên 0..60 nếu muốn).
3. **Tốc độ chạy**: Tùy chỉnh vận tốc xuất phát của xe AI (`--speed [km/h]`, mặc định 72 km/h), vận tốc tối đa (`--max-speed`), và dải tốc độ dòng xe giao thông (`--traffic-speed`).
4. **Chuẩn hóa giao diện dữ liệu trả về**: Tài liệu hóa chi tiết `(observation, reward, terminated, truncated, info, sensors)` trong `DQN.md` và code mẫu Python để các thuật toán RL dễ dàng sử dụng.

**Architecture:** 
- Môi trường `LaneObstacleEnv`: Hỗ trợ cấu hình `vehicles_count` bằng preset chuỗi (`"sparse"`, `"normal"`, `"dense"`, `"none"`) hoặc số nguyên; tham số hóa `lanes_count` (2..6), `initial_speed` (m/s & km/h), `max_speed`, `traffic_speed_min`, `traffic_speed_max`.
- Cảm biến `_sensors()`: Tính toán động theo số làn, bổ sung cảm biến tương đối (ego-centric) giúp các thuật toán RL luôn nhận diện nhất quán.
- Giao diện `HighwayView`: Tự động co giãn kích thước mặt đường theo số làn, hiển thị rõ ràng trên HUD: `LÀN: X  ·  GIAO THÔNG: [Thưa thớt | Bình thường | Dày đặc]  ·  TỐC ĐỘ: Z km/h`.
- CLI `run.py`: Tham số `--traffic` nhận cả preset (`sparse`/`thua`, `normal`/`vua`, `dense`/`day`) hoặc số nguyên (0..60); bổ sung các cờ `--lanes`, `--speed`, `--max-speed`, `--traffic-speed`.
- Tài liệu `DQN.md`: Đặc tả chi tiết cấu trúc dữ liệu trả về, kèm code mẫu Python chuẩn Gymnasium để các thuật toán khác (PPO, SAC, Custom Agent) tiêu thụ.

**Tech Stack:** Python 3.13, Gymnasium 1.4.0, Highway-Env 1.12.1, Pygame-ce 2.5.8, PyTorch 2.15.0+cpu.

**Spec:**
1. **Số làn đường**: Thêm/giảm từ 2 đến 6 làn (mặc định 3).
2. **Số lượng xe / Mật độ giao thông**:
   - `sparse` / `thua`: 8 xe (giao thông thưa thớt, dễ quan sát và vượt chướng ngại vật).
   - `normal` / `vua`: 20 xe (mật độ giao thông trung bình bình thường - mặc định).
   - `dense` / `day`: 38 xe (giao thông dày đặc, đòi hỏi lách làn và phanh né chủ động).
   - `none` / `0`: 0 xe (đường vắng, chỉ có xe hỏng).
   - Hỗ trợ số nguyên trực tiếp (ví dụ: `--traffic 15`).
3. **Tốc độ chạy**:
   - Vận tốc xuất phát của xe AI: `--speed` (km/h, mặc định 72 km/h = 20 m/s, cho phép chỉnh 30 - 120 km/h).
   - Vận tốc tối đa của xe: `--max-speed` (km/h, mặc định 108 km/h = 30 m/s, cho phép chỉnh lên tới 144 km/h).
   - Vận tốc dòng xe giao thông: `--traffic-speed` (`slow`: 36-65 km/h, `normal`: 50-95 km/h, `fast`: 75-115 km/h, hoặc min-max).
4. **Giao diện trả về**: Mô tả chuẩn Gymnasium `(obs, reward, terminated, truncated, info)` và schema chi tiết của `info["sensors"]`.

## Global Constraints

- Tương thích ngược 100%: Mặc định chạy không tham số vẫn là 3 làn đường, mật độ bình thường (20 xe), tốc độ 72 km/h (20 m/s). Các checkpoint đã huấn luyện (`models/dqn_model.pth`, `models/q_learning.npz`) tiếp tục chạy mượt mà.
- Tất cả unit tests (16 tests hiện tại + các tests mới) phải PASS 100%.
- Định dạng dữ liệu trả về phải tuân thủ chuẩn Gymnasium API.

## Review Focus

1. Preset giao thông: Truyền `--traffic sparse`, `--traffic normal`, `--traffic dense` ánh xạ chính xác số lượng xe (8, 20, 38 xe); HUD hiển thị chữ tiếng Việt rõ ràng: "Thưa thớt", "Bình thường", "Dày đặc".
2. `lanes_count = 2` (đường hẹp 2 làn): Xe chỉ đổi làn giữa 0 và 1; vật cản và xe giao thông phân bố trong 2 làn; không bị lỗi `IndexError`.
3. `lanes_count = 5` hoặc `6` (cao tốc rộng): Vật cản phân bổ ngẫu nhiên trên các làn; giao diện vẽ đầy đủ các làn và dải phân cách; xe AI né vật cản an toàn.
4. Cấu hình vận tốc: Thay đổi `--speed 90 --max-speed 120` xe bắt đầu ở 90 km/h và có thể tăng tốc lên 120 km/h; `traffic-speed` áp dụng đúng dải ngẫu nhiên cho xe khác.
5. Chiều vector cảm biến trong `DQNAgent`: Mạng nơ-ron 68 chiều luôn nhận đúng kích thước đầu vào kể cả khi số làn khác 3 nhờ cơ chế ánh xạ tương đối (ego-centric).

---

### Task 1: Mở rộng `LaneObstacleEnv` hỗ trợ số làn (2..6), preset mật độ xe và tùy biến dải tốc độ

**Files:**
- Modify: `highway_project/environment/environment.py`
- Test: `highway_project/tests/test_environment.py`

**Interfaces:**
- Consumes: Cấu hình `config`:
  - `lanes_count`: int (2..6, default 3)
  - `vehicles_count`: int | str (`"sparse"`, `"normal"`, `"dense"`, `"none"` hoặc số nguyên 0..60, default 20)
  - `initial_speed`: float (m/s, default 20.0 = 72 km/h)
  - `max_speed`: float (m/s, default 30.0 = 108 km/h)
  - `traffic_speed_min`: float (m/s, default 14.0 = ~50 km/h)
  - `traffic_speed_max`: float (m/s, default 28.0 = ~100 km/h)
- Produces: 
  - `env.reset() -> (obs: np.ndarray, info: dict)`
  - `env.step(action) -> (obs: np.ndarray, reward: float, terminated: bool, truncated: bool, info: dict)`
  - `info["sensors"]`: `{"lane": int, "lanes_count": int, "traffic_density": str, "speed": float, "speed_kmh": float, "changing": bool, "target_lane": int, "front_gap": list[float], "rear_gap": list[float], "front_closing": list[float], "rear_closing": list[float], "ego_front_gap": float, "left_clear": bool, "right_clear": bool}`

- [x] **Step 1: Viết test cho số làn động, preset mật độ xe và dải vận tốc tùy chỉnh**

Trong `highway_project/tests/test_environment.py`:
```python
def test_dynamic_lanes_traffic_presets_and_speed_config(self):
    # Test preset mật độ xe: 'sparse' (8 xe), 'dense' (38 xe)
    env_sparse = self.create(lanes_count=2, vehicles_count="sparse", initial_speed=25.0)
    obs, info = env_sparse.reset(seed=42)
    self.assertEqual(info["sensors"]["lanes_count"], 2)
    self.assertEqual(env_sparse.unwrapped.config["vehicles_count"], 8)
    self.assertEqual(info["sensors"]["traffic_density"], "sparse")
    self.assertAlmostEqual(info["speed_kmh"], 90.0, delta=2.0)
    
    env_dense = self.create(lanes_count=5, vehicles_count="dense")
    obs2, info2 = env_dense.reset(seed=42)
    self.assertEqual(info2["sensors"]["lanes_count"], 5)
    self.assertEqual(env_dense.unwrapped.config["vehicles_count"], 38)
    self.assertEqual(info2["sensors"]["traffic_density"], "dense")
```

- [x] **Step 2: Chạy test để xác nhận lỗi trước khi triển khai**

Run: `& "d:\Hoc_Tang_Cuong\.venv-highway\Scripts\python.exe" -m unittest highway_project/tests/test_environment.py`

- [x] **Step 3: Triển khai trong `highway_project/environment/environment.py`**

1. Thêm từ điển ánh xạ preset mật độ xe:
   ```python
   TRAFFIC_PRESETS = {
       "none": 0, "0": 0, "khong": 0,
       "sparse": 8, "thua": 8, "thua_thot": 8,
       "normal": 20, "vua": 20, "binh_thuong": 20,
       "dense": 38, "day": 38, "day_dac": 38,
   }
   ```
2. Trong `default_config()` và `_create_vehicles()`:
   - Chuẩn hóa `vehicles_count`: nếu truyền string thì ánh xạ qua `TRAFFIC_PRESETS`, nếu là int thì kẹp trong `0 <= count <= 60`.
   - Lưu nhãn `traffic_density`: `"sparse"` nếu $\le 10$, `"dense"` nếu $\ge 30$, `"none"` nếu $0$, còn lại `"normal"`.
   - Cho phép `2 <= self.config["lanes_count"] <= 6`.
   - Vận tốc xuất phát của ego car lấy từ `self.config["initial_speed"]`.
   - Làn xuất phát: `initial_lane_id` nằm trong `range(lanes_count)`.
   - Sinh vật cản ngẫu nhiên: `obstacle_lane = int(self.np_random.integers(0, lanes_count))`.
3. Trong `_create_traffic()`:
   - Áp dụng `traffic_speed_min` và `traffic_speed_max` cho các xe IDM khi `unpredictable_traffic = True`.
4. Trong `_sensors()` & `_info()`:
   - Đo khoảng cách trước/sau cho cả `lanes_count` làn đường.
   - Thêm `lanes_count`, `traffic_density`, `speed_kmh`, `ego_front_gap`, `left_clear`, `right_clear` vào `info` và `sensors`.

- [x] **Step 4: Chạy lại test xác nhận thành công**

Run: `& "d:\Hoc_Tang_Cuong\.venv-highway\Scripts\python.exe" -m unittest highway_project/tests/test_environment.py`
Expected: PASS.

---

### Task 2: Cập nhật giao diện Pygame `HighwayView` hiển thị số làn động và thanh thông số kịch bản

**Files:**
- Modify: `highway_project/gui/view.py`
- Test: `highway_project/tests/test_environment.py`

**Interfaces:**
- Consumes: `env.config["lanes_count"]`, `env.config["vehicles_count"]`, `info["sensors"]["traffic_density"]`, `env.vehicle.speed`.
- Produces: Giao diện đồ họa tự động vẽ đủ số làn (chiều cao đường tự động co giãn), vạch phân cách và dải HUD thông số kịch bản rõ ràng: `LÀN: X  ·  MẬT ĐỘ: [Thưa thớt | Bình thường | Dày đặc]  ·  TỐC ĐỘ: Z km/h`.

- [x] **Step 1: Viết test vẽ giao diện với các cấu hình làn và mật độ khác nhau**

Thêm `test_view_renders_various_lanes_and_traffic` trong `highway_project/tests/test_environment.py`:
```python
def test_view_renders_various_lanes_and_traffic(self):
    from highway_project.gui.view import HighwayView
    for lanes, traffic in [(2, "sparse"), (4, "normal"), (5, "dense")]:
        env = self.create(lanes_count=lanes, vehicles_count=traffic)
        obs, info = env.reset(seed=42)
        view = HighwayView(headless=True)
        view.draw(env, info, paused=False, mode="agent", last_action=1, total_reward=0.0, episode=1)
```

- [x] **Step 2: Chạy kiểm thử để quan sát trạng thái render**

Run: `& "d:\Hoc_Tang_Cuong\.venv-highway\Scripts\python.exe" -m unittest highway_project/tests/test_environment.py`

- [x] **Step 3: Triển khai cập nhật trong `highway_project/gui/view.py`**

1. Tính toán kích thước mặt đường theo số làn:
   `lanes_count = int(env.config.get("lanes_count", 3))`
   `road_height = lanes_count * self.LANE_HEIGHT`
   `road_bottom = self.ROAD_TOP + road_height`
2. Vẽ mặt đường nhựa và lề đường co giãn tương ứng.
3. Vẽ các vạch đứt phân làn cho `row in range(1, lanes_count)`.
4. Cập nhật nhãn biên làn đường: `TRÁI / LÀN 0` ở trên và `PHẢI / LÀN {lanes_count - 1}` ở dưới.
5. Hiển thị nhãn mật độ tiếng Việt trực quan:
   ```python
   density = info["sensors"].get("traffic_density", "normal")
   density_vi = {"sparse": "Thưa thớt", "normal": "Bình thường", "dense": "Dày đặc", "none": "Không có xe"}.get(density, "Bình thường")
   ```
6. Bổ sung thông tin kịch bản ở tiêu đề phụ hoặc thanh trạng thái:
   `f"Cấu hình: {lanes_count} làn đường  ·  Mật độ: {density_vi} ({env.config.get('vehicles_count', 0)} xe)  ·  Đích {env.config.get('route_length', 1000):.0f} m"`

- [x] **Step 4: Chạy lại test xác nhận PASS**

Run: `& "d:\Hoc_Tang_Cuong\.venv-highway\Scripts\python.exe" -m unittest highway_project/tests/test_environment.py`
Expected: PASS.

---

### Task 3: Bổ sung các cờ CLI trong `run.py` và tối ưu tương thích trong `DQNAgent`

**Files:**
- Modify: `highway_project/run.py`
- Modify: `highway_project/algorithms/dqn_agent.py`
- Test: `highway_project/tests/test_dqn.py`

**Interfaces:**
- Consumes CLI arguments:
  - `--lanes` (int, 2..6, default 3): Số làn đường.
  - `--traffic` (str | int, choices: `sparse`/`thua`, `normal`/`vua`, `dense`/`day`, `none`, hoặc số 0..60, default `normal`): Mật độ xe.
  - `--speed` (float, km/h, default 72.0): Vận tốc xuất phát của xe.
  - `--max-speed` (float, km/h, default 108.0): Vận tốc tối đa cho phép.
  - `--traffic-speed` (str: `slow`, `normal`, `fast`, default `normal`): Dải tốc độ xe xung quanh.
- Produces: Áp dụng đầy đủ tham số vào cấu hình `make_env`; `DQNAgent.extract_state` trích xuất vector 68 chiều ổn định không lỗi kích thước khi `lanes_count != 3`.

- [x] **Step 1: Viết test cho `DQNAgent` hoạt động trên môi trường có số làn và tốc độ thay đổi**

Trong `highway_project/tests/test_dqn.py`:
```python
def test_dqn_agent_various_lanes_and_speeds(self):
    for lanes in (2, 4, 5):
        env = make_env(config={"lanes_count": lanes, "initial_speed": 22.0, "max_speed": 32.0, "vehicles_count": "sparse"})
        obs, info = env.reset(seed=42)
        agent = DQNAgent.load()
        action = agent.act(obs, info)
        self.assertIn(action, range(5))
        obs, reward, term, trunc, info = env.step(action)
        self.assertFalse(term)
```

- [x] **Step 2: Chạy test xác nhận lỗi kích thước vector khi số làn khác 3**

Run: `& "d:\Hoc_Tang_Cuong\.venv-highway\Scripts\python.exe" -m unittest highway_project/tests/test_dqn.py`

- [x] **Step 3: Cập nhật `dqn_agent.py` và `run.py`**

1. Trong `highway_project/algorithms/dqn_agent.py`:
   - Trong `extract_state()`:
     - Với `lanes_count == 3`: Giữ nguyên vector 18 radar để tương thích tuyệt đối với trọng số mô hình đã học.
     - Với `lanes_count != 3`: Lấy 3 kênh quan sát tương đối: Làn bên trái (nếu có), Làn hiện tại, Làn bên phải (nếu có). Điền khoảng cách an toàn (180 m) nếu không có làn trái/phải. Kích thước vector cảm biến luôn đảm bảo đúng 18 phần tử và tổng vector luôn là 68 phần tử.
   - Trong `act()`:
     - Cho phép rẽ phải khi `lane < lanes_count - 1` (thay vì cố định `lane < 2`).
2. Trong `highway_project/run.py`:
   - Thêm CLI arguments:
     - `--lanes`: `type=int, choices=range(2, 7), default=3, help="Số làn đường cao tốc (2..6, mặc định 3)"`
     - `--traffic`: hỗ trợ string presets (`sparse`, `normal`, `dense`, `none`, `thua`, `vua`, `day`) hoặc số nguyên:
       ```python
       def parse_traffic(value: str) -> int:
           val = value.lower()
           mapping = {"sparse": 8, "thua": 8, "normal": 20, "vua": 20, "dense": 38, "day": 38, "none": 0}
           if val in mapping:
               return mapping[val]
           try:
               num = int(val)
               if 0 <= num <= 60:
                   return num
           except ValueError:
               pass
           raise argparse.ArgumentTypeError("Mật độ xe phải là: sparse (thưa), normal (vừa), dense (dày) hoặc số từ 0..60")
       ```
     - `--speed`: `type=float, default=72.0, help="Vận tốc xuất phát xe của bạn theo km/h (mặc định 72 km/h)"`
     - `--max-speed`: `type=float, default=108.0, help="Vận tốc tối đa cho phép theo km/h (mặc định 108 km/h)"`
     - `--traffic-speed`: `choices=["slow", "normal", "fast"], default="normal", help="Tốc độ dòng xe giao thông (slow: 36-65 km/h, normal: 50-95 km/h, fast: 75-115 km/h)"`
   - Chuyển đổi km/h sang m/s (`speed / 3.6`) và ánh xạ dải `traffic-speed` tương ứng truyền vào `make_env(config=...)`.

- [x] **Step 4: Chạy test xác nhận thành công**

Run: `& "d:\Hoc_Tang_Cuong\.venv-highway\Scripts\python.exe" -m unittest discover -s highway_project -t .`
Expected: PASS 100%.

---

### Task 4: Cập nhật tài liệu `DQN.md` và `README.md` về giao diện trả về và các tùy chọn cấu hình

**Files:**
- Modify: `highway_project/DQN.md`
- Modify: `docs/THUAT_TOAN_DQN.md`
- Modify: `highway_project/README.md`
- Modify: `README.md`

**Interfaces:**
- Đưa vào tài liệu các mục chi tiết:
  1. **Đặc tả dữ liệu trả về cho thuật toán (Return Interface Specification)**:
     - Bảng giải thích chi tiết cấu trúc `(obs, reward, terminated, truncated, info)`.
     - Chi tiết từng trường trong `info["sensors"]` (`lane`, `lanes_count`, `traffic_density`, `speed_kmh`, `front_gap`, `rear_gap`, `ego_front_gap`, `left_clear`, `right_clear`, `changing`, `target_lane`).
     - Đoạn code mẫu Python chuẩn Gymnasium minh họa cách một thuật toán bất kỳ kết nối, nhận dữ liệu, trích xuất tensor và thực thi action masking.
  2. **Hướng dẫn tùy biến kịch bản**:
     - Tùy chỉnh số làn đường (`--lanes 2..6`).
     - Tùy chỉnh mật độ xe (`--traffic sparse`, `--traffic normal`, `--traffic dense`).
     - Tùy chỉnh tốc độ chạy (`--speed`, `--max-speed`, `--traffic-speed`).
     - Các ví dụ dòng lệnh thực tế cho các kịch bản khác nhau (đường hẹp mật độ dày tốc độ chậm, cao tốc 5 làn mật độ thưa tốc độ cao).

- [x] **Step 1: Bổ sung mục "Đặc Tả Giao Diện Trả Về Của Môi Trường" trong `DQN.md` và `docs/THUAT_TOAN_DQN.md`**
- [x] **Step 2: Bổ sung mục "Tùy Biến Số Làn, Mật Độ Xe và Tốc Độ Chạy" trong `DQN.md` và `README.md`**
- [x] **Step 3: Cập nhật bảng tham số dòng lệnh trong `highway_project/README.md` và root `README.md`**
- [x] **Step 4: Kiểm tra tính toàn vẹn của markdown và liên kết**

---

### Task 5: Kiểm thử tích hợp toàn diện và mô phỏng thực nghiệm

**Files:**
- Test: Toàn bộ test suite trong `highway_project/tests/`

- [x] **Step 1: Chạy toàn bộ unit test**

Run: `& "d:\Hoc_Tang_Cuong\.venv-highway\Scripts\python.exe" -m unittest discover -s highway_project -t .`
Expected: PASS toàn bộ tất cả unit tests (tối thiểu 18 tests).

- [x] **Step 2: Chạy thử nghiệm thực tế với các kịch bản qua dòng lệnh (Headless verification)**

Run:
```powershell
# Kịch bản 1: Cao tốc hẹp 2 làn, mật độ thưa thớt, tốc độ 80 km/h
& "d:\Hoc_Tang_Cuong\.venv-highway\Scripts\python.exe" -m highway_project.run --lanes 2 --traffic sparse --speed 80 --max-speed 110 --algo dqn --mode agent --headless --steps 30

# Kịch bản 2: Cao tốc mở rộng 5 làn, mật độ dày đặc, tốc độ cao 100 km/h
& "d:\Hoc_Tang_Cuong\.venv-highway\Scripts\python.exe" -m highway_project.run --lanes 5 --traffic dense --speed 100 --max-speed 130 --traffic-speed fast --algo dqn --mode agent --headless --steps 30

# Kịch bản 3: Đường vắng không có xe, chạy thử nghiệm tốc độ
& "d:\Hoc_Tang_Cuong\.venv-highway\Scripts\python.exe" -m highway_project.run --lanes 4 --traffic none --speed 90 --algo dqn --mode agent --headless --steps 30
```
Expected: Cả 3 kịch bản đều hoàn thành thành công, xuất JSON kết quả đầy đủ thông số `distance_m`, `speed`, `speed_kmh`.
