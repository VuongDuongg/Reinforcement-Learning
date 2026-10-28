# Giải thích mã môi trường xe tự lái

Cập nhật 28/10/2026. Tài liệu này mô tả bản **HighwayEnv + Gymnasium** trong `highway_project/`; không mô tả NetLogo cũ. Slide HTML ở `index.html`, nội dung tương ứng ở `SLIDE_THUYET_TRINH_BUOI_2.md`.

## 1. Môi trường đã làm tròn vai đến đâu?

Đã có các chức năng cốt lõi để người khác tích hợp thuật toán: khởi tạo theo seed, quan sát, năm hành động, động học và va chạm, vật cản đứng yên, hàm thưởng, điều kiện kết thúc, giao diện demo và kiểm tra. Đây là nền tảng thử nghiệm cho bài tập; chưa phải mô phỏng xe thực đã được thẩm định hoặc chính sách lái thành công.

Q-learning thử chưa hội tụ. Xe có thể né rồi giảm tốc về 0, trong khi môi trường vẫn chạy đến hết giờ. **Vượt vật cản không phải điều kiện kết thúc.** Học được chính sách tới đích là trách nhiệm của thuật toán và thiết kế thí nghiệm tiếp theo.

## 2. Bản đồ mã nguồn

| File | Đọc phần nào trước? | Vai trò |
|---|---|---|
| `highway_project/environment.py` | `default_config`, `_create_vehicles`, `_rewards`, `_sensors`, `_info` | Định nghĩa môi trường |
| `highway_project/q_agent.py` | `state`, `act`, `learn` | Q-learning thử, tách khỏi môi trường |
| `highway_project/train.py` | `main`, `evaluate` | Huấn luyện, lưu mô hình, đánh giá |
| `highway_project/run.py` | `reset`, `advance`, vòng lặp sự kiện | Chạy demo bằng bàn phím hoặc model |
| `highway_project/motion.py` | `reset`, `capture`, `sample` | Nội suy giữa hai trạng thái để vẽ mượt |
| `highway_project/view.py` | `draw` | Vẽ trạng thái và thông tin; không thay vật lý |
| `highway_project/agent.py` | `CustomAgent` | Mẫu giao diện tham khảo, hiện chưa được launcher dùng |

## 3. Cấu hình: `LaneObstacleEnv.default_config()`

Vị trí: [environment.py, dòng 24](highway_project/environment.py#L24).

```python
"lanes_count": 3,
"vehicles_count": 12,
"controlled_vehicles": 1,
"duration": 60.0,
"simulation_frequency": 30,
"policy_frequency": 10,
"route_length": 650.0,
"obstacle_count": 1,
"obstacle_distance": 160.0,
```

- Một agent học điều khiển một xe. Nhiều xe giao thông xuất hiện không đồng nghĩa với học đa tác nhân.
- Một quyết định kéo dài 0,1 giây; engine thực hiện ba bước vật lý trong khoảng đó.
- Khoảng cách dùng mét, vận tốc dùng m/s; giao diện đổi sang km/h bằng `speed * 3.6`.
- Mặc định có ba làn chính; không có làn khẩn cấp hoặc quy tắc tốc độ riêng từng làn.
- Đích được tính từ điểm xuất phát, không phải cuối toàn bộ mạng đường HighwayEnv.

## 4. Tạo xe và vật cản: `_create_vehicles()`

Vị trí: [environment.py, dòng 61](highway_project/environment.py#L61).

Xe điều khiển xuất phát tại x = 80 m, mặc định ở làn giữa, tốc độ 20 m/s. Đích là:

```python
self.start_x = float(ego.position[0])
self.goal_x = self.start_x + self.config["route_length"]
```

Với cấu hình mặc định, tọa độ đích là 730 m nhưng **quãng đường cần đi là 650 m**.

`BrokenCar` kế thừa `Obstacle`, có chiều dài 5 m, rộng 2 m. Vật cản được tạo với `speed=0.0` và đưa vào `road.objects`. Nó không có bộ điều khiển tự tăng tốc, nên giữ nguyên vị trí qua các bước mô phỏng. Xe hỏng đầu tiên nằm trên làn xuất phát, cách điểm xuất phát 160 m; các vật cản sau cách nhau 180 m và được chọn làn theo seed.

Xe giao thông được tạo bằng `IDMVehicle`, có hành vi bám xe/chuyển làn của thư viện. Khi sinh, mã kiểm tra khoảng trống cùng làn để tránh đặt chồng xe hoặc vật cản. Ngẫu nhiên dùng `self.np_random`, vì vậy `reset(seed=...)` có thể tái lập kịch bản.

## 5. Hành động và chuyển trạng thái

`ACTIONS` trong `environment.py`:

| ID | Tên | Ý nghĩa |
|---|---|---|
| 0 | LANE_LEFT | Đặt làn mục tiêu sang trái |
| 1 | IDLE | Giữ mục tiêu điều khiển hiện tại |
| 2 | LANE_RIGHT | Đặt làn mục tiêu sang phải |
| 3 | FASTER | Tăng một mức tốc độ mục tiêu |
| 4 | SLOWER | Giảm một mức tốc độ mục tiêu |

Các mức tốc độ `[0, 5, 10, 15, 20, 25, 30]` cách đều vì bộ điều khiển `MDPVehicle` ánh xạ chỉ số tốc độ theo lưới đều. Không đổi tùy tiện sang một danh sách cách không đều.

Lệnh không làm xe nhảy tọa độ tức thời. Bộ điều khiển HighwayEnv biến làn/tốc độ mục tiêu thành điều khiển động học; engine bước vật lý, xử lý giao thông và va chạm. Lớp môi trường kế thừa `step()` và `reset()` từ thư viện, không tự viết lại toàn bộ engine.

## 6. Quan sát chuẩn và cảm biến bổ sung

Quan sát chuẩn có shape **(10, 5)**, kiểu `float32`. Năm cột là `presence, x, y, vx, vy`. Dòng đầu mô tả ego bằng đại lượng tuyệt đối đã chuẩn hóa; các dòng xung quanh dùng tọa độ/vận tốc tương đối. Dòng không có đối tượng được đệm 0. `include_obstacles=True` giúp xe hỏng tham gia quan sát. Số đối tượng và tầm nhìn đều có giới hạn.

Vị trí cảm biến bổ sung: [_sensors(), dòng 141](highway_project/environment.py#L141).

```python
gap = max(0.0, abs(dx) - (obj.LENGTH + ego.LENGTH) / 2)
```

`dx` là chênh lệch tọa độ tâm; `gap` trừ nửa chiều dài hai xe để xấp xỉ khoảng trống giữa thân xe. Cảm biến tìm đối tượng gần nhất trước và sau ở mỗi làn trong 180 m:

- `front_gap`, `rear_gap`: khoảng trống ba làn; giá trị 180 có thể nghĩa là không thấy đối tượng gần hơn.
- `front_closing = ego.speed - front.speed`: dương nghĩa là đang tiến gần xe trước.
- `rear_closing = rear.speed - ego.speed`: dương nghĩa là xe sau đang tiến gần.
- `lane`, `speed`, `changing`: làn, tốc độ và việc xe còn đang bám sang làn mục tiêu.

Xe chuyển làn có thể được tính là chiếm hai làn theo khoảng cách ngang đến tâm làn. Đây là cảm biến mô phỏng từ trạng thái engine, không phải camera/LiDAR thật.

Q-learning hiện **không dùng trực tiếp ma trận (10,5)** mà dùng `info["sensors"]`. Khi so sánh với DQN/PPO, phải ghi rõ đầu vào mỗi thuật toán; cùng engine chưa có nghĩa là cùng thông tin quan sát.

## 7. Hàm thưởng thống nhất giữa mã và slide

Vị trí: [_rewards(), dòng 118](highway_project/environment.py#L118) và [_reward(), dòng 138](highway_project/environment.py#L138).

Không đổi logic đang chạy trong lần chỉnh tài liệu này. Thay công thức gom nhóm khó đối chiếu bằng đúng sáu thành phần đơn giản:

```text
r_t = r_collision + r_offroad + r_speed + r_lane_change + r_goal + r_danger
```

| Tên trên slide | Khóa trong mã | Giá trị / điều kiện |
|---|---|---|
| r_collision | collision | −20 khi va chạm |
| r_offroad | offroad | −10 khi ra khỏi đường và chưa va chạm |
| r_speed | speed | +0,03 × v/30 khi còn trên đường, chưa va chạm |
| r_lane_change | lane_change | −0,02 khi yêu cầu hành động 0 hoặc 2 |
| r_goal | success | +10 khi tới đích an toàn |
| r_danger | danger | −0,15 khi gap trước <10 m hoặc TTC <2 s với closing >0,1 m/s |

Tất cả thành phần khởi tạo bằng 0. Nếu va chạm, chỉ khóa `collision` có điểm; nếu ra khỏi đường, chỉ khóa `offroad` có điểm. Chỉ nhánh còn lại mới tính tốc độ, đổi làn, đích và nguy hiểm. Như vậy công thức cộng vẫn đúng, không vừa nhận phạt va chạm vừa nhận thưởng tốc độ.

`TTC = gap / closing` chỉ có ý nghĩa ở đây khi đang tiến gần. Các khóa trong `info["rewards"]` đã là **điểm thành phần sau khi nhân trọng số**, không nhân trọng số thêm lần nữa. `_reward()` cộng chúng để trả số thưởng cuối.

Ví dụ:

- Giữ 20 m/s trên đường thoáng: `+0,02`.
- Yêu cầu đổi làn ở cùng tốc độ, không nguy hiểm: `+0,02 −0,02 = 0`.
- Giữ 20 m/s trong vùng nguy hiểm: `+0,02 −0,15 = −0,13`.
- Va chạm: `−20`, không cộng thêm thành phần khác.

Đã bỏ khỏi slide các mục cũ không có trong mã mới: thưởng ổn định riêng, phạt phanh riêng, quá tốc độ theo làn và lấn làn khẩn cấp. Phạt đổi làn tính theo **yêu cầu hành động**, không phải số lần đã chuyển làn thành công. Thưởng tốc độ tính theo bước, nên phải giữ cùng tần suất khi đối sánh. Thiết kế này chưa bảo đảm agent không học cách đứng yên.

## 8. Kết thúc: `_success()`, `_is_terminated()`, `_info()`

Vị trí: [dòng 112](highway_project/environment.py#L112), [dòng 115](highway_project/environment.py#L115), [dòng 170](highway_project/environment.py#L170).

```python
return bool(self.vehicle.position[0] >= self.goal_x
            and not self.vehicle.crashed and self.vehicle.on_road)
```

- `terminated`: tới đích hoặc trạng thái thất bại như va chạm/ra khỏi đường.
- `truncated`: đạt giới hạn thời gian; logic này kế thừa HighwayEnv.
- `is_success`: xác nhận thực sự tới đích an toàn. Nếu đích trùng đúng thời hạn, cần kiểm tra riêng cờ này; không kết luận chỉ từ `truncated`.
- `obstacles_passed`: số vật cản đã vượt, chỉ là thông tin giám sát, không dùng để kết thúc.
- `end_reason`: chuỗi giúp giao diện hiển thị nguyên nhân.

Môi trường không biết “agent đã học tốt” hay chưa. Xe tốc độ 0 nhưng chưa hết giờ vẫn là một trạng thái hợp lệ. Tránh va chạm bằng đứng yên không tính là hoàn thành nhiệm vụ.

## 9. Q-learning thử: `state()`, `act()`, `learn()`

Vị trí: [q_agent.py, dòng 26](highway_project/q_agent.py#L26), [dòng 56](highway_project/q_agent.py#L56), [dòng 66](highway_project/q_agent.py#L66).

`state()` gom cảm biến thành 648 nhóm trạng thái. `act()` dùng epsilon-greedy: khi học có xác suất chọn ngẫu nhiên; khi xem demo epsilon bằng 0 và chọn Q lớn nhất. Khi bằng điểm, ưu tiên giữ tốc rồi mới xét các hành động khác. Không có quy tắc né xe hỏng viết sẵn.

```python
target = float(reward)
if not terminated:
    target += gamma * max(Q[next_state][next_actions])
Q[state][action] += learning_rate * (target - Q[state][action])
```

Đó là cập nhật Bellman của Q-learning, với `learning_rate=0.15`, `gamma=0.98`. Không bootstrap khi terminal thực sự; time limit vẫn có thể bootstrap từ quan sát cuối trước reset. `available()` loại lệnh chuyển làn tiếp trong khi đang chuyển làn, không tự loại các hành động theo nguy cơ va chạm.

Model thử đã học 180 episode, 23.994 cập nhật; kiểm tra ban đầu trên 10 seed mới với 0 xe giao thông chưa có lượt tới đích. Đây không phải model bảo đảm lái thành công. Muốn đổi hàm thưởng hoặc cách chia trạng thái cần học lại; không coi bảng Q cũ vẫn tương đương.

## 10. Vòng học và vòng demo khác nhau thế nào?

`train.py` gọi `act → step → observe` để cập nhật Q, lưu file `.npz` và đánh giá riêng. Seed học nhỏ hơn một triệu; seed kiểm tra bắt đầu từ một triệu. Quá trình đánh giá cuối của bản thử chưa hoàn tất toàn bộ, không trình bày số liệu 6/12 xe là đã có.

`run.py` tải model rồi gọi `act → step` để xem. Hàm `advance()` không gọi học, nên xem demo không làm thay đổi trọng số. Muốn đổi thuật toán cần thay khởi tạo agent trong `run.py` và vòng học tương ứng; chỉ sửa `CustomAgent` trong `agent.py` chưa đủ vì launcher hiện dùng `QLearningAgent`.

## 11. Làm mượt hình: `MotionBuffer`

Vị trí: [motion.py, dòng 8](highway_project/motion.py#L8).

Mỗi bước engine, `capture()` giữ hai ảnh chụp số học của vị trí/hướng: trước và sau bước. Giữa hai bước, `sample(alpha)` nội suy:

```text
position_draw = (1 − alpha) × previous + alpha × current
alpha = thời gian còn trong bộ tích lũy / thời gian một bước
```

Vòng giao diện vẽ mục tiêu 60 FPS, trong khi agent vẫn quyết định ở 10 Hz. Hướng được nội suy theo cung ngắn nhất để không quay vòng khi đi qua ±π. Nội suy không sửa vị trí engine, không cải thiện thuật toán và không bảo đảm FPS thực tế luôn bằng 60. Hình có độ trễ tối đa một bước mô phỏng, khoảng 0,1 giây; khi dừng/kết thúc, hiển thị trạng thái thật.

## 12. Cách nối một thuật toán khác

```python
from highway_project.environment import make_env

env = make_env()
try:
    observation, info = env.reset(seed=42)
    while True:
        action = agent.act(observation, info)  # Số nguyên 0..4
        next_obs, reward, terminated, truncated, next_info = env.step(action)
        # Ghi kinh nghiệm/cập nhật thuật toán tại đây nếu đang huấn luyện.
        observation, info = next_obs, next_info
        if terminated or truncated:
            break
finally:
    env.close()
```

`agent` trong ví dụ là thuật toán người triển khai cung cấp. Với DQN có thể flatten ma trận thành 50 số; với PPO có thể dùng cùng đầu vào để đối sánh. Mạng xuất năm Q-value hoặc phân phối năm hành động. Không lấy nguyên mặc định 5/8 chiều và bốn hành động từ các module NetLogo cũ.

Ghi cả hai cờ kết thúc; replay/rollout phải giữ quan sát cuối trước reset. Với PPO, ngắt GAE ở ranh giới episode; không nối quỹ đạo của các episode hoặc các xe khác nhau.

## 13. Kiểm tra và phân công tiếp theo

```powershell
.\.venv-highway\Scripts\python.exe -m unittest highway_project.test_environment highway_project.test_learning -v
```

Tám kiểm tra hiện có xác minh hợp đồng môi trường, seed, vật cản, va chạm, tới đích, hết giờ, giữ tốc, cập nhật Bellman, lưu/tải Q và nội suy. Chúng không chứng minh thuật toán hội tụ hoặc môi trường hoàn chỉnh cho mọi kịch bản thực tế.

**Lê Tuấn Khanh chỉ phụ trách môi trường:** đường, xe/vật cản, cảm biến, hàm thưởng, điều kiện kết thúc và giao diện; không phụ trách cài đặt, huấn luyện hoặc tối ưu thuật toán. Sơn phụ trách lõi/tích hợp; Vượng DQN; Tuấn Anh Double Dueling DQN; Thái PPO; Sỹ baseline/đánh giá/báo cáo. Mã nền DQN/DDQN/PPO/baseline có trong `marl4av/algorithms/`, nhưng cần chuyển sang API mới và xác minh end-to-end. Q-learning thử chỉ minh họa cách nối thuật toán vào môi trường, không phải phần việc thuật toán của Khanh. Không dùng các tỷ lệ hoàn thành cũ như bằng chứng tiến độ hiện tại.

Nguồn API: [HighwayEnv](https://highway-env.farama.org/), [quan sát](https://highway-env.farama.org/observations/), [hành động](https://highway-env.farama.org/actions/). Mô tả hành vi cụ thể ở trên được đối chiếu trực tiếp mã trong dự án; đây là tài liệu giải thích, không thay đổi hàm thưởng hoặc huấn luyện thêm.
