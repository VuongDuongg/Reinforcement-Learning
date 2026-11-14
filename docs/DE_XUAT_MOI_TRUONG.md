# Đề xuất môi trường mô phỏng xe tự lái né chướng ngại vật

Ngày lập: 22/10/2026. Đây là đề xuất để lựa chọn hướng làm, chưa phải môi trường đã cài đặt hoặc kết quả thực nghiệm.

**Cập nhật 25/10/2026:** Đã triển khai môi trường HighwayEnv trong `highway_project/`, cài thư viện và kiểm tra mô phỏng; thuật toán RL để bổ sung sau. Mở `CHAY_MOI_TRUONG.cmd` và đọc [hướng dẫn bản đã triển khai](highway_project/README.md). Các mục bên dưới giữ nội dung đề xuất ban đầu để tham khảo.

## 1. Nên chọn môi trường gì?

**Đề xuất chọn HighwayEnv + Gymnasium, viết và chạy bằng Python trên máy Windows, dùng VS Code để phát triển.** Tham khảo `marl4av` của thầy về bài toán, kịch bản giao thông và cách đánh giá; không cần mặc định chọn NetLogo chỉ vì tài liệu tham khảo sử dụng nền tảng đó.

Lựa chọn này phù hợp khi bạn chưa dùng NetLogo: có thể tập trung học vòng lặp RL trong một ngôn ngữ, rồi thêm dần kịch bản né vật cản. HighwayEnv cung cấp các môi trường ra quyết định lái xe với quan sát, hành động và phần thưởng có thể cấu hình. Đây là mô phỏng phục vụ nghiên cứu thuật toán, không phải hệ thống chứng minh xe chạy an toàn ngoài thực tế. [Tài liệu HighwayEnv](https://highway-env.farama.org/)

Nếu đề bài hoặc giảng viên **bắt buộc dùng NetLogo hay bắt buộc mở rộng trực tiếp marl4av**, cần chọn theo yêu cầu đó. Trong phần tài liệu đã đọc, nhóm đang định hướng NetLogo; điều này chưa đủ để kết luận giảng viên cấm dùng môi trường khác.

## 2. Phân biệt các thành phần cho dễ hiểu

| Thành phần | Đề xuất | Dùng để làm gì? |
|---|---|---|
| Môi trường mô phỏng | HighwayEnv | Tạo đường, xe, chuyển động và va chạm |
| Giao diện tương tác RL | Gymnasium | Cung cấp `reset()` và `step(action)` |
| Ngôn ngữ | Python | Viết kịch bản, thuật toán và đánh giá |
| Công cụ lập trình | VS Code + terminal | Sửa mã, chạy chương trình, xem lỗi |
| Môi trường thư viện | `.venv-highway` riêng | Tách thư viện thử nghiệm khỏi dự án hiện tại |
| Nơi chạy ban đầu | Máy Windows cá nhân | Mở cửa sổ mô phỏng và quan sát trực tiếp |

Không cần xây website hoặc thuê máy chủ để bắt đầu. Chưa có phép đo hiệu năng trên máy của bạn, nên chưa kết luận cần GPU hay máy mạnh hơn.

## 3. So sánh các hướng phù hợp với dự án này

| Hướng | Điểm thuận lợi | Việc cần làm thêm | Đánh giá cho bạn |
|---|---|---|---|
| **HighwayEnv + Python** | Có sẵn mô phỏng cao tốc và API RL; thuận tiện viết vòng lặp huấn luyện bằng Python | Chuyển giao diện thuật toán; tự bổ sung kịch bản xe hỏng đứng yên | **Ưu tiên nếu được tự chọn nền tảng** |
| **NetLogo + marl4av** | Bám sát mã và tài liệu đang có của thầy | Học NetLogo, Python Extension và luồng giao tiếp hai phía | Chọn nếu yêu cầu bám sát nền tảng của thầy |
| **Tự viết mô phỏng Python từ đầu** | Chủ động thiết kế | Tự làm chuyển động, cảm biến, va chạm, hiển thị và kiểm chứng | Chưa nên chọn làm bước đầu |

Đánh giá trên dựa vào tình trạng dự án và việc bạn chưa dùng NetLogo, không phải kết quả đo thời gian học hay tốc độ huấn luyện.

## 4. Dùng tài liệu hiện tại làm nền như thế nào?

Qua đọc mã cục bộ, có thể giữ lại **ý tưởng bài toán**: đường nhiều làn, xe phía trước cản đường, quyết định tăng/giảm tốc và chuyển làn, so sánh chính sách bằng chỉ số an toàn và lưu thông.

Nhưng không nên xem mã Python hiện tại là có thể cắm ngay vào HighwayEnv:

- `marl4av/Expressways.nlogo` hiện thiết lập **một chiều, 3 làn chính + 1 làn khẩn cấp**, trong khi một số tài liệu vẫn mô tả hai chiều.
- Mô hình NetLogo vẫn chứa mã Python nhúng; chưa thấy lời gọi `rl_bridge`, `StateProcessor` hoặc `ObstacleManager` trong mô hình đã đọc.
- `StateProcessor` tạo vector 8 chiều, còn `rl_bridge.init_agent()` và các tác nhân học mặc định nhận 5 chiều. HighwayEnv lại có cấu trúc quan sát riêng; phải thống nhất đầu vào trước khi tái sử dụng.
- Hàm thưởng thực tế trong `get-reward` hiện cho điểm theo hành động và có phạt vào làn khẩn cấp; chưa trùng công thức thưởng liên tục trong báo cáo.
- `rl_bridge.update_agent()` chưa nhận cờ kết thúc episode. Khi chuyển môi trường phải bổ sung xử lý kết thúc và giới hạn thời gian đúng với từng thuật toán.

Vì vậy nên giữ nguyên thư mục cũ làm tài liệu tham khảo, tạo thư mục Python mới cho hướng HighwayEnv. Không ghi rằng đã tích hợp hoặc đã hội tụ khi mới chạy thử module.

## 5. Thiết kế môi trường đề xuất

### 5.1. Bản đầu tiên: chạy được và hiểu được

- Dùng `highway-v0`, cấu hình **3 làn**, khoảng **10–15 xe giao thông**, **1 xe do RL điều khiển**.
- Các xe còn lại dùng hành vi giao thông của môi trường.
- Dùng quan sát động học có sẵn; kiểm tra kích thước bằng `observation_space` trước khi xây mạng.
- Dùng hành động rời rạc mặc định: trái, giữ, phải, nhanh hơn, chậm hơn. Không tái sử dụng bảng ID 4 hành động của NetLogo.
- Chạy hành động ngẫu nhiên để hiểu API; sau đó làm một chính sách theo luật để có mốc so sánh.

HighwayEnv mặc định có bài toán tránh va chạm khi chạy giữa các xe khác; **chưa nên gọi bản mặc định là môi trường xe hỏng đứng yên theo đúng đề tài**. [Môi trường Highway](https://highway-env.farama.org/environments/highway/), [các loại hành động](https://highway-env.farama.org/actions/)

### 5.2. Bản đúng trọng tâm né chướng ngại vật

Sau khi bản cơ bản chạy ổn, tạo lớp môi trường riêng kế thừa môi trường Highway:

1. Sinh một xe hỏng hoặc đối tượng vật cản đứng yên ở làn chính. Đảm bảo đối tượng thực sự giữ vận tốc bằng 0 qua các bước mô phỏng; chỉ đặt tốc độ ban đầu bằng 0 có thể chưa đủ nếu nó vẫn tự điều khiển.
2. Ban đầu đặt vật cản ở vị trí cố định, có ít nhất một lối né; sau đó ngẫu nhiên hóa làn và khoảng cách bằng seed.
3. Giữ vật cản xuất hiện trong dữ liệu quan sát và tham gia kiểm tra va chạm.
4. Ghi nhận xe vượt qua vật cản, va chạm hoặc hết thời gian. Không mặc định hết thời gian là thành công.
5. Thêm nhiều vật cản và mật độ xe cao sau khi kịch bản đơn giản đã được kiểm chứng.

Đây là phần **cần triển khai thêm**, không chỉ là bật một tùy chọn trong cấu hình mặc định. Có thể đặt lớp tùy chỉnh trong dự án và đăng ký với Gymnasium, tránh sửa trực tiếp thư viện đã cài. [Hướng dẫn tạo môi trường riêng](https://highway-env.farama.org/make_your_own/)

### 5.3. Phạm vi tác nhân và thuật toán

**Phạm vi của Lê Tuấn Khanh:** chỉ xây dựng môi trường (đường, xe/vật cản, cảm biến, hàm thưởng, điều kiện kết thúc và giao diện). Không phụ trách cài đặt, huấn luyện hoặc tối ưu thuật toán. Phần thuật toán do Vượng (DQN), Tuấn Anh (Double Dueling DQN), Thái (PPO) và Sỹ (baseline/đánh giá) thực hiện theo phân công nhóm.

Nên bắt đầu **một tác nhân học giữa nhiều xe giao thông**. Nhiều xe xuất hiện trên đường không đồng nghĩa với nhiều tác nhân cùng học. Nếu môn học yêu cầu MARL, cần một giai đoạn riêng cho nhiều xe được điều khiển, quan sát từng xe và cách huấn luyện phối hợp.

Thứ tự đề xuất: **chính sách theo luật → DQN → Double DQN/Dueling DQN → PPO**. Với hành động rời rạc, SAC chỉ nên đưa vào khi có triển khai SAC rời rạc phù hợp; không dùng trực tiếp SAC liên tục rồi coi là tương đương.

## 6. Cài ở đâu và bắt đầu thế nào?

Chạy trên máy Windows hiện tại. Các lệnh sau là hướng dẫn cho bước triển khai sau; chưa được thực thi khi lập tài liệu. Dùng Python 3.11 làm lựa chọn khởi đầu, kiểm tra yêu cầu phiên bản của các thư viện tại lúc cài.

Trong PowerShell, tại thư mục dự án:

```powershell
cd C:\Users\Admin\Desktop\code\RL
py -3.11 -m venv .venv-highway
.\.venv-highway\Scripts\python.exe -m pip install --upgrade pip
.\.venv-highway\Scripts\python.exe -m pip install highway-env gymnasium
```

Cách gọi trực tiếp Python trong môi trường ảo không cần chạy script kích hoạt PowerShell. HighwayEnv hỗ trợ cài bằng pip theo [hướng dẫn chính thức](https://highway-env.farama.org/installation/).

Tạo `highway_project/demo.py` với ví dụ tối thiểu sau:

```python
import gymnasium as gym
import highway_env

gym.register_envs(highway_env)

env = gym.make(
    "highway-v0",
    render_mode="human",
    config={"lanes_count": 3, "vehicles_count": 15, "controlled_vehicles": 1},
)
try:
    obs, info = env.reset(seed=42)
    env.action_space.seed(42)
    print("Observation:", env.observation_space)
    print("Action:", env.action_space)
    for _ in range(300):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        if terminated or truncated:
            obs, info = env.reset()
finally:
    env.close()
```

Chạy bằng:

```powershell
.\.venv-highway\Scripts\python.exe highway_project\demo.py
```

Ví dụ chỉ kiểm tra mô phỏng, **chưa huấn luyện AI và chưa thêm vật cản riêng**. API dựa trên [ví dụ chính thức](https://highway-env.farama.org/).

Nếu máy chưa có Python 3.11, cần cài phiên bản phù hợp hoặc chọn một phiên bản đang được bộ thư viện hỗ trợ. Chưa cần cài TensorFlow chỉ để mở demo. Khi thêm thuật toán, chọn bộ thư viện học máy thống nhất; chỉ tái dùng TensorFlow hiện tại sau khi kiểm tra tương thích và sửa giao diện tác nhân.

## 7. Tổ chức phần triển khai sau khi chọn hướng

```text
RL/
├── marl4av/                    # Mã tham khảo hiện có
├── DE_XUAT_MOI_TRUONG.md        # Tài liệu này
└── highway_project/            # Thư mục đề xuất tạo ở bước triển khai
    ├── demo.py                 # Mở mô phỏng cơ bản
    ├── envs/obstacle_env.py     # Môi trường bổ sung xe hỏng
    ├── agents/                 # Chính sách theo luật, DQN, ...
    ├── train.py                # Huấn luyện
    ├── evaluate.py             # Đánh giá trên seed cố định
    ├── requirements-lock.txt   # Phiên bản thư viện đã xác minh
    └── results/                # Log, cấu hình, mô hình, biểu đồ
```

Chỉ file đề xuất này được tạo trong công việc hiện tại. Cây thư mục trên là thiết kế dự kiến.

## 8. Làm theo thứ tự nào và khi nào coi là đạt?

| Bước | Công việc | Kết quả cần thấy |
|---|---|---|
| 1 | Chạy demo ngẫu nhiên | Có đường và xe, `reset/step` hoạt động |
| 2 | Thêm vật cản cố định | Nhìn thấy vật cản đứng yên; va chạm và vượt qua được ghi nhận đúng |
| 3 | Viết chính sách theo luật | Có kết quả đối chiếu trước khi huấn luyện |
| 4 | Huấn luyện DQN | Lưu được mô hình, log phần thưởng và cấu hình |
| 5 | Đánh giá riêng | Tắt khám phá phù hợp; chạy trên các seed không dùng để huấn luyện |
| 6 | Thêm thuật toán khác | So sánh trên cùng môi trường, kịch bản và ngân sách tương tác |

Đề xuất đánh giá ban đầu trên ít nhất 30 episode kiểm tra cho mỗi kịch bản và nhiều seed huấn luyện nếu thời gian cho phép. Báo cáo tỷ lệ va chạm, tỷ lệ vượt vật cản an toàn, tốc độ trung bình, số lần đổi làn và phần thưởng; ghi rõ định nghĩa thành công và biến thiên giữa các lần chạy. Đây là kế hoạch thí nghiệm, chưa phải kết quả đạt được.

## 9. Đoạn đề xuất có thể đưa vào báo cáo

> Nhóm đề xuất sử dụng HighwayEnv kết hợp Gymnasium để xây dựng môi trường mô phỏng xe tự lái né chướng ngại vật trên đường cao tốc nhiều làn. Việc phát triển và thử nghiệm được thực hiện bằng Python trên máy tính cá nhân. Giai đoạn đầu sử dụng một xe do tác nhân học tăng cường điều khiển, trong khi các xe còn lại vận hành theo mô hình giao thông của môi trường. Nhóm sẽ bổ sung kịch bản xe hỏng đứng yên, điều chỉnh phần thưởng và đánh giá khả năng vượt vật cản an toàn. Dự án marl4av được sử dụng làm tài liệu tham khảo về kịch bản và cách tiếp cận bài toán. Lựa chọn nền tảng này cần phù hợp với yêu cầu của giảng viên; phần mở rộng đa tác nhân sẽ được thực hiện nếu thuộc phạm vi bắt buộc của đề tài.

## 10. Tài liệu tra cứu

- [HighwayEnv: tổng quan và ví dụ](https://highway-env.farama.org/)
- [Cài đặt](https://highway-env.farama.org/installation/)
- [Môi trường Highway và tham số](https://highway-env.farama.org/environments/highway/)
- [Hành động](https://highway-env.farama.org/actions/)
- [Tạo môi trường riêng](https://highway-env.farama.org/make_your_own/)
- [marl4av của thầy: nguồn tham khảo bài toán](https://github.com/thanhlnt/marl4av)

Các nhận xét về mã hiện tại dựa trên `README.md`, `TH2-Environment&Algorithms.docx.md`, `HUONG_DAN_CHAY.md`, `marl4av/Expressways.nlogo`, `rl_bridge.py`, `state_processor.py`, `obstacle_manager.py` và các module trong `algorithms/`. Không kiểm chứng lại các tuyên bố cài đặt, hội tụ hoặc kết quả thử nghiệm đã ghi trong báo cáo cũ.
