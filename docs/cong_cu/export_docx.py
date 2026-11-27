"""Script tạo file Word docx chuyên nghiệp cho tài liệu Giải Thích Code Môi Trường."""

from pathlib import Path
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

DOCX_PATH = Path(__file__).resolve().parent.parent / "moi_truong" / "GIAI_THICH_CODE_MOI_TRUONG.docx"

# Màu sắc chủ đạo (Corporate Tech Palette)
COLOR_PRIMARY = RGBColor(27, 54, 93)     # Deep Navy
COLOR_SECONDARY = RGBColor(13, 122, 117) # Teal / Dark Mint
COLOR_DARK = RGBColor(33, 37, 41)        # Text Dark Gray
COLOR_MUTED = RGBColor(108, 117, 125)    # Gray
COLOR_CODE = RGBColor(30, 30, 30)

HEX_PRIMARY = "1B365D"
HEX_SECONDARY = "0D7A75"
HEX_LIGHT_BG = "F4F7FA"
HEX_CODE_BG = "F8F9FA"
HEX_BORDER = "CCCCCC"


def set_cell_background(cell, hex_color: str):
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    cell._tc.get_or_add_tcPr().append(shd)


def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('w:top', top), ('w:bottom', bottom), ('w:left', left), ('w:right', right)]:
        node = OxmlElement(m)
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)


def add_callout(doc, text: str, title: str = "LƯU Ý QUAN TRỌNG"):
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    table.columns[0].width = Inches(6.5)
    
    cell = table.cell(0, 0)
    set_cell_background(cell, "EBF5F5")
    set_cell_margins(cell, top=140, bottom=140, left=200, right=180)
    
    # Left border teal thick
    borders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'<w:top w:val="none"/>'
        f'<w:left w:val="single" w:sz="36" w:space="0" w:color="{HEX_SECONDARY}"/>'
        f'<w:bottom w:val="none"/>'
        f'<w:right w:val="none"/>'
        f'</w:tcBorders>'
    )
    cell._tc.get_or_add_tcPr().append(borders)
    
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(4)
    run_title = p.add_run(f"📌 {title}: ")
    run_title.font.name = "Segoe UI"
    run_title.font.size = Pt(10.5)
    run_title.font.bold = True
    run_title.font.color.rgb = COLOR_SECONDARY
    
    run_text = p.add_run(text)
    run_text.font.name = "Segoe UI"
    run_text.font.size = Pt(10)
    run_text.font.color.rgb = COLOR_DARK


def add_code_block(doc, code_str: str):
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    table.columns[0].width = Inches(6.5)
    
    cell = table.cell(0, 0)
    set_cell_background(cell, HEX_CODE_BG)
    set_cell_margins(cell, top=100, bottom=100, left=140, right=140)
    
    borders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'<w:top w:val="single" w:sz="4" w:space="0" w:color="{HEX_BORDER}"/>'
        f'<w:left w:val="single" w:sz="18" w:space="0" w:color="{HEX_PRIMARY}"/>'
        f'<w:bottom w:val="single" w:sz="4" w:space="0" w:color="{HEX_BORDER}"/>'
        f'<w:right w:val="single" w:sz="4" w:space="0" w:color="{HEX_BORDER}"/>'
        f'</w:tcBorders>'
    )
    cell._tc.get_or_add_tcPr().append(borders)
    
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.15
    run = p.add_run(code_str.strip())
    run.font.name = "Consolas"
    run.font.size = Pt(9.0)
    run.font.color.rgb = COLOR_CODE


def add_styled_table(doc, headers: list[str], rows: list[list[str]], col_widths: list[float] | None = None):
    table = doc.add_table(rows=len(rows) + 1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    
    # Header row
    hdr_cells = table.rows[0].cells
    for i, title in enumerate(headers):
        hdr_cells[i].text = title
        set_cell_background(hdr_cells[i], HEX_PRIMARY)
        set_cell_margins(hdr_cells[i], top=120, bottom=120, left=120, right=120)
        p = hdr_cells[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(2)
        for r in p.runs:
            r.font.name = "Segoe UI"
            r.font.size = Pt(10)
            r.font.bold = True
            r.font.color.rgb = RGBColor(255, 255, 255)
            
    # Data rows
    for r_idx, row_data in enumerate(rows):
        row_cells = table.rows[r_idx + 1].cells
        bg_color = HEX_LIGHT_BG if r_idx % 2 == 1 else "FFFFFF"
        for c_idx, val in enumerate(row_data):
            row_cells[c_idx].text = val
            set_cell_background(row_cells[c_idx], bg_color)
            set_cell_margins(row_cells[c_idx], top=80, bottom=80, left=100, right=100)
            p = row_cells[c_idx].paragraphs[0]
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)
            p.paragraph_format.line_spacing = 1.15
            for r in p.runs:
                r.font.name = "Segoe UI"
                r.font.size = Pt(9.5)
                r.font.color.rgb = COLOR_DARK
                
    # Table borders subtle
    for row in table.rows:
        for cell in row.cells:
            borders = parse_xml(
                f'<w:tcBorders {nsdecls("w")}>'
                f'<w:top w:val="single" w:sz="4" w:space="0" w:color="{HEX_BORDER}"/>'
                f'<w:left w:val="none"/>'
                f'<w:bottom w:val="single" w:sz="4" w:space="0" w:color="{HEX_BORDER}"/>'
                f'<w:right w:val="none"/>'
                f'</w:tcBorders>'
            )
            cell._tc.get_or_add_tcPr().append(borders)
            
    # Column widths
    if col_widths:
        for row in table.rows:
            for idx, width in enumerate(col_widths):
                row.cells[idx].width = Inches(width)


def build_document():
    doc = docx.Document()
    
    # Cấu hình Margins trang (0.8 inch)
    for section in doc.sections:
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.8)
        section.right_margin = Inches(0.8)
        
    # --- TIÊU ĐỀ CHÍNH ---
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(12)
    p_title.paragraph_format.space_after = Pt(4)
    run_title = p_title.add_run("GIẢI THÍCH CHI TIẾT MÃ NGUỒN MÔI TRƯỜNG XE TỰ LÁI NÉ VẬT CẢN\n(LANE OBSTACLE ENVIRONMENT)")
    run_title.font.name = "Segoe UI"
    run_title.font.size = Pt(18)
    run_title.font.bold = True
    run_title.font.color.rgb = COLOR_PRIMARY
    
    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sub.paragraph_format.space_after = Pt(16)
    run_sub = p_sub.add_run("Tài liệu kỹ thuật giải thích thiết kế, cấu trúc mã nguồn, không gian trạng thái, hành động,\ncơ chế động học, cảm biến radar, hàm thưởng và giao diện lập trình Gymnasium API")
    run_sub.font.name = "Segoe UI"
    run_sub.font.size = Pt(11)
    run_sub.font.italic = True
    run_sub.font.color.rgb = COLOR_SECONDARY
    
    # Metadata Box
    add_callout(doc, 
        "Hệ thống: HighwayEnv (Farama Foundation) + Gymnasium 1.4.0 + PyTorch 2.x + Pygame 2.5\n"
        "Mã nguồn môi trường cốt lõi: highway_project/environment/environment.py\n"
        "Đối tượng áp dụng: Tác nhân tự hành Deep Q-Network (Double DQN), Tabular Q-learning, PPO, SAC",
        "THÔNG TIN HỆ THỐNG & NỀN TẢNG"
    )
    doc.add_paragraph()
    
    # --- MỤC 1 ---
    h1 = doc.add_heading(level=1)
    r = h1.add_run("1. TỔNG QUAN KIẾN TRÚC MÔI TRƯỜNG MÔ PHỎNG")
    r.font.name = "Segoe UI"; r.font.color.rgb = COLOR_PRIMARY; r.font.size = Pt(14); r.font.bold = True
    
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.2
    p.add_run(
        "Môi trường LaneObstacleEnv được xây dựng kế thừa từ HighwayEnv kết hợp bộ tiêu chuẩn Gymnasium API, "
        "mô phỏng tình huống xe tự hành (Ego Vehicle) di chuyển trên tuyến đường cao tốc dài 1000m với dải tốc độ cao (70 - 108 km/h). "
        "Nhiệm vụ cốt lõi của xe là nhận biết và chủ động chuyển làn né tránh các chướng ngại vật tĩnh (xe chết máy, xe tai nạn dừng khẩn cấp), "
        "đồng thời điều hướng an toàn trong dòng xe giao thông đông đúc, có tính cách khó đoán."
    )
    
    doc.add_heading(level=2).add_run("1.1. Bốn Trụ Cột Thiết Kế Của Môi Trường").font.color.rgb = COLOR_SECONDARY
    features = [
        ("Tuân thủ 100% chuẩn Gymnasium API: ", "Cung cấp đầy đủ reset() và step() với bộ trả về chuẩn (obs, reward, terminated, truncated, info) giúp mọi thuật toán RL dễ dàng kết nối."),
        ("Khả năng tùy biến kịch bản linh hoạt (Customizable Scenario): ", "Hỗ trợ thay đổi số làn từ 2 đến 6 làn đường (--lanes), mật độ xe theo 3 preset trực quan (thưa thớt / bình thường / dày đặc / không có xe), và dải tốc độ dòng xe."),
        ("Cảm biến không gian và Radar đa làn (Enhanced Telemetry): ", "Quét 180m theo từng làn đường, cung cấp cự ly an toàn tức thời ego_front_gap (O(1)), cờ left_clear, right_clear và cờ changing hỗ trợ Action Masking."),
        ("Giao diện đồ họa Pygame 60 FPS: ", "Tự động co giãn lòng đường theo số làn, kết hợp bộ đệm nội suy động học MotionBuffer tái hiện chuyển động mượt mà không bị giật khựng.")
    ]
    for b_title, b_desc in features:
        bp = doc.add_paragraph(style='List Bullet')
        bp.paragraph_format.line_spacing = 1.15
        r1 = bp.add_run(b_title); r1.font.bold = True; r1.font.name = "Segoe UI"
        r2 = bp.add_run(b_desc); r2.font.name = "Segoe UI"

    doc.add_heading(level=2).add_run("1.2. Bản Đồ File Mã Nguồn và Phân Trách Nhiệm").font.color.rgb = COLOR_SECONDARY
    file_map = [
        ["highway_project/environment/environment.py", "Môi trường lõi (Core Env)", "Định nghĩa LaneObstacleEnv, BrokenCar, cấu hình mặc định, tạo đường và vật cản, cảm biến đa làn, hàm thưởng."],
        ["highway_project/algorithms/dqn_agent.py", "Thuật toán DQN", "Cài đặt DQN 2013 (mạng đơn không Target Network, MSELoss), mạng QNetwork, ReplayBuffer, chiếu vector 68 chiều, Action Masking và chống chết máy."],
        ["highway_project/algorithms/q_agent.py", "Thuật toán Q-Learning", "Cài đặt Tabular Q-learning dạng bảng (648 nhóm trạng thái), cập nhật Bellman cổ điển."],
        ["highway_project/gui/view.py", "Giao diện Pygame", "Vẽ đồ họa mặt đường co giãn 2..6 làn, vạch phân làn, xe mô phỏng, Bảng Menu Cài Đặt Modal, thanh điều khiển 4 nút và thông báo Toast."],
        ["highway_project/gui/motion.py", "Nội suy chuyển động", "Bộ đệm MotionBuffer nội suy LERP vị trí và góc lái 60 FPS giữa 2 chu kỳ quyết định 10 Hz."],
        ["highway_project/training/train_dqn.py", "Huấn luyện DQN", "Huấn luyện mạng nơ-ron theo Curriculum Learning (tăng dần 0 -> 12 xe), lưu dqn_model.pth."],
        ["highway_project/run.py", "Khởi chạy & Vòng lặp", "Script dòng lệnh hỗ trợ CLI, vòng lặp mô phỏng, và hàm live_reconfigure tái cấu hình trực tiếp không reset xe."],
        ["highway_project/launcher.py", "Menu Console", "Trình menu console tương tác tiếng Việt thông minh, xử lý mã hóa UTF-8 chống lỗi hiển thị."],
        ["CHAY_MOI_TRUONG.cmd", "Runner duy nhất", "Script Windows một chạm: kiểm tra môi trường ảo, checkpoint mô hình, và mở giao diện mô phỏng Pygame."],
        ["CAI_DAT_MOI_TRUONG.cmd", "Setup môi trường", "Script tự động tạo môi trường ảo Python và cài đặt trọn bộ thư viện phụ thuộc."],
        ["highway_project/tests/", "Kiểm thử tự động", "Bộ 21 unit tests tự động bao quát toàn bộ hợp đồng môi trường, thuật toán DQN, GUI và Live Reconfiguration."]
    ]
    add_styled_table(doc, ["File / Đường dẫn", "Vai trò", "Mô tả chi tiết chức năng"], file_map, [2.2, 1.5, 2.8])
    doc.add_paragraph()

    # --- MỤC 2 ---
    h1 = doc.add_heading(level=1)
    r = h1.add_run("2. CẤU HÌNH MÔI TRƯỜNG: LaneObstacleEnv.default_config()")
    r.font.name = "Segoe UI"; r.font.color.rgb = COLOR_PRIMARY; r.font.size = Pt(14); r.font.bold = True
    
    doc.add_paragraph("Mọi thiết lập của môi trường được quản lý tập trung qua phương thức default_config():")
    add_code_block(doc,
"""@classmethod
def default_config(cls) -> dict:
    config = super().default_config()
    config.update({
        "lanes_count": 3,               # Số làn đường cao tốc (hỗ trợ từ 2 đến 6 làn)
        "vehicles_count": 20,           # Mật độ xe: "sparse"(8), "normal"(20), "dense"(38) hoặc số 0..60
        "controlled_vehicles": 1,       # 1 xe do thuật toán điều khiển
        "initial_lane_id": 1,           # Làn xuất phát (mặc định làn giữa)
        "initial_speed": 20.0,          # Vận tốc xuất phát (20 m/s = 72 km/h)
        "max_speed": 30.0,              # Vận tốc trần tối đa (30 m/s = 108 km/h)
        "traffic_speed_min": 14.0,      # Dải tốc độ xe giao thông (50 - 100 km/h)
        "traffic_speed_max": 28.0,
        "duration": 90.0,               # Thời gian tối đa của tập (90 giây = 900 bước)
        "simulation_frequency": 30,     # Tần số mô phỏng vật lý (30 Hz)
        "policy_frequency": 10,         # Tần số ra quyết định của AI (10 Hz = 0.1s / quyết định)
        "route_length": 1000.0,         # Chiều dài quãng đường tới đích (1000m)
        "obstacle_count": 1,            # Số lượng xe hỏng chắn làn (0..3 xe)
        "obstacle_distance": 180.0,     # Cự ly xe hỏng đầu tiên
        "obstacle_spacing": 280.0,      # Khoảng cách tối thiểu giữa các xe hỏng
        "random_obstacles": False,      # Sinh làn ngẫu nhiên cho vật cản
        "action": {"type": "DiscreteMetaAction", "target_speeds": [0, 5, 10, 15, 20, 25, 30]},
        "observation": {"type": "Kinematics", "vehicles_count": 10, "features": ["presence", "x", "y", "vx", "vy"]},
        "collision_penalty": -20.0,     # Phạt va chạm cực nặng
        "offroad_penalty": -10.0,       # Phạt lao ra khỏi đường
        "speed_weight": 0.03,           # Trọng số thưởng tốc độ cao an toàn
        "danger_penalty": -0.15,        # Phạt áp sát cự ly nguy hiểm
        "lane_change_penalty": -0.02,   # Phạt đổi làn liên tục gây mất êm ái
        "success_reward": 10.0,         # Thưởng cán đích an toàn
    })
    return config""")
    
    doc.add_paragraph()
    add_callout(doc,
        "Tần số điều khiển (policy_frequency = 10 Hz) tương ứng mỗi bước ra quyết định kéo dài dt = 0.1s. "
        "Tần số vật lý (simulation_frequency = 30 Hz) có nghĩa là engine thực hiện 3 bước tích phân số học vi phân "
        "trong mỗi chu kỳ quyết định của tác nhân. Điều này giúp tính toán va chạm và động học bám làn cực kỳ chính xác.",
        "QUAN HỆ TẦN SỐ ĐIỀU KHIỂN & VẬT LÝ"
    )
    doc.add_paragraph()

    # --- MỤC 3 ---
    h1 = doc.add_heading(level=1)
    r = h1.add_run("3. CƠ CHẾ SINH XE VÀ VẬT CẢN: _create_vehicles()")
    r.font.name = "Segoe UI"; r.font.color.rgb = COLOR_PRIMARY; r.font.size = Pt(14); r.font.bold = True

    doc.add_paragraph(
        "Phương thức _create_vehicles() chịu trách nhiệm tái tạo thế giới vật lý mỗi khi reset() được gọi:"
    )
    
    doc.add_heading(level=2).add_run("3.1. Khởi Tạo Xe Điều Khiển (Ego Vehicle)").font.color.rgb = COLOR_SECONDARY
    doc.add_paragraph(
        "Xe điều khiển được khởi tạo từ lớp MDPVehicle, đặt tại vị trí x = 80.0 m trên làn initial_lane_id (mặc định làn giữa). "
        "Thuộc tính ego.target_speed được gán trực tiếp bằng initial_speed để bộ điều khiển PID bám tốc độ ngay từ thời điểm đầu tiên, "
        "không bị độ trễ tăng tốc từ 0 m/s."
    )

    doc.add_heading(level=2).add_run("3.2. Vật Cản Cố Định (BrokenCar) & Thuật Toán Phân Đoạn (Smart Sector Distribution)").font.color.rgb = COLOR_SECONDARY
    doc.add_paragraph(
        "Lớp BrokenCar kế thừa từ Obstacle, có kích thước 5.0m x 2.0m và vận tốc speed = 0.0 m/s. "
        "Do không có bộ điều khiển gia tốc, vật cản đứng bất động tuyệt đối suốt thời gian mô phỏng."
    )
    doc.add_paragraph(
        "Khi bật cờ random_obstacles = True, môi trường áp dụng thuật toán chia phân đoạn độc lập (Sectors):"
    )
    add_code_block(doc,
"""route_len = float(self.config["route_length"])
min_start = min(130.0, max(60.0, route_len * 0.2))   # Cách điểm xuất phát tối thiểu 100m
safe_goal = min(70.0, max(30.0, route_len * 0.1))
span = max(40.0, route_len - min_start - safe_goal)
sector_len = span / count                             # Chia tuyến đường thành count sectors đều nhau

for i in range(count):
    obstacle_lane = int(self.np_random.integers(0, lanes_count)) # Sinh ngẫu nhiên làn 0..lanes_count-1
    margin = min(15.0, sector_len * 0.1)
    x_min = self.start_x + min_start + i * sector_len + margin
    x_max = self.start_x + min_start + (i + 1) * sector_len - margin
    x_pos = float(self.np_random.uniform(x_min, max(x_min + 1.0, x_max)))
    obstacle = BrokenCar.make_on_lane(self.road, ("0", "1", obstacle_lane), x_pos, speed=0.0)
    self.obstacles.append(obstacle)
    self.road.objects.append(obstacle)""")

    doc.add_heading(level=2).add_run("3.3. Dòng Xe Giao Thông IDM & Tính Cách Lái Xe Khó Đoán").font.color.rgb = COLOR_SECONDARY
    doc.add_paragraph(
        "Các xe giao thông xung quanh là các xe IDMVehicle kết hợp thuật toán chuyển làn MOBIL. "
        "Khi unpredictable_traffic = True, tính cách xe được ngẫu nhiên hóa:"
    )
    behavior_items = [
        ("Vận tốc đa dạng: ", "Vận tốc sinh ngẫu nhiên trong dải [traffic_speed_min, traffic_speed_max]."),
        ("Độ lịch sự thấp (POLITENESS = 0.0 .. 0.35): ", "Xe giao thông có xu hướng hung hăng, sẵn sàng tạt đầu hoặc chuyển làn nếu làn bên cạnh thoáng hơn."),
        ("Khả năng phanh ép (LANE_CHANGE_MAX_BRAKING_IMPOSED = 1.8 .. 3.8 m/s²): ", "Cho phép xe khác ép xe sau phải phanh gấp khi chuyển làn, tạo ra các tình huống bất ngờ đòi hỏi AI phải xử lý."),
        ("Chống sinh đè: ", "Khoảng cách khi sinh luôn đảm bảo cách xe khác và vật cản tối thiểu 28m cùng làn.")
    ]
    for b_title, b_desc in behavior_items:
        bp = doc.add_paragraph(style='List Bullet')
        bp.paragraph_format.line_spacing = 1.15
        r1 = bp.add_run(b_title); r1.font.bold = True; r1.font.name = "Segoe UI"
        r2 = bp.add_run(b_desc); r2.font.name = "Segoe UI"
    doc.add_paragraph()

    # --- MỤC 4 ---
    h1 = doc.add_heading(level=1)
    r = h1.add_run("4. KHÔNG GIAN HÀNH ĐỘNG: DiscreteMetaAction")
    r.font.name = "Segoe UI"; r.font.color.rgb = COLOR_PRIMARY; r.font.size = Pt(14); r.font.bold = True
    
    actions_data = [
        ["0", "LANE_LEFT", "Đặt mục tiêu chuyển sang làn liền kề bên trái (target_lane = lane - 1)"],
        ["1", "IDLE", "Giữ nguyên làn đường và vận tốc mục tiêu hiện tại (bám quỹ đạo ổn định)"],
        ["2", "LANE_RIGHT", "Đặt mục tiêu chuyển sang làn liền kề bên phải (target_lane = lane + 1)"],
        ["3", "FASTER", "Tăng một nấc vận tốc mục tiêu trên lưới (+5.0 m/s hoặc tương ứng)"],
        ["4", "SLOWER", "Giảm một nấc vận tốc mục tiêu trên lưới (-5.0 m/s hoặc tương ứng)"]
    ]
    add_styled_table(doc, ["ID Hành Động", "Tên Lệnh", "Ý Nghĩa Động Học Chi Tiết"], actions_data, [1.2, 1.8, 3.5])
    doc.add_paragraph()
    
    add_callout(doc,
        "Hành động chuyển làn KHÔNG làm xe nhảy tức thời tọa độ sang làn mới (Non-teleportation). "
        "Bộ điều khiển trong engine tính toán góc đánh lái psi theo đường cong mượt mà để thân xe chuyển dần sang "
        "tâm làn mới trong khoảng 1.0 - 1.5 giây. Trong suốt thời gian này, cờ sensors['changing'] = True, "
        "giúp hệ thống Action Masking khóa lệnh đổi làn tiếp theo để chống lật xe.",
        "NGUYÊN LÝ ĐỘNG HỌC CHUYỂN LÀN"
    )
    doc.add_paragraph()

    # --- MỤC 5 ---
    h1 = doc.add_heading(level=1)
    r = h1.add_run("5. KHÔNG GIAN QUAN SÁT & RADAR CẢM BIẾN ĐA LÀN: _sensors()")
    r.font.name = "Segoe UI"; r.font.color.rgb = COLOR_PRIMARY; r.font.size = Pt(14); r.font.bold = True

    doc.add_heading(level=2).add_run("5.1. Quan Sát Ma Trận Chuẩn Gymnasium Box(10, 5)").font.color.rgb = COLOR_SECONDARY
    doc.add_paragraph(
        "Quan sát obs trả về từ reset() và step() là ma trận 10 dòng x 5 cột kiểu float32:\n"
        "- Dòng 0: Đặc trưng của xe Ego (tọa độ và vận tốc tuyệt đối chuẩn hóa).\n"
        "- Dòng 1 đến 9: 9 xe hoặc vật cản gần nhất, sắp xếp theo khoảng cách tăng dần.\n"
        "- 5 thuộc tính: [presence, x, y, vx, vy] đại diện cho sự hiện diện, tọa độ dọc, tọa độ ngang và vận tốc tương đối."
    )

    doc.add_heading(level=2).add_run("5.2. Khối Radar Cảm Biến Không Gian Đa Làn (_sensors)").font.color.rgb = COLOR_SECONDARY
    doc.add_paragraph(
        "Phương thức _sensors() thực hiện quét toàn bộ không gian xung quanh trong bán kính 180m:"
    )
    add_code_block(doc,
"""def _sensors(self) -> dict:
    ego = self.vehicle
    lanes_count = int(self.config.get("lanes_count", 3))
    front = [180.0] * lanes_count
    rear = [180.0] * lanes_count
    front_closing = [0.0] * lanes_count
    rear_closing = [0.0] * lanes_count

    for obj in [*self.road.vehicles, *self.road.objects]:
        if obj is ego: continue
        dx = float(obj.position[0] - ego.position[0])
        if abs(dx) > 180: continue
        gap = max(0.0, abs(dx) - (obj.LENGTH + ego.LENGTH) / 2) # Khoảng hở thân xe thực tế
        for lane_id in range(lanes_count):
            if abs(float(obj.position[1]) - lane_id * 4) > 2.7: continue
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
    return {
        "lane": lane_id, "lanes_count": lanes_count, "traffic_density": getattr(self, "traffic_density", "normal"),
        "speed": float(ego.speed), "speed_kmh": float(ego.speed * 3.6), "changing": bool(changing),
        "target_lane": target_lane, "front_gap": front, "rear_gap": rear,
        "front_closing": front_closing, "rear_closing": rear_closing,
        "ego_front_gap": float(ego_front_gap), "left_clear": bool(left_clear), "right_clear": bool(right_clear)
    }""")
    doc.add_paragraph()

    # --- MỤC 6 ---
    h1 = doc.add_heading(level=1)
    r = h1.add_run("6. HÀM PHẦN THƯỞNG ĐA MỤC TIÊU: _rewards() & _reward()")
    r.font.name = "Segoe UI"; r.font.color.rgb = COLOR_PRIMARY; r.font.size = Pt(14); r.font.bold = True

    doc.add_paragraph(
        "Hàm phần thưởng tích hợp 6 thành phần giải tích độc lập cân bằng giữa An toàn, Tốc độ và Sự êm dịu:"
    )
    doc.add_paragraph("r_t = r_collision + r_offroad + r_speed + r_lane_change + r_success + r_danger")
    
    rewards_table = [
        ["r_collision", "collision_penalty", "-20.0", "Phạt cực nặng khi xe xảy ra va chạm. Tập mô phỏng lập tức kết thúc."],
        ["r_offroad", "offroad_penalty", "-10.0", "Phạt nặng khi xe chệch khỏi mặt đường nhựa lao ra lề cỏ."],
        ["r_speed", "speed_weight", "+0.03 * (v / 30)", "Thưởng duy trì tốc độ cao an toàn (đạt cực đại +0.03/bước khi v = 30 m/s)."],
        ["r_lane_change", "lane_change_penalty", "-0.02", "Phạt nhẹ mỗi lần đổi làn (action 0 hoặc 2), khuyến khích giữ làn ổn định."],
        ["r_success", "success_reward", "+10.0", "Thưởng lớn khi hoàn thành trọn vẹn 1000m và cán đích an toàn."],
        ["r_danger", "danger_penalty", "-0.15", "Phạt cảnh báo khi khoảng cách trước < 10m hoặc thời gian chạm va TTC < 2.0s."]
    ]
    add_styled_table(doc, ["Ký hiệu toán", "Khóa cấu hình", "Giá trị điểm", "Điều kiện áp dụng & Ý nghĩa"], rewards_table, [1.4, 1.6, 1.2, 2.3])
    doc.add_paragraph()

    add_callout(doc,
        "Nguyên tắc phân nhánh loại trừ lẫn nhau: Khi xảy ra va chạm (crashed == True), chỉ số hạng collision "
        "nhận điểm -20.0, các số hạng khác được gán bằng 0. Tác nhân không bao giờ vừa bị phạt va chạm vừa nhận điểm tốc độ.",
        "NGUYÊN TẮC PHÂN NHÁNH THƯỞNG"
    )
    doc.add_paragraph()

    # --- MỤC 7 ---
    h1 = doc.add_heading(level=1)
    r = h1.add_run("7. ĐIỀU KIỆN DỪNG VÀ SIÊU DỮ LIỆU: _info()")
    r.font.name = "Segoe UI"; r.font.color.rgb = COLOR_PRIMARY; r.font.size = Pt(14); r.font.bold = True

    doc.add_heading(level=2).add_run("7.1. Phân Biệt terminated và truncated").font.color.rgb = COLOR_SECONDARY
    term_diff = [
        ("terminated == True (Kết thúc vật lý): ", "Xảy ra khi xe va chạm, xe lao ra ngoài đường, hoặc cán đích an toàn (_success() == True). Giá trị hàm V(s') = 0, không bootstrap Bellman."),
        ("truncated == True (Hết thời gian): ", "Xảy ra khi elapsed_s >= duration (mặc định 90s). Xe vẫn hoàn toàn an toàn. Thuật toán RL vẫn được phép bootstrap V(s') = max_a' Q(s', a').")
    ]
    for b_title, b_desc in term_diff:
        bp = doc.add_paragraph(style='List Bullet')
        bp.paragraph_format.line_spacing = 1.15
        r1 = bp.add_run(b_title); r1.font.bold = True; r1.font.name = "Segoe UI"
        r2 = bp.add_run(b_desc); r2.font.name = "Segoe UI"

    doc.add_heading(level=2).add_run("7.2. Cấu Trúc Siêu Dữ Liệu info").font.color.rgb = COLOR_SECONDARY
    doc.add_paragraph(
        "Mỗi bước step(), info trả về đầy đủ các trường giám sát:\n"
        "- is_success (bool): True khi xe vượt qua vạch đích 1000m an toàn.\n"
        "- distance_m (float) & remaining_m (float): Quãng đường đã đi và cự ly còn lại tới đích.\n"
        "- obstacles_passed (int) & obstacles_total (int): Số lượng chướng ngại vật đã vượt qua / tổng số chướng ngại vật.\n"
        "- speed (m/s) & speed_kmh (km/h): Vận tốc tức thời của xe.\n"
        "- available_actions (list[int]): Danh sách hành động hợp lệ phục vụ Action Masking.\n"
        "- end_reason (str): 'collision', 'off_road', 'success', 'time_limit', hoặc 'running'.\n"
        "- sensors (dict): Khối radar không gian chi tiết."
    )
    doc.add_paragraph()

    # --- MỤC 8 ---
    h1 = doc.add_heading(level=1)
    r = h1.add_run("8. GIAO DIỆN PYGAME VÀ BỘ ĐỆM NỘI SUY 60 FPS")
    r.font.name = "Segoe UI"; r.font.color.rgb = COLOR_PRIMARY; r.font.size = Pt(14); r.font.bold = True

    doc.add_heading(level=2).add_run("8.1. Co Giãn Mặt Đường Động Trong HighwayView").font.color.rgb = COLOR_SECONDARY
    doc.add_paragraph(
        "Lớp HighwayView trong highway_project/gui/view.py tự động tính toán lại vị trí mặt đường theo lanes_count:\n"
        "- lanes_count <= 3: road_top = 294, lane_height = 72.\n"
        "- lanes_count == 4: road_top = 278, lane_height = 64.\n"
        "- lanes_count == 5: road_top = 268, lane_height = 54.\n"
        "- lanes_count == 6: road_top = 260, lane_height = 46.\n"
        "Nhờ đó lòng đường luôn hiển thị vừa vặn và thẩm mỹ, không bao giờ bị đè lên thanh điều khiển HUD bên dưới."
    )

    doc.add_heading(level=2).add_run("8.2. Nội Suy Chuyển Động Tuyến Tính 60 FPS (MotionBuffer)").font.color.rgb = COLOR_SECONDARY
    doc.add_paragraph(
        "Do AI ra quyết định ở 10 Hz (0.1s / quyết định), nếu vẽ trực tiếp xe sẽ bị giật. MotionBuffer khắc phục bằng cách "
        "chụp lại 2 trạng thái vị trí (trước và sau bước tích phân) và nội suy tuyến tính (LERP):\n"
        "p_draw = (1 - alpha) * p_prev + alpha * p_curr\n"
        "Góc quay thân xe (Heading) được nội suy theo cung lượng giác ngắn nhất Delta_theta, "
        "mang lại hình ảnh mượt mà tuyệt đối ở 60 FPS."
    )

    doc.add_heading(level=2).add_run("8.3. Bảng Menu Cài Đặt Nổi Trung Tâm (Popup Modal)").font.color.rgb = COLOR_SECONDARY
    doc.add_paragraph(
        "Nhấn phím M hoặc click '☰ CÀI ĐẶT & KỊCH BẢN (M)' để mở bảng Menu Modal nổi bật giữa màn hình với lớp phủ mờ tối (Dim Backdrop 215 alpha):\n"
        "- 1. Số làn đường cao tốc (2..6 làn).\n"
        "- 2. Mật độ xe giao thông xung quanh (Thưa 8 xe, Bình thường 20 xe, Dày 38 xe, Không có xe 0 xe).\n"
        "- 3. Vận tốc xe của bạn (54, 72, 90, 108 km/h).\n"
        "- 4. Tốc độ dòng xe (Chậm slow, Bình thường normal, Nhanh fast).\n"
        "- 5. Kịch bản mẫu tích hợp sẵn (Kịch bản 1..4).\n"
        "- 6. Chế độ điều khiển & Thuật toán (AI: DQN / Bàn phím / Q-learning), Auto-reset, Reset (R), Đóng (Esc / M)."
    )

    doc.add_heading(level=2).add_run("8.4. Cơ Chế Tái Cấu Hình Trực Tiếp (Live Reconfiguration - Không Reset Xe)").font.color.rgb = COLOR_SECONDARY
    doc.add_paragraph(
        "Hàm live_reconfigure trong run.py tái tạo trực tiếp mạng đường RoadNetwork.straight_road_network, "
        "kẹp lại chỉ số làn và vị trí y của toàn bộ xe, cập nhật vận tốc tức thời và mật độ giao thông mà TUYỆT ĐỐI KHÔNG RESET TIẾN ĐỘ XE. "
        "Xe của bạn tiếp tục lăn bánh tiến về đích, bảo toàn quãng đường đã đi (distance_m) và điểm thưởng tích lũy."
    )

    doc.add_heading(level=2).add_run("8.5. Hệ Thống Thông Báo Nổi Toast & Thanh Đáy Tinh Gọn").font.color.rgb = COLOR_SECONDARY
    doc.add_paragraph(
        "- show_toast(text, 2500ms): Hiển thị thông báo nổi viền sáng ngọc lam góc dưới bên phải phản hồi tức thì mọi thao tác.\n"
        "- Thanh đáy chính tinh gọn chỉ 4 nút: Space (Tạm dừng/Tiếp tục), Tab (AI/Phím), ☰ CÀI ĐẶT & KỊCH BẢN (M), R (Chạy lại từ đầu)."
    )
    doc.add_paragraph()

    # --- MỤC 9 ---
    h1 = doc.add_heading(level=1)
    r = h1.add_run("9. HƯỚNG DẪN KẾT NỐI THUẬT TOÁN HỌC TĂNG CƯỜNG")
    r.font.name = "Segoe UI"; r.font.color.rgb = COLOR_PRIMARY; r.font.size = Pt(14); r.font.bold = True

    doc.add_paragraph("Mã nguồn mẫu kết nối môi trường hoàn chỉnh theo chuẩn Gymnasium API:")
    add_code_block(doc,
"""from highway_project.environment.environment import make_env, ACTIONS

# 1. Khởi tạo môi trường với cấu hình mong muốn
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
    
    # 2. Action Masking: Khóa lệnh chuyển làn khi đang chuyển dở
    if sensors["changing"]:
        valid_actions = [a for a in valid_actions if a not in (0, 2)]
        
    # 3. Lựa chọn hành động dựa trên cảm biến hoặc mạng nơ-ron
    if sensors["ego_front_gap"] < 35.0:  # Phía trước có vật cản gần
        if sensors["left_clear"] and 0 in valid_actions:
            action = 0  # Lách sang làn trái
        elif sensors["right_clear"] and 2 in valid_actions:
            action = 2  # Lách sang làn phải
        else:
            action = 4  # Giảm tốc khẩn cấp
    else:
        action = 3 if sensors["speed_kmh"] < 100.0 and 3 in valid_actions else 1

    # 4. Bước môi trường
    next_obs, reward, terminated, truncated, next_info = env.step(action)
    total_reward += reward
    done = terminated or truncated

    # 5. Cập nhật kinh nghiệm vào thuật toán RL
    # replay_buffer.push(obs, action, reward, next_obs, terminated, truncated)
    obs, info = next_obs, next_info

print(f"Kết thúc tập: {info['end_reason']} | Quãng đường: {info['distance_m']:.1f}m | Reward: {total_reward:.2f}")
env.close()""")
    doc.add_paragraph()

    # --- MỤC 10 ---
    h1 = doc.add_heading(level=1)
    r = h1.add_run("10. TỔNG HỢP CHIẾN LƯỢC KIỂM THỬ TỰ ĐỘNG (UNIT TESTS)")
    r.font.name = "Segoe UI"; r.font.color.rgb = COLOR_PRIMARY; r.font.size = Pt(14); r.font.bold = True

    doc.add_paragraph(
        "Dự án được bảo vệ chặt chẽ bởi bộ 21 unit tests tự động trong highway_project/tests/:"
    )
    test_cases = [
        ["test_gymnasium_contract_and_seed", "Kiểm tra tuân thủ giao thức Gymnasium (check_env) và tính tái lập khi dùng seed."],
        ["test_obstacle_visible_stationary_and_collision_terminates", "Kiểm tra xe hỏng đứng yên tuyệt đối và va chạm kích hoạt terminated=True."],
        ["test_lane_change_can_pass_obstacle_and_reach_goal", "Kiểm tra khả năng đổi làn né vật cản và cán đích an toàn (is_success=True)."],
        ["test_timeout_is_truncation_and_not_success", "Kiểm tra hết giờ trả về truncated=True và không tính là thành công."],
        ["test_idle_keeps_speed_and_rewards_match_info", "Kiểm tra duy trì vận tốc và tính toàn vẹn của các thành phần điểm thưởng."],
        ["test_random_obstacles_placement_and_safe_spacing", "Kiểm tra vật cản sinh ngẫu nhiên cách nhau >= 150m và cách xuất phát > 100m."],
        ["test_dynamic_lanes_traffic_presets_and_speed_config", "Kiểm tra cấu hình số làn động (2, 5 làn), preset mật độ xe, và tùy biến vận tốc."],
        ["test_view_renders_various_lanes_and_traffic", "Kiểm tra giao diện Pygame vẽ chính xác cho 2, 4, 5 làn ở chế độ headless."],
        ["test_live_reconfiguration_preserves_vehicle_progress", "Xác minh đổi số làn, mật độ trực tiếp trên live road network không làm reset xe, quãng đường tiếp tục tăng."],
        ["test_parse_traffic_and_cli_options", "Kiểm tra hàm phân tích cú pháp tham số dòng lệnh --traffic."],
        ["test_dqn_agent_various_lanes_and_speeds", "Kiểm tra mô hình DQN tự động ánh xạ vector trạng thái tương đối 68 chiều trên 2, 4, 5 làn."],
        ["test_dqn_dodges_obstacle_when_danger_detected", "Kiểm tra AI tự động lách làn hoặc phanh khi phát hiện xe hỏng chắn trước mặt."],
        ["ReplayBuffer, QNetwork, Bellman Target, Save/Load", "Kiểm tra cấu trúc mạng nơ-ron, lấy mẫu bộ đệm và tính toàn vẹn trọng số mô hình."]
    ]
    add_styled_table(doc, ["Tên Bài Kiểm Thử", "Mục Tiêu & Nội Dung Xác Minh"], test_cases, [2.5, 4.0])
    doc.add_paragraph()
    
    add_callout(doc,
        "Lệnh chạy toàn bộ 21 bài kiểm thử tự động:\n"
        ".\\.venv-highway\\Scripts\\python.exe -m unittest discover -s highway_project -t .\n"
        "Kết quả kiểm thử: Ran 21 tests -> OK (PASS 100% không lỗi)",
        "KẾT QUẢ KIỂM THỬ"
    )
    
    doc.save(str(DOCX_PATH))
    print(f"Xuat file thanh cong: {DOCX_PATH}")


if __name__ == "__main__":
    build_document()

