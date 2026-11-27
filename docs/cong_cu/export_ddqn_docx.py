"""Script chuyển đổi tài liệu Markdown (.md) sang Word (.docx) chuẩn Corporate Tech.

Mặc định xuất 2 tài liệu Thuật toán 2 (Double Dueling DQN - Ngô Tuấn Anh):
- docs/thuat_toan/ddqn/GIAI_THICH_CODE_DDQN.md -> GIAI_THICH_CODE_DDQN.docx
- docs/thuat_toan/ddqn/THUAT_TOAN_DDQN.md      -> THUAT_TOAN_DDQN.docx

Cũng hỗ trợ truyền file bất kỳ qua CLI:
    python docs/cong_cu/export_ddqn_docx.py path/to/file.md
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import docx
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Inches, Pt, RGBColor

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Bảng màu Corporate Tech Palette đồng bộ với export_docx.py
COLOR_PRIMARY = RGBColor(27, 54, 93)      # Deep Navy
COLOR_SECONDARY = RGBColor(13, 122, 117)  # Teal / Dark Mint
COLOR_DARK = RGBColor(33, 37, 41)         # Dark Gray
COLOR_MUTED = RGBColor(108, 117, 125)     # Gray
COLOR_CODE = RGBColor(30, 30, 30)

HEX_PRIMARY = "1B365D"
HEX_SECONDARY = "0D7A75"
HEX_LIGHT_BG = "F4F7FA"
HEX_CODE_BG = "F8F9FA"
HEX_CALLOUT_BG = "EBF5F5"
HEX_BORDER = "CCCCCC"

DDQN_DIR = Path(__file__).resolve().parent.parent / "thuat_toan" / "ddqn"
DEFAULT_TARGETS = [
    DDQN_DIR / "GIAI_THICH_CODE_DDQN.md",
    DDQN_DIR / "THUAT_TOAN_DDQN.md",
]


def set_cell_background(cell, hex_color: str) -> None:
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    cell._tc.get_or_add_tcPr().append(shd)


def set_cell_margins(cell, top=90, bottom=90, left=130, right=130) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_mar = OxmlElement("w:tcMar")
    for m, val in [("w:top", top), ("w:bottom", bottom), ("w:left", left), ("w:right", right)]:
        node = OxmlElement(m)
        node.set(qn("w:w"), str(val))
        node.set(qn("w:type"), "dxa")
        tc_mar.append(node)
    tc_pr.append(tc_mar)


def set_cell_borders(cell, *, color=HEX_BORDER, sz="4") -> None:
    borders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'<w:top w:val="single" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:left w:val="single" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:bottom w:val="single" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:right w:val="single" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f"</w:tcBorders>"
    )
    cell._tc.get_or_add_tcPr().append(borders)


def clean_latex_math(expr: str) -> str:
    """Chuyển công thức LaTeX phổ biến sang ký tự Unicode gọn đẹp trên Word."""
    s = expr.strip()
    replacements = [
        (r"\text{", "{"),
        (r"\operatorname{", "{"),
        (r"\mathcal{A}", "A"),
        (r"\mathcal{S}", "S"),
        (r"\mathcal{P}", "P"),
        (r"\mathcal{R}", "R"),
        (r"\mathcal{L}", "L"),
        (r"\mathbb{R}", "ℝ"),
        (r"\mathbb{E}", "𝔼"),
        (r"\arg\max", "argmax"),
        (r"\max", "max"),
        (r"\sum", "Σ"),
        (r"\frac{1}{2}", "1/2"),
        (r"\frac{1}{|\mathcal{A}|}", "1/|A|"),
        (r"\frac{1}{|A|}", "1/|A|"),
        (r"\frac{1}{B}", "1/B"),
        (r"\frac{1}{\tau}", "1/τ"),
        (r"\leftarrow", "←"),
        (r"\Rightarrow", "⇒"),
        (r"\rightarrow", "→"),
        (r"\to", "→"),
        (r"\le", "≤"),
        (r"\ge", "≥"),
        (r"\neq", "≠"),
        (r"\approx", "≈"),
        (r"\times", "×"),
        (r"\cdot", "·"),
        (r"\in", "∈"),
        (r"\subseteq", "⊆"),
        (r"\dots", "…"),
        (r"\ldots", "…"),
        (r"\theta^-", "θ⁻"),
        (r"\theta", "θ"),
        (r"\tau", "τ"),
        (r"\gamma", "γ"),
        (r"\epsilon", "ε"),
        (r"\delta", "δ"),
        (r"\phi", "φ"),
        (r"\nabla", "∇"),
        (r"\big(", "("),
        (r"\big)", ")"),
        (r"\left(", "("),
        (r"\right)", ")"),
        (r"\left[", "["),
        (r"\right]", "]"),
        (r"\left\{", "{"),
        (r"\right\}", "}"),
        (r"\{", "{"),
        (r"\}", "}"),
        (r"\|", "‖"),
        (r"\;", " "),
        (r"\,", " "),
        (r"\quad", "  "),
        (r"\_", "_"),
    ]
    for old, new in replacements:
        s = s.replace(old, new)
    # Gọn \frac{a}{b} -> (a)/(b)
    s = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", s)
    # Bỏ dấu ngoặc nhọn LaTeX thừa
    s = re.sub(r"\\begin\{cases\}|\\end\{cases\}", "", s)
    s = s.replace(r"\\", " | ").replace("&", " ")
    s = re.sub(r"(?<!\\)\{([^{}]+)\}", r"\1", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def strip_markdown_links(text: str) -> str:
    return re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)


def normalize_inline_math(text: str) -> str:
    text = strip_markdown_links(text)
    text = text.replace("<br>", "\n")
    # Inline math $...$
    return re.sub(r"\$([^$\n]+)\$", lambda m: clean_latex_math(m.group(1)), text)


def add_formatted_runs(p, text: str, *, base_size: float = 10.5, base_color=COLOR_DARK, base_bold: bool = False) -> None:
    """Thêm các run hỗ trợ **bold**, *italic*, và `inline code`."""
    cleaned = normalize_inline_math(text)
    token_pattern = re.compile(r"(\*\*[^*]+\*\*|`[^`]+`|\*[^*]+\*)")
    pos = 0
    for match in token_pattern.finditer(cleaned):
        if match.start() > pos:
            r = p.add_run(cleaned[pos:match.start()])
            r.font.name = "Segoe UI"
            r.font.size = Pt(base_size)
            r.font.bold = base_bold
            r.font.color.rgb = base_color

        tok = match.group(0)
        if tok.startswith("**") and tok.endswith("**"):
            inner = tok[2:-2]
            if inner.startswith("`") and inner.endswith("`"):
                r = p.add_run(inner[1:-1])
                r.font.name = "Consolas"
                r.font.size = Pt(max(8.5, base_size - 1.0))
                r.font.bold = True
                r.font.color.rgb = COLOR_PRIMARY
            else:
                r = p.add_run(inner)
                r.font.name = "Segoe UI"
                r.font.size = Pt(base_size)
                r.font.bold = True
                r.font.color.rgb = base_color
        elif tok.startswith("`") and tok.endswith("`"):
            r = p.add_run(tok[1:-1])
            r.font.name = "Consolas"
            r.font.size = Pt(max(8.5, base_size - 1.0))
            r.font.bold = base_bold
            r.font.color.rgb = COLOR_PRIMARY
        elif tok.startswith("*") and tok.endswith("*"):
            r = p.add_run(tok[1:-1])
            r.font.name = "Segoe UI"
            r.font.size = Pt(base_size)
            r.font.italic = True
            r.font.bold = base_bold
            r.font.color.rgb = base_color
        pos = match.end()

    if pos < len(cleaned):
        r = p.add_run(cleaned[pos:])
        r.font.name = "Segoe UI"
        r.font.size = Pt(base_size)
        r.font.bold = base_bold
        r.font.color.rgb = base_color


def add_callout_box(doc, lines: list[str]) -> None:
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    table.columns[0].width = Inches(6.5)

    cell = table.cell(0, 0)
    set_cell_background(cell, HEX_CALLOUT_BG)
    set_cell_margins(cell, top=110, bottom=110, left=180, right=160)

    borders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'<w:top w:val="none"/>'
        f'<w:left w:val="single" w:sz="30" w:space="0" w:color="{HEX_SECONDARY}"/>'
        f'<w:bottom w:val="none"/>'
        f'<w:right w:val="none"/>'
        f"</w:tcBorders>"
    )
    cell._tc.get_or_add_tcPr().append(borders)

    first = True
    for line in lines:
        p = cell.paragraphs[0] if first else cell.add_paragraph()
        first = False
        p.paragraph_format.space_before = Pt(1)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.line_spacing = 1.15
        add_formatted_runs(p, line, base_size=9.5, base_color=COLOR_DARK)

    spacer = doc.add_paragraph()
    spacer.paragraph_format.space_before = Pt(0)
    spacer.paragraph_format.space_after = Pt(4)


def add_code_block(doc, code_str: str) -> None:
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    table.columns[0].width = Inches(6.5)

    cell = table.cell(0, 0)
    set_cell_background(cell, HEX_CODE_BG)
    set_cell_margins(cell, top=90, bottom=90, left=130, right=130)

    borders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'<w:top w:val="single" w:sz="4" w:space="0" w:color="{HEX_BORDER}"/>'
        f'<w:left w:val="single" w:sz="18" w:space="0" w:color="{HEX_PRIMARY}"/>'
        f'<w:bottom w:val="single" w:sz="4" w:space="0" w:color="{HEX_BORDER}"/>'
        f'<w:right w:val="single" w:sz="4" w:space="0" w:color="{HEX_BORDER}"/>'
        f"</w:tcBorders>"
    )
    cell._tc.get_or_add_tcPr().append(borders)

    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(1)
    p.paragraph_format.space_after = Pt(1)
    p.paragraph_format.line_spacing = 1.1
    run = p.add_run(code_str.rstrip())
    run.font.name = "Consolas"
    run.font.size = Pt(8.5)
    run.font.color.rgb = COLOR_CODE

    spacer = doc.add_paragraph()
    spacer.paragraph_format.space_before = Pt(0)
    spacer.paragraph_format.space_after = Pt(4)


def add_math_block(doc, math_str: str) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(6)
    run = p.add_run(clean_latex_math(math_str))
    run.font.name = "Cambria Math"
    run.font.size = Pt(10.5)
    run.font.italic = True
    run.font.color.rgb = COLOR_PRIMARY


def add_markdown_table(doc, table_lines: list[str]) -> None:
    def split_row(row_str: str) -> list[str]:
        stripped = row_str.strip().strip("|")
        return [c.strip() for c in stripped.split("|")]

    parsed = [split_row(line) for line in table_lines]
    if len(parsed) < 2:
        return

    headers = parsed[0]
    # Bỏ dòng separator | :--- | :---: |
    data_rows = [r for r in parsed[2:] if len(r) == len(headers)]
    ncols = len(headers)

    table = doc.add_table(rows=len(data_rows) + 1, cols=ncols)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True

    # Header row
    for i, header in enumerate(headers):
        cell = table.rows[0].cells[i]
        set_cell_background(cell, HEX_PRIMARY)
        set_cell_margins(cell, top=90, bottom=90, left=100, right=100)
        set_cell_borders(cell, color=HEX_PRIMARY, sz="4")
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        p = cell.paragraphs[0]
        p.paragraph_format.space_before = Pt(1)
        p.paragraph_format.space_after = Pt(1)
        add_formatted_runs(p, header, base_size=9.0, base_color=RGBColor(255, 255, 255), base_bold=True)

    # Body rows
    for r_idx, row in enumerate(data_rows):
        bg = HEX_LIGHT_BG if r_idx % 2 == 1 else "FFFFFF"
        for c_idx, val in enumerate(row):
            cell = table.rows[r_idx + 1].cells[c_idx]
            set_cell_background(cell, bg)
            set_cell_margins(cell, top=75, bottom=75, left=100, right=100)
            set_cell_borders(cell, color=HEX_BORDER, sz="4")
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)
            p.paragraph_format.line_spacing = 1.12
            add_formatted_runs(p, val, base_size=9.0, base_color=COLOR_DARK)

    spacer = doc.add_paragraph()
    spacer.paragraph_format.space_before = Pt(0)
    spacer.paragraph_format.space_after = Pt(4)


def add_page_number_footer(section) -> None:
    footer = section.footer
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = p.add_run("Nhóm 1 - 65KTPM · Double Dueling DQN (Ngô Tuấn Anh)   |   Trang ")
    run.font.name = "Segoe UI"
    run.font.size = Pt(8.5)
    run.font.color.rgb = COLOR_MUTED

    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), "PAGE")
    p._p.append(fld)


def convert_markdown_to_docx(md_path: Path, docx_path: Path | None = None) -> Path:
    if not md_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {md_path}")

    out_path = docx_path or md_path.with_suffix(".docx")
    lines = md_path.read_text(encoding="utf-8").splitlines()

    doc = docx.Document()
    for section in doc.sections:
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.85)
        section.right_margin = Inches(0.85)
        add_page_number_footer(section)

    normal_style = doc.styles["Normal"]
    normal_style.font.name = "Segoe UI"
    normal_style.font.size = Pt(10.5)
    normal_style.font.color.rgb = COLOR_DARK

    i = 0
    n = len(lines)
    while i < n:
        raw = lines[i]
        stripped = raw.strip()

        # Dòng trống hoặc đường kẻ ngang ---
        if not stripped or stripped == "---":
            i += 1
            continue

        # Khối code ```...```
        if stripped.startswith("```"):
            code_lines = []
            i += 1
            while i < n and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            i += 1  # bỏ dòng đóng ```
            add_code_block(doc, "\n".join(code_lines))
            continue

        # Khối công thức $$...$$
        if stripped.startswith("$$"):
            if stripped.endswith("$$") and len(stripped) > 4:
                add_math_block(doc, stripped[2:-2])
                i += 1
                continue
            math_lines = [stripped[2:]]
            i += 1
            while i < n and "$$" not in lines[i]:
                math_lines.append(lines[i])
                i += 1
            if i < n:
                math_lines.append(lines[i].replace("$$", ""))
                i += 1
            add_math_block(doc, " ".join(math_lines))
            continue

        # Khối trích dẫn / Callout > ...
        if stripped.startswith(">"):
            quote_lines = []
            while i < n and lines[i].strip().startswith(">"):
                q = lines[i].strip()[1:].strip()
                if q:
                    quote_lines.append(q)
                i += 1
            if quote_lines:
                add_callout_box(doc, quote_lines)
            continue

        # Bảng Markdown | ... |
        if stripped.startswith("|") and i + 1 < n and lines[i + 1].strip().startswith("|") and "---" in lines[i + 1]:
            tbl_lines = []
            while i < n and lines[i].strip().startswith("|"):
                tbl_lines.append(lines[i])
                i += 1
            add_markdown_table(doc, tbl_lines)
            continue

        # Tiêu đề #, ##, ###, ####
        heading_match = re.match(r"^(#{1,4})\s+(.*)$", stripped)
        if heading_match:
            level = len(heading_match.group(1))
            title_text = normalize_inline_math(heading_match.group(2))
            if level == 1:
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.space_before = Pt(4)
                p.paragraph_format.space_after = Pt(10)
                r = p.add_run(title_text)
                r.font.name = "Segoe UI"
                r.font.size = Pt(17)
                r.font.bold = True
                r.font.color.rgb = COLOR_PRIMARY
            else:
                h = doc.add_heading(level=min(level - 1, 3))
                h.paragraph_format.space_before = Pt(10 if level == 2 else 7)
                h.paragraph_format.space_after = Pt(4)
                r = h.add_run(title_text)
                r.font.name = "Segoe UI"
                r.font.bold = True
                if level == 2:
                    r.font.size = Pt(13.5)
                    r.font.color.rgb = COLOR_PRIMARY
                elif level == 3:
                    r.font.size = Pt(11.5)
                    r.font.color.rgb = COLOR_SECONDARY
                else:
                    r.font.size = Pt(10.5)
                    r.font.color.rgb = COLOR_DARK
            i += 1
            continue

        # Danh sách gạch đầu dòng * hoặc -
        bullet_match = re.match(r"^(\s*)[*-]\s+(.*)$", raw)
        if bullet_match:
            indent_spaces = len(bullet_match.group(1))
            content = bullet_match.group(2)
            p = doc.add_paragraph(style="List Bullet")
            if indent_spaces >= 2:
                p.paragraph_format.left_indent = Inches(0.5)
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(3)
            p.paragraph_format.line_spacing = 1.15
            add_formatted_runs(p, content, base_size=10.0)
            i += 1
            continue

        # Danh sách đánh số 1. 2. ...
        num_match = re.match(r"^(\s*)(\d+)\.\s+(.*)$", raw)
        if num_match:
            num_prefix = num_match.group(2)
            content = num_match.group(3)
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Inches(0.25)
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(3)
            p.paragraph_format.line_spacing = 1.15
            r_num = p.add_run(f"{num_prefix}. ")
            r_num.font.name = "Segoe UI"
            r_num.font.size = Pt(10.0)
            r_num.font.bold = True
            r_num.font.color.rgb = COLOR_PRIMARY
            add_formatted_runs(p, content, base_size=10.0)
            i += 1
            continue

        # Đoạn văn bản thường
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.line_spacing = 1.18
        add_formatted_runs(p, stripped, base_size=10.5)
        i += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out_path))
    print(f"Đã xuất thành công: {out_path}")
    return out_path


def main() -> None:
    targets = [Path(arg) for arg in sys.argv[1:]] if len(sys.argv) > 1 else DEFAULT_TARGETS
    for md_file in targets:
        convert_markdown_to_docx(md_file)


if __name__ == "__main__":
    main()
