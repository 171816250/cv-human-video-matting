"""Fill the Jianghan University 2026 open-source incubation application."""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "江汉大学2026年开源孵化项目-智能人像视频抠像与背景替换开源工具-申报书.docx"

PROJECT_NAME = "智能人像视频抠像与背景替换开源工具"
PROJECT_FULL_NAME = "智能人像视频抠像与背景替换开源工具（cv-human-video-matting）"
RESPONSIBLE = "【待补：项目负责人姓名】"
MEMBERS = "【待补：项目组成员姓名】"
DEPARTMENT = "江汉大学人工智能学院（待确认）"


def find_template() -> Path:
    exact = ROOT / "江汉大学2026年开源孵化项目.docx"
    if exact.exists():
        return exact
    matches = [
        path
        for path in ROOT.glob("*.docx")
        if "江汉大学2026年开源孵化项目" in path.name and "申报书" not in path.name
    ]
    if not matches:
        raise FileNotFoundError("未找到江汉大学2026年开源孵化项目申报书模板。")
    return matches[0]


def apply_run_font(run, size: float | None = None, bold: bool | None = None) -> None:
    run.font.name = "宋体"
    rpr = run._element.get_or_add_rPr()  # noqa: SLF001
    rfonts = rpr.rFonts
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:eastAsia"), "宋体")
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold


def set_paragraph(paragraph, text: str, size: float = 11, bold: bool | None = None) -> None:
    paragraph.text = text
    for run in paragraph.runs:
        apply_run_font(run, size=size, bold=bold)


def set_body_paragraph(paragraph, text: str) -> None:
    set_paragraph(paragraph, text, size=11)
    paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    paragraph.paragraph_format.line_spacing = 1.25


def set_cell(cell, text: str, size: float = 10.5, bold: bool | None = None) -> None:
    cell.text = text
    for paragraph in cell.paragraphs:
        paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        for run in paragraph.runs:
            apply_run_font(run, size=size, bold=bold)


def fill_cover(doc: Document) -> None:
    set_paragraph(doc.paragraphs[7], f"项目名称：{PROJECT_FULL_NAME}", size=14)
    set_paragraph(doc.paragraphs[8], f"项目负责人：{RESPONSIBLE}", size=14)
    set_paragraph(doc.paragraphs[9], f"项目组成员：{MEMBERS}", size=14)
    set_paragraph(doc.paragraphs[10], f"项目负责人所属部门：{DEPARTMENT}", size=14)
    set_paragraph(doc.paragraphs[12], "填报日期：2026年6月", size=14)


def project_overview_text() -> str:
    return (
        "1、研究基础\n"
        "本项目依托现有课程工程原型 cv-human-video-matting，围绕自然场景下人像视频抠像、背景替换与结果可视化开展开源孵化。"
        "原型已采用 Python、OpenCV、PyTorch 和 Gradio 构建，形成了视频 I/O、RVM 推理、alpha matte 生成、前景/背景合成、"
        "图片背景替换、alpha 可视化、side-by-side 对比、边缘羽化、形态学清理、时序平滑和浏览器可播放 MP4 转换等模块。"
        "项目已在 DLF CUDA 环境下完成 GPU smoke 测试，NVIDIA GeForce GTX 1660 Ti 可用，并建立了覆盖 CLI、pipeline、"
        "RVM 包装、后处理、Gradio 界面和视频兼容输出的 pytest 测试基础。\n\n"
        "2、项目拟完成主要功能、技术指标及预期成效\n"
        "项目拟建设为面向教学、科研训练和轻量应用演示的开源人像视频抠像工具，支持本地视频上传、CPU/GPU 设备选择、"
        "RVM 基线推理、绿幕/图片背景/alpha/前景/对比视图输出、浏览器可播放 H.264 MP4 导出、可配置边缘与时序后处理、"
        "模型替换接口和完整中文教程。2026 年度预期形成不少于 2 个正式版本发布，完善 README、安装指南、API 说明、"
        "示例教程、演示视频和贡献指南；开源生态目标为 stars≥50、forks≥10、issues/讨论≥20、contributors≥5、Pull Request≥10。"
    )


def social_value_text() -> str:
    return (
        "项目可服务于在线教学、虚拟直播、短视频制作、智能会议、数字人交互、隐私保护与校园新媒体等场景，"
        "降低自然场景人像视频抠像和背景替换技术的使用门槛。通过开源孵化，可沉淀江汉大学在计算机视觉工程化、"
        "人工智能应用开发和学生开源协作方面的实践案例，带动课程大作业原型向真实开源项目转化，形成可复用的教学资源、"
        "项目模板和校内外开发者协作社区。"
    )


def fill_project_overview_table(doc: Document) -> None:
    table = doc.tables[0]
    set_cell(table.cell(0, 1), PROJECT_FULL_NAME)
    set_cell(table.cell(1, 1), DEPARTMENT)
    set_cell(table.cell(2, 2), RESPONSIBLE)
    set_cell(table.cell(2, 4), "【待补：职务/职称】")
    set_cell(table.cell(3, 2), "【待补：联系电话】")
    set_cell(table.cell(3, 4), "【待补：电子邮箱】")
    set_cell(
        table.cell(4, 1),
        "项目负责人、学生开发者、文档维护者与测试志愿者若干（具体名单待补）。"
        "建议设置模型算法、工程开发、前端演示、文档社区四类角色。",
        size=10,
    )
    set_cell(table.cell(5, 1), "人工智能；软件研发；其他：计算机视觉开源应用")
    set_cell(table.cell(6, 1), "2026年度")
    set_cell(
        table.cell(7, 1),
        "拟托管平台：Gitee/AtomGit，GitHub 可作为镜像仓库；当前本地工程名：cv-human-video-matting。",
        size=10,
    )
    set_cell(
        table.cell(8, 1),
        "拟采用：木兰宽松许可证第2版（MulanPSL-2.0）或 Apache License 2.0；正式开源前根据学校要求确认。",
        size=10,
    )
    set_cell(table.cell(9, 1), project_overview_text(), size=10)
    set_cell(table.cell(10, 1), social_value_text(), size=10)


def fill_leader_table(doc: Document) -> None:
    table = doc.tables[1]
    values = {
        (0, 1): RESPONSIBLE,
        (0, 3): "【待补】",
        (0, 6): "【待补】",
        (1, 1): "【待补】",
        (1, 3): "【待补】",
        (2, 1): DEPARTMENT,
        (3, 1): "【待补：职务/职称】",
        (3, 3): "【待补：学历/学位】",
        (4, 1): "人工智能/计算机科学与技术/软件工程（待确认）",
        (4, 3): "计算机视觉、人工智能工程化、开源软件（待确认）",
        (5, 1): "【待补：联系电话】",
        (5, 3): "【待补：电子邮箱】",
        (6, 1): "【待补：本科/硕士/博士教育经历】",
        (7, 1): "【待补：教学、科研、工程或学生项目指导经历】",
        (8, 1): (
            "近三年以来围绕计算机视觉、人工智能应用、课程项目工程化和开源实践开展相关研发或指导工作。"
            "本项目已形成视频抠像原型、GPU 运行环境、自动化测试用例、Gradio 演示界面和项目文档，"
            "可作为开源孵化基础。具体成果清单待项目负责人补充。"
        ),
    }
    for (row, col), text in values.items():
        set_cell(table.cell(row, col), text, size=10)


def fill_member_table(doc: Document) -> None:
    table = doc.tables[2]
    rows = [
        ["【待补】", DEPARTMENT, "算法/模型开发", "【待补】", "计算机视觉/深度学习", "【待补】"],
        ["【待补】", DEPARTMENT, "工程开发/测试", "【待补】", "软件工程/Python 开发", "【待补】"],
        ["【待补】", DEPARTMENT, "前端演示/文档", "【待补】", "交互设计/技术文档", "【待补】"],
        ["【待补】", DEPARTMENT, "社区运营/开源维护", "【待补】", "开源协作/项目管理", "【待补】"],
    ]
    for row_index, values in enumerate(rows, start=1):
        for col_index, text in enumerate(values):
            set_cell(table.cell(row_index, col_index), text, size=9.5)


def background_text() -> str:
    return (
        "随着在线教学、远程会议、直播制作、短视频创作和智能交互应用快速发展，自然场景下的人像视频背景消除成为计算机视觉应用中的高频需求。"
        "传统绿幕方案对拍摄环境依赖强，商业软件通常闭源、成本较高、二次开发能力有限，难以满足高校教学、科研验证和学生创新实践中"
        "“可学习、可修改、可复现、可部署”的要求。自然场景人像视频抠像还面临动态背景、光照变化、发丝边缘、半透明衣物、快速运动和长时序闪烁等难点，"
        "既需要深度学习模型能力，也需要工程化流水线、交互界面和结果评估体系。\n\n"
        "本项目拟以开源方式孵化“智能人像视频抠像与背景替换开源工具”，围绕人工智能与软件研发方向，建设一个从模型推理、后处理、视频合成到可视化演示的完整项目。"
        "项目以 Robust Video Matting 等实时视频抠像模型作为可运行基线，以模块化工程结构支持模型替换和功能扩展，以 Gradio 工作台降低演示和使用门槛，"
        "以浏览器可播放视频输出、中文文档和自动化测试保障可复现性，并进一步规划相对距离标注等具有课程特色的扩展功能。"
        "通过开源孵化，项目可把课程大作业原型升级为面向教学示范、科研训练和开源协作的真实软件项目。"
    )


def foundation_text() -> str:
    return (
        "项目已形成可运行的本地工程原型，代码目录为 cv-human-video-matting，具备较好的继续孵化基础。当前工程采用 src 包结构组织，"
        "包含 io、matting、render、postprocess、ui、depth、utils 等模块；已实现视频探测与读写、确定性 dummy runner、RVMRunner、模型工厂、"
        "视频级处理流水线、背景颜色/图片合成、alpha 灰度输出、side-by-side 对比输出、边缘羽化、mask open/close、时序平滑、Gradio 演示界面和浏览器兼容 MP4 输出等功能。"
        "项目已建立 requirements、GPU 环境依赖说明、README、weights 说明和 Superpowers 迭代计划文档。\n\n"
        "运行基础方面，项目已在 DLF Conda 环境中验证 torch 1.13.0+cu117、torchvision 0.14.0+cu117、numpy 1.23.5 与 NVIDIA GeForce GTX 1660 Ti 可用，"
        "RVM GPU smoke 测试可处理短视频并输出可播放结果。近期测试规模已扩展至 32 个 pytest 测试，覆盖渲染、pipeline、CLI、Gradio、RVM 包装、后处理和浏览器可播放视频转换等关键环节。"
        "开源基础方面，项目已按未来 Git 仓库组织 .gitignore，排除权重、大视频和生成结果；Obsidian 笔记中已记录项目总览、技术调研、实施路线、架构设计、评价方案和 Git 规范，"
        "可继续整理为开源文档体系。"
    )


def route_text() -> str:
    return (
        "1、总体技术路线\n"
        "项目采用“输入视频 -> 人像视频抠像模型 -> alpha matte -> 后处理 -> 背景合成/可视化 -> 浏览器可播放结果输出”的流水线。"
        "模型层以 Robust Video Matting（RVM）作为实时基线，保留统一 MattingRunner 接口，支持后续接入 MatAnyone2、SAM2 辅助分割或其他高质量抠像模型。"
        "工程层将视频 I/O、模型推理、合成渲染、后处理、距离标注和前端演示解耦，确保每个模块可独立测试和替换。\n\n"
        "2、关键功能建设\n"
        "（1）核心抠像：支持本地视频输入、GPU/CPU 设备选择、RVM MobileNetV3 推理、alpha matte 输出、前景输出、绿幕替换、图片背景替换和原图/结果对比。\n"
        "（2）质量增强：提供 edge feather、mask open、mask close、temporal smooth 等可配置后处理，减轻边缘硬切、小孔洞、局部噪声和时间闪烁问题，并保留开关便于与 baseline 对比实验。\n"
        "（3）演示部署：建设 Gradio 视频工作台，支持视频文件上传、输入预览、背景图上传、推理参数控制、运行状态日志、结果预览与下载；输出统一转为浏览器兼容 H.264 MP4，提高展示稳定性。\n"
        "（4）特色扩展：规划接入 Depth Anything 等单目深度模型，在人像区域生成近/中/远相对距离标签，形成区别于普通背景替换工具的课程创新点。\n"
        "（5）开源工程：完善 README、安装教程、GPU/CPU 运行说明、示例数据生成脚本、常见问题、贡献指南、许可证、版本发布记录和自动化测试，推动项目由课程原型转为可协作维护的开源项目。\n\n"
        "3、实施计划\n"
        "第一阶段（2026年1月至3月）：完成仓库初始化、开源许可证确认、README 重构、示例数据与演示视频整理，稳定 RVM 基线和 Gradio 工作台。\n"
        "第二阶段（2026年4月至6月）：完善后处理、浏览器兼容输出、自动化测试、中文教程和插件化模型接口，发布 v0.1/v0.2 版本。\n"
        "第三阶段（2026年7月至9月）：接入相对距离标注与更多测试场景，开展性能评测、质量对比和文档增强，吸引校内学生参与 issue 与 PR。\n"
        "第四阶段（2026年10月至12月）：完成开源社区运营、版本发布、优秀开源软件项目申报材料整理和成果验收。\n\n"
        "4、创新点与可行性\n"
        "项目创新点在于将实时人像视频抠像、工程化后处理、浏览器演示和相对距离标注结合起来，面向高校教学和轻量应用场景形成可运行、可学习、可扩展的开源工具。"
        "可行性方面，项目已有可运行代码、GPU 环境、测试体系和文档基础，核心模型可使用公开开源模型，新增工作主要集中在工程完善、文档建设、社区运营和特色功能扩展，风险可控。"
    )


def budget_text() -> str:
    return (
        "经费预算建议围绕开源孵化的实际需要配置，具体额度按学校通知填写。建议预算方向包括："
        "1）GPU 计算与实验环境维护，用于模型推理测试、视频样例生成和性能评测；"
        "2）数据与案例制作，用于采集或制作合规的教学演示视频、背景素材和测试场景；"
        "3）开源文档与社区建设，用于教程编写、演示视频制作、版本发布和宣传材料；"
        "4）学生开发与测试支持，用于鼓励学生参与 issue、PR、测试与文档贡献。\n\n"
        "预期成果包括：形成一个托管在 Gitee/AtomGit 等平台的开源仓库；发布不少于 2 个版本；完成源代码、README、安装教程、用户手册、开发者文档、贡献指南、许可证和示例教程；"
        "形成不少于 3 个演示视频或案例；自动化测试不少于 40 项；完成 RVM 基线、后处理模块、浏览器可播放输出、Gradio 工作台和相对距离标注原型；"
        "达到 stars≥50、forks≥10、issues/讨论≥20、contributors≥5、Pull Request≥10 的社区孵化目标；按要求完成 2026 年度武汉市优秀开源软件项目申报支撑材料。"
    )


def fill_body(doc: Document) -> None:
    sections = {
        37: background_text(),
        41: foundation_text(),
        44: route_text(),
        47: budget_text(),
    }
    for paragraph_index, text in sections.items():
        set_body_paragraph(doc.paragraphs[paragraph_index], text)


def main() -> None:
    template = find_template()
    doc = Document(template)
    fill_cover(doc)
    fill_project_overview_table(doc)
    fill_leader_table(doc)
    fill_member_table(doc)
    fill_body(doc)
    doc.save(OUTPUT)
    print(str(OUTPUT))


if __name__ == "__main__":
    main()
