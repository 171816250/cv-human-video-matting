"""Offline, condition-matched summaries of measured run reports; no quality scores."""

from __future__ import annotations

from datetime import datetime, timezone
from html import escape
import json
import math
from pathlib import Path
import re
from statistics import median

from cv_human_video_matting.reporting import _atomic_write_text, _canonical_hash, _protect_paths


def _finite_positive(value) -> bool:
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and value > 0)


def _sha256(value) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _read_report(path: Path) -> dict:
    def invalid_number(value):
        raise ValueError(f"Non-finite JSON number: {value}")

    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            report = json.load(handle, parse_constant=invalid_number)
        if not isinstance(report, dict):
            raise ValueError("Report root must be an object.")
        if "schema_version" not in report:
            raise ValueError("Missing schema_version; this is not a run report.")
        if (not isinstance(report["schema_version"], int) or isinstance(report["schema_version"], bool)
                or report["schema_version"] not in (1, 2, 3)):
            raise ValueError(f"Unsupported run-report schema_version: {report['schema_version']}")
        if report.get("kind", "run_report") != "run_report":
            raise ValueError("The file is not a run report (kind is not run_report).")
        for name in ("input", "options", "environment", "result", "output", "code", "assets"):
            if name in report and not isinstance(report[name], dict):
                raise ValueError(f"{name} must be an object.")
        # JSON's default parser accepts overflowed numeric literals (e.g. 1e999).
        json.dumps(report, allow_nan=False)
        return report
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError(f"Cannot read run report {path}: {exc}") from exc


def _conditions(report: dict) -> tuple[dict | None, list[str], dict | None]:
    reasons = []
    source, output = report.get("input", {}), report.get("output", {})
    options, environment = report.get("options", {}), report.get("environment", {})
    code, result, assets = report.get("code", {}), report.get("result", {}), report.get("assets", {})
    if report["schema_version"] < 3:
        reasons.append("旧版报告缺少完整文件与代码追溯信息，不参与严格分组。")
    if not _sha256(source.get("sha256")):
        reasons.append("缺少有效输入文件SHA256。")
    if not isinstance(options.get("model"), str) or not options["model"]:
        reasons.append("缺少模型名称。")
    device = environment.get("device")
    if not isinstance(device, str) or re.fullmatch(r"cpu|cuda:[0-9]+", device) is None:
        reasons.append("缺少已解析的实际运行设备。")
    if (not environment.get("python") or not environment.get("platform")
            or not isinstance(environment.get("packages"), dict) or not environment["packages"]):
        reasons.append("缺少Python、操作系统或依赖版本信息。")
    if not code.get("version") or not _sha256(code.get("source_sha256")):
        reasons.append("缺少软件版本或源码SHA256。")
    output_digest = output.get("manifest_sha256") if output.get("type") == "png_sequence" else output.get("sha256")
    if output.get("type") not in ("video", "png_sequence") or not _sha256(output_digest):
        reasons.append("缺少有效输出文件或PNG序列清单哈希。")
    media_fields = ("width", "height", "fps", "frame_count", "duration_seconds")
    for name, media in (("输入", source), ("输出", output)):
        if any(not _finite_positive(media.get(field)) for field in media_fields):
            reasons.append(f"{name}尺寸、帧率、帧数或时长不完整。")
    for name in ("checkpoint", "background"):
        if options.get(name) and (not isinstance(assets.get(name), dict)
                                  or not _sha256(assets[name].get("sha256"))):
            reasons.append(f"缺少{name}文件哈希。")

    measurements = None
    required = ("frames", "elapsed_seconds", "total_seconds")
    if any(not _finite_positive(result.get(field)) for field in required):
        reasons.append("处理帧数或测量时间缺失、非正数，无法计算吞吐率。")
    elif not isinstance(result["frames"], int):
        reasons.append("处理帧数必须是整数。")
    else:
        if result["total_seconds"] + 1e-9 < result["elapsed_seconds"]:
            raise ValueError("total_seconds is shorter than elapsed_seconds.")
        measurements = {
            "frames": result["frames"], "elapsed_seconds": result["elapsed_seconds"],
            "total_seconds": result["total_seconds"],
            "processing_fps": result["frames"] / result["elapsed_seconds"],
            "total_fps": result["frames"] / result["total_seconds"],
            "metadata_seconds": result.get("metadata_seconds"),
        }
        for name, measured in (("fps", measurements["processing_fps"]),
                               ("total_fps", measurements["total_fps"])):
            if name in result and (not _finite_positive(result[name])
                                    or not math.isclose(result[name], measured, rel_tol=1e-5, abs_tol=1e-8)):
                raise ValueError(f"{name} does not match frames / measured seconds.")
    if reasons:
        return None, reasons, measurements

    conditions = {
        "input_sha256": source["sha256"],
        "input_media": {field: source[field] for field in media_fields},
        "options": options,
        "environment": environment,
        "code": code,
        "assets": assets,
        "processed_frames": result["frames"],
        "output_media": {"type": output["type"], **{field: output[field] for field in media_fields}},
        "timing_scope": report.get("timing_scope", {}),
    }
    return conditions, [], measurements


def _distribution(values: list[float]) -> dict:
    return {"median": median(values), "min": min(values), "max": max(values)}


def summarize_reports(paths) -> dict:
    """Strictly group comparable runs; retain old/incomplete reports as unverified.

    Paths are local JSON report files. Invalid JSON or inconsistent measurements
    raise ValueError naming the file. Repeated paths count once. This function
    never opens media referenced by report contents and never computes quality.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    sources = list(dict.fromkeys(Path(path).resolve() for path in paths))
    if not sources:
        raise ValueError("At least one run report is required.")
    grouped, unverified = {}, []
    for path in sources:
        report = _read_report(path)
        try:
            conditions, reasons, measurements = _conditions(report)
            row = {"source": str(path), "created_at": report.get("created_at"),
                   "input_path": report.get("input", {}).get("path"),
                   "output_path": report.get("output", {}).get("path"),
                   "measurements": measurements}
            if reasons:
                unverified.append({**row, "reasons": reasons, "options": report.get("options", {})})
                continue
            key = _canonical_hash(conditions)
            if key not in grouped:
                grouped[key] = {"group_id": key, "conditions": conditions, "runs": []}
            grouped[key]["runs"].append(row)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid run report {path}: {exc}") from exc

    groups = []
    for group in grouped.values():
        group["count"] = len(group["runs"])
        group["processing_fps"] = _distribution([
            run["measurements"]["processing_fps"] for run in group["runs"]
        ])
        group["total_fps"] = _distribution([
            run["measurements"]["total_fps"] for run in group["runs"]
        ])
        groups.append(group)
    return {
        "schema_version": 1, "kind": "evaluation_summary",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "report_count": len(sources), "group_count": len(groups),
        "comparable_report_count": sum(group["count"] for group in groups),
        "unverified_report_count": len(unverified),
        "groups": groups, "unverified": unverified, "sources": [str(path) for path in sources],
        "notes": [
            "按已记录条件分组：输入哈希、代码、全部参数、环境、实际设备、输出尺寸帧率及处理帧数须相同。",
            "默认缓存或上游预训练权重未自动核验哈希；源码指纹仅追溯代码。团队仍需固定实际权重与硬件负载。",
            "处理FPS含视频管线I/O等开销；总FPS含模型加载和交付导出，均不包含报告哈希采集与写出。",
            "组内显示重复运行的中位数、最小值和最大值；单次记录不代表长期稳定性能。",
            "旧版及身份信息不完整的报告单独列出，不与可追溯记录合并比较。",
            "哈希标识文件内容，不等同于抠像质量；本汇总不提供或推断准确率、质量分数。",
        ],
        "quality_metrics": None,
    }


def _html_summary(summary: dict) -> str:
    def text(value):
        return escape(str(value), quote=True)

    def pre(value):
        return "<pre>" + text(json.dumps(value, ensure_ascii=False, indent=2)) + "</pre>"

    def metric(values):
        return f"{values['median']:.3f} / {values['min']:.3f} / {values['max']:.3f}"

    rows, sections = [], []
    for index, group in enumerate(summary["groups"], 1):
        conditions = group["conditions"]
        rows.append(
            f"<tr><td>{index}</td><td>{text(conditions['options']['model'])}</td>"
            f"<td>{text(conditions['environment']['device'])}</td><td>{group['count']}</td>"
            f"<td>{metric(group['processing_fps'])}</td><td>{metric(group['total_fps'])}</td></tr>"
        )
        sources = "".join("<li>" + text(run["source"]) + "</li>" for run in group["runs"])
        sections.append(
            f"<details><summary>第{index}组 · 参数与来源 · {text(group['group_id'][:12])}</summary>"
            + pre(conditions) + "<h3>报告来源</h3><ul>" + sources + "</ul>"
            + "<h3>各次测量</h3>" + pre(group["runs"]) + "</details>"
        )
    unverified = "".join(
        "<article><h3>" + text(item["source"]) + "</h3><ul>"
        + "".join("<li>" + text(reason) + "</li>" for reason in item["reasons"])
        + "</ul>" + pre({"参数": item["options"], "测量": item["measurements"]}) + "</article>"
        for item in summary["unverified"]
    )
    notes = "".join("<li>" + text(note) + "</li>" for note in summary["notes"])
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'">
<title>人像视频抠像项目 · 验收运行汇总</title>
<style>body{{max-width:1120px;margin:36px auto;padding:0 24px;font-family:system-ui,'Microsoft YaHei',sans-serif;color:#183044;line-height:1.7}}
h1{{font-size:26px}}h2{{margin-top:32px;font-size:21px}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #d3dce3;padding:10px;text-align:left}}
th{{background:#eef3f7}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px;background:#f4f6f8;padding:14px}}details,article{{border:1px solid #d3dce3;padding:14px;margin:14px 0;overflow-wrap:anywhere}}
summary{{cursor:pointer;font-weight:600}}.muted{{color:#526674}}@media print{{body{{margin:0;max-width:none}}details{{break-inside:avoid}}}}</style></head>
<body><h1>人像视频抠像项目 · 验收运行汇总</h1>
<p class="muted">生成时间（UTC）：{text(summary['created_at'])}</p>
<p>共{summary['report_count']}份报告，{summary['comparable_report_count']}份进入{summary['group_count']}个同条件组；
{summary['unverified_report_count']}份因追溯信息不足单独列示。</p>
<h2>测量口径</h2><ul>{notes}</ul>
<h2>同条件运行统计</h2><p>FPS单元格依次为：中位数 / 最小值 / 最大值。不同组分别列示。</p>
<table><thead><tr><th>组</th><th>模型</th><th>实际设备</th><th>次数</th><th>处理FPS</th><th>总FPS</th></tr></thead>
<tbody>{''.join(rows) if rows else '<tr><td colspan="6">暂无满足严格分组条件的报告。</td></tr>'}</tbody></table>
<h2>参数与报告来源</h2>{''.join(sections)}
<h2>不可严格比较的记录</h2>{unverified if unverified else '<p>无。</p>'}
<p class="muted">本页为自包含静态文件，无脚本、无远程资源。报告内容按原值展示并转义。</p>
</body></html>"""


def write_evaluation_summary(paths, output_path) -> Path:
    """Create an atomic JSON or self-contained Chinese HTML report."""
    target = Path(output_path)
    if target.suffix.lower() not in {".json", ".html"}:
        raise ValueError("Evaluation summary must use a .json or .html extension.")
    summary = summarize_reports(paths)
    _protect_paths(target, summary["sources"])
    if target.suffix.lower() == ".html":
        rendered = _html_summary(summary)
    else:
        rendered = json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    return _atomic_write_text(target, rendered)
