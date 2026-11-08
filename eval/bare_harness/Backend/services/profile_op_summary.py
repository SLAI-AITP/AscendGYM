#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate MindStudio op_summary profiling data into a readable report.

Usage (command line):
    python profile_op_summary.py [folder]

Usage (as a library):
    from profile_op_summary import profile_op_summary
    result = profile_op_summary("path/to/folder")
    print(result.report)            # the Markdown text
    print(result.report_path)       # where it was written

The script searches the given folder (default: current working directory)
recursively for an ``op_summary*.csv`` file produced by MindStudio. The file
contains the same set of operator invocations repeated over several measurement
loops (8 by default). The first few loops are warm-up and are discarded; the
remaining loops are averaged per operator shape to suppress run-to-run noise.

A Markdown report (``profiling_report.md``) is written next to the CSV.

Column meanings follow ``README.txt`` shipped with the profiler output:
  * Task Duration(us): end-to-end task time (scheduling + execution).
  * aicore_time(us):    average AI Core execution time.
  * aic_mac_ratio:      fraction of AI Core time spent in the MAC (cube) unit.
  * aic_mte2_ratio:     fraction spent moving data in via MTE2 (GM -> L1).
  * cube_utilization(%): cube utilisation of the AI Core.
"""

from __future__ import annotations

import csv
import glob
import os
import statistics
import sys
from dataclasses import dataclass, field
from typing import Optional

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

DEFAULT_TOTAL_LOOPS = 8    # total rounds, including warmup
DEFAULT_WARMUP_LOOPS = 3  # leading rounds excluded from the mean

# Candidate metric columns, in preferred report order.
# (csv_header, friendly_label, decimals, description)
# `description` is the Chinese meaning of the metric (per the CANN op_summary
# field reference) and is rendered as a legend at the end of the report so each
# measured dimension is self-explanatory.
# At runtime only the columns that exist in the file AND carry at least one
# non-zero value are kept, so the report adapts to the operator kind
# (AI Core / AI Vector / AI CPU / communication ...) without code changes.
CANDIDATE_METRIC_COLUMNS = [
    # Always-present timing.
    ("Task Duration(us)", "Task Duration (us)", 3,
     "任务执行总耗时（调度时间 + 首个核从启动到最晚结束的时间），单位 μs。"),
    ("Block Dim", "Block Dim", 0,
     "任务运行的切分数量，对应任务运行时占用的核数。"),
    ("Mix Block Dim", "Mix Block Dim", 0,
     "Mix（Cube+Vector 混合）场景下任务运行的切分数量，对应占用核数。"),
    # AI Core (Cube) metrics.
    ("aicore_time(us)", "AI Core Time (us)", 3,
     "AI Core 上的平均执行耗时，基于 total_cycles 与 block dim 计算，单位 μs。"),
    ("aic_mac_time(us)", "MAC Time (us)", 3,
     "AI Core 上 MAC（Cube 矩阵乘）单元的执行耗时，单位 μs。"),
    ("aic_mac_ratio", "MAC Ratio", 3,
     "MAC（Cube 矩阵乘）单元耗时占 AI Core 总耗时的比例，反映 Cube 计算密集度。"),
    ("aic_mte1_ratio", "MTE1 Ratio", 3,
     "MTE1（L1→L0 搬运）耗时占比，反映片上数据搬运压力。"),
    ("aic_mte2_time(us)", "MTE2 Time (us)", 3,
     "MTE2（GM/外部存储→L1 搬入）耗时，单位 μs，反映从主存搬入数据的开销。"),
    ("aic_mte2_ratio", "MTE2 Ratio", 3,
     "MTE2（GM→L1 搬入）耗时占比，比例高通常意味着算子受访存带宽限制。"),
    ("aic_scalar_ratio", "Scalar Ratio", 3,
     "Scalar（标量）单元耗时占比，反映标量指令开销。"),
    ("aic_fixpipe_ratio", "Fixpipe Ratio", 3,
     "Fixpipe（定点流水/结果搬出）耗时占比。"),
    ("cube_utilization(%)", "Cube Util (%)", 2,
     "Cube 单元利用率（%），越高表示矩阵乘算力被利用得越充分。"),
    # AI Vector metrics.
    ("aiv_time(us)", "AI Vector Time (us)", 3,
     "AI Vector（AIV）上的平均执行耗时，基于 total_cycles 与 mix block dim 计算，单位 μs。"),
    ("aiv_vec_time(us)", "Vec Time (us)", 3,
     "Vector 单元执行耗时，单位 μs。"),
    ("aiv_vec_ratio", "Vec Ratio", 3,
     "Vector 单元耗时占 AIV 总耗时的比例，反映向量计算密集度。"),
    ("aiv_scalar_ratio", "AIV Scalar Ratio", 3,
     "AIV 上 Scalar（标量）单元耗时占比。"),
    ("aiv_mte2_ratio", "AIV MTE2 Ratio", 3,
     "AIV 上 MTE2（外部存储→片上搬入）耗时占比。"),
    ("aiv_mte3_ratio", "AIV MTE3 Ratio", 3,
     "AIV 上 MTE3（片上→外部存储搬出）耗时占比。"),
]

# Columns used to identify "the same operator invocation" across loops.
GROUP_KEY_COLUMNS = ["Op Type", "Input Shapes", "Input Data Types", "Output Shapes"]


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

@dataclass
class GroupStats:
    """Aggregated metrics for one operator/shape group across kept loops."""

    op_type: str
    input_shapes: str
    output_shapes: str
    input_dtypes: str
    # 算子名（CSV 的 `Op Name` 列；列缺失时为空串）。用于与 native 评测器口径对齐：
    # native 的 parse_profiling 按 Op Name（缺失退回 Op Type / Kernel Name）分组，
    # 组内取均值后跨算子求和，而本模块的分组键是 (Op Type + shapes)。
    op_name: str = ""
    samples: int = 0
    # metric label -> list of float values (one per kept loop)
    values: dict = field(default_factory=dict)

    def add(self, label: str, value: float) -> None:
        self.values.setdefault(label, []).append(value)

    def mean(self, label: str) -> float:
        vals = self.values.get(label, [])
        return statistics.fmean(vals) if vals else float("nan")

    def stdev(self, label: str) -> float:
        vals = self.values.get(label, [])
        return statistics.stdev(vals) if len(vals) > 1 else 0.0


def find_op_summary(folder: str) -> str:
    """Locate the op_summary CSV under *folder* (recursive)."""
    patterns = [
        os.path.join(folder, "**", "op_summary*.csv"),
        os.path.join(folder, "**", "op_summary*.xlsx"),
        os.path.join(folder, "**", "op_summary*.xls"),
    ]
    matches: list[str] = []
    for pat in patterns:
        matches.extend(glob.glob(pat, recursive=True))
    if not matches:
        raise FileNotFoundError(
            f"No op_summary CSV/Excel file found under: {folder}"
        )
    if len(matches) > 1:
        # Prefer the most recently modified file and warn.
        matches.sort(key=os.path.getmtime, reverse=True)
        print(f"[warn] multiple op_summary files found; using most recent:\n"
              f"       {matches[0]}", file=sys.stderr)
    return matches[0]


def read_rows(path: str) -> tuple[list[str], list[dict]]:
    """Read a CSV (or Excel) op_summary into header + list-of-dict rows."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        try:
            import openpyxl  # type: ignore
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "Excel input requires openpyxl: pip install openpyxl"
            ) from exc
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        ws = wb.active
        it = ws.iter_rows(values_only=True)
        header = [str(c).strip() if c is not None else "" for c in next(it)]
        rows = []
        for raw in it:
            if raw is None or all(c is None for c in raw):
                continue
            rows.append({h: ("" if v is None else str(v)) for h, v in zip(header, raw)})
        return header, rows

    # CSV path. Strip whitespace from cells (the profiler pads some fields).
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        header = reader.fieldnames or []
        rows = [{k: (v or "").strip() for k, v in row.items()} for row in reader]
    return header, rows


def to_float(text: str) -> float | None:
    """Parse a numeric cell, tolerating padding, commas and N/A."""
    if text is None:
        return None
    s = text.strip().strip('"').replace(",", "")
    if s == "" or s.upper() in ("N/A", "NA", "NULL", "-"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def group_key(row: dict, header: list[str]) -> tuple:
    def col(name: str) -> str:
        for h in header:
            if h.lower() == name.lower():
                return row.get(h, "")
        return ""
    return tuple(col(c) for c in GROUP_KEY_COLUMNS)


def resolve_column(header: list[str], wanted: str) -> str | None:
    """Case-insensitive lookup of a column header."""
    for h in header:
        if h.lower() == wanted.lower():
            return h
    return None


def select_metric_columns(header: list[str], rows: list[dict]) -> list[tuple]:
    """Pick the candidate metric columns that this file actually populates.

    A column is kept only if it exists in the header and at least one data row
    holds a parseable, non-zero-or-present numeric value. This lets the same
    code emit AI Core metrics for cube operators, AI Vector metrics for vector
    operators, etc., without hard-coding the operator kind.

    Returns a list of (resolved_header, label, decimals, description).
    """
    selected: list[tuple] = []
    for csv_name, label, dec, desc in CANDIDATE_METRIC_COLUMNS:
        col = resolve_column(header, csv_name)
        if col is None:
            continue
        # Keep the column only if some row carries a non-zero value. All-zero
        # columns (e.g. AI Vector metrics for a pure cube operator) are dropped
        # so the table shows only metrics relevant to this operator kind.
        if any((to_float(r.get(col, "")) or 0.0) != 0.0 for r in rows):
            selected.append((col, label, dec, desc))
    return selected


def clean_shape(input_shapes: str) -> str:
    """Strip the surrounding quotes/whitespace the profiler adds to shapes."""
    return input_shapes.strip().strip('"')


# ----------------------------------------------------------------------------
# Core aggregation
# ----------------------------------------------------------------------------

def aggregate(
    header: list[str],
    rows: list[dict],
    *,
    total_loops: int = DEFAULT_TOTAL_LOOPS,
    warmup_loops: int = DEFAULT_WARMUP_LOOPS,
) -> tuple[dict, int, int, list[tuple]]:
    """Group rows by operator/shape, skip warm-up loops, average the rest.

    Returns (groups_by_key, detected_loops, kept_loops, metric_columns), where
    metric_columns is the list of (header, label, decimals) actually present.
    """
    # Order of distinct keys = order of first appearance (one operator per loop
    # iteration). Each key reappears once per loop, so loop index = occurrence #.
    occurrences: dict[tuple, int] = {}
    groups: dict[tuple, GroupStats] = {}

    metric_columns = select_metric_columns(header, rows)
    metric_headers = [(col, label) for col, label, _dec, _desc in metric_columns]

    op_type_h = resolve_column(header, "OP Type") or resolve_column(header, "Op Type")
    op_name_h = resolve_column(header, "Op Name")
    in_shapes_h = resolve_column(header, "Input Shapes")
    out_shapes_h = resolve_column(header, "Output Shapes")
    in_dtypes_h = resolve_column(header, "Input Data Types")
    in_shapes_h = resolve_column(header, "Input Shapes")
    out_shapes_h = resolve_column(header, "Output Shapes")
    in_dtypes_h = resolve_column(header, "Input Data Types")

    if type(total_loops) is not int or total_loops < 1:
        raise ValueError(f"total_loops must be a positive integer, got {total_loops!r}")
    if type(warmup_loops) is not int or not 0 <= warmup_loops < total_loops:
        raise ValueError(
            f"warmup_loops must be in [0, total_loops), got "
            f"warmup={warmup_loops!r} total={total_loops!r}"
        )

    # 根据总行数和显式循环次数推导每个循环的行数
    total_rows = len(rows)
    if total_rows % total_loops != 0:
        raise ValueError(
            f"总行数 {total_rows} 不是循环次数 {total_loops} 的整数倍，数据可能不完整"
        )
    rows_per_loop = total_rows // total_loops
    num_cases = rows_per_loop

    # 按行号位置划分循环，循环内按 (Op Name, case_idx) 分组
    # case_idx = 循环内的行偏移，同一位置的算子在跨循环间取平均
    # 跳过前 warmup_loops 个预热循环
    groups: dict[tuple, GroupStats] = {}

    for loop_idx in range(total_loops):
        if loop_idx < warmup_loops:
            continue

        start = loop_idx * rows_per_loop
        end = start + rows_per_loop
        loop_rows = rows[start:end]

        for case_idx, row in enumerate(loop_rows):
            key = (group_key(row, header), case_idx)
            grp = groups.get(key)
            if grp is None:
                grp = GroupStats(
                    op_type=row.get(op_type_h, "") if op_type_h else "",
                    op_name=row.get(op_name_h, "") if op_name_h else "",
                    input_shapes=row.get(in_shapes_h, "") if in_shapes_h else "",
                    output_shapes=row.get(out_shapes_h, "") if out_shapes_h else "",
                    input_dtypes=row.get(in_dtypes_h, "") if in_dtypes_h else "",
                )
                groups[key] = grp
            grp.samples += 1

            for col, label in metric_headers:
                if col is None:
                    continue
                val = to_float(row.get(col, ""))
                if val is not None:
                    grp.add(label, val)

    kept = total_loops - warmup_loops
    return num_cases, total_loops, warmup_loops, kept, groups, metric_columns


# ----------------------------------------------------------------------------
# Report rendering
# ----------------------------------------------------------------------------

def fmt(value: float, decimals: int) -> str:
    if value != value:  # NaN
        return "-"
    if decimals == 0:
        return f"{value:.0f}"
    return f"{value:.{decimals}f}"


def render_report(
    groups: dict,
    csv_path: str,
    num_cases: int,
    profiling_loops: int,
    warmup_loops: int,
    kept_loops: int,
    metric_columns: list[tuple],
    op_name: Optional[str] = None,
) -> str:
    # Keep the original collection order (dict preserves first-appearance order,
    # i.e. the order operators were profiled in the file). No reordering.
    # Op Type column is always rendered — even single-op reports surface it,
    # so downstream readers don't have to cross-reference the title to know
    # what kind of kernel each row describes.
    ordered = list(groups.values())

    lines: list[str] = []

    # ---- Report title -------------------------------------------------------
    # When op_name is provided, surface it in the title so LLM / human readers
    # immediately know which operator the report describes (multiple operators
    # often share an output dir layout under different timestamps).
    if op_name:
        lines.append(f"# 算子性能 Profiling 报告 — {op_name}")
    else:
        lines.append("# 算子性能 Profiling 报告")
    lines.append("")

    # ---- Per-case averaged metrics table ------------------------------------
    lines.append("## 各测试用例性能测试结果")
    lines.append("")
    lines.append(
        f"下表为每个测试用例（test case，按输入 shape 区分）的算子性能测试结果，"
        f"共 {num_cases} 个测试用例，"
        f"采集共 {profiling_loops} 个循环，已跳过前 {warmup_loops} 个预热循环，"
        f"对后 {kept_loops} 个循环取平均以抑制噪声；每行对应一个用例，"
        f"各列含义见文末「列含义说明」。"
    )
    lines.append("")
    lead = ["Op Type", "Input Shapes", "Output Shapes"]
    headers = lead + [lbl for _c, lbl, _d, _desc in metric_columns]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")

    dur_label = "Task Duration (us)"
    for g in ordered:
        in_shape = clean_shape(g.input_shapes)
        out_shape = clean_shape(g.output_shapes)
        cells = [g.op_type, f"`{in_shape}`", f"`{out_shape}`" if out_shape else "-"]
        for _csv_name, label, dec, _desc in metric_columns:
            cells.append(fmt(g.mean(label), dec))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    # ---- Total task duration over cases -------------------------------------
    durations = [
        g.mean(dur_label) for g in ordered if g.mean(dur_label) == g.mean(dur_label)
    ]
    total_dur = sum(durations)
    lines.append(f"- **Total Task Duration:** {fmt(total_dur, 3)} us")
    # 与 native 评测器（sandbox_evaluator.parse_profiling）对齐的口径：
    # 按算子名分组（缺失退回 Op Type）→ 组内按样本数加权平均（同一算子的多次 launch / 多 shape 只计一次）
    # → 跨算子求和。上面那行是旧口径（按 (Op Type, shape, case) 分组后直接求和），保留供对照。
    per_op_total: dict[str, list[float]] = {}
    for g in ordered:
        value = g.mean(dur_label)
        if value != value:
            continue
        label = g.op_name or g.op_type or "unknown"
        weight = max(1, g.samples)
        acc = per_op_total.setdefault(label, [0.0, 0.0])
        acc[0] += value * weight
        acc[1] += weight
    aligned_us = sum(m / n for m, n in per_op_total.values()) if per_op_total else 0.0
    lines.append(f"- **Total Task Duration (native-aligned, sum of per-op mean):** "
                 f"{fmt(aligned_us, 3)} us")
    lines.append("")

    # ---- Column legend: meaning of each measured dimension ------------------
    lines.append("## 列含义说明")
    lines.append("")
    lines.append("| 列名 | 含义 |")
    lines.append("|---|---|")
    lines.append("| Op Type | 算子类型（即 kernel 内核类型；同一算子在多 shape 下被测时该列重复）。 |")
    lines.append("| Input Shapes | 算子输入张量的维度（多个张量以 `;` 分隔，每个张量的各维以 `,` 分隔）。 |")
    lines.append("| Output Shapes | 算子输出张量的维度（格式同 Input Shapes；`-` 表示 profiler 未提供）。 |")
    for _csv_name, label, _dec, desc in metric_columns:
        lines.append(f"| {label} | {desc} |")
    lines.append("")

    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------

@dataclass
class ProfileResult:
    """Outcome of one profiling run, returned by :func:`profile_op_summary`."""

    csv_path: str
    report_path: str
    report: str
    detected_loops: int
    warmup_loops: int
    kept_loops: int
    profiling_protocol: dict
    groups: dict


def profile_op_summary(
    folder: str,
    *,
    output_name: str = "profiling_report.md",
    write: bool = True,
    verbose: bool = True,
    op_name: Optional[str] = None,
    total_runs: int = DEFAULT_TOTAL_LOOPS,
    warmup_runs: int = DEFAULT_WARMUP_LOOPS,
    statistic: str = "mean",
) -> ProfileResult:
    """Run the full profiling pipeline on *folder* and return the result.

    Steps: locate the op_summary file -> read rows -> group by operator/shape ->
    skip warm-up loops -> average the remaining loops -> render a Markdown
    report (optionally written next to the source file).

    Args:
        folder: directory to search recursively for ``op_summary*.csv``.
        output_name: report file name (written beside the source CSV).
        write: if False, build the report string but don't write the file.
        verbose: if True, print progress to stdout.

    Returns:
        A :class:`ProfileResult` with the report text, paths and aggregated
        per-shape groups.

    Raises:
        NotADirectoryError: *folder* is not a directory.
        FileNotFoundError: no op_summary file found.
        ValueError: file has no data rows, or too few loops to keep any.
    """
    def log(msg: str) -> None:
        if verbose:
            print(msg)

    folder = os.path.abspath(folder)
    if not os.path.isdir(folder):
        raise NotADirectoryError(f"not a directory: {folder}")

    csv_path = find_op_summary(folder)
    log(f"[info] reading: {csv_path}")

    header, rows = read_rows(csv_path)
    if not rows:
        raise ValueError("op_summary file has no data rows")

    if statistic != "mean":
        raise ValueError(f"only statistic='mean' is supported, got {statistic!r}")
    num_cases, profiling_loops, warmup_loops, kept_loops, groups, metric_columns = aggregate(
        header,
        rows,
        total_loops=total_runs,
        warmup_loops=warmup_runs,
    )

    report = render_report(
        groups, csv_path, num_cases, profiling_loops, warmup_loops, kept_loops,
        metric_columns, op_name=op_name,
    )

    report_path = os.path.join(os.path.dirname(csv_path), output_name)
    if write:
        with open(report_path, "w", encoding="utf-8") as fh:
            fh.write(report)
        log(f"[done] report written: {report_path}")

    log(f"[info] {num_cases} cases, {profiling_loops} loops, skipped {warmup_loops} warmup, averaged last {kept_loops}")
    log(f"[info] {len(groups)} operator/shape groups")

    return ProfileResult(
        csv_path=csv_path,
        report_path=report_path,
        report=report,
        detected_loops=profiling_loops,
        warmup_loops=warmup_loops,
        kept_loops=kept_loops,
        profiling_protocol={
            "total_runs": profiling_loops,
            "warmup_runs": warmup_loops,
            "statistic": statistic,
        },
        groups=groups,
    )


# ----------------------------------------------------------------------------
# Command-line entry point
# ----------------------------------------------------------------------------

def main(argv: list[str]) -> int:
    folder = argv[1] if len(argv) > 1 else os.getcwd()
    try:
        profile_op_summary(folder)
    except (NotADirectoryError, FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
