#!/usr/bin/env python3
"""Native-bench progress monitor（按 --tag 监控，与 start_all_runs.py 同一语义）。

批次定位完全复用 `start_all_runs.py` 的规则：先找 `<runs_dir>/<tag>/`（新布局），
找不到再回退 `<runs_dir>/`（历史扁平布局），并从 `.batch-<agent>-<tag>.json` 反推 agent。
因此 `--tag` 直接用启动批次时 `start_all_runs.py --tag` 传的那个值即可。

Usage:
    # 单轮（默认行为）：打印 + 追加日志后退出
    python3 tools/monitor/native_monitor.py --tag <TAG> --single

    # 循环：每 120 秒一轮，Ctrl-C 退出
    python3 tools/monitor/native_monitor.py --tag <TAG> --loop
    python3 tools/monitor/native_monitor.py --tag <TAG> --loop --interval 60

    # 同时看多个批次
    python3 tools/monitor/native_monitor.py --tag <TAG1> --tag <TAG2> --single

    # 只打印不落盘 / 自定义输出位置
    python3 tools/monitor/native_monitor.py --tag <TAG> --single --no-log
    python3 tools/monitor/native_monitor.py --tag <TAG> --single --md /tmp/a.md --jsonl /tmp/a.jsonl

默认输出 `native_progress_<tag>.md` / `.jsonl`（多 tag 时用 `+` 连接），可用 --md/--jsonl 覆盖。

沙盒块：给了 `--sandbox-url` 就每轮实时拉取 `/status`；否则读 `/tmp/sandbox_status.json`
（该文件不存在时沙盒块留空）。两者都不涉及大模型，纯本地读取 + 一个 HTTP GET，零 token。
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import signal
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BRIDGE_DIR = Path(__file__).resolve().parents[2]
# 已知 agent kind（与 start_all_runs.py 的 --agent choices 对齐）；新增 agent 时同步这里。
if str(BRIDGE_DIR) not in sys.path:
    sys.path.insert(0, str(BRIDGE_DIR))
import start_all_runs as S  # noqa: E402  复用批次目录解析 / 状态读取 / agent 识别

MONITOR_DIR = Path(__file__).resolve().parent
AGENT_KINDS = ("claude", "codex", "opencode", "bare")
SANDBOX_STATUS = "/tmp/sandbox_status.json"
STALE_SECONDS = 150


# --- 批次定位 --------------------------------------------------------------- #
def default_runs_dir():
    """默认 runs 根目录：与调度器一致，取 op_catalog.outputs.runs_dir。"""
    try:
        catalog = json.loads(S.DEFAULT_CATALOG.read_text(encoding="utf-8"))
        return Path(catalog["outputs"]["runs_dir"])
    except (OSError, ValueError, KeyError):
        return None


def _flat_layout_has(runs_dir, tag):
    """扁平布局回退前的确认：runs_dir 里必须真的有本 tag 的痕迹。

    不能直接复用 existing_batch_dir —— 它在 `<runs_dir>/<tag>` 不存在时会无条件返回
    runs_dir 本身（为兼容历史扁平布局），于是任何不存在的 tag 都会被"找到"，
    监控会把整个 runs 根目录当成一个批次来扫。
    """
    try:
        if any(runs_dir.glob(f".batch-*-{tag}.json")):
            return True
        suffix = f"-{tag}"
        retry_re = re.compile(rf"-{re.escape(tag)}-r\d+$")
        for entry in os.listdir(runs_dir):
            if entry.endswith(suffix) or retry_re.search(entry):
                return True
    except OSError:
        return False
    return False


def resolve_batch(runs_dir, tag):
    """定位单个批次的目录；找不到返回 None。"""
    candidate = runs_dir / tag
    if candidate.is_dir():
        return candidate
    if _flat_layout_has(runs_dir, tag):
        return runs_dir
    return None


def detect_agent(batch_dir, tag):
    """从 `.batch-<agent>-<tag>.json` 反推 agent；状态文件缺失时回退到 run 目录名。"""
    suffix = f"-{tag}.json"
    candidates = []
    for path in sorted(batch_dir.glob(f".batch-*{suffix}")):
        agent = path.name[len(".batch-"):-len(suffix)]
        if agent:
            candidates.append(agent)
    if len(candidates) > 1:
        # 同 tag 被多个 agent 用过：取在当前目录里留下最多 run 的那个
        def owned(name):
            pat = re.compile(rf"-{re.escape(name)}-{re.escape(tag)}(-r\d+)?$")
            try:
                return sum(1 for entry in os.listdir(batch_dir) if pat.search(entry))
            except OSError:
                return 0
        best = max(candidates, key=owned)
        if owned(best):
            return best
    if candidates:
        return candidates[0]
    marker = f"-{tag}"
    for entry in sorted(os.listdir(batch_dir)):
        if marker not in entry:
            continue
        head = entry.rsplit(marker, 1)[0]
        if "-" in head:
            candidate = head.rsplit("-", 1)[-1]
            if candidate in AGENT_KINDS:
                return candidate
    return None


def run_dirs(batch_dir, tag):
    """批次下所有算子的 run 目录名（排除 .opencode_sessions 等非 run 目录）。"""
    found = []
    for entry in sorted(os.listdir(batch_dir)):
        if not S.looks_like_run_id(entry, tag):
            continue
        if (batch_dir / entry).is_dir():
            found.append(entry)
    return found


# --- 采样 ------------------------------------------------------------------- #
def read_procs():
    out = {}
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as handle:
                raw = handle.read().replace(b"\0", b" ").decode(errors="replace").strip()
        except OSError:
            continue
        if raw:
            out[int(pid)] = raw
    return out


_SUM_RE = re.compile(r"sum of per-op mean kernel time:\s*([\d.]+)\s*us")
# codex CLI 会把「长会话/多次压缩会降低准确度」当 item.type=error 发出来，不是真错误。
_BENIGN_ITEM_PREFIXES = ("Heads up:",)
# BOS 退化标记：adapter 可能写成字面量，也可能被 json.dumps 转义成 \uXXXX，
# 两种都要数，否则指标会静默恒为 0。
_BOS_FORMS = ("<｜begin▁of▁sentence｜>",                       # 原样
              "<\\uff5cbegin\\u2581of\\u2581sentence\\uff5c>")  # json.dumps(ensure_ascii=True)


def _mean_of(stdout_json):
    """取一次评测的总 kernel 耗时（us）。

    优先用 profiling_report 里的 'sum of per-op mean kernel time' 汇总行 —— 它与
    result.json / 分析脚本的口径一致，且不依赖内核名（早期实现只认 `| aclnn` 开头的行，
    像 HcPreSinkhorn 这类内核名会被漏掉，导致 base 显示为 '-'）。
    没有汇总行时回退为把表格里各算子行的 mean 列相加。
    """
    try:
        with open(stdout_json, encoding="utf-8", errors="replace") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return "unparsed", None
    status = payload.get("status")
    report = payload.get("profiling_report") or ""
    match = _SUM_RE.search(report)
    if match:
        return status, float(match.group(1))
    total = 0.0
    found = False
    for line in report.splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 5 or not parts[1] or parts[1] == "op" or set(parts[1]) <= {"-"}:
            continue
        try:
            total += float(parts[3])
        except ValueError:
            continue
        found = True
    return status, (total if found else None)


def scan_agent_stdout(path):
    """扫 agent stdout：BOS 退化次数 + 真实错误条目（正则只做粗筛，再解析确认）。

    返回 (bos_hits, errors, raw_type_error_hits)：raw 是未过滤的 `"type":"error"` 出现次数，
    保留下来便于对比，errors 是剔除 CLI 提示类噪声后的真实错误。
    """
    bos = 0
    errors = []
    raw = 0
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                for form in _BOS_FORMS:
                    if form in line:
                        bos += line.count(form)
                if '"type":"error"' not in line and '"type": "error"' not in line:
                    continue
                raw += 1
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                item = row.get("item") or {}
                if item.get("type") != "error":
                    continue
                message = item.get("message") or ""
                if message.startswith(_BENIGN_ITEM_PREFIXES):
                    continue
                errors.append({"item_id": item.get("id"), "message": message[:300]})
    except OSError:
        pass
    return bos, errors, raw


def bare_eval_rows(op_dir):
    """bare harness 的评测行：bare_client 把每次评测追加进 evaluations.jsonl。

    字段映射到与 native 相同的形状（duration_us 就是同一个口径的
    "sum of per-op mean kernel time"，因此 base/cand/best/gain 语义一致）。
    """
    # native 的 run 也写 evaluations.jsonl（但字段是 duration_ms + solution_id），
    # 两者共用文件名。native 布局以 evaluations/E####/ 为权威，这里必须先让路，
    # 否则 native run 会被按 bare 的 schema 读成 mean_us=None（base/best/gain 全空）。
    if any((op_dir / "evaluations").glob("E*")):
        return None
    path = op_dir / "evaluations.jsonl"
    if not path.is_file():
        return None
    rows = []
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                rows.append({"id": row.get("evaluation_id"), "kind": row.get("evaluation_kind"),
                             "status": row.get("status"), "mean_us": row.get("duration_us"),
                             "version": row.get("version")})
    except OSError:
        return None
    return rows or None


def eval_rows(op_dir):
    bare_rows = bare_eval_rows(op_dir)
    if bare_rows is not None:
        return bare_rows
    rows = []
    for edir in sorted((op_dir / "evaluations").glob("E*")):
        kind = None
        req = edir / "evaluator_request.json"
        if req.exists():
            try:
                kind = json.loads(req.read_text(encoding="utf-8", errors="replace")).get("evaluation_kind")
            except (OSError, ValueError):
                kind = "?"
        status, mean = _mean_of(str(edir / "stdout.json"))
        rows.append({"id": edir.name, "kind": kind, "status": status, "mean_us": mean})
    return rows


def op_name_of(run_id, tag, agent):
    """从 run_id = <op>-<agent>-<tag>[-rN] 反解出算子名。

    同一条 tag 可能被多个 agent 用过（批次键是 (tag, agent)），目录里会同时有
    `<op>-codex-<tag>` 和 `<op>-claude-<tag>`。只剥"检测到的那个 agent"会剥不干净，
    导致同一算子出现两行、done=x/N 的分母虚高。这里剥任意已知 agent。
    """
    base = re.sub(r"-r\d+$", "", run_id)
    base = base[: -len(tag) - 1] if base.endswith(f"-{tag}") else base
    for kind in ([agent] if agent else []) + list(AGENT_KINDS):
        if kind and base.endswith(f"-{kind}"):
            return base[: -len(kind) - 1]
    return base


def attempt_of(run_id, tag):
    """run_id 的 -rN 后缀，形如 ' (r2)'；非重试 run 返回空串。

    run 级基础设施故障重排会留下同算子的多个目录，不标注的话表格里会出现
    多行同名算子，看不出哪行是哪次尝试。
    """
    match = re.search(rf"-{re.escape(tag)}-r(\d+)$", run_id)
    return f" (r{match.group(1)})" if match else ""


def op_snapshot(op_dir, tag, agent, now):
    op_dir = Path(op_dir)
    counts = collections.Counter()
    last_type = None
    last_elapsed = None
    events_path = op_dir / "events.jsonl"
    if events_path.exists():
        with open(events_path, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                counts[row.get("event_type")] += 1
                last_type = row.get("event_type")
                last_elapsed = row.get("elapsed_sec")
    try:
        age = now - events_path.stat().st_mtime
    except OSError:
        try:
            age = now - op_dir.stat().st_mtime
        except OSError:
            age = float("inf")
    stdout_path = op_dir / "agent.stdout.jsonl"
    if not stdout_path.is_file():
        # Some harnesses write plain-text stdout; scan it best-effort.
        stdout_path = op_dir / "agent.stdout.txt"
    bos, errors, raw_errors = scan_agent_stdout(stdout_path)
    evals = eval_rows(op_dir)
    baseline = next((e["mean_us"] for e in evals if e["kind"] == "baseline" and e["mean_us"]), None)
    cands = [e for e in evals if e["kind"] == "candidate"]
    ok = [e["mean_us"] for e in cands if e["status"] == "passed" and e["mean_us"]]
    return {
        "op": op_name_of(op_dir.name, tag, agent),
        "attempt": attempt_of(op_dir.name, tag),
        "dir": str(op_dir),
        "events": sum(counts.values()),
        "native_events": counts.get("native_event", 0),
        "last_event": last_type,
        "last_elapsed_sec": last_elapsed,
        "age_sec": round(age, 1) if age != float("inf") else None,
        "done": (op_dir / "result.json").exists(),
        "baseline_us": baseline,
        "candidates": len(ok),
        "candidates_total": len(cands),
        "best_candidate_us": min(ok) if ok else None,
        "gain_pct": round((baseline - min(ok)) / baseline * 100, 2) if (ok and baseline) else None,
        "failures": collections.Counter(e["status"] for e in cands if e["status"] != "passed"),
        "bos_hits": bos,
        "type_error_hits": len(errors),
        "type_error_raw_hits": raw_errors,
        "errors": errors[:5],
        "evals": evals,
    }


def fetch_sandbox(url):
    """实时拉取沙盒 /status；失败返回 None（不中断监控）。"""
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/status", timeout=10) as resp:
            return json.load(resp)
    except (urllib.error.URLError, OSError, ValueError):
        return None


def sandbox_summary(raw):
    if not raw:
        return None
    servers = raw.get("servers", [])
    return {
        "healthy": sum(1 for s in servers if s.get("is_healthy")),
        "servers": len(servers),
        "running": sum(s.get("active_tasks", 0) for s in servers),
        "queued": raw.get("total_tasks_queued"),
        "completed_total": sum(s.get("total_tasks_completed", 0) for s in servers),
        "completed_per_card": [s.get("total_tasks_completed") for s in servers],
    }


def load_sandbox_raw(args):
    if getattr(args, "sandbox_url", None):
        return fetch_sandbox(args.sandbox_url)
    if os.path.exists(SANDBOX_STATUS):
        try:
            with open(SANDBOX_STATUS, encoding="utf-8", errors="replace") as handle:
                return json.load(handle)
        except (OSError, ValueError):
            return None
    return None


def collect(runs_dir, tags, now, sandbox_raw):
    procs = read_procs()
    batches = []
    for tag in tags:
        batch_dir = resolve_batch(runs_dir, tag)
        if batch_dir is None:
            batches.append({"tag": tag, "missing": True,
                            "tried": [str(runs_dir / tag), str(runs_dir)]})
            continue
        agent = detect_agent(batch_dir, tag)
        dirs = run_dirs(batch_dir, tag)
        agent_pids = [p for p, c in procs.items() if S.native_agent_kind(c)]
        if agent:
            agent_pids = [p for p in agent_pids if S.native_agent_kind(procs[p]) == agent]
        manager_pids = [p for p, c in procs.items()
                        if "op_manager.py run" in c and f"--runs-dir {batch_dir}" in c]
        sched_pids = [p for p, c in procs.items()
                      if "start_all_runs.py" in c and f"--tag {tag}" in c]
        ops = [op_snapshot(batch_dir / d, tag, agent, now) for d in dirs]
        # 已登记但还没建目录的算子（排队中）也列出来，否则进度会看着像"少了几个算子"
        state = S.load_batch_state(batch_dir, agent, tag) if agent else None
        planned = (state or {}).get("planned") or \
            [row.get("run_id") for row in (state or {}).get("runs", [])]
        have = set(dirs)
        for run_id in planned:
            if not run_id or run_id in have or (batch_dir / run_id).is_dir():
                continue
            have.add(run_id)
            ops.append({
                "op": op_name_of(run_id, tag, agent), "attempt": attempt_of(run_id, tag),
                "dir": None, "queued": True,
                "events": 0, "native_events": 0, "last_event": None, "last_elapsed_sec": None,
                "age_sec": None, "done": False, "baseline_us": None, "candidates": 0,
                "candidates_total": 0,
                "best_candidate_us": None, "gain_pct": None,
                "failures": collections.Counter(), "bos_hits": 0,
                "type_error_hits": 0, "type_error_raw_hits": 0, "errors": [], "evals": [],
            })
        # 有进展的（run/DONE/stale）排前面，queue 的排最后：否则新批次一打开满屏都是 "-"
        ops.sort(key=lambda o: (bool(o.get("queued")), o["op"], o.get("attempt") or ""))
        batches.append({
            "tag": tag,
            "agent": agent,
            "root": str(batch_dir),
            "scheduler_pids": sched_pids,
            "manager_count": len(manager_pids),
            "agent_proc_count": len(agent_pids),
            "ops_total": len(ops),
            "ops_done": sum(1 for o in ops if o["done"]),
            "ops_queued": sum(1 for o in ops if o.get("queued")),
            "ops": ops,
            "bos_hits": sum(o["bos_hits"] for o in ops),
            "type_error_hits": sum(o["type_error_hits"] for o in ops),
        })
    return {
        "ts": now,
        "ts_text": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
        "tags": list(tags),
        "sandbox": sandbox_summary(sandbox_raw),
        "batches": batches,
    }


# --- 渲染 ------------------------------------------------------------------- #
def fmt_us(value):
    return "-" if not value else f"{value:.3f}"


def _flag_of(op):
    if op.get("queued"):
        return "queue"
    if op["done"]:
        return "DONE"
    age = op["age_sec"]
    return "stale" if (age is None or age > STALE_SECONDS) else "run"


def _op_line(op):
    flag = _flag_of(op)
    age = op["age_sec"]
    fails = ",".join(f"{k}x{v}" for k, v in op["failures"].items()) or "-"
    gain = "-" if op["gain_pct"] is None else f"{op['gain_pct']:+.2f}%"
    label = op["op"] + (op.get("attempt") or "")
    return (f"   {label:26} {flag:5} events={op['events']:5} base={fmt_us(op['baseline_us']):>8} "
            f"cand={op['candidates']}/{op.get('candidates_total', op['candidates'])}"
            f"{'':<3} best={fmt_us(op['best_candidate_us']):>8} "
            f"gain={gain:>7} age={(age if age is not None else -1):5.0f}s fails[{fails}]")


def render_text(snap):
    lines = [f"== snapshot {snap['ts_text']}"]
    sb = snap.get("sandbox")
    if sb:
        lines.append(
            f"   sandbox: healthy={sb['healthy']}/{sb['servers']} running={sb['running']} "
            f"queued={sb['queued']} completed_total={sb['completed_total']}"
        )
    for b in snap["batches"]:
        if b.get("missing"):
            lines.append(f"\n-- [{b['tag']}] 批次目录不存在（已尝试 {b['tried']}）")
            continue
        lines.append(
            f"\n-- [{b['tag']}] agent={b['agent']} {b['root']}\n"
            f"   scheduler={b['scheduler_pids']} op_manager={b['manager_count']} "
            f"agent_proc={b['agent_proc_count']} done={b['ops_done']}/{b['ops_total']} "
            f"BOS={b['bos_hits']} type_error={b['type_error_hits']}"
        )
        lines.extend(_op_line(o) for o in b["ops"])
    return "\n".join(lines)


def render_markdown(snap, round_no, note=None):
    out = [f"\n## 第 {round_no} 轮 — {snap['ts_text']}\n"]
    if note:
        out.append(f"> {note}\n")
    sb = snap.get("sandbox")
    if sb:
        out.append(
            f"- 沙盒：healthy {sb['healthy']}/{sb['servers']}，running {sb['running']}，"
            f"queued {sb['queued']}，累计完成 {sb['completed_total']}（分卡 {sb['completed_per_card']}）"
        )
    for b in snap["batches"]:
        if b.get("missing"):
            out.append(f"\n- [{b['tag']}] 批次目录不存在（已尝试 {b['tried']}）")
            continue
        out.append(
            f"\n### {b['tag']}（{b['ops_done']}/{b['ops_total']} 完成，agent={b['agent']}，"
            f"op_manager={b['manager_count']}，agent 进程={b['agent_proc_count']}，"
            f"scheduler PID={b['scheduler_pids']}）\n"
        )
        out.append("| 算子 | 状态 | 事件数 | 基线(us) | 候选数 | 最佳候选(us) | 较基线 | 失败明细 | 无事件时长 |")
        out.append("|---|---|---|---|---|---|---|---|---|")
        for o in b["ops"]:
            flag = _flag_of(o)
            age = o["age_sec"]
            gain = "-" if o["gain_pct"] is None else f"{o['gain_pct']:+.2f}%"
            fails = ", ".join(f"{k}×{v}" for k, v in o["failures"].items()) or "-"
            age_text = "?" if age is None else f"{age:.0f}s"
            out.append(
                f"| {o['op'] + (o.get('attempt') or '')} | {flag} | {o['events']} | "
                f"{fmt_us(o['baseline_us'])} | {o['candidates']} | "
                f"{fmt_us(o['best_candidate_us'])} | {gain} | {fails} | {age_text} |"
            )
        out.append(f"\n- 异常扫描：BOS 退化 {b['bos_hits']} 次，`type=error` {b['type_error_hits']} 次")
        out.append(f"- 目录：`{b['root']}`")
    return "\n".join(out) + "\n"


def next_round_no(path):
    if not path.exists():
        return 1
    with open(path, encoding="utf-8", errors="replace") as handle:
        return sum(1 for line in handle if line.startswith("## 第 ")) + 1


def default_log_paths(tags):
    stem = tags[0] if len(tags) == 1 else "+".join(tags)
    return MONITOR_DIR / f"native_progress_{stem}.md", MONITOR_DIR / f"native_progress_{stem}.jsonl"


# --- 主流程 ----------------------------------------------------------------- #
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="native 批次进度监控（按 --tag，与 start_all_runs.py 同一语义）",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", action="append", default=None, metavar="TAG",
                        help="批次 tag，可重复传以同时监控多批（与 start_all_runs.py --tag 一致）")
    parser.add_argument("--runs-dir", dest="runs_dir",
                        help="runs 根目录（默认取 op_catalog.outputs.runs_dir）")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--single", action="store_true",
                      help="只采集一轮后退出（未指定 --loop 时的默认行为）")
    mode.add_argument("--loop", action="store_true",
                      help="持续循环采集，Ctrl-C 退出")
    parser.add_argument("--interval", type=float, default=120.0, metavar="SECONDS",
                        help="--loop 下每轮之间的间隔秒数（默认 120）")
    parser.add_argument("--sandbox-url", dest="sandbox_url", metavar="URL",
                        help="每轮实时拉取该沙盒的 /status；缺省则读 /tmp/sandbox_status.json")
    parser.add_argument("--no-log", action="store_true", help="只打印，不追加 md/jsonl")
    parser.add_argument("--md", help="markdown 输出路径（默认 native_progress_<tag>.md）")
    parser.add_argument("--jsonl", help="jsonl 输出路径（默认 native_progress_<tag>.jsonl）")
    args = parser.parse_args(argv)
    if not args.tag:
        parser.error("至少需要一个 --tag（与启动批次时 start_all_runs.py --tag 的值相同）")
    args.tag = list(dict.fromkeys(args.tag))
    if args.interval <= 0:
        parser.error("--interval 必须 > 0")
    return args


def do_round(args, runs_dir, md_path, jsonl_path, round_no, note=None):
    snap = collect(runs_dir, args.tag, time.time(), load_sandbox_raw(args))
    print(render_text(snap))
    if args.no_log:
        return round_no
    if round_no == 1:
        md_path.parent.mkdir(parents=True, exist_ok=True)
    with open(md_path, "a", encoding="utf-8") as handle:
        handle.write(render_markdown(snap, round_no, note))
    with open(jsonl_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(snap, ensure_ascii=False) + "\n")
    print(f"\n[logged] round {round_no} -> {md_path} / {jsonl_path}")
    return round_no + 1


def main(argv=None):
    args = parse_args(argv)
    runs_dir = Path(args.runs_dir) if args.runs_dir else default_runs_dir()
    if runs_dir is None:
        raise SystemExit("无法确定 runs 根目录：请用 --runs-dir 指定，"
                         "或确保 configs/op_catalog.json 里有 outputs.runs_dir")
    md_path = Path(args.md) if args.md else default_log_paths(args.tag)[0]
    jsonl_path = Path(args.jsonl) if args.jsonl else default_log_paths(args.tag)[1]

    missing = [tag for tag in args.tag if resolve_batch(runs_dir, tag) is None]
    for tag in missing:
        print(f"warning: 找不到 tag={tag} 的批次目录（已尝试 {runs_dir / tag} 与 {runs_dir}）",
              file=sys.stderr)

    if not args.loop:
        do_round(args, runs_dir, md_path, jsonl_path, 1)
        return 0

    def _stop(_signum, _frame):
        raise KeyboardInterrupt

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _stop)
    print(f"loop 模式：每 {args.interval:g}s 一轮，Ctrl-C 退出；tag={', '.join(args.tag)}")
    round_no = 1
    try:
        while True:
            round_no = do_round(args, runs_dir, md_path, jsonl_path, round_no)
            if args.interval > 0:
                time.sleep(args.interval)
    except KeyboardInterrupt:
        print(f"\n已停止（共 {round_no - 1} 轮）")
        return 0


if __name__ == "__main__":
    sys.exit(main())
