#!/usr/bin/env python3
"""批量后台启动 Ascend 算子优化评测（所有算子一次拉起，并发受控）。

为什么需要它：op_manager.py 一次只启动一个 run；17 个算子逐个手敲既费时，
又容易在 SSH 断开时被 SIGHUP 杀掉。本脚本负责：
  1. 按 op_catalog.resolved.json 自动枚举全部已解析算子；
  2. 以并发上限（默认 4，匹配沙盒 4 个 NPU 节点）排队启动；
  3. 每个 run 用独立会话（start_new_session）启动，SSH 断线不影响；
  4. 已完成（有 result.json）的 run 自动跳过，同 tag 重跑即可续跑剩余算子；
  5. 提供 --status / --stop / --dry-run。

常用命令（在 ${ASCEND_GYM_WORKSPACE}/bridge 下执行）：

  # 1) 一次性后台启动全部 17 个算子（默认 claude；预算取 op_catalog 默认档）
  TAG=$(date +%Y%m%d-%H%M%S)
  setsid nohup python3 start_all_runs.py --agent claude --tag "$TAG" \
      > ../runs/batch-$TAG.console.log 2>&1 < /dev/null &

  # 2) 只跑部分算子 / 指定预算（standard 档示例）
  setsid nohup python3 start_all_runs.py --agent claude --tag "$TAG" \
      --ops add_rms_norm,swi_glu,rms_norm \
      --candidate-evaluations-max 8 --max-model-turns 200 --final-reprofile-count 3 &

  # 2b) 调整墙钟（每个 run 的搜索时长上限，默认取 op_catalog 的 10800s）
  setsid nohup python3 start_all_runs.py --agent claude --tag "$TAG" \
      --wall-clock-seconds 21600 ...      # 6 小时
  #    --wall-clock-seconds 0 ...         # 0 = 关闭墙钟限制（不推荐无人值守时使用）

  # 2c) 调整所选 agent 的单轮模型调用超时（默认 claude/codex 1200s，opencode 1800s）
  setsid nohup python3 start_all_runs.py --agent claude --tag "$TAG" \
      --agent-timeout-seconds 2400 ...    # 单轮最多等 40 分钟

  # 3) 查看进度 / 停止未完成任务（SSH 重连后执行）
  python3 start_all_runs.py --status --agent claude --tag "$TAG"
  python3 start_all_runs.py --stop   --agent claude --tag "$TAG"

  # 4) 只预览要执行什么（不启动、不写产物）
  python3 start_all_runs.py --dry-run --ops all --agent claude

  # 4b) 切换到新 native harness（原生 coding agent，不再走 wrapper/单步工具协议）
  #     停止请用 --stop（会先写 STOP 文件让 native run 自行收尾）。
  python3 start_all_runs.py --agent codex --harness native --tag "$TAG" ...
  python3 start_all_runs.py --status --agent codex --harness native --tag "$TAG"
  python3 start_all_runs.py --stop   --agent codex --harness native --tag "$TAG"

  # 5) 断线/失败后补齐：同一 tag 再启动一次，只跑未完成的；
  #    半途中断（输出目录存在但没有 result.json）加 --retry-incomplete 自动换 -r2 重跑
  setsid nohup python3 start_all_runs.py --agent claude --tag "$TAG" --retry-incomplete ...

说明：本脚本是"调度器"，自身需要保持存活；请用 setsid + nohup 启动。
即使调度器被杀，已启动的 run 仍在各自会话中继续运行，可用 --status 查看。
"""
import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import op_manager

BRIDGE_DIR = Path(__file__).resolve().parent
BRIDGE_CONFIG_DIR = BRIDGE_DIR / "configs"
DEFAULT_CATALOG = BRIDGE_CONFIG_DIR / "op_catalog.json"
POLL_SECONDS = 3.0
PROGRESS_HEARTBEAT_SECONDS = 300.0
# native harness 支持 STOP 文件（受控停止）：写完先等它自己收尾，再退回发信号。
NATIVE_STOP_GRACE_SECONDS = 45.0
# 退回发信号后的两段收尾：先 SIGTERM，仍不走的再 SIGKILL。
STOP_TERM_GRACE_SECONDS = 8.0
STOP_KILL_GRACE_SECONDS = 5.0


def fmt_duration(seconds):
    seconds = int(max(0, round(seconds)))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}"


def progress_bar(done, total, width=24):
    total = max(1, total)
    filled = min(width, max(0, int(round(width * done / total))))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def effective_config(catalog, args):
    """op_catalog 默认值 + 命令行覆盖，返回本批次实际生效的配置。"""
    budget = dict(catalog.get("budget_defaults", {}))
    for field in ("candidate_evaluations_max", "max_model_turns", "final_reprofile_count",
                  "wall_clock_seconds_max", "candidate_solutions_target"):
        value = getattr(args, field, None)
        if value is not None:
            budget[field] = value
    requested_model = getattr(args, "model", None) or getattr(args, "bare_model", None)
    model_spec = op_manager.resolve_model(catalog, name=requested_model,
                                          url=getattr(args, "model_url", None),
                                          raw_name=getattr(args, "model_name", None))
    harness = getattr(args, "harness", "native")
    if harness == "bare":
        # bare harness 的"agent"是裸模型，模型表在 catalog.bare 段（现统一到 models 注册表）。
        agent = {"kind": "bare", "model": op_manager.resolve_bare_model(catalog, requested_model)}
    else:
        agent = dict(catalog["agents"][args.agent])
        if model_spec is not None:
            agent["model_spec"] = model_spec          # 仅用于日志/回显，不进 run config
    if getattr(args, "agent_timeout_seconds", None) is not None and "timeout_seconds" in agent:
        agent["timeout_seconds"] = args.agent_timeout_seconds
    profiling_runs = op_manager.resolve_profiling_runs(args.profiling_runs)
    warmup_runs = op_manager.resolve_warmup_runs(getattr(args, "warmup_runs", None))
    sandbox_default = "http://127.0.0.1:9000"
    return {
        "budget": budget,
        "agent": agent,
        "backend": catalog["backend"],
        "profiling_runs": profiling_runs,
        "warmup_runs": warmup_runs,
        "sandbox_url": args.sandbox_url or os.environ.get("SANDBOX_URL") or sandbox_default,
        "parallel": args.parallel,
        "metrics": catalog.get("metrics_defaults", {}),
    }


def overridden_fields(args):
    names = ("candidate_evaluations_max", "max_model_turns", "final_reprofile_count",
             "wall_clock_seconds_max", "agent_timeout_seconds", "profiling_runs", "sandbox_url",
             "candidate_solutions_target", "warmup_runs", "model", "model_url", "model_name")
    return [name for name in names if getattr(args, name, None) is not None]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, payload):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def pid_alive(pid):
    return isinstance(pid, int) and Path(f"/proc/{pid}").exists()


def proc_stat(pid):
    """读 /proc/<pid>/stat → (state, ppid, pgid)；失败返回 None。

    身份判定优先走 /proc 而不是 os.getpgid/os.kill：受限容器里 syscall 可能被
    拒绝（进程号在容器 PID namespace 里不存在），但 /proc 文件仍然可读。
    """
    try:
        with open(f"/proc/{pid}/stat", "r", encoding="utf-8", errors="replace") as fh:
            _, _, tail = fh.read().rpartition(")")
    except OSError:
        return None
    parts = tail.split()
    if len(parts) < 3:
        return None
    try:
        return parts[0], int(parts[1]), int(parts[2])
    except ValueError:
        return None


def proc_alive(pid):
    """进程存在且不是僵尸（僵尸进程等父进程回收，不算"仍在运行"）。"""
    stat = proc_stat(pid)
    return bool(stat) and stat[0] != "Z"


def pgid_of(pid):
    stat = proc_stat(pid)
    if stat:
        return stat[2]
    try:
        return os.getpgid(pid)
    except OSError:
        return None


def proc_cmdline(pid):
    """读取 /proc/<pid>/cmdline；进程不存在/无权限时返回空串。"""
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as fh:
            return fh.read().replace(b"\x00", b" ").decode("utf-8", "replace").strip()
    except OSError:
        return ""


def looks_like_run_process(cmd):
    """run 进程树的身份特征（op_manager/native_bench/wrapper/opencode/评测桥）。"""
    return any(token in cmd for token in
               ("op_manager.py", "native_bench",
                "sandbox_evaluator", "opencode run"))


def cmd_has_tag(cmd, tag):
    """cmdline 里是否有 `--tag <tag>` / `--tag=<tag>`（避免 tagA 误配 tagA2）。"""
    return re.search(r"--tag(?:=|\s+)" + re.escape(tag) + r"(?:\s|$)", cmd) is not None


def cmd_has_agent(cmd, agent):
    """cmdline 里是否有 `--agent <agent>`：同 tag 不同 agent 的批次不能互相收尾。"""
    return re.search(r"--agent(?:=|\s+)" + re.escape(agent) + r"(?:\s|$)", cmd) is not None


def cmd_has_run_id(cmd, run_id):
    """cmdline 里是否有完整的 run_id（允许后面跟 .json/.log 等扩展名）。"""
    return re.search(re.escape(run_id) + r"(?:\.[A-Za-z0-9_]+)?(?:\s|$)", cmd) is not None


def proc_environ(pid):
    """读 /proc/<pid>/environ（无权限/已退出返回空串）。

    native harness 给 agent CLI 注入 NATIVE_BENCH_RUN_ID / NATIVE_BENCH_CONNECTION，
    agent 的 bash 子进程会继承，所以这是没有 pid 文件时最可靠的身份锚点。
    """
    try:
        with open(f"/proc/{pid}/environ", "rb") as fh:
            return fh.read().replace(b"\x00", b"\n").decode("utf-8", "replace")
    except OSError:
        return ""


def proc_cwd(pid):
    """读 /proc/<pid>/cwd（无权限/已退出返回空串）；agent CLI 的 cwd 在 <run>/workspace 下。"""
    try:
        return os.readlink(f"/proc/{pid}/cwd")
    except OSError:
        return ""


def children_map():
    """一次全表扫描得到 pid -> [子进程 pid]。"""
    children = {}
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        pid = int(name)
        stat = proc_stat(pid)
        if stat:
            children.setdefault(stat[1], []).append(pid)
    return children


def ancestor_pids():
    """当前进程的祖先链：--stop 不能把调用它的 shell / agent 一起当成目标。"""
    ancestors, current = set(), os.getppid()
    while current > 1 and current not in ancestors:
        ancestors.add(current)
        stat = proc_stat(current)
        if not stat:
            break
        current = stat[1]
    return ancestors


# native harness 的 agent CLI argv 特征（native_bench/adapters.py 生成，供孤儿识别）。
NATIVE_AGENT_SIGNATURES = (
    ("claude", ("-p", "--output-format stream-json", "--verbose", "--include-partial-messages")),
    ("codex", ("exec", "--json", "--skip-git-repo-check")),
    ("opencode", ("run", "--format json")),
    # bare harness 的"agent 进程"是 k-lens 的裸客户端（`python -m bare_client.runner ...`）。
    ("bare", ("-m", "bare_client.runner")),
)


def native_agent_kind(cmd):
    """识别 native harness 拉起的 agent CLI；不是 native agent 返回 None。"""
    for kind, tokens in NATIVE_AGENT_SIGNATURES:
        if all(token in cmd for token in tokens):
            return kind
    return None


def looks_like_run_id(name, tag, agent=None):
    """run_id = <op>-<agent>-<tag>（重跑为 ...-<tag>-r2）；给了 agent 就要求 agent 也对上。"""
    if not name or not tag:
        return False
    agent_part = re.escape(agent) if agent else r"[A-Za-z][\w]*"
    return re.fullmatch(r".+-" + agent_part + r"-" + re.escape(tag) + r"(?:-r\d+)?",
                        name) is not None


def cmdline_run_ids(cmd, tag, agent=None):
    """从 cmdline 抽取 run-id：--run-id X / --config <runs_dir>/X.json / --output <runs_dir>/X。"""
    names = list(re.findall(r"--run-id(?:=|\s+)(\S+)", cmd))
    for value in re.findall(r"--(?:config|output)(?:=|\s+)(\S+)", cmd):
        name = Path(value).name
        names.append(name[:-5] if name.endswith(".json") else name)
    return [name for name in dict.fromkeys(names) if looks_like_run_id(name, tag, agent)]


def environ_run_ids(env_text, tag, agent=None):
    """从 /proc/<pid>/environ 抽取 native run-id（NATIVE_BENCH_RUN_ID / CONNECTION 路径）。"""
    if not env_text:
        return []
    names = list(re.findall(r"NATIVE_BENCH_RUN_ID=(\S+)", env_text))
    names += [Path(value).parent.name
              for value in re.findall(r"NATIVE_BENCH_CONNECTION=(\S+)", env_text)]
    return [name for name in dict.fromkeys(names) if looks_like_run_id(name, tag, agent)]


def text_run_dirs(text, batch_dir, tag, agent=None):
    """从 cmdline / 环境变量里抽出 `<batch_dir>/<run_id>[...]` 形式的 run-id。

    只认批次目录下的第一层路径，避免在 batch_dir 恰好是 runs 根目录（旧的扁平布局）
    或 tag 写错时把别的批次进程一起算进来。
    """
    prefix = str(Path(batch_dir)).rstrip("/") + "/"
    found = set()
    for token in re.findall(r"[^\s,;'\"]+", text or ""):
        if not token.startswith(prefix):
            continue
        name = token[len(prefix):].split("/", 1)[0]
        if name.endswith(".json"):
            name = name[:-5]
        if looks_like_run_id(name, tag, agent):
            found.add(name)
    return found


def find_batch_processes(tag, run_ids, batch_dir, exclude=(), agent=None):
    """按"批次身份"定位进程，不依赖 .batch-*.json / *.pid（它们在 NFS 上可能已丢）。

    命中任一锚点即算批次进程：
      1) cmdline 里的完整 run-id 或 --run-id/--config/--output 指向的 run-id；
      2) cmdline/环境变量里有 `<batch_dir>/<run_id>` 形式的路径（--runs-dir/--config/--output、
         NATIVE_BENCH_CONNECTION 等）；
      3) cmdline 含 --tag <tag>（调度器；给了 agent 时会要求 --agent 也对上）；
      4) 环境变量里有 NATIVE_BENCH_RUN_ID / NATIVE_BENCH_CONNECTION 对应的 run-id；
      5) cwd 落在批次目录下（agent CLI 的 workspace）。
    再把命中进程的全部后代并进来：native harness 的 agent CLI 由 native_bench 用
    start_new_session 拉起（独立 session/进程组），它的 bash 子进程还会再开 session，
    只杀 run 的进程组杀不到它们。

    返回 (matched, orphans, foreign)：
      matched = {pid: {"why": 命中依据, "runs": {run_id, ...}}}
      orphans = {pid: 说明} —— 父进程已退出的 native agent CLI（argv 特征 + ppid=1），
                且环境变量/cwd/cmdline 能归属到本批次 → 本批次的遗留，随 --stop 一并清掉。
      foreign = {pid: 说明} —— 同样是孤儿 agent，但归属不到本批次：默认只报告，
                需要清理时显式加 --gc-orphan-agents，避免 --stop 误伤其它批次。
    """
    batch = str(Path(batch_dir))
    known = list(dict.fromkeys(run_ids))
    excluded = set(exclude) | {os.getpid()} | ancestor_pids()
    children = children_map()
    matched, orphans, foreign = {}, {}, {}
    for name in sorted(os.listdir("/proc"), key=lambda item: (len(item), item)):
        if not name.isdigit():
            continue
        pid = int(name)
        if pid in excluded or not proc_alive(pid):
            continue
        cmd = proc_cmdline(pid)
        if not cmd or "--stop" in cmd or "--status" in cmd:
            continue  # 正在执行的 stop/status 自身及其 shell 包装
        env, cwd = proc_environ(pid), proc_cwd(pid)
        runs, why = set(), []
        hits = [rid for rid in known if cmd_has_run_id(cmd, rid) or rid in env]
        if not hits and cwd:
            hits = [rid for rid in known if cwd.startswith(f"{batch}/{rid}/")]
        for run_id in hits:
            runs.add(run_id)
        if runs:
            why.append("run-id")
        found = [rid for rid in cmdline_run_ids(cmd, tag, agent) + environ_run_ids(env, tag, agent)
                 if rid in known or (Path(batch) / rid).is_dir()]
        if set(found) - runs:
            runs |= set(found)
            why.append("run-id(cmdline/env)")
        if text_run_dirs(cmd, batch, tag, agent) - runs:
            runs |= text_run_dirs(cmd, batch, tag, agent)
            why.append("run 目录路径")
        if tag and cmd_has_tag(cmd, tag) and (agent is None or cmd_has_agent(cmd, agent)):
            why.append("--tag")
        if text_run_dirs(env, batch, tag, agent) - runs:
            runs |= text_run_dirs(env, batch, tag, agent)
            why.append("环境变量")
        if cwd and cwd.startswith(batch + os.sep) \
                and (native_agent_kind(cmd) or looks_like_run_process(cmd)):
            why.append("cwd")
        if why:
            matched[pid] = {"why": "+".join(dict.fromkeys(why)), "runs": runs}

    # 孤儿 native agent：父进程已退出（上一轮 stop 的遗留），argv 是 harness 的 agent CLI 形状。
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        pid = int(name)
        if pid in matched or pid in excluded or not proc_alive(pid):
            continue
        stat, cmd = proc_stat(pid), proc_cmdline(pid)
        kind = native_agent_kind(cmd)
        if not (kind and stat and stat[1] == 1):
            continue
        env, cwd = proc_environ(pid), proc_cwd(pid)
        runs = {rid for rid in known if rid in env or rid in cmd
                or cwd.startswith(f"{batch}/{rid}/")}
        runs |= {rid for rid in cmdline_run_ids(cmd, tag, agent) + environ_run_ids(env, tag, agent)
                 if rid in known}
        runs |= text_run_dirs(env, batch, tag, agent)
        if runs or text_run_dirs(cmd, batch, tag, agent):
            runs |= text_run_dirs(cmd, batch, tag, agent)
            orphans[pid] = {"why": f"孤儿 {kind} agent（本批次遗留，父进程已退出）", "runs": runs}
        else:
            foreign[pid] = f"孤儿 {kind} agent（归属不到本批次，父进程已退出）"

    # 后代：身份继承父进程，单独标注（进程组杀不到的 agent CLI / bash 子进程都在这里）。
    queue = list(matched) + list(orphans)
    while queue:
        parent = queue.pop()
        for child in children.get(parent, ()):
            if child in matched or child in excluded or not proc_alive(child):
                continue
            runs = set((matched.get(parent) or orphans.get(parent) or {}).get("runs") or set())
            child_cmd = proc_cmdline(child)
            for run_id in cmdline_run_ids(child_cmd, tag, agent) \
                    + environ_run_ids(proc_environ(child), tag, agent):
                if run_id in known or (Path(batch) / run_id).is_dir():
                    runs.add(run_id)
            prefix = (matched.get(parent) or orphans.get(parent) or {}).get("why") \
                or f"孤儿 agent({parent})"
            matched[child] = {"why": f"{prefix} 的后代", "runs": runs}
            queue.append(child)

    # 归属不到本批次的孤儿（默认只报告）——连它的子进程一起标出来，便于 --gc-orphan-agents。
    queue = list(foreign)
    while queue:
        parent = queue.pop()
        for child in children.get(parent, ()):
            if child in matched or child in orphans or child in foreign \
                    or child in excluded or not proc_alive(child):
                continue
            foreign[child] = f"孤儿 agent({parent}) 的后代"
            queue.append(child)
    return matched, orphans, foreign


def signal_batch_processes(pids, sig):
    """给批次进程发信号：自己是组长的整组发（顺带覆盖同组子孙），其余逐个发。

    返回 (已发的进程组, 已发的单进程)——只作为日志。
    """
    my_pgid = pgid_of(os.getpid()) or os.getpid()
    groups, singles = [], []
    for pid in sorted(pids):
        pgid = pgid_of(pid)
        if pgid == pid and pgid not in (my_pgid, 1):
            try:
                os.killpg(pgid, sig)
                groups.append(pgid)
                continue
            except OSError:
                pass
        try:
            os.kill(pid, sig)
            singles.append(pid)
        except OSError:
            pass
    return groups, singles


class Logger:
    def __init__(self, path):
        self.fh = open(path, "a", encoding="utf-8", buffering=1)

    def __call__(self, message):
        stamp = time.strftime("%H:%M:%S")
        line = f"[{stamp}] {message}"
        print(line, flush=True)
        self.fh.write(line + "\n")

    def close(self):
        self.fh.close()


def resolved_ops(catalog_path):
    catalog = read_json(catalog_path)
    resolved = read_json(Path(catalog_path).with_name("op_catalog.resolved.json"))
    names = [item["name"] for item in resolved["ops"] if item.get("status") == "resolved"]
    if not names:
        raise SystemExit("op_catalog.resolved.json 中没有 resolved 状态的算子")
    return catalog, names


def select_ops(spec, names):
    if spec.strip().lower() in {"all", ""}:
        return names
    wanted = [x.strip() for x in spec.replace(" ", ",").split(",") if x.strip()]
    unknown = [x for x in wanted if x not in names]
    if unknown:
        raise SystemExit(f"未知算子: {unknown}；可用: {names}")
    return [x for x in names if x in wanted]


def run_state(runs_dir, run_id):
    """done / incomplete / free"""
    if (runs_dir / run_id / "result.json").is_file():
        return "done"
    if (runs_dir / run_id).exists() or (runs_dir / f"{run_id}.json").exists():
        return "incomplete"
    return "free"


def allocate_run_id(runs_dir, op, agent, tag, retry_incomplete):
    base = f"{op}-{agent}-{tag}"
    state = run_state(runs_dir, base)
    if state == "free":
        return base
    if state == "done":
        return None  # 已完成，跳过
    if not retry_incomplete:
        return "INCOMPLETE"
    index = 2
    while index < 1000:
        candidate = f"{base}-r{index}"
        state = run_state(runs_dir, candidate)
        if state == "free":
            return candidate
        if state == "done":
            return None
        index += 1
    raise SystemExit(f"{op}: 无法分配新的 run-id（-r2..-r999 都被占用）")


def known_run_ids(runs_dir, ops, agent, tag):
    pattern = re.compile("^(?:" + "|".join(re.escape(op) for op in ops) +
                         ")-" + re.escape(agent) + "-" + re.escape(tag) + "(-r\\d+)?$")
    found = []
    if runs_dir.is_dir():
        for entry in sorted(runs_dir.iterdir()):
            if entry.is_dir() and pattern.match(entry.name):
                found.append(entry.name)
    return found


def run_artifact_paths(runs_dir, run_id):
    """一个 run 落在 runs_dir 下的全部产物：run 目录 + .json/.pid/.launch.log 三个旁挂文件。

    四者缺一不可：少删 run 目录会被 native harness 的存在性检查拒绝
    （op_manager.py 要求 --output 目录必须不存在）；少删 .json/.pid 会让
    run_state 把算子误判成 incomplete，.launch.log 以 "ab" 打开则会旧日志续写。
    """
    return [runs_dir / run_id, runs_dir / f"{run_id}.json",
            runs_dir / f"{run_id}.pid", runs_dir / f"{run_id}.launch.log"]


def purge_run_id_list(runs_dir, op, agent, tag):
    """该算子在本 tag 下的 run_id 列表：base + 历史 -rN 变体。

    复用 known_run_ids（与 --status/--stop 的识别口径一致）而不是自己写 glob：
    它按 `{op}-{agent}-{tag}(-r\\d+)?` 精确匹配，因此只会产出本批次本算子的名字，
    不接受任意路径输入。
    """
    base = f"{op}-{agent}-{tag}"
    ids = known_run_ids(runs_dir, [op], agent, tag)
    if base not in ids:
        ids.append(base)
    return ids


def collect_purge_targets(runs_dir, op, agent, tag):
    """该算子在本 tag 下的全部已存在产物（含历史 -rN 变体），去重且保序。"""
    seen, out = set(), []
    for run_id in purge_run_id_list(runs_dir, op, agent, tag):
        for path in run_artifact_paths(runs_dir, run_id):
            if path in seen:
                continue
            if path.exists() or path.is_symlink():
                seen.add(path)
                out.append(path)
    return out


def live_run_ids(runs_dir, previous, agent, tag, run_ids):
    """本 tag 下真正还在跑的 run_id（用于"有活任务就拒绝续测/重测"的守卫）。

    两个来源取并集，比单独用 pid_alive 可靠：
      A) find_batch_processes 按 cmdline/env/cwd 身份锚定，能抓到状态文件里
         没有记录（或 pid 已陈旧）但实际仍在运行的 run；
      B) 批次状态文件里 proc_alive 为真的行（proc_alive 排除僵尸进程，
         避免 pid_alive 的"PID 被复用即视为存活"误判）。
    """
    live = set()
    matched, orphans, _ = find_batch_processes(tag, run_ids, runs_dir, agent=agent)
    for info in list(matched.values()) + list(orphans.values()):
        runs = info.get("runs") if isinstance(info, dict) else None
        if runs:
            live.update(runs)
    for row in (previous or {}).get("runs", []):
        run_id, pid = row.get("run_id"), row.get("pid")
        if run_id and isinstance(pid, int) and proc_alive(pid):
            live.add(run_id)
    return sorted(live)


def purge_run_artifacts(paths):
    """删除给定产物路径，返回实际删除成功的路径字符串列表。"""
    deleted = []
    for path in paths:
        try:
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)
        except OSError as exc:
            raise SystemExit(f"删除失败 {path}: {exc}") from exc
        deleted.append(str(path))
    return deleted


def write_retry_audit(runs_dir, args, ops_named, deleted, confirmed):
    """把本次续测/重测删了哪些路径追加写入 <batch_dir>/.retry-audit.jsonl。

    硬删除没有回收站，这份审计日志是事后追溯的唯一依据。
    """
    record = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "tag": args.tag, "agent": args.agent, "harness": args.harness,
        "ops": list(ops_named), "deleted": list(deleted),
        "deleted_count": len(deleted), "dry_run": False,
        "confirmed": confirmed, "scheduler_pid": os.getpid(),
    }
    path = runs_dir / ".retry-audit.jsonl"
    try:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as exc:
        print(f"warning: 审计日志写入失败 {path}: {exc}", flush=True)
        return None
    return path


def prune_batch_state(runs_dir, agent, tag, purged_run_ids):
    """把被删除的 run_id 从批次状态的 planned/runs 里剔掉。

    状态文件的 runs[] 只增不删（spawn 时 append），同一 run_id 重跑会留下
    指向死进程的陈旧行，进而让 live 守卫误判。仅在确实删过东西时调用。
    """
    state = load_batch_state(runs_dir, agent, tag)
    if not state:
        return None
    purged = set(purged_run_ids)
    kept_runs = [row for row in state.get("runs", []) if row.get("run_id") not in purged]
    removed = len(state.get("runs", [])) - len(kept_runs)
    state["runs"] = kept_runs
    if state.get("planned"):
        state["planned"] = [rid for rid in state["planned"] if rid not in purged]
    if removed:
        save_batch_state(runs_dir, agent, tag, state)
    return removed


def confirm_purge(args, total, log):
    """删除前的二次确认。返回 (是否继续, 确认方式)。

    非交互环境（非 tty）下没有可阻塞的确认入口，必须显式 --yes，否则拒绝 —— 宁可
    让批处理脚本失败，也不能让它无人值守地删掉整批结果。
    """
    if args.yes:
        return True, "--yes"
    if not sys.stdin.isatty():
        log("非交互环境：确认删除请加 --yes（或先用 --dry-run 预览将删除的路径）")
        return False, None
    try:
        reply = input(f"确认删除上述 {total} 项产物并用同一 run-id 原地重跑？[y/N] ")
    except EOFError:
        log("读取确认输入失败（stdin 已关闭）：如确认请加 --yes")
        return False, None
    if reply.strip().lower() in ("y", "yes"):
        return True, "tty"
    return False, None


def next_retry_run_id(runs_dir, run_id):
    """给已结束的 run 分配下一个 -rN 重试编号（base、-r2、-r3 … 依次递增）。"""
    match = re.match(r"^(?P<base>.+?)(?:-r(?P<index>\d+))?$", run_id)
    base = match.group("base")
    index = int(match.group("index") or 1) + 1
    while index < 1000:
        candidate = base if index == 1 else f"{base}-r{index}"
        if run_state(runs_dir, candidate) == "free":
            return candidate
        index += 1
    raise SystemExit(f"{run_id}: 无法分配新的重试 run-id（-r2..-r999 都被占用）")


# --- 基础设施故障判定（run 级重试的依据）---------------------------------------
# 评测期间的沙盒/设备故障会被 native harness 升级成整个 run 失败
# （termination_reason=evaluation_state_unknown），随后 run 有了 result.json，
# 批次就把它当成"已完成"永不再碰。这类失败与算子质量无关，换时间/换设备重跑
# 很有机会拿到真实结论，所以单独识别出来重排。
INFRA_TERMINATION_REASONS = ("evaluation_state_unknown", "infrastructure_error")
# bare_client 的传输不确定语义（Backend 通道无 request_id、不重试）：
# 评测通道不确定 → evaluation_state_unknown；改代码通道不确定 → tool_state_unknown。
BARE_INFRA_TERMINATION_REASONS = ("evaluation_state_unknown", "tool_state_unknown",
                                  "context_unrecoverable")   # k-lens-v2 上下文护栏的终止


def read_eval_rows(run_dir):
    path = run_dir / "evaluations.jsonl"
    if not path.is_file():
        return []
    rows = []
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
    except OSError:
        return []
    return rows


def is_infra_row(row):
    """该次评测是否命中基础设施故障（新旧评测器都认）。"""
    raw = row.get("raw") or {}
    return bool(raw.get("infra_error")) or row.get("status") == "sandbox_error"


def infra_failure_reason(runs_dir, run_id):
    """是基础设施故障导致的失败 -> 返回原因串；否则返回 None（真实结论）。"""
    run_dir = runs_dir / run_id
    result = run_dir / "result.json"
    if not result.is_file():
        return None
    try:
        data = read_json(result)
    except (OSError, ValueError):
        return None
    if data.get("success") is True:
        return None
    stop = data.get("termination_reason")
    if stop in INFRA_TERMINATION_REASONS or stop in BARE_INFRA_TERMINATION_REASONS:
        return f"termination_reason={stop}"
    # 收尾原因看着正常，但最后一次评测撞上了基础设施故障（例如 final_verification 期间）。
    # 若故障之后出现过任何一次真实判定，说明 run 已恢复，失败是真实结论，不再重排。
    rows = read_eval_rows(run_dir)
    last_infra = None
    for index, row in enumerate(rows):
        if is_infra_row(row):
            last_infra = index
    if last_infra is None:
        return None
    if any(row.get("status") == "passed" for row in rows[last_infra + 1:]):
        return None
    eid = rows[last_infra].get("evaluation_id") or "?"
    return f"{eid} 之后无有效判定（基础设施故障）"


# --- agent 侧协议错误判定（同样按 run 级重排处理）----------------------------
# 典型形态：模型生成长工具调用时被截断，会话历史里留下一个 arguments 不是合法 JSON 的
# function_call；下一轮请求原样回传，上游（vLLM 的 Responses 实现）用 json.loads 解析
# 历史 tool_call 参数时抛错并整包 400（BadRequestError: Unterminated string ...），
# codex 遂把整个 turn 判失败 —— termination_reason=agent_error、候选 0。
# 这类失败与算子质量无关，且脏状态只存在于那一个会话里：换新 run-id（全新会话）重跑
# 即可绕开，所以和基础设施故障一样重排，而不是直接判死。
AGENT_PROTOCOL_ERROR_PATTERNS = (
    "BadRequestError",
    "Unterminated string",
    "is not valid JSON",
    "Invalid HTTP request received",
)


def read_run_error_blob(run_dir):
    """把 run 目录里可能记录 agent 报错的文本拼起来（result.json 的 usage + agent.stderr）。"""
    chunks = []
    result = run_dir / "result.json"
    if result.is_file():
        try:
            data = read_json(result)
        except (OSError, ValueError):
            data = {}
        chunks.append(json.dumps(data.get("usage") or {}, ensure_ascii=False))
        chunks.append(json.dumps(data.get("error") or "", ensure_ascii=False))
    stderr = run_dir / "agent.stderr.txt"
    if stderr.is_file():
        try:
            chunks.append(stderr.read_text(encoding="utf-8", errors="replace")[-20000:])
        except OSError:
            pass
    return "\n".join(chunks)


def agent_protocol_failure_reason(runs_dir, run_id):
    """agent 侧协议/请求体错误导致的失败 -> 返回原因串；否则返回 None。"""
    run_dir = runs_dir / run_id
    result = run_dir / "result.json"
    if not result.is_file():
        return None
    try:
        data = read_json(result)
    except (OSError, ValueError):
        return None
    if data.get("success") is True:
        return None
    if data.get("termination_reason") not in ("agent_error", "gateway_error"):
        return None
    blob = read_run_error_blob(run_dir).lower()
    for pattern in AGENT_PROTOCOL_ERROR_PATTERNS:
        if pattern.lower() in blob:
            return f"agent 协议错误（{pattern}）→ 换新会话重跑"
    return None


def load_batch_state(runs_dir, agent, tag):
    path = runs_dir / f".batch-{agent}-{tag}.json"
    if not path.is_file():
        return None
    try:
        return read_json(path)
    except (OSError, ValueError):
        return None  # 状态文件损坏（NFS 半写）时退回按进程/目录身份定位


def save_batch_state(runs_dir, agent, tag, payload):
    write_json(runs_dir / f".batch-{agent}-{tag}.json", payload)


def native_result_detail(data):
    """native harness 的 result.json 摘要（字段与 v4 不同，见 native_bench/runner.py）。"""
    baseline = data.get("baseline") or {}
    best = data.get("best_verified") or {}
    speedup = None
    if isinstance(baseline.get("duration_ms"), (int, float)) \
            and isinstance(best.get("duration_ms"), (int, float)) and best["duration_ms"]:
        speedup = baseline["duration_ms"] / best["duration_ms"]
    tokens = "tokens=ok" if (data.get("usage") or {}).get("available") else "tokens=n/a"
    detail = (f"native ok={data.get('success')} cand={data.get('candidate_evaluations')} "
              f"stop={data.get('termination_reason')} {tokens}")
    if isinstance(speedup, (int, float)):
        detail += f" speedup={speedup:.4f}"
    return detail


def bare_result_detail(data):
    """bare harness 的 result.json 摘要（k-lens bare_client 的 schema）。"""
    baseline = data.get("baseline") or {}
    best = data.get("best_verified") or {}
    duration = lambda row: row.get("duration_us") if isinstance(row, dict) else None
    speedup = None
    if isinstance(duration(baseline), (int, float)) and isinstance(duration(best), (int, float)) \
            and duration(best):
        speedup = duration(baseline) / duration(best)
    final = (data.get("final_verification") or {}).get("valid")
    detail = (f"bare ok={data.get('success')} cand={data.get('candidate_evaluations')} "
              f"stop={data.get('termination_reason')} final={'ok' if final else 'n/a'}")
    if isinstance(speedup, (int, float)):
        detail += f" speedup={speedup:.4f}"
    return detail


def run_summary(runs_dir, run_id):
    """返回 (state, detail)：用于 --status。"""
    run_dir = runs_dir / run_id
    pid_file = runs_dir / f"{run_id}.pid"
    pid = None
    if pid_file.is_file():
        try:
            pid = int(pid_file.read_text(encoding="utf-8").strip())
        except ValueError:
            pid = None
    alive = pid_alive(pid)
    last_event = "-"
    events = run_dir / "events.jsonl"
    if events.is_file():
        try:
            with events.open("rb") as fh:
                fh.seek(max(0, events.stat().st_size - 8192))
                tail = fh.read().decode("utf-8", "replace").strip().splitlines()
            if tail:
                last = json.loads(tail[-1])
                last_event = f"{last.get('event_type', '?')}@{last.get('timestamp_utc', '')[11:19]}"
        except (ValueError, OSError):
            last_event = "unreadable"
    result = run_dir / "result.json"
    if result.is_file():
        data = read_json(result)
        if data.get("schema_version") == "native-bench-result-v1":
            return "finished", native_result_detail(data)
        if data.get("schema_version") == "klens-bare-client-result-v1":
            return "finished", bare_result_detail(data)
        verification = data.get("final_verification") or {}
        speedup = verification.get("speedup_vs_baseline") or data.get("speedup")
        detail = (f"final={data.get('operational_status')} cand={data.get('candidate_eval_count')} "
                  f"turns={data.get('llm_rounds')} speedup={speedup:.4f}" if isinstance(speedup, (int, float))
                  else f"final={data.get('operational_status')} cand={data.get('candidate_eval_count')} turns={data.get('llm_rounds')}")
        return "finished", detail
    if alive:
        return "running", f"pid={pid} last={last_event}"
    if run_dir.exists():
        return "stopped", f"last={last_event}"
    return "missing", "-"


def request_native_stop(runs_dir, run_ids, harness=None):
    """对 native run 写 STOP 文件：先关闭评测入口、给 agent 收尾时间，再回退为发信号。

    只依赖 run 目录本身：manifest.json 在 NFS 上可能已被误删，此时用
    harness 提示或 native 产物特征（events.jsonl/agent.stdout.jsonl/prompt.txt/launch.json）
    判断身份，避免给 v4 run 误写 STOP。返回 (已写 STOP 的 run_id, 跳过说明)。
    """
    runs_dir = Path(runs_dir)
    requested, skipped = [], []
    for run_id in run_ids:
        run_dir = runs_dir / run_id
        if (run_dir / "result.json").is_file():
            skipped.append(f"{run_id}(已自行结束)")
            continue
        if not run_dir.is_dir():
            skipped.append(f"{run_id}(目录不存在)")
            continue
        is_native = harness == "native"
        if not is_native:
            manifest = run_dir / "manifest.json"
            try:
                payload = read_json(manifest) if manifest.is_file() else {}
            except (OSError, ValueError):
                payload = {}
            is_native = payload.get("schema_version") == "native-bench-run-v1"
            if not is_native and not manifest.is_file():
                markers = ("events.jsonl", "agent.stdout.jsonl", "prompt.txt", "launch.json")
                is_native = sum((run_dir / name).is_file() for name in markers) >= 2
        if not is_native:
            skipped.append(f"{run_id}(非 native run，未写 STOP)")
            continue
        stop_file = run_dir / "STOP"
        if not stop_file.exists():
            try:
                write_json(stop_file, {"run_id": run_id, "reason": "manual_stop"})
            except OSError as exc:
                skipped.append(f"{run_id}(STOP 写入失败: {exc})")
                continue
        requested.append(run_id)
    return requested, skipped


def wait_native_stop(runs_dir, run_ids, timeout):
    """等 STOP 生效（result.json 落盘或该 run 的进程全部退出）；返回仍未结束的 run_id。"""
    runs_dir = Path(runs_dir)
    deadline = time.monotonic() + timeout
    pending = list(run_ids)
    while pending and time.monotonic() < deadline:
        matched, _, _ = find_batch_processes("", pending, runs_dir)
        live = {rid for info in matched.values() for rid in info["runs"]}
        pending = [rid for rid in pending
                   if not (runs_dir / rid / "result.json").is_file() and rid in live]
        if not pending:
            return []
        time.sleep(1.0)
    return pending


def batch_dir_for(runs_dir, tag):
    """本批次专属目录 <runs_dir>/<tag>/，所有过程文件与结果都放这里。"""
    batch_dir = runs_dir / tag
    batch_dir.mkdir(parents=True, exist_ok=True)
    return batch_dir


def existing_batch_dir(runs_dir, tag):
    """优先用新布局 <runs_dir>/<tag>/；找不到时回退旧的扁平布局（兼容历史批次）。"""
    candidate = runs_dir / tag
    return candidate if candidate.is_dir() else runs_dir


def batch_rows(batch_dir, agent, tag, ops):
    """批次内每个 run 的状态行与计数；--status 与 tools/task_dispatcher.py 共用一份口径。

    batch_dir 由调用方解析（--status 用 existing_batch_dir 兼容扁平布局；调度器用
    monitor.resolve_batch，以便区分"批次还不存在"）。没有任何运行记录时返回 None。
    """
    state = load_batch_state(batch_dir, agent, tag)
    registered = []
    if state:
        registered = [row["run_id"] for row in state.get("runs", [])]
        # 完整计划：优先用批次启动时写入的 planned 列表；老批次没有则按 --ops 推导。
        planned = state.get("planned") or [f"{op}-{agent}-{tag}" for op in ops]
        run_ids = list(dict.fromkeys(list(planned) + registered))
    else:
        run_ids = known_run_ids(batch_dir, ops, agent, tag)
        if not run_ids:
            return None
    # 进程身份兜底：pid 文件/状态文件丢失时也能看出"其实还在跑"。
    matched, orphans, foreign = find_batch_processes(tag, run_ids, batch_dir, agent=agent)
    rows = []
    for run_id in run_ids:
        state_name, detail = run_summary(batch_dir, run_id)
        if run_id not in registered and state_name == "missing":
            state_name, detail = "queued", "尚未启动（等待前面的算子）"
        if state_name in {"missing", "stopped"}:
            live = sorted(pid for pid, info in matched.items() if run_id in info["runs"])
            if live:
                state_name, detail = "running", f"pid={live[0]}（按进程身份识别）"
        rows.append((run_id, state_name, detail))
    counts = {name: sum(1 for _, state_name, _ in rows if state_name == name)
              for name in ("finished", "running", "stopped", "queued")}
    recorded = (state or {}).get("scheduler_pid")
    spid = int(recorded) if isinstance(recorded, int) or (
        isinstance(recorded, str) and recorded.isdigit()) else None
    live_sched = sorted({pid for pid in [spid] + list(matched) if pid
                         and "start_all_runs.py" in proc_cmdline(pid)})
    return {"state": state, "rows": rows, "counts": counts, "total": len(rows),
            "matched": matched, "orphans": orphans, "foreign": foreign,
            "scheduler_pid": spid, "live_sched": live_sched}


def cmd_status(args, catalog, ops):
    runs_dir = Path(args.runs_dir) if args.runs_dir else Path(catalog["outputs"]["runs_dir"])
    runs_dir = existing_batch_dir(runs_dir, args.tag)
    data = batch_rows(runs_dir, args.agent, args.tag, ops)
    if data is None:
        raise SystemExit(f"没有找到 tag={args.tag} agent={args.agent} 的运行记录")
    state, rows = data["state"], data["rows"]
    matched, orphans, foreign = data["matched"], data["orphans"], data["foreign"]
    run_ids = [run_id for run_id, _, _ in rows]
    finished = data["counts"]["finished"]
    running_n = data["counts"]["running"]
    stopped = data["counts"]["stopped"]
    queued = data["counts"]["queued"]
    width = max(len(x) for x in run_ids) + 2
    harness = (state or {}).get("harness")
    print(f"tag={args.tag} agent={args.agent}"
          + (f" harness={harness}" if harness else "")
          + f" runs_dir={runs_dir}")
    spid, live_sched = data["scheduler_pid"], data["live_sched"]
    if spid or live_sched:
        shown = ", ".join(f"pid={pid}" for pid in live_sched) if live_sched else f"pid={spid}"
        print(f"调度器: {shown} ({'运行中' if live_sched else '已退出'})"
              + ("；--stop 会一并停止" if live_sched else ""))
    if orphans:
        print("本批次遗留孤儿 agent: " + ", ".join(f"pid={pid}" for pid in sorted(orphans))
              + "；--stop 会一并清理")
    if foreign:
        print("归属不到本批次的孤儿 agent: " + ", ".join(f"pid={pid}" for pid in sorted(foreign))
              + "；--stop 默认不动，可用 --gc-orphan-agents 清理")
    percent = (finished / len(rows) * 100) if rows else 100.0
    print(f"进度: {progress_bar(finished, len(rows))} {finished}/{len(rows)} ({percent:5.1f}%) "
          f"| 完成 {finished} 运行 {running_n} 中断 {stopped} "
          f"待启动 {queued}")
    for run_id, state_name, detail in rows:
        print(f"{run_id.ljust(width)}{state_name:<10}{detail}")
    return 0


def describe_process(pid):
    """一行进程描述：pid/pgid + 截断的 cmdline。"""
    return f"pid={pid} pgid={pgid_of(pid)} {proc_cmdline(pid)[:110]}"


def sweep_batch_processes(tag, run_ids, batch_dir, exclude=(), agent=None,
                          gc_orphans=False, report=print):
    """按身份收尾本批次进程：SIGTERM（进程组优先）→ 复扫 → SIGKILL → 复扫。

    每一轮都重新做身份匹配，所以 agent CLI 在收尾过程中新拉起的 bash 子进程、
    以及父进程已被杀而"转孤儿"的 agent CLI，都会被下一轮捕获。
    gc_orphans=True 时连同"归属不到本批次"的孤儿 agent 一起清理。
    返回最终幸存 {pid: 说明}。
    """
    excluded = set(exclude)
    for sig, wait_seconds, label in ((signal.SIGTERM, STOP_TERM_GRACE_SECONDS, "SIGTERM"),
                                     (signal.SIGKILL, STOP_KILL_GRACE_SECONDS, "SIGKILL")):
        matched, orphans, foreign = find_batch_processes(
            tag, run_ids, batch_dir, exclude=excluded, agent=agent)
        targets = dict(matched)
        targets.update(orphans)
        if gc_orphans:
            for pid, why in foreign.items():
                targets.setdefault(pid, {"why": why, "runs": set()})
        if not targets:
            break
        groups, singles = signal_batch_processes(list(targets), sig)
        report(f"{label}: 进程组 {groups or '无'} / 单进程 {singles or '无'}"
               f"（批次进程 {len(targets)} 个）")
        deadline = time.monotonic() + wait_seconds
        while time.monotonic() < deadline:
            time.sleep(0.5)
            again, again_orphans, again_foreign = find_batch_processes(
                tag, run_ids, batch_dir, exclude=excluded, agent=agent)
            if not again and not again_orphans and (not gc_orphans or not again_foreign):
                break
    matched, orphans, foreign = find_batch_processes(
        tag, run_ids, batch_dir, exclude=excluded, agent=agent)
    survivors = {pid: info["why"] for pid, info in matched.items()}
    survivors.update({pid: info["why"] for pid, info in orphans.items()})
    return survivors


def cmd_stop(args, catalog, ops):
    """停止一个批次：不依赖 .batch-*.json / *.pid（NFS 上可能已丢），按进程身份兜底。"""
    runs_dir = Path(args.runs_dir) if args.runs_dir else Path(catalog["outputs"]["runs_dir"])
    runs_dir = existing_batch_dir(runs_dir, args.tag)
    state = load_batch_state(runs_dir, args.agent, args.tag)
    state_file = runs_dir / f".batch-{args.agent}-{args.tag}.json"
    run_ids = [row["run_id"] for row in state.get("runs", [])] if state \
        else known_run_ids(runs_dir, ops, args.agent, args.tag)
    harness = (state or {}).get("harness")

    # 身份定位：run 目录名 + 进程 cmdline/环境变量/cwd（状态文件丢了也能定位）。
    matched, orphans, foreign = find_batch_processes(args.tag, run_ids, runs_dir,
                                                     agent=args.agent)
    discovered = set()
    for pid in matched:
        cmd, env = proc_cmdline(pid), proc_environ(pid)
        discovered.update(rid for rid in cmdline_run_ids(cmd, args.tag) + environ_run_ids(env, args.tag)
                          if rid in run_ids or (runs_dir / rid).is_dir())
    run_ids = list(dict.fromkeys(list(run_ids) + sorted(discovered)))
    if not run_ids and not matched and not orphans and not foreign:
        raise SystemExit(f"没有找到 tag={args.tag} agent={args.agent} 的运行记录或进程"
                         f"（runs_dir={runs_dir}）")

    if args.dry_run:
        print(f"[dry-run] tag={args.tag} agent={args.agent} runs_dir={runs_dir}")
        if not state_file.is_file():
            print("[dry-run] 批次状态文件缺失：按进程/目录身份定位（加固后覆盖的场景）")
        print(f"[dry-run] run-id ({len(run_ids)}): " + (", ".join(run_ids) if run_ids else "无"))
        print(f"[dry-run] 批次进程 ({len(matched)}):")
        for pid, info in sorted(matched.items()):
            print(f"  {describe_process(pid)}  依据={info['why']} "
                  f"runs={','.join(sorted(info['runs'])) or '-'}")
        if orphans:
            print(f"[dry-run] 本批次遗留的孤儿 agent ({len(orphans)}):")
            for pid, info in sorted(orphans.items()):
                print(f"  {describe_process(pid)}  {info['why']}")
        if foreign:
            print(f"[dry-run] 归属不到本批次的孤儿 agent ({len(foreign)}，默认不处理):")
            for pid, why in sorted(foreign.items()):
                print(f"  {describe_process(pid)}  {why}")
        print("[dry-run] 未发送任何信号、未写 STOP。")
        return 0

    # 1) 先停调度器：否则杀掉当前 run 后它还会从队列里继续启动下一个算子。
    scheduler_pids = sorted(pid for pid in matched if "start_all_runs.py" in proc_cmdline(pid))
    if scheduler_pids:
        signal_batch_processes(scheduler_pids, signal.SIGTERM)
        print("已停止调度器: " + ", ".join(f"pid={pid}" for pid in scheduler_pids))
        time.sleep(min(2.0, STOP_TERM_GRACE_SECONDS))

    # 2) native run 写 STOP（受控停止：关评测入口 → 宽限期 → 由 runner 收紧 agent）。
    graceful, stop_skipped = request_native_stop(runs_dir, run_ids, harness=harness)
    if graceful:
        print(f"native STOP 已写入（受控停止，最多等 {int(NATIVE_STOP_GRACE_SECONDS)}s）: "
              + ", ".join(graceful))
        still = wait_native_stop(runs_dir, graceful, NATIVE_STOP_GRACE_SECONDS)
        done = [rid for rid in graceful if rid not in still]
        if done:
            print("native 已自行收尾: " + ", ".join(done))
        if still:
            print("native 未在宽限期内结束，回退为按进程树停止: " + ", ".join(still))
    if stop_skipped:
        print("未写 STOP: " + ", ".join(stop_skipped))

    # 3) 进程树收尾：覆盖 op_manager / native_bench / agent CLI 及其 bash 子进程。
    gc_orphans = bool(getattr(args, "gc_orphan_agents", False))
    if foreign and not gc_orphans:
        print(f"另有 {len(foreign)} 个归属不到本批次的孤儿 agent（未处理，需要清理加 "
              f"--gc-orphan-agents）: " + ", ".join(f"pid={pid}" for pid in sorted(foreign)))
    survivors = sweep_batch_processes(args.tag, run_ids, runs_dir,
                                      agent=args.agent, gc_orphans=gc_orphans, report=print)
    if survivors:
        print("仍有残留进程（SIGKILL 后仍存活）:")
        for pid, why in sorted(survivors.items()):
            print(f"  {describe_process(pid)}  {why}")
    else:
        print("残留进程: 无（本批次进程已全部退出）")
    finished = sum(1 for rid in run_ids if (runs_dir / rid / "result.json").is_file())
    print(f"结果: run={len(run_ids)} 个，其中已落 result.json={finished} 个。")
    print("续测/重测（会用同一 run-id 删除并重建产物，删除前需确认）：")
    print(f"  python3 start_all_runs.py --agent {args.agent} --harness "
          f"{(state or {}).get('harness') or 'native'} --tag {args.tag} "
          f"--ops <算子1,算子2> --retry-incomplete")
    return 1 if survivors else 0


def cmd_launch(args, catalog, ops):
    base_runs_dir = Path(args.runs_dir) if args.runs_dir else Path(catalog["outputs"]["runs_dir"])
    new_dir = base_runs_dir / args.tag
    if new_dir.is_dir():
        runs_dir = new_dir
    elif load_batch_state(base_runs_dir, args.agent, args.tag) \
            or known_run_ids(base_runs_dir, ops, args.agent, args.tag):
        runs_dir = base_runs_dir  # 旧扁平布局的历史批次：留在原目录续跑，避免产生重复结果
    else:
        runs_dir = new_dir if args.dry_run else batch_dir_for(base_runs_dir, args.tag)
    batch_dir = runs_dir  # 新布局：本批次的配置/日志/pid/产物全部落在 <runs_dir>/<tag>/
    legacy_layout = batch_dir == base_runs_dir
    if args.dry_run:
        log = lambda message: print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)
    else:
        log = Logger(runs_dir / f"batch-{args.agent}-{args.tag}.log")

        def _handle_signal(signum, _frame):
            log(f"收到信号 {signum}：调度器退出；已启动的 run 在独立会话中继续运行。"
                f"如需一并停止：python3 start_all_runs.py --stop --agent {args.agent} --tag {args.tag}")
            raise SystemExit(0)

        for _sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(_sig, _handle_signal)
    log(f"batch start: agent={args.agent} harness={args.harness} tag={args.tag} parallel={args.parallel} "
        f"ops={len(ops)} pid={os.getpid()} runs_dir={base_runs_dir} batch_dir={batch_dir}"
        + ("（沿用旧扁平布局）" if legacy_layout else ""))
    effective = effective_config(catalog, args)
    budget = effective["budget"]
    agent_cfg = effective["agent"]
    metrics = effective["metrics"]
    wall = budget.get("wall_clock_seconds_max")
    if isinstance(wall, (int, float)) and wall:
        wall_text = f"{wall}s({wall / 3600:.1f}h)"
    elif wall == 0:
        wall_text = "0(不限)"
    else:
        wall_text = "-"
    epsilon = budget.get("min_improvement_epsilon")
    epsilon_text = f"{epsilon * 100:.1f}%" if isinstance(epsilon, (int, float)) else "-"
    log("配置（已解析：op_catalog 默认值 + 命令行覆盖）")
    if args.harness == "native":
        target = budget.get("candidate_solutions_target")
        target_text = f" 软目标={target}" if target is not None else " 软目标=-（默认=硬上限）"
        limit_kind = "native" if args.harness == "native" else "bare"
        log(f"  budget : 候选={budget.get('candidate_evaluations_max')} "
            f"复测={budget.get('final_reprofile_count')}{target_text}"
            f"（{limit_kind}：无轮次/墙钟/token 上限；max_model_turns={budget.get('max_model_turns')}、"
            f"墙钟={wall_text}、噪声门控={epsilon_text} 不生效）")
    else:
        log(f"  budget : 候选={budget.get('candidate_evaluations_max')} 轮次={budget.get('max_model_turns')} "
            f"复测={budget.get('final_reprofile_count')} 墙钟={wall_text} "
            f"噪声门控={epsilon_text}")
    agent_impl = agent_cfg.get("executable") or ("wrapper" if agent_cfg.get("command") else "-")
    warmup_runs = effective["warmup_runs"]
    protocol_supported = op_manager.native_supports_profiling(op_manager.native_root(catalog))
    if args.harness == "bare":
        model = agent_cfg.get("model") or {}
        bare_backend = (catalog.get("bare") or {}).get("backend") or {}
        log(f"  agent  : bare 裸模型={model.get('key')}（{model.get('name')} @ {model.get('url')}）；"
            f"k-lens Backend={(catalog.get('bare') or {}).get('backend_root')}")
        log(f"  backend: k-lens Backend（每 run 独立进程 + 独立 op-path-version；"
            f"compile_timeout={bare_backend.get('compile_timeout_ms')}ms、"
            f"run_timeout={bare_backend.get('run_timeout_ms')}ms）")
        protocol = (catalog.get("bare") or {}).get("profiling") or {}
        log(f"  runtime: profiling 固定协议 {json.dumps(protocol, ensure_ascii=False)}"
            f"（total_runs 含预热，k-lens 客户端逐字校验） 并发={args.parallel} "
            f"沙盒={effective['sandbox_url']}")
    else:
        if args.harness == "native":
            log(f"  agent  : {args.agent} 原生 CLI（native；无单轮超时，wrapper 不参与）")
        else:
            log(f"  agent  : {args.agent} 单轮超时={agent_cfg.get('timeout_seconds')}s ({agent_impl})")
        log(f"  backend: 单次评测超时={effective['backend'].get('timeout_seconds')}s"
            + ("（native evaluator）" if args.harness == "native" else ""))
        log(f"  runtime: profiling 测量={effective['profiling_runs']} 预热={warmup_runs} "
            f"共={effective['profiling_runs'] + warmup_runs} 轮"
            + (f"（下发固定协议 total_runs={effective['profiling_runs'] + warmup_runs}，"
               f"评测器回显 raw.profiling_protocol）" if protocol_supported else "（该 harness 不支持固定协议）")
            + f" 并发={args.parallel} 沙盒={effective['sandbox_url']}")
    log(f"  runs   : 本批产物目录={batch_dir}（配置/日志/pid/结果都在这里）")
    log(f"  metrics: time_checkpoints_sec={metrics.get('time_checkpoints_sec')} "
        f"token_checkpoints={metrics.get('token_checkpoints')}")
    overrides = overridden_fields(args)
    log("  overrides: " + (", ".join(overrides) if overrides else "无（全部取 op_catalog 默认值）"))
    if args.harness == "native":
        native_agent = dict(op_manager.NATIVE_AGENT_DEFAULTS.get(args.agent, {}))
        native_agent["kind"] = args.agent
        spec = (effective.get("agent") or {}).get("model_spec")
        if spec is not None:
            # 回显必须与 op_manager 实际下发的 argv 一致（opencode 是 <provider>/<name>）
            native_agent["model"] = (f"{spec['provider']}/{spec['name']}"
                                     if args.agent == "opencode" else spec["name"])
        else:
            model, _ = op_manager.native_agent_model(args.agent, catalog)
            if model:
                native_agent["model"] = model
        log("  config.json: " + json.dumps({
            "harness": "native", "agent": native_agent,
            "evaluator": {"command": [arg.replace("{python}", sys.executable)
                                      for arg in (effective["backend"].get("command") or [])],
                          "evidence_kind": effective["backend"].get("evidence_kind"),
                          "per_call_timeout_seconds": effective["backend"].get("timeout_seconds"),
                          **({"profiling": {"total_runs": effective["profiling_runs"] + warmup_runs,
                                            "warmup_runs": warmup_runs, "statistic": "mean"}}
                             if protocol_supported else {})},
            "stop": {"candidate_evaluations_max": budget.get("candidate_evaluations_max"),
                     "final_reprofile_count": budget.get("final_reprofile_count")},
            **({"search": {"candidate_solutions_target": budget["candidate_solutions_target"]}}
               if budget.get("candidate_solutions_target") is not None else {}),
            "profiling_runs": effective["profiling_runs"], "sandbox_url": effective["sandbox_url"],
            "parallel": args.parallel,
        }, ensure_ascii=False))
    elif args.harness == "bare":
        log("  config.json: " + json.dumps({
            "harness": "bare", "agent": agent_cfg,
            "stop": {"candidate_evaluations_max": budget.get("candidate_evaluations_max"),
                     "final_reprofile_count": budget.get("final_reprofile_count")},
            "profiling": (catalog.get("bare") or {}).get("profiling"),
            "sandbox_url": effective["sandbox_url"], "parallel": args.parallel,
        }, ensure_ascii=False))
    else:
        log("  config.json: " + json.dumps({
            "harness": args.harness, "budget": budget, "agent": agent_cfg,
            "backend": effective["backend"],
            "profiling_runs": effective["profiling_runs"], "sandbox_url": effective["sandbox_url"],
            "parallel": args.parallel,
        }, ensure_ascii=False))
    if args.harness == "native":
        if args.max_model_turns is not None or args.wall_clock_seconds_max is not None \
                or args.agent_timeout_seconds is not None:
            log("  warning: native harness 不接受 --max-model-turns / --wall-clock-seconds / "
                "--agent-timeout-seconds，这几个覆盖值不会生效")
        spec = (effective.get("agent") or {}).get("model_spec")
        if spec is not None:
            log(f"本批次 {args.agent}(native) 模型: " + op_manager.model_summary(spec)
                + "（命令行/环境变量指定；端点只在本批次生效，不写全局配置）")
            model, source = spec["name"], "命令行指定"
        else:
            model, source = op_manager.native_agent_model(args.agent, catalog)
        if spec is not None:
            pass                                  # 上面已经打过"命令行指定"那行，避免重复
        elif args.agent == "opencode":
            log("本批次 opencode(native) 模型: "
                + (f"{model}（来源: {source}）" if model else "未固定（使用 opencode 自带配置）"))
        elif model:
            log(f"本批次 {args.agent}(native) 模型: {model}（来源: {source}）")

    previous = load_batch_state(runs_dir, args.agent, args.tag)
    planned_ids = known_run_ids(runs_dir, ops, args.agent, args.tag) + \
        [row["run_id"] for row in (previous or {}).get("runs", []) if row.get("run_id")]
    live = live_run_ids(runs_dir, previous, args.agent, args.tag, planned_ids)
    # 续测/重测是「先删后建」，绝不能和活着的 run 抢同一批产物：此时 --force 也不放行，
    # 必须由人先手动停止，确认批次真的静下来了再重跑。
    if live and (args.retry_incomplete or not args.force):
        log(f"该 tag 仍有 {len(live)} 个 run 在运行: {live}")
        if args.retry_incomplete:
            log("续测/重测会删除并重建这些算子的产物，因此不会自动替你停止它们。")
        log("请先手动停止本批次（停止命令）：")
        log(f"  python3 start_all_runs.py --agent {args.agent} --harness {args.harness} "
            f"--tag {args.tag} --stop")
        log("可先加 --dry-run 预览将停止哪些进程；确认批次已停止后，再重新执行本次命令。")
        return 2

    # 续测/重测：先把目标算子的旧产物删干净，再用同一 run_id 原地重跑。
    purged_ids = set()
    if args.retry_incomplete:
        purge_plan = [(op, collect_purge_targets(runs_dir, op, args.agent, args.tag)) for op in ops]
        total = sum(len(paths) for _, paths in purge_plan)
        for op, paths in purge_plan:
            if paths:
                log(f"重测 {op}: 待删除 {len(paths)} 项")
                for path in paths:
                    log(f"    - {path}")
        log(f"续测/重测: 队列 {len(ops)} 个算子，其中 {sum(1 for _, p in purge_plan if p)} "
            f"个已有旧产物，共 {total} 项待删除")
        if total:
            if args.ops.strip().lower() in {"all", ""} and len(ops) > 1 and not args.yes:
                log(f"--ops all 配合 --retry-incomplete 会清空整批（{total} 项）。"
                    f"如确需如此请加 --yes，或用 --ops 指定要重测的算子。")
                return 3
            if not args.dry_run:
                ok, confirmed = confirm_purge(args, total, log)
                if not ok:
                    log("已取消：未删除任何产物，未启动任何 run。")
                    return 3
                deleted = []
                for op, paths in purge_plan:
                    if paths:
                        purged_ids.update(purge_run_id_list(runs_dir, op, args.agent, args.tag))
                        deleted.extend(purge_run_artifacts(paths))
                write_retry_audit(runs_dir, args, ops, deleted, confirmed)
                removed = prune_batch_state(runs_dir, args.agent, args.tag, purged_ids)
                # 必须重新读：state 是从内存里的 previous 重建的，不刷新就会在 spawn 时
                # 把刚剔掉的陈旧行又写回文件，让 live 守卫继续误判。
                previous = load_batch_state(runs_dir, args.agent, args.tag)
                log(f"已删除 {len(deleted)} 项产物，清理批次状态行 {removed or 0} 条"
                    f"（审计: {runs_dir / '.retry-audit.jsonl'}）")
        else:
            log("没有已存在的产物需要删除，按全新 run 启动")

    jobs, skipped = [], []
    for op in ops:
        if args.retry_incomplete:
            # 旧产物已删（或 dry-run 下视为将删），按 base run_id 原地重跑：
            # 不走 allocate_run_id，避免它把 done 判成"跳过"、或把残留判成 -r2。
            jobs.append((op, f"{op}-{args.agent}-{args.tag}"))
            continue
        run_id = allocate_run_id(runs_dir, op, args.agent, args.tag, False)
        if run_id is None:
            skipped.append((op, "已完成"))
        elif run_id == "INCOMPLETE":
            skipped.append((op, "输出目录已存在但未完成（加 --retry-incomplete 删旧产物后原地重跑）"))
        else:
            jobs.append((op, run_id))
    for op, reason in skipped:
        log(f"skip {op}: {reason}")
    log(f"队列: 本批执行 {len(jobs)} 个算子，跳过 {len(skipped)} 个")
    if not jobs:
        log("没有需要启动的算子")
        return 0

    def build_command(op, run_id):
        """构造一个 run 的启动命令（首次与 run 级重试共用，保证参数完全一致）。"""
        command = [sys.executable, str(BRIDGE_DIR / "op_manager.py"), "run",
                   "--op", op, "--agent", args.agent, "--run-id", run_id,
                   "--runs-dir", str(runs_dir), "--harness", args.harness]
        if getattr(args, "model", None):
            command += ["--model", args.model]
        if getattr(args, "model_url", None):
            command += ["--model-url", args.model_url]
        if getattr(args, "model_name", None):
            command += ["--model-name", args.model_name]
        if args.harness == "bare" and getattr(args, "bare_model", None):
            command += ["--bare-model", args.bare_model]
        if args.candidate_evaluations_max is not None:
            command += ["--candidate-evaluations-max", str(args.candidate_evaluations_max)]
        if args.candidate_solutions_target is not None:
            command += ["--candidate-solutions-target", str(args.candidate_solutions_target)]
        if args.max_model_turns is not None:
            command += ["--max-model-turns", str(args.max_model_turns)]
        if args.final_reprofile_count is not None:
            command += ["--final-reprofile-count", str(args.final_reprofile_count)]
        if args.wall_clock_seconds_max is not None:
            command += ["--wall-clock-seconds", str(args.wall_clock_seconds_max)]
        if args.agent_timeout_seconds is not None:
            command += ["--agent-timeout-seconds", str(args.agent_timeout_seconds)]
        if args.profiling_runs is not None:
            command += ["--profiling-runs", str(args.profiling_runs)]
        if args.warmup_runs is not None:
            command += ["--warmup-runs", str(args.warmup_runs)]
        return command

    # build_command 已经带上全部覆盖参数，这里不要再追加一遍（历史上重复 append 过，
    # 导致首启 argv 与 requeue_infra 重试路径的 argv 不一致，--dry-run 打印的也不是真实命令）。
    commands = {run_id: build_command(op, run_id) for op, run_id in jobs}

    if args.dry_run:
        log("dry-run: 以下命令不会真正执行")
        for op, run_id in jobs:
            log("  " + " ".join(commands[run_id]))
            log(f"    log: {runs_dir / (run_id + '.launch.log')}")
        return 0

    env = os.environ.copy()
    if args.sandbox_url:
        env["SANDBOX_URL"] = args.sandbox_url
    # 让子进程的环境变量与下发到 config 的协议保持一致（评测器以请求协议为准，env 只作回退）
    env["SANDBOX_PROFILING_RUNS"] = str(effective["profiling_runs"])
    env["SANDBOX_WARMUP_RUNS"] = str(effective["warmup_runs"])
    planned_ids = [row["run_id"] for row in (previous or {}).get("runs", [])] + \
        [run_id for _, run_id in jobs]
    state = {"agent": args.agent, "harness": args.harness, "tag": args.tag,
             "scheduler_pid": os.getpid(),
             "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
             "parallel": args.parallel, "planned": list(dict.fromkeys(planned_ids)),
             "runs": list((previous or {}).get("runs", []))}
    # pending 元素为 (op, run_id, not_before)：not_before 是 monotonic 秒，
    # 用于可重试失败（基础设施故障 / agent 协议错误）后「先缓一缓再重跑」的延时重排。
    pending = [(op, run_id, 0.0) for op, run_id in jobs]
    running = {}  # pid -> dict
    total = len(jobs)
    attempts = {op: 0 for op, _ in jobs}  # op -> 已用掉的 run 级重试次数
    started_at = time.monotonic()
    stats = {"ok": 0, "fail": 0, "requeue": 0, "durations": []}
    last_progress = {"at": started_at}

    def show_progress(prefix="进度: "):
        done = stats["ok"] + stats["fail"]
        elapsed = time.monotonic() - started_at
        eta = "-"
        if stats["durations"]:
            average = sum(stats["durations"]) / len(stats["durations"])
            eta = fmt_duration(average * max(0, total - done) / max(1, args.parallel))
        last_progress["at"] = time.monotonic()
        percent = (done / total * 100) if total else 100.0
        extra = f" 重排 {stats['requeue']}" if stats["requeue"] else ""
        log(f"{prefix}{progress_bar(done, total)} {done}/{total} ({percent:5.1f}%) "
            f"| 完成 {stats['ok']} 失败 {stats['fail']}{extra} 运行 {len(running)} 排队 {len(pending)} "
            f"| 已用 {fmt_duration(elapsed)} | ETA {eta}")

    def spawn(op, run_id):
        log_path = runs_dir / f"{run_id}.launch.log"
        handle = open(log_path, "ab")
        process = subprocess.Popen(
            commands[run_id], cwd=str(BRIDGE_DIR), env=env,
            stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT,
            start_new_session=True)
        (runs_dir / f"{run_id}.pid").write_text(str(process.pid), encoding="utf-8")
        row = {"op": op, "run_id": run_id, "pid": process.pid, "log": str(log_path),
               "attempt": attempts.get(op, 0)}
        state["runs"].append(row)
        save_batch_state(runs_dir, args.agent, args.tag, state)
        running[process.pid] = {"process": process, "handle": handle,
                                "started": time.monotonic(), **row}
        log(f"start {op} -> {run_id} pid={process.pid} log={log_path.name}")

    def requeue_infra(op, run_id, reason):
        """可重试失败（基础设施故障 / agent 协议错误）-> 换 run-id 延时重排；次数用尽返回 False。"""
        if args.max_run_retries <= 0:
            return False
        if attempts.get(op, 0) >= args.max_run_retries:
            log(f"retry {op} -> {run_id} 已达上限 {args.max_run_retries} 次，不再重排（{reason}）")
            return False
        attempts[op] = attempts.get(op, 0) + 1
        new_id = next_retry_run_id(runs_dir, run_id)
        commands[new_id] = build_command(op, new_id)
        pending.append((op, new_id, time.monotonic() + max(0.0, args.run_retry_delay)))
        stats["requeue"] += 1
        delay_note = f"，{args.run_retry_delay:.0f}s 后启动" if args.run_retry_delay > 0 else ""
        log(f"requeue {op} -> {new_id} [{attempts[op]}/{args.max_run_retries}]{delay_note}（{reason}）")
        return True

    def take_ready_pending(now):
        """取一个到期的排队项；未到期的不阻塞其它可运行项。"""
        for index, item in enumerate(pending):
            if item[2] <= now:
                return pending.pop(index)
        return None

    show_progress()
    while pending or running:
        while len(running) < args.parallel:
            item = take_ready_pending(time.monotonic())
            if item is None:
                break
            spawn(item[0], item[1])
        time.sleep(POLL_SECONDS)
        for pid in list(running):
            process = running[pid]["process"]
            code = process.poll()
            if code is None:
                continue
            running[pid]["handle"].close()
            row = running.pop(pid)
            duration = time.monotonic() - row.get("started", time.monotonic())
            stats["durations"].append(duration)
            log(f"exit {row['op']} -> {row['run_id']} rc={code} 用时={fmt_duration(duration)}")
            # 退出码 0 只表示"run 正常收尾"，不代表算子有提升；基础设施故障可能两种码都能出现，
            # 所以按产物判定，再看是否需要换时间/换设备重跑。
            # 除基础设施故障外，agent 侧协议错误（会话历史被污染的 400 → turn.failed）同样重排。
            infra_reason = infra_failure_reason(runs_dir, row["run_id"])
            agent_reason = None if infra_reason else agent_protocol_failure_reason(runs_dir, row["run_id"])
            reason = infra_reason or agent_reason
            if reason and requeue_infra(row["op"], row["run_id"], reason):
                total += 1
            else:
                stats["ok" if code == 0 else "fail"] += 1
                if reason:
                    kind = "基础设施故障" if infra_reason else "agent 协议错误"
                    log(f"  {row['run_id']} 因{kind}失败且不再重排（{reason}）")
            show_progress()
        if time.monotonic() - last_progress["at"] >= PROGRESS_HEARTBEAT_SECONDS:
            show_progress("心跳: ")
    show_progress("完成: ")
    extra = f" 重排 {stats['requeue']}" if stats["requeue"] else ""
    log(f"batch done: 完成 {stats['ok']} 失败 {stats['fail']}{extra} "
        f"用时 {fmt_duration(time.monotonic() - started_at)}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="批量后台启动算子优化评测（并发受控，断线可存活）",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG))
    parser.add_argument("--ops", default="all", help="all 或逗号分隔的算子名（默认 all）")
    parser.add_argument("--agent", default=None,
                        choices=["claude", "codex", "opencode", "bare"],
                        help="bare 只用于 --harness bare；缺省时按 harness 推断")
    parser.add_argument("--model", dest="model",
                        help="本批次使用的模型（op_catalog.models 的键，如 qwen3.5-122b-a10b / glm-5.3）；"
                             "不传则沿用各 agent 的全局配置")
    parser.add_argument("--model-url", dest="model_url",
                        help="直接指定 OpenAI 兼容端点覆盖注册表（如 http://127.0.0.1:8008/v1）")
    parser.add_argument("--model-name", dest="model_name",
                        help="直接指定 served model name 覆盖注册表")
    parser.add_argument("--bare-model", dest="bare_model",
                        help="--model 的别名（兼容旧用法）；--harness bare 的裸模型名")
    parser.add_argument("--harness", default="native", choices=list(op_manager.HARNESS_KINDS),
                        help="native=native_bench harness（原生 CLI，无 token/墙钟/轮数上限，"
                             "逐候选 token 由完成时模型 usage 计数）；bare=k-lens-v2 bare_client（纯模型 + 工具池）")
    parser.add_argument("--tag", default=time.strftime("%Y%m%d-%H%M%S"),
                        help="批次标识，run-id = <op>-<agent>-<tag>（默认当前时间）")
    parser.add_argument("--parallel", type=int, default=4, help="并发 run 数（默认 4，对应 4 个 NPU 节点）")
    parser.add_argument("--candidate-evaluations-max", dest="candidate_evaluations_max", type=int)
    parser.add_argument("--candidate-solutions-target", dest="candidate_solutions_target", type=int,
                        help="native 软目标：希望 agent 提交的实质不同方案数（默认=硬上限；需 v2+ harness）")
    parser.add_argument("--max-model-turns", dest="max_model_turns", type=int)
    parser.add_argument("--final-reprofile-count", dest="final_reprofile_count", type=int)
    parser.add_argument("--wall-clock-seconds", "--wall-clock-seconds-max",
                        dest="wall_clock_seconds_max", type=int, metavar="SECONDS",
                        help="每个 run 的搜索墙钟上限（秒），覆盖 op_catalog 默认值；0=关闭限制")
    parser.add_argument("--agent-timeout-seconds", dest="agent_timeout_seconds", type=int,
                        metavar="SECONDS",
                        help="所选 agent 的单轮模型调用超时（秒），覆盖 op_catalog 的 agents.<agent>.timeout_seconds")
    parser.add_argument("--max-run-retries", dest="max_run_retries", type=int, default=2,
                        metavar="N",
                        help="run 因基础设施故障（沙盒/设备故障）或 agent 协议错误（上游 400 → turn.failed）失败后，自动换 -rN 重跑的次数上限"
                             "（默认 2；0=关闭 run 级重试）。重跑的 run 只在拿到真实判定后才计入结果")
    parser.add_argument("--run-retry-delay", dest="run_retry_delay", type=float, default=120.0,
                        metavar="SECONDS",
                        help="run 级重试前等待秒数（默认 120），给故障设备留出恢复时间")
    parser.add_argument("--profiling-runs", dest="profiling_runs", type=int)
    parser.add_argument("--warmup-runs", dest="warmup_runs", type=int,
                        help="profiling 预热轮数（不含在 --profiling-runs 内）；"
                             "默认取 SANDBOX_WARMUP_RUNS，缺省 3")
    parser.add_argument("--sandbox-url", dest="sandbox_url", help="覆盖 SANDBOX_URL，默认 http://127.0.0.1:9000")
    parser.add_argument("--runs-dir", dest="runs_dir", help="覆盖 runs 目录（默认取 op_catalog.outputs.runs_dir）")
    parser.add_argument("--retry-incomplete", action="store_true",
                        help="续测/重测 --ops 指定的算子：先删除其在本 tag 下的全部旧产物"
                             "（run 目录 + .json/.pid/.launch.log，含历史 -rN 变体），"
                             "再用同一 run-id 原地重跑。删除前会列出清单并要求确认；"
                             "本 tag 仍有 run 在运行时拒绝执行（需先 --stop），--force 不生效")
    parser.add_argument("--yes", action="store_true",
                        help="跳过 --retry-incomplete 的删除确认（非交互环境必需；"
                             "配合 --ops all 时也是强制要求）")
    parser.add_argument("--force", action="store_true",
                        help="忽略同 tag 仍在运行的检查（普通启动用；"
                             "配合 --retry-incomplete 无效，仍会拒绝）")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印将执行的命令；配合 --stop 时只列出将收尾的进程")
    parser.add_argument("--status", action="store_true", help="查看该 tag 的进度后退出")
    parser.add_argument("--stop", action="store_true",
                        help="停止该 tag 的调度器与全部 run（含 agent CLI 及其子进程）后退出；"
                             "不依赖 .batch-*.json/*.pid，先写 native STOP 再 SIGTERM→SIGKILL 收尾")
    parser.add_argument("--gc-orphan-agents", action="store_true",
                        help="配合 --stop：连归属不到本批次的孤儿 native agent 一起清理"
                             "（默认只报告，避免误伤其它批次）")
    args = parser.parse_args()
    if args.parallel < 1:
        raise SystemExit("--parallel 必须 >= 1")
    if not args.agent:
        args.agent = "bare" if args.harness == "bare" else "claude"
    if args.harness == "bare" and args.agent != "bare":
        raise SystemExit(f"--harness bare 只接受 --agent bare（当前 {args.agent}）")
    if args.agent == "bare" and args.harness != "bare":
        raise SystemExit(f"--agent bare 只能配 --harness bare（当前 harness={args.harness}）")
    if args.agent == "opencode" and args.harness != "native":
        raise SystemExit(f"--agent opencode 只能配 --harness native（当前 harness={args.harness}）")
    catalog, names = resolved_ops(args.catalog)
    ops = select_ops(args.ops, names)
    if args.status:
        return cmd_status(args, catalog, ops)
    if args.stop:
        return cmd_stop(args, catalog, ops)
    return cmd_launch(args, catalog, ops)


if __name__ == "__main__":
    sys.exit(main())
