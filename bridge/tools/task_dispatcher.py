#!/usr/bin/env python3
"""任务提交调度器：等指定 tag 的批次跑完，立刻串行提交下一条命令。

用途：把"等上一批评测结束 → 换模型/换算子 → 启动下一批"这类链条交给队列，
不用人守着 `--status`。队列持久化在磁盘上，daemon 重启后从状态文件继续，
不会重复启动已经在跑的作业。

    # 1) 提交（可一次提交多条；也可多次 submit 追加）
    python3 tools/task_dispatcher.py submit --queue main --label next-batch \\
        --after-tag <UPSTREAM-TAG> --after-agent codex \\
        --cmd "python3 start_all_runs.py --agent claude --harness native --tag claude-next-001 --ops all --parallel 4"

    # 2) 起 daemon（后台常驻，自己 daemon 化）
    python3 tools/task_dispatcher.py run --queue main --daemon

    # 3) 看队列 / 停 daemon
    python3 tools/task_dispatcher.py list --queue main
    python3 tools/task_dispatcher.py stop --queue main

语义（三条口径，务必先看）：

* **完成判定**：依赖批次里 `running == 0 且 queued == 0`，连续 2 次轮询都成立即放行。
  等价于"每个 run 都落盘了 result.json / 没有 run 还会被启动"。注意本仓库里 infra
  故障的 run 也会写 result.json（写出后永远算完成），所以**放行不代表上游是成功的**：
  放行时会把上游每个 run 的 `success` / `termination_reason` 汇总写进队列状态与日志。
* **串行**：同一队列内一次只跑一个作业（严格 FIFO：队首依赖没满足时后面等着）。
  不限制其它批次同时在跑——不同批次本来就是并存的。
* **命令**：`--cmd` 是交给 `bash -lc` 的整条 shell 命令，退出码取最后一条命令的
  退出码（管道里就是最后一个命令的）。要 `pipefail` 请自己写在命令里。
* **禁止复用依赖的 tag**（submit 默认直接拒绝，`--allow-tag-reuse` 可放行）：
  `start_all_runs.py` 的批次键是 `(tag, agent)`，**只有同 agent 复用 tag 才会被拒（rc=2）**；
  换一个 agent 用同一个 tag 会被当成另一个批次、写进同一个批次目录（`.batch-<agent>-<tag>.json`
  并存），于是两条链并行跑同一批算子名。2026-09-23 就是这么误起过一个 claude 批次。

产物（默认 `--state-dir <runs_dir>/.dispatcher/`）：

    <state-dir>/<queue>.json          队列状态（原子写，唯一事实来源）
    <state-dir>/<queue>.jsonl         状态迁移审计（追加写）
    <state-dir>/<queue>.daemon.log    daemon 日志（依赖轮询、放行、退出）
    <state-dir>/<queue>.daemon.pid    daemon pid（用进程身份校验，NFS 丢文件也不误判）
    <state-dir>/<queue>/<id>-<label>.log   每个作业的 stdout/stderr

退出码：run 0=正常（--once 跑无可跑 / daemon 收 SIGTERM）；2=已有 daemon 在跑；3=队列不存在。
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
BRIDGE = TOOLS.parent
sys.path.insert(0, str(BRIDGE))                  # start_all_runs / op_manager
sys.path.insert(0, str(TOOLS / "monitor"))       # native_monitor：resolve_batch / detect_agent
import native_monitor as M  # noqa: E402
import start_all_runs as S  # noqa: E402

STATE_DIR_NAME = ".dispatcher"
POLL_DEFAULT = 60.0           # 依赖轮询间隔（秒）；批次以小时计，60s 足够"立刻"
STABLE_HITS = 2               # 连续多少次"没有在跑的 run"才判完成（与 watch_batch_then_launch.sh 一致）
MAX_WAIT_DEFAULT = 48 * 3600  # 单条依赖最多等多久，超时把作业标 skipped，不永远堵队列
HEARTBEAT_SECONDS = 300.0     # 依赖没变化时的心跳日志间隔
TERMINAL_STATES = ("done", "failed", "skipped", "cancelled")
UPSTREAM_INFRA_REASONS = ("evaluation_state_unknown", "infrastructure_error")


def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def die(message, code=1):
    """带退出码的错误：2=已有 daemon 在跑（对照 start_all_runs.py 的"仍有 run 在跑"），3=队列不存在。"""
    print(message, file=sys.stderr)
    raise SystemExit(code)


# ---------------------------------------------------------------------------
# 队列状态
# ---------------------------------------------------------------------------
def catalog_defaults(catalog_path):
    """返回 (runs_dir, op 名列表)：与调度器/监控同源，取 op_catalog 的输出目录。"""
    catalog = S.read_json(catalog_path)
    runs_dir = Path(catalog["outputs"]["runs_dir"])
    ops = []
    resolved = Path(catalog_path).resolve().parent / "op_catalog.resolved.json"
    for candidate in (resolved, Path(catalog_path)):
        if not candidate.is_file():
            continue
        try:
            payload = S.read_json(candidate)
        except (OSError, ValueError):
            continue
        ops = [entry["name"] for entry in payload.get("ops", []) if entry.get("name")]
        if ops:
            break
    return runs_dir, ops


def state_dir_for(args, runs_dir):
    return Path(args.state_dir) if args.state_dir else runs_dir / STATE_DIR_NAME


def queue_path(state_dir, queue):
    return state_dir / f"{queue}.json"


def load_queue(state_dir, queue):
    path = queue_path(state_dir, queue)
    return S.read_json(path) if path.is_file() else None


def merge_foreign_jobs(state_dir, queue, payload):
    """把别的进程（submit / cancel）写进队列文件的改动并进来。

    daemon 是常驻进程、手里有一份内存副本，每轮都会把整个队列写回文件；如果不先合并，
    一次外部 `submit` 会被下一轮 save 悄悄覆盖（实测踩过：job 2 提交成功但被 daemon 抹掉）。
    """
    disk = load_queue(state_dir, queue)
    if disk is None:
        return []
    known = {job["id"]: job for job in payload["jobs"]}
    added = []
    for job in disk.get("jobs", []):
        current = known.get(job.get("id"))
        if current is None:
            payload["jobs"].append(job)
            added.append(job["id"])
        elif current["state"] in ("pending", "waiting") and job.get("state") in TERMINAL_STATES:
            current.update(job)  # 外部 cancel：daemon 还没推进它就采纳
    if added:
        payload["jobs"].sort(key=lambda job: job["id"])
    return added


def save_queue(state_dir, queue, payload):
    merge_foreign_jobs(state_dir, queue, payload)
    payload["updated"] = now_iso()
    state_dir.mkdir(parents=True, exist_ok=True)
    S.write_json(queue_path(state_dir, queue), payload)


def audit(state_dir, queue, event, **fields):
    """追加一行审计：状态迁移/放行/退出都留痕，便于事后复盘。"""
    state_dir.mkdir(parents=True, exist_ok=True)
    row = {"ts": now_iso(), "event": event, **fields}
    with (state_dir / f"{queue}.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def daemon_log(state_dir, queue, message):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    path = state_dir / f"{queue}.daemon.log"
    if stdout_is(path):
        return  # 后台模式下 daemon 的 stdout 已经重定向到这个文件，别再写一遍
    try:
        state_dir.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:
        pass


def stdout_is(path):
    """stdout 是否已经指向这个文件（后台 daemon 的日志重定向场景）。"""
    try:
        fd, target = os.fstat(1), os.stat(path)
    except OSError:
        return False
    return (fd.st_dev, fd.st_ino) == (target.st_dev, target.st_ino)


# ---------------------------------------------------------------------------
# 依赖闸门
# ---------------------------------------------------------------------------
def gate_check(dep, runs_dir, ops):
    """查一条依赖的状态。返回 {"status": "done"|"waiting"|"notfound", "detail", ...}。

    复用 start_all_runs.batch_rows（--status 的同一套口径：批次状态文件 + run 目录 +
    进程身份兜底），所以"队列什么时候放行"和"人敲 --status 看到什么"永远一致。
    """
    tag = dep.get("tag")
    batch_dir = M.resolve_batch(runs_dir, tag)
    if batch_dir is None:
        return {"status": "notfound",
                "detail": f"找不到 tag={tag} 的批次（{runs_dir}/{tag}/ 不存在，根目录也没有它的痕迹）"}
    agent = dep.get("agent") or M.detect_agent(batch_dir, tag)
    if not agent:
        return {"status": "notfound",
                "detail": f"无法判定 tag={tag} 的 agent（缺 .batch-<agent>-{tag}.json，run 目录名里也没有 agent）"}
    data = S.batch_rows(batch_dir, agent, tag, ops)
    if data is None:
        return {"status": "notfound",
                "detail": f"批次目录 {batch_dir} 里没有任何 run 记录（agent={agent}）"}
    counts = data["counts"]
    never = [run_id for run_id, state_name, _ in data["rows"] if state_name in ("stopped", "missing")]
    # 没有 result.json 又不会再启动的 run（stopped/missing）：调度器还活着时继续等它
    # （可能正处在"刚 spawn、目录还没建"的窗口里），调度器已经退了就说明这批永远不会再产出，
    # 不阻塞放行，但要如实标出来。
    stalled = never if data["live_sched"] else []
    pending = counts["running"] + counts["queued"] + len(stalled)
    detail = (f"agent={agent} 完成 {counts['finished']}/{data['total']} "
              f"运行 {counts['running']} 待启动 {counts['queued']} 中断 {counts['stopped']}")
    if stalled:
        detail += f" | {len(stalled)} 个 run 无 result.json 且调度器仍在（继续等）"
    elif never:
        detail += f" | 无 result.json 的 run {len(never)} 个（调度器已退出，不会再产出）"
    if data["live_sched"]:
        detail += f" | 调度器 pid={data['live_sched']}"
    return {"status": "waiting" if pending else "done", "detail": detail,
            "batch_dir": str(batch_dir), "agent": agent, "total": data["total"],
            "never": never, "rows": data["rows"]}


def upstream_verdict(gates):
    """放行时汇总上游批次的实际结果：成功/失败/infra 终止/从未产出（多依赖时合并统计）。"""
    ok = failed = infra = runs = never_produced = 0
    reasons = Counter()
    for gate in gates:
        batch_dir = Path(gate["batch_dir"])
        runs += len(gate.get("rows", []))
        never_produced += len(gate.get("never", []))
        for run_id, _state_name, _detail in gate.get("rows", []):
            path = batch_dir / run_id / "result.json"
            if not path.is_file():
                continue
            try:
                data = S.read_json(path)
            except (OSError, ValueError):
                continue
            reasons[str(data.get("termination_reason"))] += 1
            if data.get("success") is True:
                ok += 1
            elif data.get("success") is False:
                failed += 1
            if data.get("termination_reason") in UPSTREAM_INFRA_REASONS:
                infra += 1
    return {"runs": runs, "success": ok, "failed": failed, "infra_terminated": infra,
            "never_produced": never_produced, "termination_reasons": dict(reasons)}


# ---------------------------------------------------------------------------
# 作业启动 / 存活判定
# ---------------------------------------------------------------------------
def safe_label(label):
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(label))[:48] or "job"


def start_job(job, state_dir, queue):
    """串行启动一个作业：独立会话 + 日志重定向，daemon 退出不影响它。"""
    job_dir = state_dir / queue
    job_dir.mkdir(parents=True, exist_ok=True)
    log_path = job_dir / f"{job['id']:03d}-{safe_label(job['label'])}.log"
    env = dict(os.environ)
    env.update(job.get("env") or {})
    cwd = job.get("cwd") or str(BRIDGE)
    with log_path.open("ab") as log:
        log.write(f"\n==== job {job['id']} ({job['label']}) start {now_iso()} cwd={cwd}\n"
                  f"$ {job['cmd']}\n".encode())
        log.flush()
        proc = subprocess.Popen(["/bin/bash", "-lc", job["cmd"]], cwd=cwd, env=env,
                                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
    return proc, log_path


def job_alive(job):
    """作业进程是否还活着：/proc 存活 + cmdline 特征，避免 pid 复用误判。"""
    pid = job.get("pid")
    if not isinstance(pid, int) or not S.proc_alive(pid):
        return False
    cmdline = S.proc_cmdline(pid)
    if not cmdline:
        return True  # 进程在但读不到 cmdline（容器权限）：保守认为还活着
    probe = " ".join((job.get("cmd") or "").split()[:3])
    return not probe or probe in cmdline or "bash" in cmdline


def reap_child(pid):
    """非阻塞回收直接子进程拿退出码；daemon 重启后接管的作业拿不到（返回 None）。"""
    if not isinstance(pid, int):
        return None
    try:
        done, status = os.waitpid(pid, os.WNOHANG)
    except (ChildProcessError, OSError):
        return None
    return os.waitstatus_to_exitcode(status) if done else None


def rc_hint(cmd, rc):
    """对 start_all_runs.py 的退出码给出人话解释（rc 本身的含义容易误读）。"""
    if "start_all_runs.py" not in cmd or rc == 0:
        return ""
    return {2: "（该 tag 仍有 run 在跑，或参数/预检被拒）",
            3: "（purge 取消：--ops all + --retry-incomplete 需要 --yes）"}.get(rc, "")


# ---------------------------------------------------------------------------
# daemon 生命周期
# ---------------------------------------------------------------------------
def daemon_pid_path(state_dir, queue):
    return state_dir / f"{queue}.daemon.pid"


def daemon_pids(state_dir, queue):
    """正在跑本队列的 daemon pid：按 cmdline 校验身份，pid 文件丢了也能认出来。"""
    path = daemon_pid_path(state_dir, queue)
    if not path.is_file():
        return []
    try:
        recorded = [int(x) for x in path.read_text(encoding="utf-8").split()]
    except (OSError, ValueError):
        return []
    alive = []
    for pid in recorded:
        if not S.proc_alive(pid):
            continue
        cmdline = S.proc_cmdline(pid)
        if not cmdline:
            alive.append(pid)  # 读不到 cmdline（容器权限）：保守认为 daemon 还活着
            continue
        if "task_dispatcher.py" in cmdline and f"--queue {queue}" in cmdline:
            alive.append(pid)
    return sorted(set(alive))


def acquire_daemon(state_dir, queue, force):
    live = daemon_pids(state_dir, queue)
    if live and not force:
        return None, live
    state_dir.mkdir(parents=True, exist_ok=True)
    daemon_pid_path(state_dir, queue).write_text(f"{os.getpid()}\n", encoding="utf-8")
    return os.getpid(), live


def release_daemon(state_dir, queue):
    path = daemon_pid_path(state_dir, queue)
    try:
        if path.read_text(encoding="utf-8").strip() == str(os.getpid()):
            path.unlink()
    except OSError:
        pass


def recover_running_jobs(payload, state_dir, queue):
    """daemon 重启后的收尾：还在跑的作业继续接管，已退出又没结论的标 failed。"""
    for job in payload["jobs"]:
        if job.get("state") != "running":
            continue
        if job_alive(job):
            daemon_log(state_dir, queue,
                       f"job {job['id']} ({job['label']}) 仍在运行 pid={job['pid']}，继续接管")
            continue
        job["state"] = "failed"
        job["ended"] = now_iso()
        job["note"] = "daemon 重启时作业已退出且未记录结论：退出码不可知（不是它父进程），请查作业日志"
        audit(state_dir, queue, "job_interrupted", id=job["id"], label=job["label"])
        daemon_log(state_dir, queue, f"job {job['id']} ({job['label']}) 已被中断，标记 failed")


def current_state(payload):
    for job in payload["jobs"]:
        if job["state"] == "running":
            return f"running job {job['id']}"
        if job["state"] in ("pending", "waiting"):
            return f"{job['state']} job {job['id']}"
    return "空闲"


def elapsed_text(job):
    started = job.get("started_epoch")
    if not isinstance(started, (int, float)):
        return "?"
    secs = max(0, int(time.time() - started))
    return f"{secs // 3600}:{(secs % 3600) // 60:02d}:{secs % 60:02d}"


def waiting_seconds(dep):
    started = dep.get("wait_started_epoch")
    return 0.0 if not isinstance(started, (int, float)) else max(0.0, time.time() - started)


def step(payload, state_dir, queue, runs_dir, ops):
    """推进一步：先收尾运行中的作业，否则尝试启动队首。

    返回"还有事情在做"（作业在跑 / 刚启动 / 刚收尾），--once 靠它决定何时退出：
    只有队首依赖没满足、且没有作业在跑时才返回 False。
    """
    running = [job for job in payload["jobs"] if job["state"] == "running"]
    if running:
        job = running[0]
        if job_alive(job):
            return True
        rc = reap_child(job.get("pid"))
        if rc is None and job.get("returncode") is not None:
            rc = job["returncode"]
        job["returncode"] = rc
        job["ended"] = now_iso()
        job["state"] = "done" if rc == 0 else "failed"
        if rc is None:
            job["note"] = "退出码不可知（作业不是本 daemon 的子进程），按 failed 处理"
        audit(state_dir, queue, "job_" + job["state"], id=job["id"], label=job["label"], returncode=rc)
        daemon_log(state_dir, queue, f"job {job['id']} ({job['label']}) 结束 rc={rc}"
                                     f"{rc_hint(job['cmd'], rc)} 用时 {elapsed_text(job)}")
        return True

    head = next((job for job in payload["jobs"] if job["state"] in ("pending", "waiting")), None)
    if head is None:
        return False

    gates = {}
    blocked = None
    for dep in head.get("after") or []:
        gate = gate_check(dep, runs_dir, ops)
        gates[dep["tag"]] = gate
        dep["status"] = gate["status"]
        dep["detail"] = gate["detail"]
        if gate["status"] == "done":
            dep["hits"] = dep.get("hits", 0) + 1
            if dep["hits"] < STABLE_HITS and blocked is None:
                blocked = (dep, gate, f"已无在跑的 run，再确认 {STABLE_HITS - dep['hits']} 次后放行")
            continue
        dep["hits"] = 0
        if not isinstance(dep.get("wait_started_epoch"), (int, float)):
            dep["wait_started_epoch"] = time.time()
        elif waiting_seconds(dep) > dep.get("max_wait_sec", MAX_WAIT_DEFAULT):
            head["state"] = "skipped"
            head["ended"] = now_iso()
            head["note"] = (f"依赖 tag={dep['tag']} 等 {waiting_seconds(dep) / 3600:.1f}h 超时："
                            f"{gate['detail']}")
            audit(state_dir, queue, "job_skipped", id=head["id"], label=head["label"],
                  reason=head["note"])
            daemon_log(state_dir, queue, f"job {head['id']} ({head['label']}) 依赖超时，跳过")
            return True
        if blocked is None:
            blocked = (dep, gate, None)

    if blocked is not None:
        dep, gate, tail = blocked
        head["state"] = "waiting"
        head["note"] = f"等 tag={dep['tag']}：{tail + '；' if tail else ''}{gate['detail']}"
        if dep.get("last_logged") != gate["detail"]:
            dep["last_logged"] = gate["detail"]
            daemon_log(state_dir, queue, f"job {head['id']} ({head['label']}) {head['note']}")
        return False

    verdict = None
    if gates:
        verdict = upstream_verdict(list(gates.values()))
        daemon_log(state_dir, queue,
                   f"依赖满足，放行 job {head['id']} ({head['label']})：上游 {verdict['runs']} 个 run，"
                   f"成功 {verdict['success']} / 失败 {verdict['failed']} / "
                   f"infra 终止 {verdict['infra_terminated']} / 未产出 {verdict['never_produced']}；"
                   f"终止原因 {verdict['termination_reasons']}")

    if head.get("dry_run"):
        head["state"] = "skipped"
        head["ended"] = now_iso()
        head["note"] = "dry-run：依赖已满足，未真正执行"
        head["gate_verdict"] = verdict
        audit(state_dir, queue, "job_dry_run", id=head["id"], label=head["label"], cmd=head["cmd"])
        daemon_log(state_dir, queue, f"[dry-run] 到点了，将执行：{head['cmd']}")
        return True

    proc, log_path = start_job(head, state_dir, queue)
    head.update({"state": "running", "pid": proc.pid, "started": now_iso(),
                 "started_epoch": time.time(), "log": str(log_path), "gate_verdict": verdict})
    audit(state_dir, queue, "job_started", id=head["id"], label=head["label"], pid=proc.pid,
          cmd=head["cmd"], gate_verdict=verdict)
    daemon_log(state_dir, queue, f"启动 job {head['id']} ({head['label']}) pid={proc.pid} log={log_path}")
    return True


def main_loop(state_dir, queue, payload, runs_dir, ops, args):
    daemon_log(state_dir, queue, f"daemon 启动 pid={os.getpid()} queue={queue} "
                                 f"interval={args.interval}s runs_dir={runs_dir}")
    last_heartbeat = time.monotonic()
    stop = {"flag": False}

    def _term(signum, frame):
        stop["flag"] = True
        daemon_log(state_dir, queue, f"收到信号 {signum}：daemon 退出，已启动的作业继续运行")

    signal.signal(signal.SIGTERM, _term)
    signal.signal(signal.SIGINT, _term)

    while not stop["flag"]:
        added = merge_foreign_jobs(state_dir, queue, payload)
        if added:
            daemon_log(state_dir, queue, f"从队列文件并入新作业 {added}")
        progressed = step(payload, state_dir, queue, runs_dir, ops)
        save_queue(state_dir, queue, payload)
        if args.once and not progressed:
            daemon_log(state_dir, queue, "--once：没有可推进的作业，退出")
            return 0
        if not args.once and time.monotonic() - last_heartbeat > HEARTBEAT_SECONDS:
            done = sum(1 for job in payload["jobs"] if job["state"] in TERMINAL_STATES)
            daemon_log(state_dir, queue, f"心跳: 共 {len(payload['jobs'])} 个作业，已结束 {done}，"
                                         f"当前 {current_state(payload)}")
            last_heartbeat = time.monotonic()
        slept = 0.0
        while slept < args.interval and not stop["flag"]:
            tick = min(1.0, args.interval - slept)
            time.sleep(tick)
            slept += tick
    return 0


# ---------------------------------------------------------------------------
# 子命令
# ---------------------------------------------------------------------------
def parse_env(pairs):
    env = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise SystemExit(f"--env 需要 KEY=VALUE 形式：{pair!r}")
        key, value = pair.split("=", 1)
        env[key] = value
    return env


def normalize_deps(after):
    """依赖可以是 "tag" 字符串或 {tag, agent, poll_sec, max_wait_sec} 字典。"""
    deps = []
    for dep in after:
        if isinstance(dep, str):
            deps.append({"tag": dep})
        elif isinstance(dep, dict) and dep.get("tag"):
            deps.append(dict(dep))
        else:
            raise SystemExit(f"依赖格式不对（需要 tag 字符串或含 tag 的对象）：{dep!r}")
    for dep in deps:
        dep.setdefault("agent", None)
        dep.setdefault("max_wait_sec", MAX_WAIT_DEFAULT)
    return deps


def cmd_submit(args, runs_dir, ops):
    state_dir = state_dir_for(args, runs_dir)
    payload = load_queue(state_dir, args.queue)
    if payload is None:
        payload = {"queue": args.queue, "created": now_iso(), "runs_dir": str(runs_dir), "jobs": []}
    specs = []
    if args.spec:
        loaded = S.read_json(args.spec)
        specs = loaded if isinstance(loaded, list) else [loaded]
    if args.cmd:
        specs.append({"label": args.label, "cmd": args.cmd, "cwd": args.cwd,
                      "env": parse_env(args.env),
                      "after": [{"tag": tag, "agent": args.after_agent,
                                 "max_wait_sec": args.max_wait} for tag in (args.after_tag or [])],
                      "dry_run": args.dry_run})
    if not specs:
        raise SystemExit("submit 需要 --cmd 或 --spec（两者可同时给，会按顺序入队）")
    next_id = max([job["id"] for job in payload["jobs"]], default=0) + 1
    # 先全部校验（安全闸门/必填项），再统一入队与打印：避免"已加入队列"喊完才发现被拒。
    prepared = []
    for spec in specs:
        if not spec.get("cmd"):
            raise SystemExit("每个作业都必须有 cmd")
        job = {"id": next_id, "label": spec.get("label") or f"job-{next_id}",
               "cmd": spec["cmd"], "cwd": spec.get("cwd") or str(BRIDGE),
               "env": spec.get("env") or {}, "after": normalize_deps(spec.get("after") or []),
               "state": "pending", "created": now_iso(), "pid": None, "returncode": None,
               "note": "", "dry_run": bool(spec.get("dry_run"))}
        check_cmd_safety(job, args)
        prepared.append(job)
        next_id += 1
    for job in prepared:
        payload["jobs"].append(job)
        audit(state_dir, args.queue, "job_submitted", id=job["id"], label=job["label"],
              cmd=job["cmd"], after=[dep["tag"] for dep in job["after"]])
        print(f"已加入队列 {args.queue}: job {job['id']} ({job['label']}) "
              + (f"依赖 tag={[dep['tag'] for dep in job['after']]}" if job["after"] else "无依赖，立即执行")
              + (" [dry-run]" if job["dry_run"] else ""))
    save_queue(state_dir, args.queue, payload)
    return 0


def check_cmd_safety(job, args):
    """提交时拦两类会真出事的命令：复用依赖 tag、以及漏了 --sandbox-url 踩到坏节点。"""
    for dep in job["after"]:
        if S.cmd_has_tag(job["cmd"], dep["tag"]) and not args.allow_tag_reuse:
            die(f"命令里复用了依赖的 tag（--tag {dep['tag']}）。\n"
                f"  批次键是 (tag, agent)：同 agent 复用会被 start_all_runs.py 拒绝（rc=2），"
                f"换个 agent 复用则会**另起一个批次并写进同一个批次目录**（两条链并行跑同一批算子）。\n"
                f"  新批次请换一个新的 --tag；确实要复用请显式加 --allow-tag-reuse。", 1)
    env = dict(os.environ)
    env.update(job.get("env") or {})
    if ("start_all_runs.py" in job["cmd"] and "--sandbox-url" not in job["cmd"]
            and not env.get("SANDBOX_URL")):
        print("  ⚠ 命令没写 --sandbox-url，环境里也没有 SANDBOX_URL："
              "会落到 bridge 默认沙盒（127.0.0.1:9000，已知对所有作业返回坏结果）。"
              "建议显式带上 --sandbox-url http://127.0.0.1:9000。", file=sys.stderr)


def cmd_run(args, runs_dir, ops):
    state_dir = state_dir_for(args, runs_dir)
    payload = load_queue(state_dir, args.queue)
    if payload is None:
        die(f"队列不存在：{queue_path(state_dir, args.queue)}（先 submit）", 3)
    if args.daemon:
        live = daemon_pids(state_dir, args.queue)
        if live and not args.force:
            die(f"队列 {args.queue} 已有 daemon 在跑（pid={live}），要再起一个请加 --force", 2)
        state_dir.mkdir(parents=True, exist_ok=True)
        argv = [sys.executable, str(Path(__file__).resolve()), "run", "--queue", args.queue,
                "--interval", str(args.interval), "--state-dir", str(state_dir),
                "--runs-dir", str(runs_dir)]
        if args.once:
            argv.append("--once")
        with (state_dir / f"{args.queue}.daemon.log").open("ab") as log:
            proc = subprocess.Popen(argv, cwd=str(BRIDGE), stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(f"daemon 已后台启动 pid={proc.pid} queue={args.queue} "
              f"log={state_dir / (args.queue + '.daemon.log')}")
        return 0
    pid, live = acquire_daemon(state_dir, args.queue, args.force)
    if pid is None:
        die(f"队列 {args.queue} 已有 daemon 在跑（pid={live}），要再起一个请加 --force", 2)
    try:
        recover_running_jobs(payload, state_dir, args.queue)
        return main_loop(state_dir, args.queue, payload, runs_dir, ops, args)
    finally:
        release_daemon(state_dir, args.queue)


def cmd_list(args, runs_dir, ops):
    state_dir = state_dir_for(args, runs_dir)
    payload = load_queue(state_dir, args.queue)
    if payload is None:
        die(f"队列不存在：{queue_path(state_dir, args.queue)}", 3)
    live = daemon_pids(state_dir, args.queue)
    if args.json:
        print(json.dumps({"queue": payload.get("queue"), "daemon_pids": live,
                          "jobs": payload["jobs"]}, ensure_ascii=False, indent=2))
        return 0
    print(f"队列 {args.queue}  runs_dir={payload.get('runs_dir')}  "
          f"daemon={'pid=' + ','.join(map(str, live)) if live else '未运行'}")
    head_id = next((job["id"] for job in payload["jobs"] if job["state"] not in TERMINAL_STATES), None)
    for job in payload["jobs"]:
        print(f"{job['id']:>3}  {job['state']:<9} {job['label'][:24]:<24} "
              f"rc={str(job.get('returncode')):<5} {job['cmd'][:60]}")
        if job["id"] != head_id and job["state"] in ("pending", "waiting") and not any(
                dep.get("status") for dep in job.get("after") or []):
            print("       └ 严格 FIFO：等前面的作业先结束（依赖状态轮到它时才查）")
        for dep in job.get("after") or []:
            mark = f" {dep.get('hits', 0)}/{STABLE_HITS}" if dep.get("status") == "done" else ""
            print(f"       └ 依赖 tag={dep['tag']} [{dep.get('status', '?')}{mark}] {dep.get('detail', '')}")
        if job.get("gate_verdict"):
            verdict = job["gate_verdict"]
            print(f"       └ 上游结果: 成功 {verdict['success']} / 失败 {verdict['failed']} / "
                  f"infra 终止 {verdict['infra_terminated']} / 未产出 {verdict['never_produced']}")
        if job.get("note"):
            print(f"       └ {job['note']}")
        if job.get("log"):
            print(f"       └ log={job['log']}")
    return 0


def kill_job(job):
    """干掉作业进程组：agent/评测器会自己再开 session，所以按进程组尽力而为。"""
    pid = job.get("pid")
    if not isinstance(pid, int):
        return
    pgid = S.pgid_of(pid) or pid
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pgid, sig)
        except OSError:
            try:
                os.kill(pid, sig)
            except OSError:
                return
        for _ in range(20):
            if not S.proc_alive(pid):
                return
            time.sleep(0.5)


def cmd_cancel(args, runs_dir, ops):
    state_dir = state_dir_for(args, runs_dir)
    payload = load_queue(state_dir, args.queue)
    if payload is None:
        die(f"队列不存在：{queue_path(state_dir, args.queue)}", 3)
    wanted = set(args.id or [])
    targets = [job for job in payload["jobs"] if args.all or job["id"] in wanted]
    if not targets:
        raise SystemExit("没有匹配的作业（用 --id N 或 --all）")
    changed = 0
    for job in targets:
        if job["state"] in TERMINAL_STATES:
            print(f"job {job['id']} 已是 {job['state']}，跳过")
            continue
        if job["state"] == "running":
            if not args.force:
                print(f"job {job['id']} 正在运行；要连它一起停请加 --force（SIGTERM 它的进程组）",
                      file=sys.stderr)
                continue
            kill_job(job)
        job["state"] = "cancelled"
        job["ended"] = now_iso()
        job["note"] = "手动取消"
        audit(state_dir, args.queue, "job_cancelled", id=job["id"], label=job["label"],
              forced=bool(args.force))
        print(f"已取消 job {job['id']} ({job['label']})")
        changed += 1
    save_queue(state_dir, args.queue, payload)
    return 0 if changed else 1


def cmd_stop(args, runs_dir, ops):
    """停 daemon（默认不动正在跑的作业，与 start_all_runs.py 的语义一致）。"""
    state_dir = state_dir_for(args, runs_dir)
    live = daemon_pids(state_dir, args.queue)
    if not live:
        print(f"队列 {args.queue} 没有在跑的 daemon")
        return 0
    for pid in live:
        try:
            os.kill(pid, signal.SIGTERM)
            print(f"已向 daemon pid={pid} 发 SIGTERM（正在跑的作业不受影响；"
                  f"要连作业一起停用 cancel --force）")
        except OSError as exc:
            print(f"pid={pid} 发信号失败: {exc}", file=sys.stderr)
    if args.wait:
        deadline = time.monotonic() + args.wait
        while time.monotonic() < deadline and daemon_pids(state_dir, args.queue):
            time.sleep(1.0)
        left = daemon_pids(state_dir, args.queue)
        print("daemon 已退出" if not left else f"仍在退出中: {left}")
    return 0


# ---------------------------------------------------------------------------
def build_parser():
    parser = argparse.ArgumentParser(description="任务提交调度器：等 tag 批次跑完，串行提交下一条命令")
    # 全局参数在子命令前后都能写：子命令侧用 SUPPRESS 默认值，避免它把主命令解析出的值盖掉。
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--catalog", default=argparse.SUPPRESS)
    common.add_argument("--runs-dir", default=argparse.SUPPRESS, help="覆盖 op_catalog.outputs.runs_dir")
    common.add_argument("--state-dir", default=argparse.SUPPRESS,
                        help=f"队列状态目录，默认 <runs_dir>/{STATE_DIR_NAME}/")
    parser.add_argument("--catalog", default=str(BRIDGE / "configs" / "op_catalog.json"))
    parser.add_argument("--runs-dir", default=None, help="覆盖 op_catalog.outputs.runs_dir")
    parser.add_argument("--state-dir", default=None,
                        help=f"队列状态目录，默认 <runs_dir>/{STATE_DIR_NAME}/")
    sub = parser.add_subparsers(dest="action", required=True)

    p = sub.add_parser("submit", help="把命令加进队列（可一次多条）", parents=[common])
    p.add_argument("--queue", default="default")
    p.add_argument("--label", default=None)
    p.add_argument("--cmd", default=None, help="交给 bash -lc 的整条 shell 命令")
    p.add_argument("--spec", default=None, help="JSON 文件：单个作业对象或对象数组（批量提交）")
    p.add_argument("--after-tag", action="append", default=None,
                   help="依赖的批次 tag，可重复；全部满足才放行。不给则立即执行")
    p.add_argument("--after-agent", default=None, help="依赖批次的 agent；默认自动判定")
    p.add_argument("--max-wait", type=float, default=MAX_WAIT_DEFAULT,
                   help=f"依赖最多等多久（秒），默认 {int(MAX_WAIT_DEFAULT)}，超时标 skipped")
    p.add_argument("--cwd", default=None)
    p.add_argument("--env", action="append", default=None, help="KEY=VALUE，可重复")
    p.add_argument("--dry-run", action="store_true", help="只记录不执行：依赖满足后打印命令并结束")
    p.add_argument("--allow-tag-reuse", action="store_true",
                   help="允许命令复用依赖的 tag（默认拒绝：会另起一个批次写进同一目录）")
    p.set_defaults(func=cmd_submit)

    p = sub.add_parser("run", help="起 daemon：轮询依赖并串行执行队列", parents=[common])
    p.add_argument("--queue", default="default")
    p.add_argument("--interval", type=float, default=POLL_DEFAULT, help="轮询间隔（秒），默认 60")
    p.add_argument("--daemon", action="store_true", help="后台常驻（自己 daemon 化后立即返回）")
    p.add_argument("--once", action="store_true", help="把当前能推进的作业跑完就退出")
    p.add_argument("--force", action="store_true", help="已有 daemon 时也强行再起一个")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("list", help="查看队列", parents=[common])
    p.add_argument("--queue", default="default")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("cancel", help="取消还没跑的作业（运行中的要 --force）", parents=[common])
    p.add_argument("--queue", default="default")
    p.add_argument("--id", type=int, action="append", default=None)
    p.add_argument("--all", action="store_true")
    p.add_argument("--force", action="store_true", help="连正在运行的作业一起停")
    p.set_defaults(func=cmd_cancel)

    p = sub.add_parser("stop", help="停 daemon（默认不动正在跑的作业）", parents=[common])
    p.add_argument("--queue", default="default")
    p.add_argument("--wait", type=float, default=0.0, help="等 daemon 退出的秒数")
    p.set_defaults(func=cmd_stop)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    runs_dir, ops = catalog_defaults(args.catalog)
    if args.runs_dir:
        runs_dir = Path(args.runs_dir)
    if not ops:
        raise SystemExit(f"没能从 {args.catalog} / op_catalog.resolved.json 读到 op 列表")
    return args.func(args, runs_dir, ops)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BrokenPipeError:  # `list | head` 这类下游提前关闭管道：静默收尾，不吐 traceback
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        sys.exit(0)
