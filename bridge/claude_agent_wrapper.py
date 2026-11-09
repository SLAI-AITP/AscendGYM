#!/usr/bin/env python3
"""native harness 下 Claude Code 的保底重试 wrapper（bridge 侧，不修改 harness）。

native_bench 用 `agent.executable` 直接拉起本文件，argv 由 harness 生成：

    <this wrapper> -p --output-format stream-json --verbose --include-partial-messages
                   [--allowedTools ...] [--permission-mode ...] [--model ...] [extra_args...]

prompt 通过 stdin 传入，stdout 必须是逐行 JSON 的 Claude stream-json。

本 wrapper 只做一件事：把 stdout/stderr 原样、实时透传给 harness，同时在 Claude Code
自己以「可恢复」方式结束时（is_error=true / subtype=error_during_execution /
terminal_reason=aborted_streaming / 非 0 退出 / 流被截断没有 result 行），用
`--resume <session_id>` 在同一个 session 上续跑，避免服务端一次断流就让整轮 run 报废。

与 native harness 的契约（改动前请先读 runner.py / usage.py）：

* 只有最终一次 attempt 成功（进程 rc=0 且末条 result 无 is_error）才以 0 退出；
  全部 attempt 失败时按最后一次的真实 rc（非 0）退出 —— harness 仍记为 agent_error，
  不会把失败伪装成成功。
* harness 用 `_read_stream` 逐行 json.loads 统计 usage，因此 stdout 必须逐行原样转发、
  不能被本 wrapper 吞掉或改写；重试日志一律走 stderr（harness 汇总到 agent.stderr.txt）。
* harness 在 stop 条件满足时会 `killpg` 整个进程组；本 wrapper 不给子进程新建会话，
  保持同组，信号能同时收敛 wrapper 与 claude。
* 人工停止（<output>/STOP 文件）或 harness 已关闭 admission（/status.stop_required）
  时不再重试。

可用环境变量（默认值即线上配置）：

* `NATIVE_CLAUDE_RETRY=0`   由 op_manager 读取：不注入本 wrapper（完全回退原生 CLI）。
* `CLAUDE_AGENT_BIN`        真实 claude CLI，默认 `claude`。
* `CLAUDE_AGENT_RETRY_MAX`  额外重试次数，默认 2（总 attempt 最多 3）。
* `CLAUDE_AGENT_RETRY_DELAY` 首次重试前等待秒数，默认 15（按 attempt 线性退避）。
* `CLAUDE_AGENT_RETRY_FRESH_FALLBACK=1` 会话无法 resume 时，允许用原始 prompt 重开一轮
  （默认关闭：重开会丢掉已完成的进度并重复花费 token）。
"""

import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

SELF = Path(__file__).resolve()

# 这些收尾原因意味着「连接/流被服务端打断」，任务本身没有结论，值得在同一 session 续跑。
RETRYABLE_TERMINALS = (
    "aborted_streaming", "aborted", "aborted_tools", "stream_error",
    "stream_disconnected", "connection_error", "overloaded", "server_error",
)
CONTINUE_PROMPT = (
    "上一次会话在流式返回中被服务端中断（aborted_streaming）。请接着上一次的进度继续完成任务："
    "先核对 workspace 里的当前代码和已有评测结果，再从断开处继续，不要从头重做整个任务。"
)
DEFAULT_RETRY_MAX = 2
DEFAULT_RETRY_DELAY = 15.0
MAX_RETRY_MAX = 10


def note(message):
    """重试日志只写 stderr，绝不污染 stdout 的 stream-json。"""
    print(f"[claude-retry] {message}", file=sys.stderr, flush=True)


def env_flag(name, default="1"):
    value = os.environ.get(name, default)
    return str(value).strip().lower() not in ("", "0", "false", "no", "off")


def env_int(name, default, low, high):
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        value = int(str(raw).strip())
    except ValueError:
        note(f"{name}={raw!r} 不是整数，回退 {default}")
        return default
    if value < low or value > high:
        note(f"{name}={value} 超出 [{low}, {high}]，回退 {default}")
        return default
    return value


def env_float(name, default, low):
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        value = float(str(raw).strip())
    except ValueError:
        note(f"{name}={raw!r} 不是数字，回退 {default}")
        return default
    return value if value >= low else default


def real_claude():
    configured = os.environ.get("CLAUDE_AGENT_BIN", "claude").strip() or "claude"
    found = shutil.which(configured) or configured
    candidate = Path(found)
    try:
        if candidate.exists() and candidate.resolve() == SELF:
            note("CLAUDE_AGENT_BIN 指向 wrapper 自身，拒绝启动（避免递归）")
            raise SystemExit(2)
    except OSError:
        pass
    return found


def output_root():
    """harness 固定用 cwd=<output>/workspace，STOP/launch.json 都在 output 根目录。"""
    cwd = Path.cwd().resolve()
    for candidate in (cwd, cwd.parent, cwd.parent.parent):
        if (candidate / "launch.json").exists() or (candidate / "connection.json").exists():
            return candidate
    return cwd.parent if cwd.name == "workspace" else cwd


def harness_stop_reason():
    """读 harness 的 /status（只读，不消耗评测预算）；任何异常都返回 None。"""
    path = os.environ.get("NATIVE_BENCH_CONNECTION")
    if not path or not Path(path).exists():
        return None
    try:
        conn = json.loads(Path(path).read_text(encoding="utf-8"))
        request = urllib.request.Request(
            conn["url"] + "/status", data=b"{}",
            headers={"Authorization": "Bearer " + conn["token"], "Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=5) as response:
            data = json.load(response)
    except (OSError, ValueError, KeyError, urllib.error.URLError):
        return None
    return data.get("stop_reason") if data.get("stop_required") else None


def stop_requested(reason):
    """返回停止原因字符串；None 表示可以继续重试。"""
    root = output_root()
    if (root / "STOP").exists():
        return "harness STOP 文件（人工停止）"
    closed = harness_stop_reason()
    if closed:
        return f"harness 已关闭 admission（{closed}）"
    return None


class Attempt:
    """一次 claude 进程的收尾状态，全部来自转发的 stream-json 行。"""

    def __init__(self, index, resumed):
        self.index = index
        self.resumed = resumed
        self.session_id = None
        self.saw_result = False
        self.result_error = False
        self.subtype = None
        self.terminal_reason = None
        self.num_turns = None
        self.duration_api_ms = None
        self.cost_usd = None
        self.rc = None

    def observe(self, record):
        typ = record.get("type")
        if isinstance(record.get("session_id"), str) and self.session_id is None:
            self.session_id = record["session_id"]
        if typ == "system" and record.get("subtype") == "init":
            if record.get("session_id"):
                self.session_id = record["session_id"]
        elif typ == "result":
            self.saw_result = True
            self.result_error = bool(record.get("is_error"))
            self.subtype = record.get("subtype")
            self.terminal_reason = record.get("terminal_reason")
            if record.get("session_id"):
                self.session_id = record["session_id"]
            self.num_turns = record.get("num_turns")
            self.duration_api_ms = record.get("duration_api_ms")
            self.cost_usd = record.get("total_cost_usd")

    def failure(self):
        """返回可重试的失败原因；None 表示这次 attempt 正常收尾。"""
        if self.rc == 0 and self.saw_result and not self.result_error:
            return None
        if not self.saw_result:
            return f"no_final_result(rc={self.rc})"
        if self.result_error:
            return self.terminal_reason or self.subtype or "result_is_error"
        return f"exit_rc_{self.rc}"

    def summary(self):
        return {"attempt": self.index, "resumed": self.resumed, "rc": self.rc,
                "subtype": self.subtype, "terminal_reason": self.terminal_reason,
                "session_id": self.session_id, "num_turns": self.num_turns,
                "duration_api_ms": self.duration_api_ms, "cost_usd": self.cost_usd}


def pump_stderr(pipe):
    """claude 的 stderr 原样转发到 harness 的 agent.stderr.txt。"""
    target = sys.stderr.buffer
    try:
        for raw in pipe:
            target.write(raw)
            target.flush()
    except (BrokenPipeError, ValueError):
        pass
    finally:
        try:
            pipe.close()
        except OSError:
            pass


def run_once(argv, prompt, attempt, state):
    """跑一次 claude：转发 stream-json，返回收尾状态。"""
    try:
        child = subprocess.Popen(
            argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd=str(Path.cwd()), env=os.environ.copy())
    except OSError as exc:
        note(f"无法启动 {argv[0]}: {exc}")
        attempt.rc = 127
        return attempt
    state.child = child
    stderr_thread = threading.Thread(target=pump_stderr, args=(child.stderr,), daemon=True)
    stderr_thread.start()
    try:
        try:
            child.stdin.write(prompt.encode("utf-8"))
            child.stdin.close()
        except (BrokenPipeError, OSError):
            try:
                child.stdin.close()
            except OSError:
                pass
        out = sys.stdout.buffer
        for raw in child.stdout:
            # 必须先原样转发（harness 逐行 json.loads 统计 usage），再解析。
            out.write(raw)
            out.flush()
            line = raw.decode("utf-8", "replace").strip()
            if not line:
                continue
            try:
                attempt.observe(json.loads(line))
            except ValueError:
                continue
        attempt.rc = child.wait()
    except KeyboardInterrupt:
        attempt.rc = child.poll()
        if attempt.rc is None:
            attempt.rc = -2
    finally:
        state.child = None
        stderr_thread.join(timeout=5)
        for handle in (child.stdin, child.stdout, child.stderr):
            try:
                if handle is not None and not handle.closed:
                    handle.close()
            except OSError:
                pass
    return attempt


def install_signals(state):
    def handler(signum, frame):
        state.interrupted = signum
        note(f"收到信号 {signum}，转发给 claude 子进程（不重试）")
        child = state.child
        if child is not None and child.poll() is None:
            try:
                child.send_signal(signum)
            except OSError:
                pass
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        try:
            signal.signal(signum, handler)
        except (ValueError, OSError):
            pass


def write_retry_report(root, payload):
    """best-effort 审计文件：<output>/bridge_retry.json（不影响 harness 判定）。"""
    try:
        (root / "bridge_retry.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except OSError:
        pass


def main(argv):
    harness_argv = list(argv[1:])
    claude = real_claude()
    if harness_argv and harness_argv[0] == "--version":
        return subprocess.run([claude, "--version"]).returncode
    if os.environ.get("NATIVE_CLAUDE_RETRY", "").strip().lower() in ("0", "false", "no", "off"):
        os.execv(claude, [claude] + harness_argv)  # 显式关闭：纯透传
    prompt = sys.stdin.read()
    base_argv = [claude] + harness_argv
    retries = env_int("CLAUDE_AGENT_RETRY_MAX", DEFAULT_RETRY_MAX, 0, MAX_RETRY_MAX)
    delay = env_float("CLAUDE_AGENT_RETRY_DELAY", DEFAULT_RETRY_DELAY, 0.0)
    fresh_fallback = env_flag("CLAUDE_AGENT_RETRY_FRESH_FALLBACK", "0")
    root = output_root()
    state = type("State", (), {"child": None, "interrupted": None})()
    install_signals(state)
    note(f"启动 claude={claude}，最多 {retries} 次重试，间隔 {delay:g}s，output={root}")
    attempts, session_id, rc = [], None, 1
    for index in range(1, retries + 2):
        resumed = index > 1
        run_argv, run_prompt = list(base_argv), prompt
        if resumed and session_id:
            run_argv += ["--resume", session_id]
            run_prompt = CONTINUE_PROMPT
        elif resumed:
            if not fresh_fallback:
                note("无 session_id 可 resume，且未开启 CLAUDE_AGENT_RETRY_FRESH_FALLBACK，停止重试")
                break
            note("无 session_id 可 resume，按 fresh fallback 用原始 prompt 重开一轮")
        attempt = run_once(run_argv, run_prompt, Attempt(index, resumed), state)
        attempts.append(attempt)
        rc = attempt.rc if attempt.rc is not None else 1
        if attempt.session_id:
            session_id = attempt.session_id
        failure = attempt.failure()
        note(f"attempt {index}/{retries + 1} 收尾 rc={rc} subtype={attempt.subtype} "
             f"terminal_reason={attempt.terminal_reason} session={attempt.session_id} "
             f"turns={attempt.num_turns} api_ms={attempt.duration_api_ms} cost_usd={attempt.cost_usd} "
             f"failure={failure}")
        if failure is None:
            rc = 0
            break
        if state.interrupted:
            note("wrapper 已被要求停止，不再重试")
            rc = 128 + int(state.interrupted) if rc == 0 else rc
            break
        if index > retries:
            note(f"重试次数已用尽（{retries} 次），按最后一次 rc={rc} 退出")
            rc = rc if rc != 0 else 1
            break
        blocked = stop_requested(failure)
        if blocked:
            note(f"不重试：{blocked}")
            rc = rc if rc != 0 else 1
            break
        wait = delay * index
        if wait > 0:
            note(f"等待 {wait:g}s 后以 --resume 续跑（原因 {failure}）")
            deadline = time.monotonic() + wait
            while time.monotonic() < deadline and not state.interrupted:
                time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
            if state.interrupted:
                note("等待期间被要求停止，不再重试")
                rc = rc if rc != 0 else 1
                break
    write_retry_report(root, {
        "wrapper": str(SELF), "claude": claude, "retry_max": retries, "retry_delay": delay,
        "fresh_fallback": fresh_fallback, "session_id": session_id, "exit_code": rc,
        "attempts": [attempt.summary() for attempt in attempts],
    })
    if rc == 0:
        note(f"claude 正常收尾（attempt 数 {len(attempts)}）")
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv))
