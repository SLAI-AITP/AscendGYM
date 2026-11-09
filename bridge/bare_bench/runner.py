#!/usr/bin/env python3
"""bare_bench：用 k-lens `bare_client` 跑一次「裸模型 + 工具池」算子优化评测。

与 native_bench 的分工：native_bench 自己就是评测网关（下发 client.py 入口、收集
evaluations/E####）；bare_client 自带工具池（k-lens Backend 的 `/api/`），因此这里
只做三件事 —— **起 Backend、拉起 bare_client、把产物接进 bridge 契约**。

    python3 -m bare_bench run --config <runs_dir>/<run_id>.json --output <runs_dir>/<run_id>

产物落点（bridge 侧读者按这些路径取值）：

    <output>/prompt.txt               ← bare_client 写（软链）
    <output>/config.resolved.json     ← bare_client 写（软链）
    <output>/evaluations.jsonl        ← bare_client 写（软链，逐条追加，监控实时可读）
    <output>/trajectory.jsonl         ← bare_client 写（软链）
    <output>/result.json              ← bare_client 写（软链，schema klens-bare-client-result-v1）
    <output>/usage.json               ← bare_client 写（软链）
    <output>/events.jsonl             ← 本模块由 trajectory 转写（monitor 的 staleness/turn 口径）
    <output>/agent.stdout.jsonl       ← bare_client 的 stdout
    <output>/agent.stderr.txt         ← bare_client 的 stderr
    <output>/backend.log              ← Backend 进程日志
    <output>/manifest.json            ← 本模块写的 run 元信息
    <output>/_stage/                  ← 暂存区：op 源码、版本根、profiling 产物
    <output>/_client/                 ← bare_client 的真实输出目录

不复制、不改写 bare_client 的任何评测证据：真实文件都在 `_client/`，run 根下是软链。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

BRIDGE = Path(__file__).resolve().parent.parent
if str(BRIDGE) not in sys.path:
    sys.path.insert(0, str(BRIDGE))

from op_manager import redact_credentials, write_private_text

RUNNER_DIR = Path(__file__).resolve().parent / "sandbox_runner"
CLIENT_LINK_FILES = ("prompt.txt", "config.resolved.json", "evaluations.jsonl",
                     "trajectory.jsonl", "result.json", "usage.json")
TOOL_RESULT_PREFIX = "Tool execution result:"
# client 能力探测缓存：k-lens 与 k-lens-v2 的参数集不同（如 --context-* 只有 v2 有），
# 直接下发老客户端不认的参数会让 argparse 退出，所以按 runner.py 里的参数名探测后再生效。
_CLIENT_FLAG_CACHE: dict[tuple[str, str], bool] = {}


def client_flag_supported(client_root, flag: str) -> bool:
    """该 client 的 bare_client/runner.py 是否声明了某个 CLI 参数。"""
    key = (str(client_root), flag)
    if key not in _CLIENT_FLAG_CACHE:
        runner = Path(client_root) / "bare_client" / "runner.py"
        try:
            text = runner.read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = ""
        _CLIENT_FLAG_CACHE[key] = f'"{flag}"' in text
    return _CLIENT_FLAG_CACHE[key]


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def expand_path(value):
    return os.path.expanduser(os.path.expandvars(str(value))) if value else value


def write_json(path, value):
    write_private_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


# --- 阶段 1：把 bridge 的算子包摆成 k-lens Backend 认识的布局 -------------------- #
def stage_operator(cfg, run_root):
    """<stage>/op_path/<op_name>/<op_snake>/{op_host,op_kernel,...} + <op_snake>_test.py

    Backend 的 `_resolve_op_root` 找的是 `<root>/<op_name>/<to_snake_case_compact(op_name)>/`，
    测试脚本首选 `<op_dir>/<op_snake>_test.py`（Backend 自己有 fallback，但显式放好更稳）。
    """
    op_name = cfg["input"]["op_name"]
    source = Path(expand_path(cfg["input"]["source_dir"]))
    if not source.is_dir():
        raise SystemExit(f"source_dir 不存在: {source}")
    snake = cfg["input"].get("op_snake") or op_name
    op_dir = run_root / "_stage" / "op_path" / op_name / snake
    if op_dir.exists():
        shutil.rmtree(op_dir)
    op_dir.parent.mkdir(parents=True, exist_ok=True)

    # op_speed_test/ 是 bridge 包里的测试目录，测试脚本单独放到 preferred 位置，
    # 不放进去，避免给沙盒多传一份（与 bridge 的 collect_files 排除 test_dir 口径一致）。
    skip = {"op_speed_test"}
    shutil.copytree(source, op_dir, ignore=shutil.ignore_patterns(*skip))

    tests = sorted((source / "op_speed_test").glob("*.py")) if (source / "op_speed_test").is_dir() else []
    explicit = cfg["input"].get("test_script")
    if explicit:
        # config 里通常是包内相对路径（如 op_speed_test/<op>_custom_test.py）
        explicit_path = Path(expand_path(explicit))
        tests = [explicit_path if explicit_path.is_absolute() else source / explicit_path]
    if not tests:
        raise SystemExit(f"未找到测试脚本：{source}/op_speed_test/*.py（可用 input.test_script 指定）")
    # Backend 找的是 <root>/<op_name>/<op_snake>_test.py：算子目录的**同级**，不是目录内。
    target = op_dir.parent / f"{snake}_test.py"
    shutil.copyfile(tests[0], target)

    version_root = run_root / "_stage" / "op_path_version"
    if version_root.exists():
        shutil.rmtree(version_root)
    version_root.mkdir(parents=True, exist_ok=True)
    return {"op_dir": str(op_dir), "op_path_root": str(op_dir.parent.parent),
            "version_root": str(version_root), "test_script": str(target),
            "staged_from": str(source), "staged_test_from": str(tests[0])}


# --- 阶段 2：Backend 生命周期 -------------------------------------------------- #
def backend_env(cfg, staged):
    env = os.environ.copy()
    sandbox = cfg["sandbox"]
    backend = cfg["backend"]
    profiling = cfg["profiling"]
    env["PROFILING_HOST"] = str(sandbox["host"])
    env["PROFILING_PORT"] = str(sandbox["port"])
    env["PROFILING_ASCEND_C_PATH"] = str(RUNNER_DIR)
    # Backend 会 `cd $PROFILING_ASCEND_C_PATH` 后执行该相对路径；runner 自包含，无需 shim。
    env["PROFILING_MAIN_PY_PATH"] = "./main.py"
    env["PROFILING_COMPILE_TIMEOUT"] = str(backend.get("compile_timeout_ms", 1800000))
    env["PROFILING_RUN_TIMEOUT"] = str(backend.get("run_timeout_ms", 3600000))
    env["PROFILING_RUNS"] = str(profiling["total_runs"])
    env["PROFILING_WARMUP_RUNS"] = str(profiling["warmup_runs"])
    env["PROFILING_MIN_IMPROVEMENT_EPSILON"] = str(backend.get("min_improvement_epsilon", 0.0))
    env["RUS_OP_PATH"] = staged["op_path_root"]
    env["RUS_OP_PATH_VERSION"] = staged["version_root"]
    env["SANDBOX_PATH"] = str(RUNNER_DIR)          # Backend 里 `conda run` 那条死路径用不到，指过来即可
    env["BARE_OPS_REPO"] = str(cfg["input"].get("ops_repo") or "")
    if cfg["input"].get("data_dir"):
        env["BARE_TEST_DATA_DIR"] = expand_path(str(cfg["input"]["data_dir"]))
    # Backend 的 profiling 子进程直接用当前解释器跑 runner，必须能 import requests。
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def client_env(cfg):
    """bare_client 的环境：把注册表里的 api_key 传下去（裸模型链路原本不传 key）。

    bare_client 的取 key 顺序是 `os.environ[$api_key_env]` → `config["api_key"]`，
    而 `bare_bench` 拉起它时不写任何 key 字段，所以**外部 API 一律 401**
    （实测 2026-09-25：gpt-5.6-terra @ az.gptplus5.com 返回
    `HTTPError: 401 Client Error: Unauthorized`）。本地部署用占位 key `any-value`
    且服务端不校验，此前不带 Authorization 头也能跑通，故这里的规则是：
    非空且非占位值才注入（`vllm` 档的 api_key_env 缺省即 `API_KEY`），
    本地 run 的行为保持不变。
    """
    env = os.environ.copy()
    key = str((cfg.get("agent", {}).get("model") or {}).get("api_key") or "").strip()
    if key and key != "any-value":
        env["API_KEY"] = key
    return env


def wait_health(port, timeout, proc_log, required_tools=("finish_task",)):
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.monotonic() + timeout
    last = ""
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                payload = json.load(response)
            tools = payload.get("available_tools") or []
            missing = [name for name in required_tools if name not in tools]
            if missing:
                raise SystemExit(f"Backend /health 缺少必需工具 {missing}: {tools}")
            return payload
        except (urllib.error.URLError, OSError, ValueError) as exc:
            last = f"{type(exc).__name__}: {exc}"
            if proc_log.exists() and proc_log.stat().st_size:
                tail = proc_log.read_text(encoding="utf-8", errors="replace")[-800:]
                if "Traceback" in tail or "Error" in tail:
                    raise SystemExit(f"Backend 启动失败（{last}）\n--- backend.log ---\n{tail}")
            time.sleep(0.5)
    raise SystemExit(f"Backend 在 {timeout}s 内未就绪: {url} ({last})")


def stop_backend(proc):
    if proc.poll() is not None:
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(os.getpgid(proc.pid), sig)
        except (ProcessLookupError, PermissionError):
            return
        try:
            proc.wait(timeout=10)
            return
        except subprocess.TimeoutExpired:
            continue


# --- 阶段 3：trajectory → bridge 事件 ----------------------------------------- #
def classify(row):
    message = row.get("message") or {}
    role = message.get("role")
    content = message.get("content")
    text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
    if role == "assistant":
        name = None
        marker = "<tool_call>"
        if marker in text:
            try:
                payload = json.loads(text.split(marker, 1)[1].split("</tool_call>", 1)[0].strip())
                name = payload.get("name")
            except (ValueError, AttributeError):
                name = None
        return ("bare_tool_call" if name else "bare_message"), {"tool": name} if name else {}
    if role == "user" and isinstance(text, str) and text.startswith(TOOL_RESULT_PREFIX):
        return "bare_tool_result", {}
    return "bare_message", {}


def tail_trajectory(trajectory, events, run_id, stop_event):
    """把 bare_client 的 trajectory.jsonl 逐行转写成 bridge 事件流（供 monitor 读）。

    注意：必须用 readline() 而不是 `for line in handle` —— 文本模式下正在迭代时
    调用 tell() 会抛 OSError("telling position disabled by next() call")，而那个异常
    会被这里的 OSError 兜底悄悄吃掉，表现为事件流永远为空。
    """
    position = 0
    seq = 0
    while True:
        try:
            with trajectory.open(encoding="utf-8", errors="replace") as handle:
                handle.seek(position)
                while True:
                    line = handle.readline()
                    if not line:
                        break
                    if not line.endswith("\n"):   # 半行：等下一轮补齐再读
                        break
                    position = handle.tell()
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except ValueError:
                        continue
                    event_type, extra = classify(row)
                    seq += 1
                    record = {"event_id": seq, "run_id": run_id, "event_type": event_type,
                              "elapsed_sec": row.get("elapsed_seconds", 0.0),
                              "timestamp_utc": now_iso(), "sequence": row.get("sequence")}
                    if extra:
                        record.update(extra)
                    if row.get("usage"):
                        record["usage"] = row["usage"]
                    with events.open("a", encoding="utf-8") as out:
                        out.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:
            # 真实存在的文件读不了才是问题；首次写入前的 FileNotFoundError 属正常。
            if trajectory.exists():
                print(f"[bare_bench] trajectory 转写失败: {type(exc).__name__}: {exc}", file=sys.stderr)
        if stop_event.wait(1.0):
            return


# --- 主流程 -------------------------------------------------------------------- #
def run(cfg, run_root):
    if run_root.exists():
        raise SystemExit(f"bare harness 要求 --output 目录必须不存在: {run_root}")
    run_root.mkdir(parents=True)
    run_id = cfg.get("run_id") or run_root.name
    stop_event = threading.Event()

    staged = stage_operator(cfg, run_root)
    log(f"staged op: {staged['op_dir']} (test={Path(staged['test_script']).name}, "
        f"repo={cfg['input'].get('ops_repo')})")

    client_dir = run_root / "_client"
    port = cfg["backend"].get("port") or free_port()
    backend_root = Path(expand_path(cfg["backend"]["root"]))
    python = cfg["backend"].get("python") or sys.executable
    backend_log = run_root / "backend.log"
    cmd = [python, str(backend_root / "app.py"), "--port", str(port),
           "--op-path", staged["op_path_root"], "--op-path-version", staged["version_root"],
           "--min-improvement-epsilon", str(cfg["backend"].get("min_improvement_epsilon", 0.0))]
    # 工具池默认只认 OpenAI JSON；Qwen/GLM 系还会输出 <function=…><parameter=…> 原生形式，
    # 必须在池侧开 --native-tool 才会解析（Backend README: "Native-tool parsing is opt-in"）。
    tool_format = (cfg["agent"].get("model") or {}).get("tool_format") or "json"
    if tool_format == "xml":
        cmd.append("--native-tool")
    log(f"starting Backend: port={port} root={backend_root} tool_format={tool_format}"
        + ("（已开 --native-tool）" if tool_format == "xml" else ""))
    handle = backend_log.open("w", encoding="utf-8")
    backend_proc = subprocess.Popen(cmd, cwd=str(backend_root), env=backend_env(cfg, staged),
                                    stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        health = wait_health(port, cfg["backend"].get("start_timeout_seconds", 90), backend_log)
        log(f"Backend ready: {len(health.get('available_tools') or [])} tools, "
            f"sandbox={cfg['sandbox']['host']}:{cfg['sandbox']['port']}")

        events = run_root / "events.jsonl"
        write_json(run_root / "manifest.json", {
            "schema_version": "bare-bench-run-v1", "run_id": run_id,
            "harness": "bare", "op_name": cfg["input"]["op_name"],
            "model": redact_credentials(cfg["agent"]["model"]), "sandbox": cfg["sandbox"],
            "backend_port": port, "staged": staged,
            "profiling": cfg["profiling"], "stop": cfg["stop"], "client_dir": str(client_dir),
        })
        with events.open("a", encoding="utf-8") as out:
            out.write(json.dumps({"event_id": 0, "run_id": run_id, "event_type": "agent_started",
                                  "elapsed_sec": 0.0, "timestamp_utc": now_iso()},
                                 ensure_ascii=False) + "\n")

        client_root = Path(expand_path(cfg["client"]["root"]))
        client_cmd = [python, "-m", "bare_client.runner",
                      "--op-name", cfg["input"]["op_name"],
                      "--spec", expand_path(cfg["input"]["spec_path"]),
                      "--output", str(client_dir),
                      "--tool-api", f"http://127.0.0.1:{port}/api/",
                      "--llm-provider", cfg["agent"]["model"].get("provider", "vllm"),
                      "--model-url", cfg["agent"]["model"]["url"],
                      "--model-name", cfg["agent"]["model"]["name"],
                      "--candidate-evaluations-max", str(cfg["stop"]["candidate_evaluations_max"]),
                      "--candidate-solutions-target", str(cfg["stop"]["candidate_solutions_target"]),
                      "--final-reprofile-count", str(cfg["stop"]["final_reprofile_count"]),
                      "--profiling-protocol", json.dumps(cfg["profiling"]),
                      "--backend-timeout-seconds", str(cfg["client"].get("backend_timeout_seconds", 3600))]
        # 注册表 models.<key>.protocol_correction_max（可选）：模型"偶尔只回文本不发工具调用"时
        # 的纠偏次数上限；缺省不传，用 bare_client 自己的默认值（2）。
        _pcm = (cfg["agent"].get("model") or {}).get("protocol_correction_max")
        if _pcm is not None:
            client_cmd += ["--protocol-correction-max", str(int(_pcm))]
        # 注册表 models.<key>.context（可选）：k-lens-v2 的上下文护栏要按真实部署窗口计算
        # 触发线（缺省 131072，对 200k+ 的模型偏小）。k-lens 的 client 没有该参数，
        # 探测到才下发，避免 argparse 退出。
        _ctx = (cfg["agent"].get("model") or {}).get("context")
        if _ctx and client_flag_supported(client_root, "--context-window-tokens"):
            client_cmd += ["--context-window-tokens", str(int(_ctx))]
        editable = cfg["input"].get("editable_paths")
        tests = cfg["input"].get("test_paths")
        if editable:
            path = run_root / "editable_paths.json"
            write_json(path, editable)
            client_cmd += ["--editable-paths-file", str(path)]
        if tests:
            path = run_root / "test_paths.json"
            write_json(path, tests)
            client_cmd += ["--test-paths-file", str(path)]

        for name in CLIENT_LINK_FILES:                      # 软链：run 根下即可见，且单一真相
            link = run_root / name
            if not link.exists():
                link.symlink_to(client_dir / name)

        watcher = threading.Thread(target=tail_trajectory,
                                   args=(client_dir / "trajectory.jsonl", events, run_id, stop_event),
                                   daemon=True)
        watcher.start()

        log(f"launching bare_client: model={cfg['agent']['model']['name']} "
            f"budget={cfg['stop']['candidate_evaluations_max']}")
        stdout_path = run_root / "agent.stdout.jsonl"
        stderr_path = run_root / "agent.stderr.txt"
        with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open("w", encoding="utf-8") as err:
            proc = subprocess.run(client_cmd, cwd=str(client_root), stdout=out, stderr=err,
                                  env=client_env(cfg))
        stop_event.set()
        watcher.join(timeout=5)

        with events.open("a", encoding="utf-8") as out:
            out.write(json.dumps({"event_id": 10 ** 6, "run_id": run_id, "event_type": "run_finished",
                                  "elapsed_sec": 0.0, "timestamp_utc": now_iso(),
                                  "exit_code": proc.returncode}, ensure_ascii=False) + "\n")
    finally:
        stop_backend(backend_proc)
        handle.close()

    result_path = client_dir / "result.json"
    if result_path.exists():
        result = read_json(result_path)
        log(f"bare_client 结束: success={result.get('success')} "
            f"reason={result.get('termination_reason')} "
            f"candidates={result.get('candidate_evaluations')}")
        return 0 if result.get("success") else 1
    log("bare_client 未写出 result.json（异常退出）")
    return proc.returncode or 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run", help="run one bare_client evaluation")
    run_parser.add_argument("--config", required=True)
    run_parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    cfg = read_json(args.config)
    return run(cfg, Path(args.output))


if __name__ == "__main__":
    sys.exit(main())
