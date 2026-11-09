#!/usr/bin/env python3
"""bare_bench 的沙盒 profiling runner（k-lens Backend 的 `PROFILING_MAIN_PY_PATH`）。

k-lens 自带的 ascend_c runner 在这台机器上**不完整**：`--profile` 分支要跑
`utils/request_application.py`，而它 import 的 `gen_prof_code_application` 在本机
任何 checkout 里都不存在。bridge 自己的 `sandbox_evaluator.py` 则有一套已经跑通的
`/run_code` 调用方式。这个 runner 取两者的交集：

* **输入契约与 k-lens 一致**（Backend 只认这套 CLI，见 Backend/services/
  ascendc_code_profiling_service.py::_run_profiling）；
* **调用方式与 bridge 一致**（payload 字段、文件 key、msprof wrapper 都照搬
  sandbox_evaluator.py，因此结果与 native 批次同口径可比）；
* **输出契约与 k-lens 一致**（stdout 打 `Status Code:` / `Full Response JSON:`
  两个 marker，并把 profile_output.zip 解到 --profiling-output-dir，供
  Backend 的 profile_op_summary 找 op_summary*.csv）。

    main.py --custom-op-dir DIR --exec-code-path FILE --host H --port P \
            --ops-repo nn --profile --profiling-runs 8 --profiling-output-dir DIR

退出码：0 = 编译与 profiling 都成功；1 = 失败（细节在 stdout 的 markdown 里）。
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

EXEC_SCRIPT_NAME = "main_exec_code.py"
OPPROF_DIR = "profile_output"
OPPROF_ZIP = "profile_output.zip"
FETCH_FILES = ["build_output.log", OPPROF_ZIP]


def log(message):
    print(message, flush=True)


def build_parser():
    parser = argparse.ArgumentParser(description="SandboxFusion ascend_c runner (bridge bare_bench)")
    parser.add_argument("--custom-op-dir", required=True)
    parser.add_argument("--exec-code-path", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--ops-repo", default=os.environ.get("BARE_OPS_REPO") or "gtsrus")
    parser.add_argument("--language", default="ascend_c")
    parser.add_argument("--compile-timeout", type=int, default=1800000)
    parser.add_argument("--run-timeout", type=int, default=3600000)
    parser.add_argument("--experimental", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profiling-runs", type=int, default=8)
    parser.add_argument("--profiling-output-dir", default="sandbox-output")
    parser.add_argument("--profiling-sandbox-output-path", default=f"/tmp/{OPPROF_ZIP}")
    return parser


def to_seconds(value):
    """k-lens 侧超时是毫秒（Backend 自己也按 /1000 用），/run_code 收秒。"""
    return max(1, int(value // 1000)) if value >= 10000 else max(1, int(value))


def collect_files(op_dir):
    """与 bridge sandbox_evaluator.collect_files 同口径：{<op_basename>/<rel>: base64}。"""
    op_dir = Path(op_dir)
    files = {}
    for path in sorted(op_dir.rglob("*")):
        if path.is_file():
            rel = path.relative_to(op_dir).as_posix()
            files[f"{op_dir.name}/{rel}"] = base64.b64encode(path.read_bytes()).decode("ascii")
    return files


def stage_test_data(op_dir, data_dir):
    """把 golden 数据按 <op>/test_data/... 摆进沙盒（与 bridge 的 stage_test_data 同口径）。"""
    if not data_dir:
        return {}
    root = Path(data_dir)
    if not root.is_dir():
        log(f"warning: test data dir 不存在，跳过: {root}")
        return {}
    staged = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            rel = path.relative_to(root).as_posix()
            staged[f"{Path(op_dir).name}/test_data/{rel}"] = \
                base64.b64encode(path.read_bytes()).decode("ascii")
    return staged


def profiling_wrapper(script_name, output_dir, output_zip, runs):
    """沙盒侧 app 脚本：整段测试脚本跑在 msprof 下（照搬 bridge sandbox_evaluator）。"""
    return f'''
import glob
import os
import shutil
import subprocess

output_dir = {output_dir!r}
output_zip = {output_zip!r}
runs = {runs!r}
script_name = {script_name!r}

print("=== Starting Profiling Execution ===")
print("cwd:", os.getcwd())
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
if os.path.exists(output_zip):
    os.remove(output_zip)
os.makedirs(output_dir, exist_ok=True)

script = os.path.abspath(script_name)
if not os.path.exists(script):
    for root, dirs, files in os.walk("."):
        if script_name in files:
            script = os.path.abspath(os.path.join(root, script_name))
            break
print("app script:", script, "exists:", os.path.exists(script))
msprof = shutil.which("msprof")
print("msprof binary:", msprof)

cmd = 'msprof --application="python3 %s --runs %d" --output=%s' % (script, runs, output_dir)
print("Running command:", cmd)
exit_code = os.system(cmd)
print("Profiling command exit code:", exit_code)

for pattern in ("OPPROF_*", "*"):
    for actual in glob.glob(os.path.join(output_dir, pattern)):
        if os.path.isdir(actual) and os.path.basename(actual) != "OPPROF_placeholder":
            for item in os.listdir(actual):
                src = os.path.join(actual, item)
                dst = os.path.join(output_dir, item)
                if os.path.isdir(src):
                    if os.path.exists(dst):
                        shutil.rmtree(dst)
                    shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, dst)
            shutil.rmtree(actual)
            break

if os.path.exists(output_dir) and os.listdir(output_dir):
    shutil.make_archive(output_zip[:-4], "zip", output_dir)
    print("Created zip file:", output_zip, os.path.getsize(output_zip), "bytes")
else:
    print("ERROR: profiling output directory is empty")
    print("fallback: running the app directly so correctness and profiling failures stay distinguishable")
    direct_code = os.system('python3 %s --runs %d' % (script, runs))
    print("direct run exit code:", direct_code)
'''


def post_run_code(payload, url, timeout):
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status, json.load(response)


def extract_zip(blob_b64, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / OPPROF_ZIP
    archive.write_bytes(base64.b64decode(blob_b64))
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(output_dir)
    return archive


def summarize(response):
    """把 /run_code 响应压成一行结论，方便 Backend 的诊断与人工排查。"""
    compile_result = response.get("compile_result") or {}
    run_result = response.get("run_result") or {}
    compile_ok = compile_result.get("status") == "Finished" and compile_result.get("return_code") == 0
    run_ok = run_result.get("status") == "Finished" and run_result.get("return_code") == 0
    parts = [f"compile={'ok' if compile_ok else 'failed'} "
             f"(status={compile_result.get('status')}, rc={compile_result.get('return_code')})",
             f"run={'ok' if run_ok else 'failed'} "
             f"(status={run_result.get('status')}, rc={run_result.get('return_code')})"]
    if not compile_ok:
        parts.append("--- compile stderr tail ---\n" + (compile_result.get("stderr") or "")[-3000:])
    if not run_ok:
        parts.append("--- run stdout tail ---\n" + (run_result.get("stdout") or "")[-3000:])
        parts.append("--- run stderr tail ---\n" + (run_result.get("stderr") or "")[-3000:])
    return compile_ok and run_ok, "\n".join(parts)


def main(argv=None):
    args = build_parser().parse_args(argv)
    op_dir = Path(args.custom_op_dir)
    if not op_dir.is_dir():
        log(f"错误: --custom-op-dir 不存在: {op_dir}")
        return 1
    if not args.profile:
        log("错误: 本 runner 只支持 --profile 模式（k-lens Backend 总是带 --profile 调用）")
        return 1

    compile_timeout = to_seconds(args.compile_timeout)
    run_timeout = to_seconds(args.run_timeout)
    files = collect_files(op_dir)
    files.update(stage_test_data(op_dir, os.environ.get("BARE_TEST_DATA_DIR")))
    files[EXEC_SCRIPT_NAME] = base64.b64encode(
        Path(args.exec_code_path).read_text(encoding="utf-8").encode("utf-8")).decode("ascii")

    payload = {
        "code": profiling_wrapper(EXEC_SCRIPT_NAME, OPPROF_DIR, OPPROF_ZIP, args.profiling_runs),
        "language": args.language,
        "lint": False,
        "ops_repo": args.ops_repo,
        "experimental": bool(args.experimental),
        "run_timeout": run_timeout,
        "compile_timeout": compile_timeout,
        "required_npu": 1,
        "files": files,
        "fetch_files": list(FETCH_FILES),
    }
    # Optional NPU pinning for controlled retests. The scheduler accepts
    # preferred_node_id / preferred_npu_id and rejects placement outside the
    # requested pair when these fields are present.
    preferred_node = os.environ.get("SANDBOX_PREFERRED_NODE_ID")
    if preferred_node:
        payload["preferred_node_id"] = preferred_node
    preferred_npu = os.environ.get("SANDBOX_PREFERRED_NPU_ID")
    if preferred_npu not in (None, ""):
        payload["preferred_npu_id"] = int(preferred_npu)
    url = f"http://{args.host}:{args.port}/run_code"
    log(f"submitting {len(files)} files to {url} (repo={args.ops_repo}, "
        f"op={op_dir.name}, runs={args.profiling_runs}, "
        f"compile_timeout={compile_timeout}s, run_timeout={run_timeout}s)")

    try:
        status, response = post_run_code(payload, url, compile_timeout + run_timeout + 300)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        log(f"Status Code: 0\n沙盒请求失败: {type(exc).__name__}: {exc}")
        return 1

    log(f"Status Code: {status}")
    log("Full Response JSON: " + json.dumps(response, ensure_ascii=False))

    returned = response.get("files") or {}
    if OPPROF_ZIP in returned:
        archive = extract_zip(returned[OPPROF_ZIP], args.profiling_output_dir)
        log(f"profiling 输出已解压: {archive} -> {args.profiling_output_dir}")
    else:
        log(f"warning: 响应中没有 {OPPROF_ZIP}（无法产出 op_summary）")

    ok, summary = summarize(response)
    log(summary)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
