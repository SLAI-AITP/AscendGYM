#!/usr/bin/env python3
"""SandboxFusion evaluator bridge for native_bench `command` backends.

stdin : ascend-gym-evaluator-v1 request
        {"schema_version", "evaluation_kind", "source_dir", "artifacts_dir"}
        可选 "profiling": {"total_runs", "warmup_runs", "statistic"}
        （native v2+ harness 的固定采样协议；总轮数含预热，只支持 statistic=mean）
stdout: native-bench Evaluation object (see eval/native_harness/core/contract.py)

采样协议优先级：请求里的 "profiling" > SANDBOX_PROFILING_RUNS/SANDBOX_WARMUP_RUNS 环境变量。
请求带协议时，响应会在 raw.profiling_protocol 原样回显；协议非法则拒绝评测（exit 2），
避免"配置写一套、实际按另一套采样"的混合测量。

设备故障的归属判定（2026-09-22 真机重放实测）：baseline 已通过时某个候选触发 5070xx，
故障会稳定跟着**该候选代码**复现（同一候选跨多张 NPU 均复现，而同算子 baseline 在各张 NPU
上都复现不了），属于候选实现缺陷而非沙盒故障。两类故障因此走不同路径，详见 main() 中的注释。

The bridge packs the staged operator sources, sends them to the distributed
sandbox proxy (/run_code, language=ascend_c) together with the read-only test
script, fetches the msprof profiling archive and converts `Task Duration(us)`
samples into the single `duration_ms` number the harness expects.

Configuration is environment-only; nothing here chooses endpoints per turn:
  SANDBOX_URL             default http://127.0.0.1:9000
  SANDBOX_OPS_REPO        default nn
  SANDBOX_OP_NAME         default add_rms_norm
  SANDBOX_COMPILE_TIMEOUT default 1800 (s)
  SANDBOX_RUN_TIMEOUT     default 3600 (s)
  SANDBOX_PROFILING_RUNS  default 8（测量轮数；请求未带 profiling 时生效）
  SANDBOX_WARMUP_RUNS     default 3（测试脚本总共跑 PROFILING_RUNS+WARMUP_RUNS 轮，
                          前 WARMUP_RUNS 轮的 profiling 样本在聚合时被丢弃，只对剩余轮次取平均）
  SANDBOX_EVAL_RETRIES    default 3（额外重试次数；命中设备故障、传输故障或"无判定"时重跑整个评测）
  SANDBOX_EVAL_RETRY_BACKOFF  default 20（秒，首次重试前等待，之后按 2 倍递增）
  SANDBOX_EVAL_RETRY_MAX_BACKOFF  default 300（秒，单次重试等待上限）
  SANDBOX_EVAL_BUDGET_SECONDS  default 4800（秒，本评测器自身的时间预算上限；必须小于
                          harness 的 evaluator.per_call_timeout_seconds，否则会被 harness 先 SIGKILL，
                          那次评测就变成 state_unknown 并作废整个 run）
  SANDBOX_INFRA_MAX_STREAK  default 3（连续多少次基础设施故障后放弃并如实上报 sandbox_error。
                          未达阈值时返回非致命的 SANDBOX_INFRA_KEEPALIVE_STATUS，把设备故障原样
                          交还给 agent，让 run 继续——候选样本不会因为沙盒抖动被丢弃）
  SANDBOX_INFRA_KEEPALIVE_STATUS  default profiling_failed（未达阈值时对 harness 使用的非致命状态；
                          必须避开 sandbox_error/timeout，它们是 harness 的致命状态）
  SANDBOX_INFRA_GUIDANCE  可选，覆盖回传给 agent 的处置指引文本（默认见 INFRA_GUIDANCE_DEFAULT）
  SANDBOX_CANDIDATE_FAULT_STATUS   default test_failed（判定为"候选实现把设备打挂"时返回的状态；
                          必须避开 sandbox_error/timeout）
  SANDBOX_CANDIDATE_FAULT_GUIDANCE 可选，覆盖该情形下的诊断模板（占位符 {cause}/{attempts} 按次填充）
  SANDBOX_CANDIDATE_EVAL_RETRIES   default 1（baseline 已通过时的整评测重试次数；这类故障稳定复现，
                          给满 EVAL_RETRIES 只是烧 NPU 时间，留 1 次兜偶发抖动）
  SANDBOX_DEVICE_ERROR_PATTERN 可选，覆盖设备故障正则（默认匹配 5070xx、
                          vector core timeout、aivec error、AclrtSynchronizeDeviceWithTimeout 等）
  SANDBOX_NPU_TYPES       optional, comma separated (e.g. ascend910_93)
  SANDBOX_EXPERIMENTAL    set to 1 for operators under */experimental/
  SANDBOX_TEST_DATA_DIR   host-side test_data/ directory staged into the sandbox
                          as <op_name>/test_data/ so the frozen test can find it
  SANDBOX_TEST_DATA_CASES "all" (default), "auto" (pick smallest cases up to the
                          byte budget) or an explicit id list like "0,1,4"
  SANDBOX_TEST_DATA_MAX_BYTES  byte budget for "auto" case selection (default 12MB)
  SANDBOX_SUCCESS_PATTERN / SANDBOX_FAILURE_PATTERN  regex overrides for the
                          correctness verdict in run stdout
"""
import base64
import csv
import io
import json
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

SANDBOX_URL = os.environ.get("SANDBOX_URL", "http://127.0.0.1:9000").rstrip("/")
OPS_REPO = os.environ.get("SANDBOX_OPS_REPO", "nn")
OP_NAME = os.environ.get("SANDBOX_OP_NAME", "add_rms_norm")
COMPILE_TIMEOUT = int(os.environ.get("SANDBOX_COMPILE_TIMEOUT", "1800"))
RUN_TIMEOUT = int(os.environ.get("SANDBOX_RUN_TIMEOUT", "3600"))
PROFILING_RUNS = int(os.environ.get("SANDBOX_PROFILING_RUNS", "8"))
WARMUP_RUNS = max(0, int(os.environ.get("SANDBOX_WARMUP_RUNS", "3")))
EVAL_RETRIES = max(0, int(os.environ.get("SANDBOX_EVAL_RETRIES", "3")))
EVAL_RETRY_BACKOFF = max(0.0, float(os.environ.get("SANDBOX_EVAL_RETRY_BACKOFF", "20")))
EVAL_RETRY_MAX_BACKOFF = max(0.0, float(os.environ.get("SANDBOX_EVAL_RETRY_MAX_BACKOFF", "300")))
# 本评测器自身的墙钟预算：harness 会在 evaluator.per_call_timeout_seconds 处 SIGKILL 我们，
# 所以单次 HTTP 超时与重试等待都必须留在这条线以内（默认 5400 - 600 余量）。
EVAL_BUDGET_SECONDS = max(0.0, float(os.environ.get("SANDBOX_EVAL_BUDGET_SECONDS", "4800")))
INFRA_MAX_STREAK = max(1, int(os.environ.get("SANDBOX_INFRA_MAX_STREAK", "3")))
# 未达放弃阈值时回给 harness 的非致命状态；sandbox_error/timeout 是 harness 的致命状态，
# 一旦返回就会 close_admission 并强杀 agent，所以绝不能出现在这里。
_INFRA_NONFATAL_CHOICES = ("compile_failed", "test_failed", "test_mismatch", "profiling_failed")
INFRA_KEEPALIVE_STATUS = (os.environ.get("SANDBOX_INFRA_KEEPALIVE_STATUS", "profiling_failed").strip()
                          or "profiling_failed")
if INFRA_KEEPALIVE_STATUS not in _INFRA_NONFATAL_CHOICES:
    # 常量区在 log() 之前，不能在这里打日志；非法值静默退回默认（合法值由 __main__ 记录一次）
    INFRA_KEEPALIVE_STATUS = "profiling_failed"
INFRA_GUIDANCE_DEFAULT = (
    "这是沙盒/设备侧的基础设施故障，不是你的算子代码问题。请勿修改代码："
    "稍等片刻后用相同代码重新提交一次 candidate 评测即可。若连续多次仍是同一故障，"
    "不要继续重试，直接简要说明情况并结束本轮。"
)
INFRA_GUIDANCE = os.environ.get("SANDBOX_INFRA_GUIDANCE", "").strip() or INFRA_GUIDANCE_DEFAULT
# 候选实现把设备打挂时的判定与措辞。实测（2026-09-22，真机重放）表明：baseline 已通过的
# 前提下某个候选触发设备故障，故障会稳定跟在该候选代码上复现（跨多张 NPU 均复现），
# 属于候选实现缺陷而非沙盒故障——原样重交必然再次触发，必须明确劝阻。
CANDIDATE_FAULT_STATUS = os.environ.get("SANDBOX_CANDIDATE_FAULT_STATUS", "test_failed").strip() or "test_failed"
if CANDIDATE_FAULT_STATUS not in _INFRA_NONFATAL_CHOICES:
    CANDIDATE_FAULT_STATUS = "test_failed"
CANDIDATE_FAULT_GUIDANCE_DEFAULT = (
    "本次评测在你的候选实现执行期间触发了 NPU 设备故障：\n{cause}\n"
    "本 run 的 baseline 已通过，且该故障在 {attempts} 次重试中稳定复现，"
    "因此判定为候选实现缺陷，而不是沙盒/设备侧的基础设施故障。\n"
    "请勿原样重新提交当前版本：相同代码会再次触发同一故障，并浪费候选评测额度。\n"
    "请先定位并修复导致设备异常的缺陷（常见原因：越界访存、非法指令、UB 空间越界），"
    "确认修复后再提交新的候选版本。"
)
CANDIDATE_FAULT_GUIDANCE = (os.environ.get("SANDBOX_CANDIDATE_FAULT_GUIDANCE", "").strip()
                            or CANDIDATE_FAULT_GUIDANCE_DEFAULT)
# baseline 已通过时，候选触发设备故障几乎必然是候选实现缺陷（且稳定复现），
# 为此保留 3 次整评测重试只是在烧 NPU 时间（实测每次故障重试约 150s）。
# 单独给这一类更小的重试预算：仍留 1 次重试兜住偶发抖动，但不再拖满 EVAL_RETRIES。
CANDIDATE_EVAL_RETRIES = max(0, int(os.environ.get("SANDBOX_CANDIDATE_EVAL_RETRIES", "1")))
NPU_TYPES = [t for t in os.environ.get("SANDBOX_NPU_TYPES", "").replace(",", " ").split() if t]
EXPERIMENTAL = os.environ.get("SANDBOX_EXPERIMENTAL", "").strip().lower() in {"1", "true", "yes"}
TEST_DATA_DIR = os.environ.get("SANDBOX_TEST_DATA_DIR", "").strip()
TEST_DATA_CASES = os.environ.get("SANDBOX_TEST_DATA_CASES", "").strip()
TEST_DATA_MAX_BYTES = int(os.environ.get("SANDBOX_TEST_DATA_MAX_BYTES", "12000000"))
SUCCESS_RE = re.compile(os.environ.get(
    "SANDBOX_SUCCESS_PATTERN",
    r"All tests PASSED|ALL CHECKS PASSED|\b\d+\s+passed\b|checks passed"))
FAILURE_RE = re.compile(os.environ.get(
    "SANDBOX_FAILURE_PATTERN",
    r"REGRESSION DETECTED|\[FAIL\]|\b[1-9]\d*\s+(failed|error)\b|Traceback \(most recent call last\)"))

# 设备/驱动级故障：与候选代码质量无关，重跑通常可通过；连续命中则按基础设施
# 错误上报，绝不能记成候选的 test_failed / test_mismatch。
# 数值分支覆盖整个 5070xx 家族（507015/507035 等都出现过；只写 50703\d 会漏掉 507015）。
DEVICE_ERROR_RE = re.compile(os.environ.get(
    "SANDBOX_DEVICE_ERROR_PATTERN",
    r"5070\d\d"
    r"|vector core (?:timeout|execution is abnormal|execution timed out)"
    r"|aivec error"
    r"|AclrtSynchronizeDeviceWithTimeout"
    r"|rtStreamSynchronize execution failed"
    r"|rtDeviceSynchronizeWithTimeout"
    r"|ACL stream synchronize failed"
    r"|npuSynchronizeDevice"), re.IGNORECASE)

TEST_DIR = "op_speed_test"
EXEC_SCRIPT_NAME = "main_exec_code.py"


def log(message):
    print(message, file=sys.stderr, flush=True)


def profiling_wrapper(script_name, output_dir, output_zip, runs):
    """Wrap the read-only test script so the whole application runs under msprof."""
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
try:
    print("cwd entries:", sorted(os.listdir("."))[:60])
except OSError as exc:
    print("cwd listing failed:", exc)
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
if msprof:
    try:
        probe = subprocess.run([msprof, "--version"], capture_output=True, text=True, timeout=60)
        print("msprof version rc:", probe.returncode,
              ((probe.stdout or "") + (probe.stderr or ""))[-300:])
    except Exception as exc:
        print("msprof version probe failed:", exc)

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


def collect_files(source_dir, test_dir):
    """Pack staged sources as {op_name/relative/path: base64}."""
    files = {}
    source = Path(source_dir)
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(source).as_posix()
        if test_dir and (rel == test_dir or rel.startswith(test_dir + "/")):
            continue
        files[f"{OP_NAME}/{rel}"] = base64.b64encode(path.read_bytes()).decode("ascii")
    return files


def read_exec_code(source_dir, test_dir):
    tests = sorted((Path(source_dir) / test_dir).glob("*.py"))
    if not tests:
        raise RuntimeError(f"no test script found under {test_dir}/")
    return tests[0].read_text(encoding="utf-8")


def stage_test_data():
    """Stage golden data as <op_name>/test_data/... exactly like the local layout.

    The frozen tests resolve data through resolve_data_root(), which looks for
    CWD/<op>/test_data or <script dir>/test_data. Staging under <op>/test_data
    covers both the "sandbox cwd" and "op dir cwd" variants. When only a subset
    of cases is staged, cases_manifest.json is filtered too, because the tests
    take their case-id list from that manifest.
    """
    if not TEST_DATA_DIR:
        return {}, {}
    root = Path(TEST_DATA_DIR)
    if not root.is_dir():
        raise RuntimeError(f"SANDBOX_TEST_DATA_DIR not found: {root}")

    case_bytes = {}
    for path in root.rglob("case_*.pt"):
        match = re.fullmatch(r"case_(\d+)\.pt", path.name)
        if match:
            case_id = int(match.group(1))
            case_bytes[case_id] = case_bytes.get(case_id, 0) + path.stat().st_size
    all_ids = sorted(case_bytes)

    selection = None
    spec = TEST_DATA_CASES.lower()
    if spec and spec not in {"all", ""}:
        if spec == "auto":
            selection, used = [], 0
            for case_id in sorted(all_ids, key=lambda i: case_bytes[i]):
                if selection and used + case_bytes[case_id] > TEST_DATA_MAX_BYTES:
                    continue
                selection.append(case_id)
                used += case_bytes[case_id]
            selection = sorted(selection)
        else:
            wanted = {int(token) for token in re.split(r"[,\s]+", spec) if token.strip()}
            selection = [case_id for case_id in all_ids if case_id in wanted]

    staged, manifest_source = {}, None
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        match = re.fullmatch(r"case_(\d+)\.pt", path.name)
        if match and selection is not None and int(match.group(1)) not in selection:
            continue
        rel = path.relative_to(root).as_posix()
        if rel == "cases_manifest.json":
            manifest_source = path
            continue
        staged[f"{OP_NAME}/test_data/{rel}"] = base64.b64encode(path.read_bytes()).decode("ascii")

    if manifest_source is not None:
        if selection is None:
            blob = manifest_source.read_bytes()
        else:
            document = json.loads(manifest_source.read_text(encoding="utf-8"))
            if isinstance(document.get("cases"), list):
                document["cases"] = [c for c in document["cases"] if c.get("id") in set(selection)]
            document["staged_cases"] = selection
            blob = json.dumps(document, ensure_ascii=False, indent=2).encode("utf-8")
        staged[f"{OP_NAME}/test_data/cases_manifest.json"] = base64.b64encode(blob).decode("ascii")

    staged_ids = selection if selection is not None else all_ids
    summary = {
        "dir": str(root),
        "cases_available": all_ids,
        "cases_staged": staged_ids,
        "bytes_available": sum(case_bytes.values()),
        "bytes_staged": sum(case_bytes[i] for i in staged_ids),
    }
    return staged, summary


def post_run_code(payload, timeout):
    request = urllib.request.Request(
        SANDBOX_URL + "/run_code",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def try_post_run_code(payload, timeout):
    """post_run_code 的容错版：返回 (response, error)。

    传输故障（HTTP 状态、连接失败、超时、响应不可解析）不再是「直接作废整次评测」，
    而是返回 (None, 原因)，由调用方按可重试的基础设施故障处理。
    `timeout` <= 0 说明预算已耗尽，直接放弃本次尝试。
    """
    if timeout <= 0:
        return None, f"evaluation budget exhausted before dispatch (timeout={timeout:.0f}s)"
    try:
        return post_run_code(payload, timeout), None
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = tail(exc.read().decode("utf-8", "replace") if exc.fp else "", 2000)
        except Exception:  # noqa: BLE001 - 诊断信息尽力而为，不能因为读 body 失败再抛
            body = ""
        return None, f"sandbox HTTP {exc.code}: {body}"
    except urllib.error.URLError as exc:
        return None, f"sandbox connection error: {exc.reason}"
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


def dump_b64(artifacts, name, blob):
    try:
        data = base64.b64decode(blob)
    except Exception:
        return None
    target = artifacts / name
    target.write_bytes(data)
    return data


def mean_column(rows, candidates):
    for key in candidates:
        values = [row[key] for row in rows]
        numbers = []
        for value in values:
            try:
                numbers.append(float(str(value).replace(",", "")))
            except (TypeError, ValueError):
                pass
        if numbers:
            return statistics.fmean(numbers), len(numbers)
    return None, 0


def parse_profiling(zip_path, artifacts, total_runs=None, warmup_runs=0):
    """Return (duration_ms, report, hardware_metrics) or (None, report, {})."""
    extract_dir = artifacts / "profile_output"
    extract_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(extract_dir)
    csv_files = sorted(extract_dir.rglob("*.csv"))
    report = ["# SandboxFusion profiling", "", f"archive: {zip_path.name}", ""]
    if not csv_files:
        report.append("no CSV files found in profiling archive")
        return None, "\n".join(report), {}

    summary_file, summary_rows, duration_key = None, None, None
    for candidate in csv_files:
        with candidate.open(newline="", encoding="utf-8", errors="replace") as handle:
            reader = csv.DictReader(handle)
            rows = [row for row in reader]
        if not rows:
            continue
        duration_keys = [k for k in rows[0] if k and "task duration" in k.lower()]
        if not duration_keys:
            continue
        op_keys = [k for k in rows[0] if k and k.strip().lower() in ("op name", "op type", "kernel name")]
        if not op_keys:
            op_keys = [k for k in rows[0] if k and "name" in k.lower()]
        summary_file, summary_rows, duration_key = candidate, rows, duration_keys[0]
        report.append(f"summary csv: {candidate.relative_to(extract_dir)}")
        report.append(f"op column: {op_keys[0] if op_keys else '(none)'}")
        report.append(f"duration column: {duration_key}")
        break

    if summary_rows is None:
        report.append("no op_summary style CSV with Task Duration column found")
        report.append("files: " + ", ".join(p.name for p in csv_files[:20]))
        return None, "\n".join(report), {}

    op_key = [k for k in summary_rows[0] if k and k.strip().lower() in ("op name", "op type", "kernel name")]
    op_key = op_key[0] if op_key else None
    task_type_key = next((k for k in summary_rows[0] if k and k.strip().lower() == "task type"), None)

    start_key = next((k for k in summary_rows[0] if k and "task start time" in k.lower()), None)
    per_op = {}
    for row in summary_rows:
        value = row.get(duration_key)
        try:
            duration = float(str(value).replace(",", ""))
        except (TypeError, ValueError):
            continue
        if duration <= 0:
            continue
        if task_type_key and "AI" not in str(row.get(task_type_key, "")).upper():
            continue
        label = str(row.get(op_key, "unknown")) if op_key else "unknown"
        try:
            start = float(str(row.get(start_key)).replace(",", "")) if start_key else 0.0
        except (TypeError, ValueError):
            start = 0.0
        per_op.setdefault(label, []).append((start, duration))

    if not per_op:
        report.append("no positive kernel duration samples found")
        return None, "\n".join(report), {}

    # 丢弃前 warmup_runs 轮（每轮样本数 = 该 kernel 总样本数 / 总轮数），只对剩余轮次取平均。
    dropped_samples = dropped_runs = 0
    used_per_op = {}
    for label, items in per_op.items():
        items.sort()
        samples = [d for _, d in items]
        if warmup_runs and total_runs and total_runs > 1 and len(samples) % total_runs == 0:
            per_run = len(samples) // total_runs
            drop_n = min(len(samples) - 1, warmup_runs * per_run) if len(samples) > 1 else 0
            used_per_op[label] = samples[drop_n:]
            dropped_samples += drop_n
            dropped_runs = warmup_runs
        else:
            used_per_op[label] = samples

    total_us = 0.0
    report.append("")
    if total_runs:
        report.append(f"profiling runs: total={total_runs}, warmup dropped={warmup_runs}, "
                      f"measured={max(0, total_runs - warmup_runs)}")
    report.append("| op | samples(used/total) | mean us | min us | max us |")
    report.append("|---|---|---|---|---|")
    for label, samples in sorted(used_per_op.items()):
        mean = statistics.fmean(samples)
        total_us += mean
        total_samples = len(per_op[label])
        report.append(f"| {label} | {len(samples)}/{total_samples} | {mean:.3f} | "
                      f"{min(samples):.3f} | {max(samples):.3f} |")
    report.append("")
    report.append(f"sum of per-op mean kernel time: {total_us:.3f} us")
    report.append(f"samples used: {sum(len(v) for v in used_per_op.values())} "
                  f"(dropped {dropped_samples} warmup samples from {dropped_runs} runs)")

    hardware = {}
    aicore, _ = mean_column(summary_rows, [k for k in summary_rows[0] if k and "aicore time" in k.lower()])
    aiv, _ = mean_column(summary_rows, [k for k in summary_rows[0] if k and "aiv time" in k.lower()])
    block, _ = mean_column(summary_rows, [k for k in summary_rows[0] if k and "block dim" in k.lower()])
    if aicore is not None:
        hardware["aicore_time_us"] = aicore
    if aiv is not None:
        hardware["aiv_time_us"] = aiv
    if block is not None:
        hardware["core_count"] = block
    report.append("")
    report.append("hardware metrics: " + json.dumps(hardware, ensure_ascii=False))
    return total_us / 1000.0, "\n".join(report), hardware


def tail(text, limit=4000):
    text = text or ""
    return text[-limit:]


# 构建/测试输出里，真正的失败未必在末尾：CMake 报错后仍会继续构建其余 target
# （实测一份 126KB 的 build_output.log，报错在第 459 行，最后 4000 字符全是报错之后
# 的 protobuf host 构建噪音），失败用例也可能出现在中途，甚至就在开头（例如
# `op main_exec_code.py: dir=<missing>`）。只取尾部会切掉这些行；而下发完整日志
# 不可行 —— 失败的评测背后 45 份 build log 中位 150KB、最大 528KB，约 3.7 万 token/份。
#
# 因此取三个互补窗口，任一都能独立命中，不依赖任何单一措辞：
#   head   —— 无条件保留开头，兜住"错误就在最开头"的情况，无论它长什么样
#   errors —— 评测器自己的判定标记（SUCCESS_RE/FAILURE_RE/DEVICE_ERROR_RE）加上
#             通用构建错误形态，命中中段
#   tail   —— 兜住最终判定与只出现在末尾的设备故障码
BUILD_ERROR_RE = re.compile(
    # 评测器据以判定的标记：按构造必然相关，优先级最高
    r"REGRESSION DETECTED|\[FAIL\]|\b[1-9]\d*\s+(failed|error)\b|"
    r"Traceback \(most recent call last\)|"
    # 逐用例结果表。表格用裸 FAIL，既不匹配 \[FAIL\] 也不匹配 "N failed"，
    # 而这张表——哪个用例挂了、max_diff 多少——正是 agent 修复所需的信息。
    r"^\s*case[ _]?\S*\s|\bFAIL\b|\bFAILED\b|"
    # 设备故障码，使 agent 在内联文本里也看得到，而不只是归因行
    r"5070\d\d|aivec error|vector core (?:timeout|execution is abnormal)|"
    r"AclrtSynchronizeDeviceWithTimeout|ACL stream synchronize failed|"
    # 通用构建失败形态
    r"error:|fatal error|CMake Error|undefined reference|"
    r"gmake.*\*\*\*|make.*\*\*\* Error|Error \d+$|"
    r"not declared in this scope|cannot find|No such file|"
    r"错误|失败|异常", re.I)
# CMake 的特性探测会打印 "- Failed"，那不是错误；Git fetch 失败是可恢复的网络噪音
BUILD_ERROR_NOISE_RE = re.compile(
    r"Performing Test|Git fetch failed|-- (Found|Could NOT find|Configuring done)", re.I)
# 与失败行同属一个失败区域、但自身不含错误关键字的行（结果表的数据行、分隔线、
# traceback 体、编译器回显的源码行、make 链）。用于判断 head 窗口是否被切断。
REGION_CONT_RE = re.compile(
    r"^\s*case[ _]?\S*\s|^[-=]{5,}\s*$|^\s*\d+\s*\||^\s*File \"|^\s*\^+\s*$|"
    r"^\s*(gmake|make)\b|^\s*(at |in )\S|\bError\b|^\s*Traceback|Golden data|"
    r"^\s*\[ERROR\]|^\s*\[WARNING\]|^\s*\|", re.I)


# 三个窗口的字符数由实测定：对 109 份唯一的失败日志（54 次失败评测，按内容去重，
# 去掉重试造成的重复）测量「决定性失败行」的位置分布——
#   head：run_stdout 的首个失败行 min 1,718 / 中位 1,937 / max 3,012 字符
#         -> 2,000 只覆盖 51%，4,000 覆盖 100%
#   tail：run_stdout 末个失败行距结尾 min 1,709 / 中位 5,172 / max 5,182 字符
#         -> 4,000 只覆盖 24%，6,000 覆盖 100%
#   context：失败区域长度中位 5 行、p90 26 行、max 97 行
#         -> ±1 行只覆盖 52.5%，±12 行覆盖 91.8%
#   error cap：24,000 已经给到 91.8%；加到 32,000/40,000 一个百分点都不涨，
#        只是把 p95 从 34K 推到 36-40K，故停在 24,000
# 注意 build_output.log 既不靠 head 也不靠 tail：它的首个失败行中位在第 62,328
# 字符（日志中位 15 万字符），距离结尾中位 12,440 字符 —— 对它只有 errors 窗口有效。
#
# 三段按行号划分、互不重复（见 excerpt）：实测未去重时 run_stdout 有 37.8% 的
# 下发量是 head/tail/errors 各发一遍的同一批行。去重后 62 份唯一失败日志：
# 决定性锚点行 ±12 行 100% 可见，整段失败区域 98.9% 可见，下发量中位 9,940 字符。
HEAD_LIMIT = 4000
TAIL_LIMIT = 6000
ERR_CONTEXT = 12
ERR_CHAR_CAP = 24000
# 含错误关键字的行在 head 窗口内不做常规截短（见 clip_long_lines），只设防病态硬上限
ERROR_LINE_HARD_CAP = 4000


def _error_index_set(lines, limit, context, exclude=None):
    """承载失败信息的行号集合（各带 context 行上下文）。

    匹配行多于 limit 时取「最先 limit/2 条 + 最后 limit/2 条」：致命错误通常排在
    一串 warning 之后，只取前 N 条会把真正的失败点挤掉。相邻命中行的上下文窗口
    会合并，因此一整张结果表只算一个窗口、不会按行长线性膨胀。

    exclude 是已被 head/tail 原样覆盖的行号。**必须先剔除再计数**：否则当命中行
    有上百条时，「取两端」选中的恰好全落在 head/tail 里，中间那段既没进 head、
    也没进 tail、又没能进 errors，整段消失（实测 34 行的空洞）。
    """
    exclude = exclude or set()
    hits = [i for i, l in enumerate(lines)
            if BUILD_ERROR_RE.search(l) and not BUILD_ERROR_NOISE_RE.search(l)]
    if not hits:
        return set()
    uncovered = [i for i in hits if i not in exclude]
    if len(uncovered) > limit:
        half = limit // 2
        hits = uncovered[:half] + uncovered[-half:]
    elif uncovered:
        hits = uncovered
    else:
        hits = hits[:limit]     # 全部命中都在 head/tail 里：无需再补
    keep = set()
    for i in hits:
        keep.update(range(max(0, i - context), min(len(lines), i + context + 1)))
    return keep - exclude


def error_lines(text, limit=25, context=ERR_CONTEXT, char_cap=ERR_CHAR_CAP):
    """错误行块（各带 context 行上下文），没有就返回空串。"""
    lines = (text or "").splitlines()
    keep = _error_index_set(lines, limit, context)
    if not keep:
        return ""
    return "\n".join(lines[i] for i in sorted(keep))[:char_cap]


def clip_long_lines(text, max_line=500, max_lines=500):
    """把超长行截短，避免一条噪音行独占整个 head 窗口。

    实测有 stdout 以 `cwd entries: [...]` 开头，单行 1300+ 字符，把 head 窗口
    全部吃掉，紧跟其后的逐用例结果表一行都进不来。

    max_lines 只是防病态的阀门：head 的长度已由字符预算（HEAD_LIMIT + 扩展）
    限死，行数上限若取得太小会反过来切断窗口——实测 max_lines=90 会把设备故障
    stderr 里第 90 行之后的错误行整批丢掉。
    """
    out = []
    for i, l in enumerate((text or "").splitlines()):
        if i >= max_lines:
            out.append("…")
            break
        if len(l) <= max_line:
            out.append(l)
        elif BUILD_ERROR_RE.search(l) or REGION_CONT_RE.search(l):
            # 承载错误的行不能截：实测设备故障行有 507-560 字符，按 500 截会切掉
            # 尾部的 error code。只做一个防病态的硬上限。
            out.append(l if len(l) <= ERROR_LINE_HARD_CAP
                       else l[:ERROR_LINE_HARD_CAP] + f" …〔本行共 {len(l)} 字符，已截断〕")
        else:
            out.append(l[:max_line] + f" …〔本行共 {len(l)} 字符，已截断〕")
    return "\n".join(out)


def _head_line_end(lines, head_limit, tail_start, extend_cap):
    """head 窗口吃掉的行数 [0, end)。

    先按字符预算装行，然后——若末尾仍落在失败区域内——继续吸收该区域的行。
    实测有 4 份日志的失败区域紧贴文件开头（偏移 1-15 行）却长达 66-97 行，
    按字符截断会在区域中间切断，而区域内没有可供 errors 窗口匹配的行，
    两边都落空。扩展永不越过 tail 窗口起点，否则 head 与 tail 会互相覆盖。
    """
    used = 0
    end = 0
    for i, l in enumerate(lines):
        if used + len(l) + 1 > head_limit:
            break
        used += len(l) + 1
        end = i + 1
    if end == 0 and lines:                      # 首行就超预算，至少给一行
        end = 1
    extra = 0
    while end < tail_start:
        l = lines[end]
        if not (BUILD_ERROR_RE.search(l) or REGION_CONT_RE.search(l)):
            break
        extra += len(l) + 1
        end += 1
        if extra >= extend_cap:
            break
    return end


def _tail_line_start(lines, tail_limit):
    """tail 窗口的起始行号 [start, len(lines))。"""
    used = 0
    start = len(lines)
    for i in range(len(lines) - 1, -1, -1):
        if used + len(lines[i]) + 1 > tail_limit:
            break
        used += len(lines[i]) + 1
        start = i
    return start


def head_window(text, head_limit=HEAD_LIMIT, extend_cap=16000):
    """head 窗口文本（对外保留的字符串接口，内部走行号）。"""
    text = text or ""
    if len(text) <= head_limit:
        return text
    lines = text.splitlines()
    end = _head_line_end(lines, head_limit, len(lines), extend_cap)
    return "\n".join(lines[:end])


def excerpt(text, label, head_limit=HEAD_LIMIT, tail_limit=TAIL_LIMIT):
    """开头 + 判定/错误行 + 结尾，三段互不重复。

    尾部始终原样保留（设备故障码等只在末尾出现的信号不会丢），开头也始终保留
    （措辞不在任何模式里、位置又恰好在最开头的错误不会丢）。错误行是纯增补。
    三段按行号划分，errors 只负责 head 与 tail 都没覆盖到的行——否则日志末尾的
    `gmake: *** Error 2` 这类命中行会被 head/errors/tail 各发一遍：实测在
    run_stdout 上曾有 37.8% 的下发量是纯重复。
    head 窗口内会截短超长行，使窗口不被单条噪音行占满。
    """
    text = text or ""
    if len(text) <= head_limit + tail_limit:
        # 整个文本就比窗口还短，不做任何省略（也与旧格式逐字节一致）
        return f"{label} tail: {text}"
    lines = text.splitlines()
    n = len(lines)
    tail_start = _tail_line_start(lines, tail_limit)
    head_end = _head_line_end(lines, head_limit, tail_start, 16000)
    covered = set(range(0, head_end)) | set(range(tail_start, n))

    # 命中行 ±context；已由 head/tail 覆盖的行不再重复下发
    err_idx = sorted(_error_index_set(lines, 25, ERR_CONTEXT, exclude=covered))

    parts = [f"{label} head:\n{clip_long_lines(chr(10).join(lines[:head_end]))}"]
    if err_idx:
        block = "\n".join(lines[i] for i in err_idx)[:ERR_CHAR_CAP]
        parts.append(f"{label} errors:\n{block}")
    parts.append(f"{label} tail: {chr(10).join(lines[tail_start:])}")
    return "\n".join(parts)


def resolve_profiling(request):
    """把请求里的固定采样协议解析成 (测量轮数, 预热轮数, 回显协议, 错误信息)。

    请求未带 profiling 时返回环境变量定义的旧行为（回显协议为 None）。
    协议非法时不猜、不放宽：返回错误信息，由调用方拒绝整次评测。
    """
    block = request.get("profiling")
    if block is None:
        return PROFILING_RUNS, WARMUP_RUNS, None, None
    if not isinstance(block, dict) or set(block) != {"total_runs", "warmup_runs", "statistic"}:
        return None, None, None, ("profiling 必须是 {total_runs, warmup_runs, statistic}，"
                                  f"当前 {block!r}")
    total, warmup = block["total_runs"], block["warmup_runs"]
    if type(total) is not int or total < 1:
        return None, None, None, f"profiling.total_runs 必须是正整数，当前 {total!r}"
    if type(warmup) is not int or not 0 <= warmup < total:
        return None, None, None, (f"profiling.warmup_runs 必须是 [0, total_runs) 内的整数，"
                                  f"当前 warmup={warmup!r} total={total!r}")
    if block["statistic"] != "mean":
        return None, None, None, ("本评测器只实现 profiling.statistic=mean，"
                                  f"当前 {block['statistic']!r}")
    return total - warmup, warmup, dict(block), None


def run_root_from_artifacts(artifacts):
    """<run_root>/evaluations/E#### -> <run_root>；目录结构不符时退回 artifacts 自身。

    连续故障计数存在 run 根目录，才能跨同一次 run 的多次评测累加。
    """
    parent = artifacts.parent
    if parent.name == "evaluations":
        return parent.parent
    return artifacts


def infra_state_path(artifacts):
    return run_root_from_artifacts(artifacts) / "infra_retry_state.json"


def baseline_passed(artifacts):
    """本 run 的 baseline 是否已经 passed（读 run 根目录的 evaluations.jsonl）。

    用来区分两类设备故障：baseline 已通过的前提下某个候选触发故障，故障几乎只可能
    来自候选实现本身；baseline 尚未通过（或 baseline 自己就故障）时，才归为基础设施问题。
    """
    path = run_root_from_artifacts(artifacts) / "evaluations.jsonl"
    if not path.is_file():
        return False
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return False
    for line in lines:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get("evaluation_kind") == "baseline":
            return row.get("status") == "passed"
    return False


def load_infra_streak(artifacts):
    """读「连续基础设施故障」计数；缺失或损坏一律当 0（宁可多给一次机会）。"""
    try:
        data = json.loads(infra_state_path(artifacts).read_text(encoding="utf-8"))
        return max(0, int(data.get("consecutive_infra_faults", 0)))
    except (OSError, ValueError, TypeError):
        return 0


def save_infra_streak(artifacts, streak, cause=None):
    """原子写入连续故障计数。写不进去也不影响本次判定，只是下次从头计数。"""
    path = infra_state_path(artifacts)
    payload = {"consecutive_infra_faults": int(streak),
               "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if cause:
        payload["last_cause"] = cause
    try:
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)
    except OSError as exc:
        log(f"[infra] 无法写入重试状态 {path}: {exc}（本次判定不受影响）")


def transport_outcome(message):
    """把「评测器与沙盒之间的传输故障」表达成可重试的基础设施结果。

    以前 HTTPError/URLError/超时直接落到 main() 的 except 里，立刻定成致命的
    sandbox_error 且不重试；现在与设备故障走同一条重试/降级路径。
    """
    info = {"attempt": None, "assigned_server": None, "total_time": None, "compile_ok": None,
            "run_status": None, "run_rc": None, "success_marker": None, "failure_marker": None,
            "device_error": [], "missing_verdict": False, "transport_error": message}
    return {"status": "infrastructure_suspect", "correctness_passed": None, "duration_ms": None,
            "profiling_report": "", "hardware_metrics": {},
            "diagnostic": f"evaluator/sandbox transport failure: {message}", "info": info}


def classify_attempt(artifacts, response, attempt, measured_runs=None, warmup_runs=None):
    """跑完一次沙盒尝试后的落盘与判定，返回 harness 需要的字段 + info。

    判定顺序：编译失败 -> 设备故障/无判定（可重试的基础设施信号）-> 真实测试失败
    -> profiling 失败 -> 通过。设备故障与"无判定"不会被记成候选失败。
    """
    measured_runs = PROFILING_RUNS if measured_runs is None else measured_runs
    warmup_runs = WARMUP_RUNS if warmup_runs is None else warmup_runs
    returned = response.get("files") or {}
    for name, blob in returned.items():
        dump_b64(artifacts, name.replace("/", "_"), blob)
    slim = dict(response)
    slim["files"] = {name: f"<{len(base64.b64decode(blob)) if blob else 0} bytes>"
                     for name, blob in returned.items()}
    response_name = "sandbox_response.json" if attempt == 1 else f"sandbox_response.attempt{attempt}.json"
    (artifacts / response_name).write_text(json.dumps(slim, ensure_ascii=False, indent=2), encoding="utf-8")

    compile_res = response.get("compile_result") or {}
    run_res = response.get("run_result") or {}
    out = run_res.get("stdout") or ""
    err = run_res.get("stderr") or ""
    suffix = "" if attempt == 1 else f".attempt{attempt}"
    (artifacts / f"compile_stdout{suffix}.txt").write_text(compile_res.get("stdout") or "", encoding="utf-8")
    (artifacts / f"compile_stderr{suffix}.txt").write_text(compile_res.get("stderr") or "", encoding="utf-8")
    (artifacts / f"run_stdout{suffix}.txt").write_text(out, encoding="utf-8")
    (artifacts / f"run_stderr{suffix}.txt").write_text(err, encoding="utf-8")
    build_log = artifacts / f"build_output{suffix}.log"
    if "build_output.log" in returned:
        build_log.write_text(base64.b64decode(returned["build_output.log"]).decode("utf-8", "replace"),
                             encoding="utf-8")

    compile_ok = compile_res.get("status") == "Finished" and compile_res.get("return_code") == 0
    rc = run_res.get("return_code")
    success_marker = SUCCESS_RE.search(out)
    failure_marker = FAILURE_RE.search(out)
    build_log_text = build_log.read_text(encoding="utf-8", errors="replace") if build_log.is_file() else ""
    # 设备故障可能只出现在编译输出或 build_output.log 里（run stdout 干净），
    # 只扫 run stdout/stderr 会把它误判成候选的 compile_failed / test_failed。
    device_scan = "\n".join((out, err, compile_res.get("stdout") or "",
                             compile_res.get("stderr") or "", build_log_text))
    device_hits = sorted({m.group(0) for m in DEVICE_ERROR_RE.finditer(device_scan)})
    missing_verdict = rc == 0 and success_marker is None and failure_marker is None
    info = {
        "attempt": attempt,
        "assigned_server": response.get("assigned_server"),
        "total_time": response.get("total_time"),
        "compile_ok": compile_ok,
        "run_status": run_res.get("status"),
        "run_rc": rc,
        "success_marker": success_marker.group(0) if success_marker else None,
        "failure_marker": failure_marker.group(0) if failure_marker else None,
        "device_error": device_hits,
        "missing_verdict": missing_verdict,
    }
    verdict_line = (f"success marker: {info['success_marker'] or 'none'}; "
                    f"failure marker: {info['failure_marker'] or 'none'}")
    outcome = {"status": "sandbox_error", "correctness_passed": None, "duration_ms": None,
               "profiling_report": "", "hardware_metrics": {}, "diagnostic": "", "info": info}

    if not compile_ok:
        if device_hits:
            # 设备故障发生在编译阶段：这是基础设施问题，不是候选代码编译不过。
            cause = "device error during compile: " + ", ".join(device_hits)
            outcome["status"] = "infrastructure_suspect"
            outcome["diagnostic"] = (
                f"compile status={compile_res.get('status')} rc={compile_res.get('return_code')}\n"
                f"retryable infrastructure signal: {cause}\n"
                f"stderr: {tail(compile_res.get('stderr'))}\n"
                + excerpt(build_log_text, "build log"))
            return outcome
        outcome["status"] = "compile_failed"
        outcome["diagnostic"] = (
            f"compile status={compile_res.get('status')} rc={compile_res.get('return_code')}\n"
            f"stderr: {tail(compile_res.get('stderr'))}\n"
            + excerpt(build_log_text, "build log"))
        return outcome

    if device_hits or missing_verdict:
        cause = ("device error: " + ", ".join(device_hits)) if device_hits else "no verdict marker"
        outcome["status"] = "infrastructure_suspect"
        outcome["diagnostic"] = (
            f"run status={run_res.get('status')} rc={rc}\n"
            f"retryable infrastructure signal: {cause}\n"
            f"{verdict_line}\n"
            + excerpt(out, "stdout") + "\n"
            + excerpt(err, "stderr"))
        return outcome

    if rc != 0 or success_marker is None or failure_marker is not None:
        outcome["status"] = "test_mismatch" if failure_marker else "test_failed"
        outcome["correctness_passed"] = False
        outcome["diagnostic"] = (
            f"run status={run_res.get('status')} rc={rc}\n{verdict_line}\n"
            + excerpt(out, "stdout") + "\n"
            + excerpt(err, "stderr"))
        return outcome

    if "profile_output.zip" not in returned:
        outcome["status"] = "profiling_failed"
        outcome["diagnostic"] = ("run passed but the profiling archive was not returned; available files: "
                                 + ", ".join(returned))
        return outcome

    duration_ms, report, hardware = parse_profiling(
        artifacts / "profile_output.zip", artifacts,
        total_runs=measured_runs + warmup_runs, warmup_runs=warmup_runs)
    if duration_ms is None:
        outcome["status"] = "profiling_failed"
        outcome["profiling_report"] = report
        outcome["diagnostic"] = "profiling archive could not be converted into kernel durations"
        return outcome

    outcome.update({
        "status": "passed",
        "correctness_passed": True,
        "duration_ms": duration_ms,
        "profiling_report": report,
        "hardware_metrics": hardware,
        "diagnostic": "correctness passed; kernel durations parsed from msprof op summary",
    })
    return outcome


def main():
    request = json.load(sys.stdin)
    kind = request["evaluation_kind"]
    source = Path(request["source_dir"]).resolve()
    artifacts = Path(request["artifacts_dir"]).resolve()
    artifacts.mkdir(parents=True, exist_ok=True)

    result = {
        "status": "sandbox_error",
        "correctness_passed": None,
        "duration_ms": None,
        "profiling_report": "",
        "hardware_metrics": {},
        "diagnostic": "",
        "raw": {"sandbox_url": SANDBOX_URL, "ops_repo": OPS_REPO, "evaluation_kind": kind},
    }

    measured_runs, warmup_runs, protocol, protocol_error = resolve_profiling(request)
    if protocol_error:
        result["diagnostic"] = ("拒绝评测：请求携带的 profiling 协议非法或本评测器不支持 —— "
                                + protocol_error)
        result["raw"]["profiling_rejected"] = request.get("profiling")
        log(result["diagnostic"])
        print(json.dumps(result, ensure_ascii=False))
        sys.exit(2)
    if protocol is not None:
        # native v2+ 要求原样回显，否则 harness 判定为"混合测量"并拒绝结果。
        result["raw"]["profiling_protocol"] = protocol
        if (measured_runs, warmup_runs) != (PROFILING_RUNS, WARMUP_RUNS):
            log(f"profiling: 请求协议覆盖环境变量 "
                f"(env measured={PROFILING_RUNS} warmup={WARMUP_RUNS} -> "
                f"request measured={measured_runs} warmup={warmup_runs})")

    try:
        files = collect_files(source, TEST_DIR)
        exec_code = read_exec_code(source, TEST_DIR)
        data_files, data_summary = stage_test_data()
        files.update(data_files)
        total_runs = measured_runs + warmup_runs
        log(f"profiling: measured_runs={measured_runs} warmup_runs={warmup_runs} total_runs={total_runs}"
            + ("（来自请求协议）" if protocol is not None else "（来自环境变量）"))
        server_code = profiling_wrapper(EXEC_SCRIPT_NAME, "profile_output", "profile_output.zip", total_runs)
        files[EXEC_SCRIPT_NAME] = base64.b64encode(exec_code.encode("utf-8")).decode("ascii")
        payload = {
            "code": server_code,
            "language": "ascend_c",
            "lint": False,
            "ops_repo": OPS_REPO,
            "experimental": EXPERIMENTAL,
            "run_timeout": RUN_TIMEOUT,
            "compile_timeout": COMPILE_TIMEOUT,
            "required_npu": 1,
            "files": files,
            "fetch_files": ["build_output.log", "profile_output.zip"],
        }
        if NPU_TYPES:
            payload["required_npu_types"] = NPU_TYPES
        preferred_node = os.environ.get("SANDBOX_PREFERRED_NODE_ID")
        if preferred_node:
            payload["preferred_node_id"] = preferred_node
        preferred_npu = os.environ.get("SANDBOX_PREFERRED_NPU_ID")
        if preferred_npu not in (None, ""):
            payload["preferred_npu_id"] = int(preferred_npu)
        result["raw"]["file_count"] = len(files)
        if data_summary:
            result["raw"]["test_data"] = data_summary
            log(f"staged golden cases {data_summary['cases_staged']} "
                f"({data_summary['bytes_staged']} bytes of {data_summary['bytes_available']})")
        log(f"submitting {len(files)} files to {SANDBOX_URL}/run_code "
            f"(kind={kind}, repo={OPS_REPO}, op={OP_NAME})")

        # baseline 已通过 + 本次是候选：故障大概率是候选实现缺陷，用更小的重试预算
        # （仍留 CANDIDATE_EVAL_RETRIES 次重试兜住偶发抖动，但不再为确定性故障拖满 4 次）
        if kind == "candidate" and baseline_passed(artifacts):
            attempts = CANDIDATE_EVAL_RETRIES + 1
        else:
            attempts = EVAL_RETRIES + 1
        attempt_infos, outcome = [], None
        started_at = time.monotonic()
        natural_timeout = COMPILE_TIMEOUT + RUN_TIMEOUT + 120
        for attempt in range(1, attempts + 1):
            # 单次 HTTP 超时必须留在预算内，否则会被 harness 先 SIGKILL，
            # 那次评测就变成 state_unknown 并作废整个 run。
            remaining = EVAL_BUDGET_SECONDS - (time.monotonic() - started_at)
            response, transport_error = try_post_run_code(payload, min(natural_timeout, remaining))
            if response is None:
                outcome = transport_outcome(transport_error)
            else:
                result["raw"]["assigned_server"] = response.get("assigned_server")
                result["raw"]["total_time"] = response.get("total_time")
                outcome = classify_attempt(artifacts, response, attempt,
                                           measured_runs=measured_runs, warmup_runs=warmup_runs)
            info = outcome["info"]
            attempt_infos.append({k: v for k, v in info.items() if k != "attempt"})
            returned = (response or {}).get("files") or {}
            result["raw"].setdefault("files", {}).update(
                {name: f"<{len(base64.b64decode(blob)) if blob else 0} bytes>"
                 for name, blob in returned.items()})
            if outcome["status"] != "infrastructure_suspect" or attempt >= attempts:
                break
            backoff = min(EVAL_RETRY_BACKOFF * (2 ** (attempt - 1)), EVAL_RETRY_MAX_BACKOFF)
            if time.monotonic() - started_at + backoff >= EVAL_BUDGET_SECONDS:
                log(f"[infra] 预算 {EVAL_BUDGET_SECONDS:.0f}s 不足以再等 {backoff:.0f}s，停止重试")
                break
            cause = ", ".join(info["device_error"]) or info.get("transport_error") or "no verdict marker"
            log(f"[retry] attempt {attempt}/{attempts} hit {cause}; "
                f"re-running the whole evaluation in {backoff:.0f}s")
            if backoff:
                time.sleep(backoff)
        result["raw"]["attempts"] = attempt_infos
        result["raw"]["eval_elapsed_seconds"] = round(time.monotonic() - started_at, 1)
        if outcome["status"] == "infrastructure_suspect":
            cause = ", ".join(outcome["info"]["device_error"]) or outcome["info"].get("transport_error") or "no verdict marker"
            log(f"[infrastructure] giving up after {len(attempt_infos)} attempt(s): {cause}")
            # 归因：baseline 已通过 + 本次是候选 + 命中设备故障 => 候选实现把设备打挂。
            # 这类故障稳定跟代码复现，必须判成候选失败（非致命）并明确劝阻原样重交，
            # 既不能算成基础设施（会误导 agent "等待后重交同一版本"），也不能让它反复消耗额度。
            if kind == "candidate" and outcome["info"]["device_error"] and baseline_passed(artifacts):
                result["status"] = CANDIDATE_FAULT_STATUS
                result["correctness_passed"] = False
                result["raw"]["device_fault_attributed_to"] = "candidate"
                result["diagnostic"] = (
                    "[DEVICE-FAULT] "
                    + CANDIDATE_FAULT_GUIDANCE.format(cause=cause, attempts=len(attempt_infos))
                    + "\n\n原始诊断：\n" + outcome["diagnostic"])
                log(f"[device-fault] baseline 已通过，判定为候选实现缺陷"
                    f"（{CANDIDATE_FAULT_STATUS}，不按基础设施处理，不计入连续故障）")
            else:
                # 本评测器自己的重试已用尽。是否作废整个 run 由「连续故障计数」决定：
                #   未达阈值 -> 返回非致命状态，把设备故障原样交还 agent，让它稍后用相同代码重交，
                #               run 得以继续，已通过验证的候选不会被丢弃；
                #   达到阈值 -> 如实返回 sandbox_error（harness 会收尾该 run），
                #               随后由批次层的 run 级重试换时间/换设备重跑。
                streak = load_infra_streak(artifacts) + 1
                save_infra_streak(artifacts, streak, cause)
                result["raw"]["infra_error"] = True
                result["raw"]["infra_streak"] = streak
                result["correctness_passed"] = None
                if streak >= INFRA_MAX_STREAK:
                    result["status"] = "sandbox_error"
                    result["diagnostic"] = (
                        f"[{cause}] 评测连续 {len(attempt_infos)} 次命中基础设施故障，且本 run 已累计 "
                        f"{streak} 次连续故障（阈值 {INFRA_MAX_STREAK}）。已按基础设施错误如实上报，"
                        f"由批次层换时间/换设备重跑；不记为候选失败，也不作为 baseline 判据。\n"
                        + outcome["diagnostic"])
                    log(f"[infrastructure] 连续故障 {streak} 次达到阈值 {INFRA_MAX_STREAK}，"
                        f"交还 harness 收尾该 run")
                else:
                    result["status"] = INFRA_KEEPALIVE_STATUS
                    result["diagnostic"] = (
                        f"[INFRA] {cause}\n"
                        f"本 run 连续基础设施故障 {streak}/{INFRA_MAX_STREAK} 次，"
                        f"本次评测未取得有效判定（不记为候选失败）。\n"
                        f"{INFRA_GUIDANCE}\n"
                        + outcome["diagnostic"])
                    log(f"[infrastructure] 连续故障 {streak}/{INFRA_MAX_STREAK} 次，"
                        f"以非致命状态 {INFRA_KEEPALIVE_STATUS} 交还 agent 继续")
        else:
            save_infra_streak(artifacts, 0)  # 拿到真实判定即清零连续计数
            for key in ("status", "correctness_passed", "duration_ms",
                        "profiling_report", "hardware_metrics", "diagnostic"):
                if key in outcome:
                    result[key] = outcome[key]
    except Exception as exc:  # noqa: BLE001 - the harness records the diagnostic verbatim
        result["status"] = "sandbox_error"
        result["diagnostic"] = f"{type(exc).__name__}: {exc}"

    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
