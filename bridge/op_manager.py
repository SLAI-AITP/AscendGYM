#!/usr/bin/env python3
"""Operator catalog manager for the Ascend C optimization pipeline.

Reads a declarative operator list (`op_catalog.json`), resolves every operator
to its source tree / frozen test / golden data inside the given repositories,
then builds harness-ready input packages and run configs.

Subcommands:
  scan     resolve the catalog and write op_catalog.resolved.json
  build    materialise text-only input packages + spec files
  validate preflight every built package with the real harness constructor
  config   emit a harness run config for one operator and one agent
  run      config + launch the harness with the right sandbox environment

`validate` / `config` / `run` 支持 native 与 bare：native 生成 native_bench 的
input/agent/evaluator/stop 配置，bare 生成 k-lens-v2 bare_client 配置。native 没有
token/墙钟/轮数上限，旧预算字段会被显式忽略并记录日志。

Only stdlib is required.
"""
import argparse
import concurrent.futures
import fnmatch
import json
import os
import re
import shutil
import socket
import secrets
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BRIDGE_CONFIG_DIR = Path(__file__).resolve().with_name("configs")
DEFAULT_CATALOG = BRIDGE_CONFIG_DIR / "op_catalog.json"
DEFAULT_PYTHON = sys.executable

# --harness 的取值：native / bare
#   native = 原生 coding-agent harness（native_bench）
# native 的代码根目录需同时含 native_bench/ 与 core/（后者提供 metrics/contract）。
# 可用环境变量 NATIVE_HARNESS_ROOT 或 op_catalog.json 的 outputs.native_root 覆盖。
DEFAULT_NATIVE_ROOT = Path(__file__).resolve().parent.parent / "eval" / "native_harness"
# bare = k-lens-v2 bare_client（纯模型 + 工具池）：agent 是裸模型，评测网关是 k-lens-v2 Backend。
# 与 native 的区别见 bare_bench/runner.py 的模块注释。
HARNESS_KINDS = ("native", "bare")

# native claude 的保底重试 wrapper：Claude Code 自己以 is_error / aborted_streaming 收尾时
# （服务端流式中断），harness 会把它当 agent_error 强制收尾、跳过末次复评、状态记 unknown。
# 这里把 claude 的可执行文件换成 bridge 侧 wrapper，由它在同一个 session 上用 --resume
# 续跑，只有最终一次 attempt 仍失败才把非 0 退出码交回 harness（语义不变）。
# 可用 NATIVE_CLAUDE_RETRY=0 关闭，回退原生 CLI 直接拉起。
CLAUDE_RETRY_WRAPPER = Path(__file__).resolve().with_name("claude_agent_wrapper.py")

# native harness 直接拉起原生 CLI（不再经过 bridge wrapper），因此权限/网络设置必须
# 写进 run config：codex 需要访问 127.0.0.1 上的评测入口，claude 需要 Bash 等原生工具。
NATIVE_AGENT_DEFAULTS = {
    "claude": {
        "permission_mode": "acceptEdits",
        "allowed_tools": ["Read", "Edit", "Write", "Glob", "Grep", "Bash"],
    },
    "codex": {
        "sandbox": "workspace-write",
        "extra_args": ["-c", "sandbox_workspace_write.network_access=true",
                       "-c", 'approval_policy="never"'],
    },
    # opencode 在非交互 run 中无法弹出审批：任何需要 ask 的权限请求都会被自动拒绝
    # （实测会把 workspace/.bench/* 直接判成需审批目录，导致 agent 一开始就读不到任务描述）。
    # --auto 只放行"未被显式 deny"的请求，语义上与 codex 的 approval_policy=never、
    # claude 的 allowed_tools 一致，属于显式配置而非绕过 harness。
    "opencode": {
        "extra_args": ["--auto"],
    },
}
# native 的 metrics 只接受这些键（token/时间检查点是事后评分观察点，不是停止条件）。
NATIVE_METRIC_KEYS = ("token_checkpoints", "time_checkpoints_sec", "recipe")


def claude_retry_enabled():
    return os.environ.get("NATIVE_CLAUDE_RETRY", "1").strip().lower() not in ("0", "false", "no", "off")


def inject_claude_retry(agent, log_protocol=True):
    """给 native claude 注入保底重试 wrapper（agent.executable 是 harness 认可的字段）。

    wrapper 缺失/不可执行时静默回退原生 CLI，避免因为一个可选增强把整批评测卡死。
    """
    if agent.get("kind") != "claude":
        return
    if not claude_retry_enabled():
        if log_protocol:
            log("native claude 保底重试: 已关闭（NATIVE_CLAUDE_RETRY=0），直接拉起原生 claude CLI")
        return
    if not CLAUDE_RETRY_WRAPPER.is_file():
        if log_protocol:
            log(f"warning: 未找到保底重试 wrapper {CLAUDE_RETRY_WRAPPER}，回退原生 claude CLI")
        return
    if not os.access(CLAUDE_RETRY_WRAPPER, os.X_OK):
        if log_protocol:
            log(f"warning: 保底重试 wrapper 不可执行（chmod +x {CLAUDE_RETRY_WRAPPER}），回退原生 claude CLI")
        return
    agent["executable"] = str(CLAUDE_RETRY_WRAPPER)
    if log_protocol:
        retry_max = os.environ.get("CLAUDE_AGENT_RETRY_MAX", "2")
        retry_delay = os.environ.get("CLAUDE_AGENT_RETRY_DELAY", "15")
        real_bin = os.environ.get("CLAUDE_AGENT_BIN", "claude")
        log(f"native claude 保底重试: 已启用（真实 CLI={real_bin}，同一 session --resume 续跑，"
            f"最多 {retry_max} 次重试 / 间隔 {retry_delay}s；仅当 claude 以 is_error、"
            f"aborted_streaming 等异常收尾时触发，人工 STOP 与已关闭 admission 不重试；"
            f"wrapper={CLAUDE_RETRY_WRAPPER}）")


def log(message):
    print(redact_credentials(message), flush=True)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def expand_env(value):
    """Expand ${VAR}/$VAR in config values (URLs, keys)."""
    return os.path.expandvars(value) if isinstance(value, str) else value


def expand_path(value):
    """Expand environment variables and ~ in filesystem paths."""
    return os.path.expanduser(expand_env(value)) if isinstance(value, str) else value


def redact_credentials(value):
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            normalized = re.sub(r"[^a-z0-9]", "", str(key).lower())
            sensitive = (normalized in {"authorization", "proxyauthorization", "password", "secret", "credential", "credentials", "token", "auth"}
                         or normalized.endswith(("apikey", "authtoken", "accesstoken", "refreshtoken", "bearertoken", "usagetoken")))
            result[key] = "[REDACTED]" if sensitive else redact_credentials(item)
        return result
    if isinstance(value, (list, tuple)):
        return [redact_credentials(item) for item in value]
    if isinstance(value, str):
        value = re.sub(r"(https?://)[^/\s@]+@", r"\1[REDACTED]@", value)
        value = re.sub(r"([?&](?:api[_-]?key|key|token|access_token|auth|signature)=)[^&#\s]*",
                       r"\1[REDACTED]", value, flags=re.IGNORECASE)
    return value


def write_private_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        os.fchmod(handle.fileno(), 0o600)
        handle.write(text)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_private_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def strip_jsonc(text):
    """Remove // and /* */ comments outside strings (JSONC support)."""
    out, index, size = [], 0, len(text)
    in_string = escaped = False
    while index < size:
        char = text[index]
        if in_string:
            out.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue
        if char == '"':
            in_string = True
            out.append(char)
            index += 1
            continue
        if char == "/" and index + 1 < size and text[index + 1] == "/":
            while index < size and text[index] != "\n":
                index += 1
            continue
        if char == "/" and index + 1 < size and text[index + 1] == "*":
            index += 2
            while index + 1 < size and not (text[index] == "*" and text[index + 1] == "/"):
                index += 1
            index += 2
            continue
        out.append(char)
        index += 1
    return "".join(out)


def load_jsonc(path):
    """Parse JSON or JSONC (comments / trailing commas tolerated)."""
    text = Path(path).read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        cleaned = re.sub(r",(\s*[}\]])", r"\1", strip_jsonc(text))
        return json.loads(cleaned)


def global_opencode_config():
    """Locate the user-level opencode config managed by switch_model.sh."""
    candidates = []
    for key in ("MODEL_CONF_OPENCODE_FILE", "OPENCODE_GLOBAL_CONFIG"):
        value = os.environ.get(key)
        if value:
            candidates.append(Path(expand_path(value)))
    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        candidates += [Path(xdg) / "opencode/opencode.json", Path(xdg) / "opencode/opencode.jsonc"]
    home = Path.home()
    candidates += [home / ".config/opencode/opencode.json", home / ".config/opencode/opencode.jsonc"]
    candidates.append(Path.home() / ".config/opencode/opencode.json")
    for path in candidates:
        if path.is_file():
            return path
    return None


def is_text_file(path):
    try:
        path.read_bytes().decode("utf-8")
        return True
    except (UnicodeDecodeError, OSError):
        return False


def forbidden(name, patterns):
    return any(fnmatch.fnmatch(name, pattern) for pattern in patterns)


def collect_manifest_files(root, package_cfg):
    """Return (files, editable, skipped, total_bytes) for the text-only package."""
    keep = package_cfg["keep_top_level"]
    optional = package_cfg.get("optional_top_level", [])
    patterns = package_cfg.get("forbid_names", [])
    editable_prefixes = tuple(package_cfg.get("editable_prefixes", ["op_host/", "op_kernel/"]))
    files, editable, skipped = {}, [], []
    total = 0
    for top in list(keep) + list(optional):
        entry = root / top
        if not entry.exists():
            continue
        candidates = [entry] if entry.is_file() else sorted(p for p in entry.rglob("*") if p.is_file())
        for path in candidates:
            rel = path.relative_to(root).as_posix()
            if any(part == "__pycache__" for part in Path(rel).parts) or forbidden(path.name, patterns):
                skipped.append(rel)
                continue
            if not is_text_file(path):
                skipped.append(rel + " (non-utf8)")
                continue
            files[rel] = path
            total += path.stat().st_size
            if rel.startswith(editable_prefixes):
                editable.append(rel)
    return files, sorted(editable), skipped, total


def find_test_script(op_dir, op_name):
    direct = op_dir / f"{op_name}_test.py"
    if direct.is_file():
        return direct
    matches = sorted(op_dir.glob("*_test.py"))
    return matches[0] if matches else None


def test_kind(path):
    text = path.read_text(encoding="utf-8", errors="replace")
    if "pytest.main" in text or "import pytest" in text:
        return "pytest"
    if "ALL CHECKS PASSED" in text or "All tests PASSED" in text:
        return "golden-compare"
    return "custom"


def discover_data(op_dir):
    data = op_dir / "test_data"
    if not data.is_dir():
        return None
    info = {"dir": str(data.resolve()), "camel": None, "cases": [], "bytes": 0}
    for sub in ("golden_result", "golden_input"):
        base = data / sub
        if not base.is_dir():
            continue
        for camel_dir in sorted(p for p in base.iterdir() if p.is_dir()):
            if info["camel"] is None:
                info["camel"] = camel_dir.name
    if info["camel"]:
        result_dir = data / "golden_result" / info["camel"]
        cases = []
        for case in sorted(result_dir.glob("case_*.pt")) if result_dir.is_dir() else []:
            match = re.fullmatch(r"case_(\d+)\.pt", case.name)
            if not match:
                continue
            index = int(match.group(1))
            size = case.stat().st_size
            input_case = data / "golden_input" / info["camel"] / case.name
            if input_case.is_file():
                size += input_case.stat().st_size
            cases.append({"id": index, "bytes": size})
        info["cases"] = cases
        info["bytes"] = sum(c["bytes"] for c in cases)
    manifest = data / "cases_manifest.json"
    if manifest.is_file():
        info["manifest"] = str(manifest.resolve())
        info["total_files_bytes"] = sum(p.stat().st_size for p in data.rglob("*") if p.is_file())
    return info


def candidate_priority(op_entry, roots):
    prefer = op_entry.get("prefer", [])
    order = {root_id: index for index, root_id in enumerate(prefer)}
    fallback = {root["id"]: len(order) + index for index, root in enumerate(roots)}
    return lambda root_id: order.get(root_id, fallback[root_id])


def resolve_op(op_entry, roots, source_root):
    op_name = op_entry["name"]
    rank = candidate_priority(op_entry, roots)
    candidates = []
    for root in sorted(roots, key=lambda r: rank(r["id"])):
        op_dir = source_root / root["path"] / op_name
        if not op_dir.is_dir():
            continue
        test = find_test_script(op_dir, op_name)
        data = discover_data(op_dir)
        candidates.append({
            "root_id": root["id"],
            "dir": str(op_dir.resolve()),
            "ops_repo": root["ops_repo"],
            "bucket": root["bucket"],
            "experimental": root["bucket"] == "experimental",
            "has_op_host": (op_dir / "op_host").is_dir(),
            "has_op_kernel": (op_dir / "op_kernel").is_dir(),
            "test": str(test.resolve()) if test else None,
            "test_name": test.name if test else None,
            "test_kind": test_kind(test) if test else None,
            "data": data,
        })

    buildable = [c for c in candidates if c["has_op_host"] and c["has_op_kernel"]]
    warnings = []
    if not buildable:
        return {"name": op_name, "status": "unresolved", "candidates": candidates,
                "warnings": ["no candidate has op_host + op_kernel"]}

    source = buildable[0]
    test_candidate = next((c for c in buildable if c["test"]), None)
    data_candidate = next((c for c in buildable if c["data"]), None)
    if test_candidate is None:
        warnings.append("no *_test.py found in any candidate")
    if data_candidate is None:
        warnings.append("no test_data/ (golden) found in any candidate")
    if test_candidate and data_candidate and test_candidate is not data_candidate:
        warnings.append(f"test from {test_candidate['root_id']} but golden data from {data_candidate['root_id']}")

    package_files, editable, skipped, total_bytes = collect_manifest_files(Path(source["dir"]), CATALOG["package"])
    if total_bytes > CATALOG["package"]["max_bytes"]:
        warnings.append(f"agent package {total_bytes} bytes exceeds max_bytes")
    if len(package_files) > CATALOG["package"]["max_files"]:
        warnings.append(f"agent package {len(package_files)} files exceeds max_files")
    if skipped:
        warnings.append(f"{len(skipped)} non-text/forbidden files skipped from agent package")
    if data_candidate and data_candidate["data"] and data_candidate["data"]["bytes"] > 20_000_000:
        warnings.append(
            f"golden data {data_candidate['data']['bytes']} bytes > 20MB, bridge must trim cases"
        )

    return {
        "name": op_name,
        "status": "resolved",
        "prefer": op_entry.get("prefer", []),
        "data_cases": op_entry.get("data_cases"),
        "source_dir": source["dir"],
        "source_root_id": source["root_id"],
        "ops_repo": source["ops_repo"],
        "bucket": source["bucket"],
        "experimental": source["experimental"],
        "test": {
            "path": test_candidate["test"] if test_candidate else None,
            "name": test_candidate["test_name"] if test_candidate else None,
            "kind": test_candidate["test_kind"] if test_candidate else None,
            "from_root": test_candidate["root_id"] if test_candidate else None,
        },
        "data": data_candidate["data"] if data_candidate else None,
        "agent_package": {
            "files": len(package_files),
            "bytes": total_bytes,
            "editable": editable,
            "skipped": skipped,
        },
        "alternatives": [c["root_id"] for c in candidates],
        "candidates": candidates,
        "warnings": warnings,
    }


def resolve_catalog(catalog, catalog_path):
    roots = catalog["search_roots"]
    source_root = Path(expand_path(catalog["source_root"]))
    resolved = [resolve_op(entry, roots, source_root) for entry in catalog["ops"]]
    return {
        "schema_version": "ascend-gym-op-catalog-resolved-v1",
        "catalog": str(Path(catalog_path).resolve()),
        "catalog_name": catalog.get("name"),
        "source_root": str(source_root),
        "generated_from": catalog.get("name"),
        "ops": resolved,
    }


def op_by_name(catalog, name):
    for entry in catalog["ops"]:
        if entry["name"] == name:
            return entry
    raise SystemExit(f"operator not in catalog: {name}")


def spec_text(resolved, catalog):
    name = resolved["name"]
    source = Path(resolved["source_dir"])
    readme = source / "README.md"
    description = ""
    if readme.is_file():
        text = readme.read_text(encoding="utf-8", errors="replace").strip()
        description = text[:2500]
    lines = [
        f"# 算子优化任务：{name}",
        "",
        "## 来源",
        f"- 源码目录：`{resolved['source_dir']}`",
        f"- 仓库分类：{resolved['bucket']} / ops_repo=`{resolved['ops_repo']}`",
        f"- 冻结测试：`{resolved['test']['name']}`（{resolved['test']['kind']}）",
    ]
    if resolved.get("data"):
        data = resolved["data"]
        case_ids = [c["id"] for c in data["cases"]]
        lines.append(f"- 金标数据：`{data['dir']}`，{len(case_ids)} 个用例 {case_ids}")
    lines += [
        "",
        "## 目标",
        "在不改变算子接口、shape、dtype、计算语义与数值容差的前提下，降低该算子各测试用例的平均 kernel 耗时之和。",
        "",
        "## 允许修改范围",
        "- 只允许修改 `op_host/` 与 `op_kernel/` 下的实现文件。",
        "- 不得修改 `op_speed_test/` 下的测试脚本、金标数据、评测器与预算。",
        "- 每次修改后必须通过 harness 提交真实编译/精度/profiling 评测，不能只给计划。",
        "",
        "## 评测方式",
        "- 沙盒在 NPU（ascend910_93）上编译算子并运行冻结测试。",
        "- 通过正确性测试后，用 msprof 采集各算子 kernel 的 Task Duration，指标为每个算子均值的总和。",
        "- 编译失败或精度失败时，harness 会返回编译日志/失败用例，基于真实日志修复。",
    ]
    if description:
        lines += ["", "## 算子说明（摘自 README）", "", "```", description, "```"]
    return "\n".join(lines) + "\n"


def package_layout(catalog, name):
    """返回 (package_dir, spec_path, meta_path)。

    优先“按算子隔离”布局（catalog.package_layout=per_op，或已存在 <inputs_dir>/<name>/package.json）：
        <inputs_dir>/<name>/{agent/, spec.md, package.json, test_data/}
    兼容旧的扁平布局：
        <inputs_dir>/<name>_agent、<name>_spec.md、<name>.package.json
    """
    inputs_dir = Path(expand_path(catalog["outputs"]["inputs_dir"]))
    isolated = inputs_dir / name
    per_op = str(catalog.get("package_layout", "")).lower() in {"per_op", "isolated"}
    if per_op or (isolated / "package.json").is_file():
        return isolated / "agent", isolated / "spec.md", isolated / "package.json"
    return inputs_dir / f"{name}_agent", inputs_dir / f"{name}_spec.md", inputs_dir / f"{name}.package.json"


def package_paths(catalog, name):
    package_dir, spec_path, _ = package_layout(catalog, name)
    return package_dir, spec_path


def build_package(catalog, resolved, force=False):
    name = resolved["name"]
    if resolved["status"] != "resolved":
        return {"name": name, "status": "skipped", "reason": "unresolved"}
    package_dir, spec_path, meta_path = package_layout(catalog, name)
    if package_dir.exists() and not force:
        return {"name": name, "status": "exists", "package": str(package_dir)}

    source = Path(resolved["source_dir"])
    files, editable, skipped, total = collect_manifest_files(source, catalog["package"])
    if not editable:
        return {"name": name, "status": "error", "reason": "no editable files under op_host/ or op_kernel/"}
    if total > catalog["package"]["max_bytes"] or len(files) > catalog["package"]["max_files"]:
        return {"name": name, "status": "error",
                "reason": f"package limit exceeded: {len(files)} files / {total} bytes"}

    if package_dir.exists():
        shutil.rmtree(package_dir)
    for rel, path in sorted(files.items()):
        target = package_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)

    test_dir = package_dir / catalog["package"]["test_dir"]
    test_dir.mkdir(parents=True, exist_ok=True)
    test_source = Path(resolved["test"]["path"]) if resolved["test"]["path"] else None
    if test_source is None:
        return {"name": name, "status": "error", "reason": "no frozen test script"}
    test_target = test_dir / test_source.name
    shutil.copy2(test_source, test_target)

    spec_path.write_text(spec_text(resolved, catalog), encoding="utf-8")
    metadata = {
        "schema_version": "ascend-gym-op-package-v1",
        "op_name": name,
        "source_dir": str(package_dir),
        "spec_path": str(spec_path),
        "original_source": resolved["source_dir"],
        "ops_repo": resolved["ops_repo"],
        "bucket": resolved["bucket"],
        "experimental": resolved["experimental"],
        "test_paths": [f"{catalog['package']['test_dir']}/{test_source.name}"],
        "test_kind": resolved["test"]["kind"],
        "test_source": resolved["test"]["path"],
        "data_dir": resolved["data"]["dir"] if resolved["data"] else None,
        "data_cases": resolved.get("data_cases"),
        "data_case_bytes": {str(c["id"]): c["bytes"] for c in (resolved["data"]["cases"] if resolved["data"] else [])},
        "editable_paths": editable,
        "files": len(files) + 1,
        "bytes": total + test_target.stat().st_size,
        "skipped_files": skipped,
        "warnings": resolved["warnings"],
    }
    write_json(meta_path, metadata)
    return {"name": name, "status": "built", "package": str(package_dir), "files": metadata["files"],
            "bytes": metadata["bytes"], "data": metadata["data_dir"], "warnings": resolved["warnings"]}


def metadata_for_existing(catalog, resolved):
    """Adopt a pre-existing package (e.g. the earlier add_rms_norm_agent run)."""
    name = resolved["name"]
    package_dir, spec_path, metadata_path = package_layout(catalog, name)
    if not package_dir.is_dir() or metadata_path.is_file():
        return metadata_path.is_file()
    tests = sorted((package_dir / catalog["package"]["test_dir"]).glob("*.py"))
    if not tests:
        return False
    editable = sorted(
        p.relative_to(package_dir).as_posix()
        for prefix in ("op_host", "op_kernel")
        for p in (package_dir / prefix).rglob("*") if p.is_file()
    )
    metadata = {
        "schema_version": "ascend-gym-op-package-v1",
        "op_name": name,
        "source_dir": str(package_dir),
        "spec_path": str(spec_path),
        "original_source": resolved["source_dir"],
        "ops_repo": resolved["ops_repo"],
        "bucket": resolved["bucket"],
        "experimental": resolved["experimental"],
        "test_paths": [f"{catalog['package']['test_dir']}/{tests[0].name}"],
        "test_kind": resolved["test"]["kind"],
        "test_source": resolved["test"]["path"],
        "data_dir": resolved["data"]["dir"] if resolved["data"] else None,
        "data_cases": resolved.get("data_cases"),
        "data_case_bytes": {str(c["id"]): c["bytes"] for c in (resolved["data"]["cases"] if resolved["data"] else [])},
        "editable_paths": editable,
        "files": sum(1 for p in package_dir.rglob("*") if p.is_file()),
        "bytes": sum(p.stat().st_size for p in package_dir.rglob("*") if p.is_file()),
        "skipped_files": [],
        "warnings": resolved["warnings"] + ["metadata adopted for pre-existing package"],
    }
    write_json(metadata_path, metadata)
    return True


def load_native_package(root):
    """把 native harness 代码根加入 sys.path 并导入 NativeRun（清理指向其它副本的同名模块）。"""
    root = Path(root).resolve()
    prefix = str(root)
    for module_name in list(sys.modules):
        if not (module_name == "native_bench" or module_name.startswith("native_bench.")
                or module_name == "core" or module_name.startswith("core.")):
            continue
        module_path = getattr(sys.modules[module_name], "__file__", None) or ""
        if module_path and not str(Path(module_path).resolve()).startswith(prefix):
            del sys.modules[module_name]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from native_bench.runner import NativeRun
    return NativeRun


def validate_package_native(catalog, resolved, metadata, package_dir, spec_path):
    """native preflight：用 NativeRun 构造器做配置/schema/输入包校验（不调用模型与沙盒）。"""
    name = resolved["name"]
    try:
        root = require_native_harness(catalog)
    except SystemExit as exc:
        return {"name": name, "status": "failed", "error": str(exc).splitlines()[0]}
    budget = {
        "candidate_evaluations_max": max(1, int(catalog["budget_defaults"].get(
            "candidate_evaluations_max", 1))),
        "final_reprofile_count": max(0, int(catalog["budget_defaults"].get("final_reprofile_count", 1))),
    }
    default_target = catalog.get("budget_defaults", {}).get("candidate_solutions_target")
    if isinstance(default_target, int) and not isinstance(default_target, bool):
        budget["candidate_solutions_target"] = default_target
    try:
        config = native_run_config(catalog, resolved, "codex", f"preflight-native-{name}",
                                   budget, metadata, package_dir, spec_path,
                                   profiling_measured=resolve_profiling_runs(None),
                                   profiling_warmup=resolve_warmup_runs(None),
                                   log_protocol=False)
        NativeRun = load_native_package(root)
        import tempfile
        with tempfile.TemporaryDirectory(prefix=f"preflight-native-{name}-") as tmp:
            NativeRun(config, Path(tmp) / "run")
        return {"name": name, "status": "ok", "files": metadata["files"],
                "bytes": metadata["bytes"], "harness": "native"}
    except Exception as exc:  # noqa: BLE001
        return {"name": name, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}


def validate_package(catalog, resolved, harness="native"):
    """Run the real harness preflight against the built package (no sandbox calls)."""
    name = resolved["name"]
    package_dir, spec_path, package_json = package_layout(catalog, name)
    if not package_dir.is_dir() or not package_json.is_file():
        return {"name": name, "status": "missing"}
    metadata = read_json(package_json)
    if harness == "native":
        return validate_package_native(catalog, resolved, metadata, package_dir, spec_path)
    if harness == "bare":
        return {
            "name": name, "status": "skipped",
            "reason": "bare harness has no offline preflight",
        }
    return {
        "name": name, "status": "failed",
        "error": "harness not in supported list: native, bare",
    }
def make_run_config(catalog, resolved, agent_name, run_id, budget_overrides, output_dir,
                    agent_timeout_seconds=None, harness="native",
                    profiling_measured=None, profiling_warmup=None, sandbox_url=None,
                    bare_model=None, model_spec=None, claude_settings=None):
    name = resolved["name"]
    if harness != "bare" and agent_name not in catalog["agents"]:
        raise SystemExit(f"unknown agent: {agent_name} (have {sorted(catalog['agents'])})")
    if harness not in HARNESS_KINDS:
        raise SystemExit(f"unknown harness: {harness} (have {list(HARNESS_KINDS)})")
    package_dir, spec_path, meta_path = package_layout(catalog, name)
    metadata = read_json(meta_path)
    budget = dict(catalog["budget_defaults"])
    budget.update(budget_overrides or {})
    if harness == "bare":
        if agent_timeout_seconds is not None:
            log("warning: bare harness 没有单轮 agent 超时；--agent-timeout-seconds 已忽略")
        config = bare_run_config(catalog, resolved, bare_model, run_id, budget, metadata,
                                 package_dir, spec_path,
                                 profiling_measured=profiling_measured,
                                 profiling_warmup=profiling_warmup, sandbox_url=sandbox_url,
                                 model_spec=model_spec)
        write_json(output_dir, config)
        return config
    if harness == "native":
        if agent_timeout_seconds is not None:
            log("warning: native harness 没有单轮 agent 超时（只有停止后的 shutdown_grace）；"
                "--agent-timeout-seconds 已忽略")
        config = native_run_config(catalog, resolved, agent_name, run_id, budget, metadata,
                                   package_dir, spec_path,
                                   profiling_measured=profiling_measured,
                                   profiling_warmup=profiling_warmup,
                                   model_spec=model_spec, claude_settings=claude_settings)
        write_json(output_dir, config)
        return config
    raise SystemExit(
        "harness not in supported list: native, bare"
    )
    if "candidate_solutions_target" in budget:
        log("warning: --candidate-solutions-target 只对 native harness 生效；该字段已忽略")
        budget.pop("candidate_solutions_target")
    agent_config = dict(catalog["agents"][agent_name])
    if agent_timeout_seconds is not None:
        if isinstance(agent_timeout_seconds, bool) or not isinstance(agent_timeout_seconds, int) \
                or agent_timeout_seconds <= 0:
            raise SystemExit(
                f"--agent-timeout-seconds 必须是正整数（秒），当前值 {agent_timeout_seconds!r}")
        agent_config["timeout_seconds"] = agent_timeout_seconds
    config = {
        "run_id": run_id,
        "input": {
            "op_name": name,
            "kind": "ascendc",
            "source_dir": str(package_dir),
            "spec_path": str(spec_path),
            "editable_paths": metadata["editable_paths"],
            "test_paths": metadata["test_paths"],
        },
        "budget": budget,
        "metrics": catalog["metrics_defaults"],
        "agent": agent_config,
        "backend": catalog["backend"],
    }
    wall = budget.get("wall_clock_seconds_max")
    agent_timeout = agent_config.get("timeout_seconds")
    checkpoints = (config["metrics"].get("time_checkpoints_sec") or [])
    if isinstance(wall, (int, float)) and not isinstance(wall, bool) and wall < 0:
        raise SystemExit(f"wall_clock_seconds_max 必须 >= 0（0=关闭限制），当前值 {wall}")
    if isinstance(wall, (int, float)) and not isinstance(wall, bool) and wall \
            and isinstance(agent_timeout, (int, float)) and not isinstance(agent_timeout, bool) \
            and agent_timeout >= wall:
        log(f"warning: {agent_name} 的单轮超时 {agent_timeout}s >= wall_clock_seconds_max {wall}s；"
            f"单轮请求可能晚于墙钟才被中断")
    if isinstance(wall, (int, float)) and not isinstance(wall, bool) and wall and checkpoints \
            and wall > max(checkpoints):
        log(f"warning: wall_clock_seconds_max={wall}s 超过 metrics.time_checkpoints_sec 的最大检查点 "
            f"{max(checkpoints)}s；STS_time 的横轴不会覆盖完整搜索窗口"
            f"（可在 op_catalog.json 的 metrics_defaults.time_checkpoints_sec 里追加检查点）")
    write_json(output_dir, config)
    return config


DEFAULT_INFRA_RETRY = {
    "eval_retries": 3,
    "eval_retry_backoff_seconds": 20,
    "eval_retry_max_backoff_seconds": 300,
    "eval_budget_seconds": 4800,
    "max_streak": 3,
    "agent_sleep_seconds": 60,
    "keepalive_status": "profiling_failed",
    "candidate_fault_status": "test_failed",
    "guidance": "",
    "candidate_fault_guidance": "",
}


def infra_retry_policy(catalog):
    """基础设施故障处置策略：catalog.infra_retry 为默认值，同名 SANDBOX_* 环境变量可覆盖。

    评测器与 prompt 共用这一份策略，避免"prompt 说 3 次、评测器按 5 次算"这类漂移。
    """
    policy = dict(DEFAULT_INFRA_RETRY)
    section = catalog.get("infra_retry")
    if isinstance(section, dict):
        policy.update({k: v for k, v in section.items() if not str(k).startswith("_")})
    overrides = {
        "eval_retries": "SANDBOX_EVAL_RETRIES",
        "eval_retry_backoff_seconds": "SANDBOX_EVAL_RETRY_BACKOFF",
        "eval_retry_max_backoff_seconds": "SANDBOX_EVAL_RETRY_MAX_BACKOFF",
        "eval_budget_seconds": "SANDBOX_EVAL_BUDGET_SECONDS",
        "max_streak": "SANDBOX_INFRA_MAX_STREAK",
        "keepalive_status": "SANDBOX_INFRA_KEEPALIVE_STATUS",
        "candidate_fault_status": "SANDBOX_CANDIDATE_FAULT_STATUS",
        "guidance": "SANDBOX_INFRA_GUIDANCE",
        "candidate_fault_guidance": "SANDBOX_CANDIDATE_FAULT_GUIDANCE",
    }
    for key, env_name in overrides.items():
        raw = os.environ.get(env_name)
        if raw is not None and str(raw).strip() != "":
            policy[key] = raw
    # 评测器的墙钟预算必须留在 harness 的 per_call_timeout_seconds 以内，
    # 否则会被 harness 先 SIGKILL，那次评测退回 state_unknown 并作废整个 run。
    budget = catalog.get("backend", {}).get("timeout_seconds", 5400)
    try:
        policy["eval_budget_seconds"] = min(float(policy["eval_budget_seconds"]), float(budget) - 300.0)
    except (TypeError, ValueError):
        policy["eval_budget_seconds"] = float(DEFAULT_INFRA_RETRY["eval_budget_seconds"])
    # 在这里就把 {sleep}/{max_streak} 渲染掉：guidance 同时会被 sandbox_env 下发给评测器，
    # 评测器只是原样回显，不会（也不该）自己格式化——否则 agent 在诊断里会看到裸占位符。
    policy["guidance_template"] = str(policy.get("guidance") or "").strip()
    policy["guidance"] = render_guidance(policy)
    return policy


def render_guidance(policy):
    """把处置指引里的 {sleep}/{max_streak} 换成实际取值；模板写错时原样返回。"""
    template = str(policy.get("guidance_template") or policy.get("guidance") or "").strip()
    if not template:
        return ""
    try:
        return template.format(sleep=int(float(policy["agent_sleep_seconds"])),
                               max_streak=int(float(policy["max_streak"])))
    except (KeyError, ValueError, IndexError, TypeError):
        return template


def infra_task_note(policy, run_id):
    """写进 config.task_note 的补充说明 —— harness 会把它原样附在 agent 的初始 prompt 后面。

    这是 bridge 侧唯一被 harness 认可的 prompt 注入通道：策略本身来自 op_catalog.infra_retry，
    不在代码里硬编码，改配置即可调整等待时长与放弃阈值。
    """
    note = f"本 run 由 ascend_gym_bridge 以 native harness 启动：run_id={run_id}。"
    guidance = str(policy.get("guidance") or "").strip()  # infra_retry_policy 已渲染过占位符
    if guidance:
        note += "\n\n基础设施故障处置（沙盒/设备问题时适用）：" + guidance
    return note


def sandbox_env(catalog, resolved, metadata, profiling_runs=None, warmup_runs=None):
    env = {}
    env["SANDBOX_URL"] = expand_env(os.environ.get("SANDBOX_URL", "http://127.0.0.1:9000"))
    env["SANDBOX_OPS_REPO"] = resolved["ops_repo"]
    env["SANDBOX_OP_NAME"] = resolved["name"]
    env["SANDBOX_PROFILING_RUNS"] = str(resolve_profiling_runs(profiling_runs))
    # 测试脚本总共跑 (测量轮数 + 预热轮数)，聚合时丢弃前 WARMUP 轮（默认 8+3，丢 3）
    # native v2+ 会把等价协议写进 config 的 evaluator.profiling，评测器以请求协议为准。
    env["SANDBOX_WARMUP_RUNS"] = str(resolve_warmup_runs(warmup_runs))
    # 设备故障/传输故障/无判定时的整评测重试与降级策略（sandbox_evaluator.py 读取，见 USAGE）
    policy = infra_retry_policy(catalog)
    env["SANDBOX_EVAL_RETRIES"] = str(int(float(policy["eval_retries"])))
    env["SANDBOX_EVAL_RETRY_BACKOFF"] = str(float(policy["eval_retry_backoff_seconds"]))
    env["SANDBOX_EVAL_RETRY_MAX_BACKOFF"] = str(float(policy["eval_retry_max_backoff_seconds"]))
    env["SANDBOX_EVAL_BUDGET_SECONDS"] = str(float(policy["eval_budget_seconds"]))
    env["SANDBOX_INFRA_MAX_STREAK"] = str(int(float(policy["max_streak"])))
    env["SANDBOX_INFRA_KEEPALIVE_STATUS"] = str(policy["keepalive_status"])
    env["SANDBOX_INFRA_GUIDANCE"] = str(policy.get("guidance") or "")  # 已渲染 {sleep}/{max_streak}
    # 候选把设备打挂时的判定与措辞。这里是**模板**：{cause}/{attempts} 只有评测时刻才知道，
    # 由 sandbox_evaluator.py 在写诊断时填充，所以此处不要渲染。
    env["SANDBOX_CANDIDATE_FAULT_STATUS"] = str(policy["candidate_fault_status"])
    env["SANDBOX_CANDIDATE_FAULT_GUIDANCE"] = str(policy.get("candidate_fault_guidance") or "")
    if metadata.get("experimental"):
        env["SANDBOX_EXPERIMENTAL"] = "1"
    if metadata.get("data_dir"):
        env["SANDBOX_TEST_DATA_DIR"] = expand_path(metadata["data_dir"])
        # 默认 "all"：把该算子 test_data 下的全部 case 都传进沙盒，不做体积裁剪。
        # 需要裁剪时在 package.json 里显式写 data_cases: "auto" 或 "0,1,4"。
        cases = resolved.get("data_cases") or metadata.get("data_cases")
        if isinstance(cases, (list, tuple)):
            cases = ",".join(str(c) for c in cases)
        env["SANDBOX_TEST_DATA_CASES"] = str(cases or "all")
    return env


def native_root(catalog):
    """native harness 的代码根目录（需同时含 native_bench/ 与 core/）。"""
    value = os.environ.get("NATIVE_HARNESS_ROOT")
    if not value and isinstance(catalog, dict):
        value = (catalog.get("outputs") or {}).get("native_root")
    return Path(expand_path(value)) if value else DEFAULT_NATIVE_ROOT


def native_harness_ready(root):
    root = Path(root)
    return ((root / "native_bench" / "runner.py").is_file()
            and (root / "core" / "contract.py").is_file())


def native_supports_search(root):
    """v2+ 的 native config 才接受顶层 search；v1 会把未知字段当错误拒绝整份配置。"""
    config_py = Path(root) / "native_bench" / "config.py"
    try:
        return "candidate_solutions_target" in config_py.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False


def native_supports_profiling(root):
    """v2+ 才接受 evaluator.profiling 固定采样协议（并要求评测器在响应里原样回显）。"""
    root = Path(root)
    config_py = root / "native_bench" / "config.py"
    try:
        text = config_py.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return (root / "core" / "profiling.py").is_file() and "profiling" in text


def resolve_profiling_runs(value=None):
    """测量轮数：显式参数 > SANDBOX_PROFILING_RUNS > 8。"""
    if value is None:
        value = os.environ.get("SANDBOX_PROFILING_RUNS") or None
    if value is None:
        return 8
    try:
        runs = int(value)
    except (TypeError, ValueError):
        raise SystemExit(f"测量轮数必须是整数，当前 {value!r}（--profiling-runs / SANDBOX_PROFILING_RUNS）")
    if runs < 1:
        raise SystemExit(f"测量轮数必须 >= 1，当前 {runs}（--profiling-runs / SANDBOX_PROFILING_RUNS）")
    return runs


def resolve_warmup_runs(value=None):
    """预热轮数：显式参数 > SANDBOX_WARMUP_RUNS > 3。"""
    if value is None:
        value = os.environ.get("SANDBOX_WARMUP_RUNS") or None
    if value is None:
        return 3
    try:
        runs = int(value)
    except (TypeError, ValueError):
        raise SystemExit(f"预热轮数必须是整数，当前 {value!r}（--warmup-runs / SANDBOX_WARMUP_RUNS）")
    if runs < 0:
        raise SystemExit(f"预热轮数必须 >= 0，当前 {runs}（--warmup-runs / SANDBOX_WARMUP_RUNS）")
    return runs


def require_native_harness(catalog):
    root = native_root(catalog)
    if not native_harness_ready(root):
        raise SystemExit(
            f"native harness 不可用: {root}\n"
            f"  需要 <root>/native_bench/ 和 <root>/core/（该 root 会被加入 sys.path）。\n"
            f"  可用环境变量 NATIVE_HARNESS_ROOT 或 op_catalog.json 的 outputs.native_root 覆盖。")
    return root


def native_agent_model(agent_name, catalog):
    """native 模式下的模型固定值（wrapper 已不参与，改由 config 的 agent.model 传递）。"""
    configured = ((catalog.get("agents") or {}).get(agent_name) or {}).get("model")
    if configured:
        return configured, f"op_catalog agents.{agent_name}.model"
    env_key = {"codex": "CODEX_AGENT_MODEL", "opencode": "OPENCODE_AGENT_MODEL"}.get(agent_name)
    if env_key and os.environ.get(env_key):
        return os.environ[env_key], f"环境变量 {env_key}"
    if agent_name == "opencode":
        candidates = []
        pinned = os.environ.get("OPENCODE_CONFIG")
        if pinned:
            candidates.append(Path(pinned))
        source = global_opencode_config()
        if source is not None:
            candidates.append(source)
        for path in candidates:
            try:
                model = (load_jsonc(path) or {}).get("model")
            except (OSError, ValueError):
                model = None
            if model:
                return model, str(path)
    return None, None


def models_registry(catalog):
    """统一模型注册表（op_catalog.json 的 models 段）。"""
    return (catalog.get("models") or {})


def resolve_model(catalog, name=None, url=None, raw_name=None, default_key=None):
    """把「本次用哪个模型」解析成一份完整描述；返回 None 表示不指定（沿用各 agent 现状）。

    优先级：--model <注册表名> → --model-url/--model-name（原始覆盖）→ 环境变量
    MODEL/MODEL_URL/MODEL_NAME（兼容旧的 BARE_MODEL*）→ default_key（catalog 默认）。
    命令行与 env 都解析不到时返回 None，四个 agent 各自继续读自己的全局配置（零回归）。

    返回 dict：{key, provider, name, openai_base, anthropic_base, wire_api, context, api_key, url}
    其中 `url` 是 bare/chat 侧完整地址（openai_base + /chat/completions）。
    """
    registry = models_registry(catalog)
    legacy = (catalog.get("bare") or {}).get("models") or {}   # 旧位置，仍兼容
    key = (name or os.environ.get("MODEL") or os.environ.get("BARE_MODEL") or default_key or "").strip()
    url = (url or os.environ.get("MODEL_URL") or os.environ.get("BARE_MODEL_URL") or "").strip()
    raw_name = (raw_name or os.environ.get("MODEL_NAME") or os.environ.get("BARE_MODEL_NAME") or "").strip()

    if key and key not in registry and key not in legacy:
        raise SystemExit(
            f"未知模型: {key!r}\n"
            f"  op_catalog.models 可选: {sorted(registry) or '(空)'}\n"
            f"  也可用原始参数：--model-url http://host:port/v1 --model-name <served-name>")
    spec = {}
    if key in registry:
        spec = dict(registry[key])
    elif key in legacy:                      # 兼容 bare.models 条目
        entry = dict(legacy[key])
        spec = {"provider": entry.get("provider", "local"), "name": entry.get("name", key),
                "openai_base": (entry.get("url") or "").rsplit("/chat/completions", 1)[0]}
    if not spec and not (url or raw_name):
        return None                          # 不指定 → 走各自现状

    spec.setdefault("provider", "local")
    if url:
        spec["openai_base"] = url
    if raw_name:
        spec["name"] = raw_name
    spec.setdefault("name", key)
    base = (spec.get("openai_base") or "").rstrip("/")
    if not base:
        raise SystemExit("模型端点缺失：请用 --model <注册表名> 或 --model-url 指定")
    spec["openai_base"] = base
    if url:
        # 显式 --model-url 覆盖端点时，anthropic_base 也必须跟着走：
        # 注册表条目里的 anthropic_base 可能指向别处（或只是占位主机），
        # 沿用它会让 claude 探测/连到一个与 openai_base 不同的上游。
        spec["anthropic_base"] = base[:-3].rstrip("/") if base.endswith("/v1") else base
    else:
        spec.setdefault("anthropic_base", base[:-3].rstrip("/") if base.endswith("/v1") else base)
    spec.setdefault("wire_api", "responses")
    spec.setdefault("context", None)
    spec.setdefault("api_key", "any-value")
    spec.setdefault("tool_format", "json")     # xml = Qwen/GLM 原生 <function=…> 形式
    spec["key"] = key or spec["name"]
    spec["url"] = f"{base}/chat/completions"
    for field in ("openai_base", "anthropic_base", "url", "api_key"):
        if isinstance(spec.get(field), str):
            spec[field] = expand_env(spec[field])
    if spec.get("api_key_env"):
        env_name = spec["api_key_env"]
        token = os.environ.get(env_name)
        if not token:
            raise SystemExit(f"模型密钥环境变量未设置或为空: {env_name}")
        spec["api_key"] = token
    return spec


BARE_CLIENT_PROVIDERS = ("vllm", "openrouter", "openai", "glm-api", "glm-client")


def bare_model_payload(spec):
    """把一份 resolve_model() 结果转成 bare_bench 消费的 {provider,name,url,key} 形状。

    provider 必须落进 bare_client 的 `--llm-provider` 取值域。注册表里的 "local" 是
    codex / opencode 配置用的 provider 名（它们的配置块就叫 local），bare_client 不认，
    这里统一映射成本地 OpenAI-compatible 的 "vllm"，否则 bare_bench 会把 "local"
    原样传给 bare_client，argparse 直接拒绝（invalid choice: 'local'）。
    """
    provider = spec["provider"] if spec["provider"] in BARE_CLIENT_PROVIDERS else "vllm"
    return {"provider": provider, "name": spec["name"], "url": spec["url"],
            "key": spec["key"], "context": spec.get("context"), "api_key": spec.get("api_key"),
            "provider_source": spec["provider"], "tool_format": spec.get("tool_format") or "json",
            # 透传给 bare_client 的 --protocol-correction-max（缺省时用 client 自己的默认值 2）：
            # "只回文本不发 tool call" 的模型需要更多纠偏机会，见 CHANGELOG 2026-09-27。
            "protocol_correction_max": spec.get("protocol_correction_max")}


def resolve_bare_model(catalog, name=None, spec=None):
    """bare harness 的模型；``spec`` 为已解析结果时直接转换（保留 --model-url 覆盖）。"""
    if spec is None:
        spec = resolve_model(catalog, name=name, default_key=(catalog.get("bare") or {}).get("default_model"))
    if spec is None:
        raise SystemExit("bare harness 需要显式模型：--model <注册表名>（或 --bare-model）")
    return bare_model_payload(spec)


def model_token(spec):
    """本地 vLLM 不校验 token，但 Claude Code 必须有非空凭据才会启动。"""
    return spec.get("api_key") or "any-value"


def model_summary(spec):
    """给日志/产物用的一行摘要。"""
    if not spec:
        return "-"
    return (f"{spec['key']}（name={spec['name']} base={spec['openai_base']} "
            f"wire={spec.get('wire_api')} ctx={spec.get('context') or '?'}）")


# --- per-run 模型落地：每个 run 一份独立文件，绝不改全局配置 -------------------- #

def per_run_dir(runs_dir, run_id):
    """per-run 私有目录（<runs_dir>/.per-run/<run_id>/），与 output_dir 无关。

    native/bare 要求 --output 目录在启动前不存在，所以这些文件不能写进 output_dir。
    """
    path = Path(runs_dir) / ".per-run" / run_id
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.chmod(0o700)
    return path


def write_claude_settings(runs_dir, run_id, spec):
    """写 per-run Claude Code settings（`claude --settings <file>` 会覆盖用户级同名 env）。

    键与取值照搬 ${ASCEND_GYM_WORKSPACE}/scripts/switch_model.sh 的 mutate_claude：其中
    CLAUDE_CODE_MODEL_CAPABILITIES 用来关掉 mid-conv-system / tool-change / per-turn-effort
    三类特性——它们会往 messages 里插 role:"system"，严格实现的 vLLM 会直接 400。
    key（模型名）必须小写，否则静默失配。
    """
    env = {
        "ANTHROPIC_BASE_URL": spec["anthropic_base"],
        "ANTHROPIC_API_KEY": "",
        "ANTHROPIC_MODEL": spec["name"],
        "ANTHROPIC_DEFAULT_OPUS_MODEL": spec["name"],
        "ANTHROPIC_DEFAULT_SONNET_MODEL": spec["name"],
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": spec["name"],
        "CLAUDE_CODE_SUBAGENT_MODEL": spec["name"],
        "CLAUDE_CODE_MODEL_CAPABILITIES":
            f"{spec['name'].lower()}=-mid_conv_system,-mid_conv_tool_change,-per_turn_effort",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "CLAUDE_STREAM_FIRST_BYTE_TIMEOUT_MS": "300000",
        "API_TIMEOUT_MS": "3600000",
    }
    if spec.get("context"):
        env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] = str(spec["context"])
    for key, value in (spec.get("claude_env") or {}).items():
        env[str(key)] = str(value)
    env["ANTHROPIC_BASE_URL"] = spec["anthropic_base"]
    env["ANTHROPIC_AUTH_TOKEN"] = model_token(spec)
    path = per_run_dir(runs_dir, run_id) / "claude-settings.json"
    write_json(path, {"env": env})
    os.chmod(path, 0o600)                    # 内含 token
    return path


def _toml_set(text, key, value):
    """在 TOML 文本里替换/追加一个顶层 `key = value`（不引入 toml 依赖）。"""
    pattern = re.compile(rf"^{re.escape(key)}\s*=.*$", re.MULTILINE)
    line = f"{key} = {value}"
    return pattern.sub(line, text, count=1) if pattern.search(text) else text.rstrip("\n") + f"\n{line}\n"


def write_codex_home(runs_dir, run_id, spec, template_home):
    """写 per-run CODEX_HOME（config.toml + models.json），让 codex 只认本次模型/端点。

    以当前生效的 CODEX_HOME/config.toml 为模板做文本替换（保留用户的其它设置），
    只改 model / model_provider / model_context_window / model_catalog_json 与
    目标 provider 的 base_url、wire_api。`model_catalog_json` 必须是绝对路径：
    codex 解析其中的 `~` 用的是 $HOME，不是 $CODEX_HOME。
    """
    home = per_run_dir(runs_dir, run_id) / "codex-home"
    home.mkdir(parents=True, exist_ok=True)
    provider = spec["provider"]
    template = Path(expand_path(template_home)) / "config.toml"
    try:
        text = template.read_text(encoding="utf-8")
    except OSError:
        text = ""
    if not text.strip():
        text = ('model_provider = "local"\n\n[model_providers.local]\nname = "local"\n')
    text = _toml_set(text, "model", json.dumps(spec["name"]))
    text = _toml_set(text, "model_provider", json.dumps(provider))
    if spec.get("context"):
        text = _toml_set(text, "model_context_window", str(int(spec["context"])))
    text = _toml_set(text, "model_catalog_json", json.dumps(str(home / "models.json")))
    # 抹掉模板里所有 [model_providers.*] 段：别的 provider 可能带着真实 API key
    # （例如全局配置里的 deepseek），per-run home 不应该把它复制进 run 产物。
    text = re.sub(r"^\[model_providers\.[^\]]+\]\s*$.*?(?=^\[|\Z)", "", text,
                  flags=re.MULTILINE | re.DOTALL).rstrip() + "\n"
    section = (f"\n[model_providers.{provider}]\nname = {json.dumps(provider)}\n"
               f"base_url = {json.dumps(spec['openai_base'])}\n"
               f"wire_api = {json.dumps(spec.get('wire_api') or 'responses')}\n"
               f"experimental_bearer_token = {json.dumps(model_token(spec))}\n"
               f"requires_openai_auth = false\n"
               f"supports_websockets = false\n"
               # codex 默认约 5 分钟无 SSE 事件即断开；大 prompt 的 prefill 可能更久。
               f"stream_idle_timeout_ms = 1800000\n")
    text = text.rstrip("\n") + "\n" + section
    write_private_text(home / "config.toml", text if text.endswith("\n") else text + "\n")

    catalog = {"models": []}
    source = Path(expand_path(template_home)) / "models.json"
    if source.is_file():
        try:
            catalog = json.loads(source.read_text(encoding="utf-8"))
        except ValueError:
            catalog = {"models": []}
    entries = catalog.setdefault("models", [])
    entry = next((e for e in entries if isinstance(e, dict) and e.get("slug") == spec["name"]), None)
    if entry is None:
        # codex 对目录条目有一堆必填字段（如 supported_reasoning_levels），自己拼一条最小条目
        # 会被 `failed to parse model_catalog_json ... missing field` 直接拒掉。
        # 先在其他已知 catalog 里找同名条目整条拿来用，否则用现有条目做字段模板只换身份字段。
        for candidate in (Path.home() / ".codex/models.json",):
            try:
                doc = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            found = next((e for e in (doc.get("models") or [])
                          if isinstance(e, dict) and e.get("slug") == spec["name"]), None)
            if found is not None:
                entry = dict(found)
                break
        if entry is None:
            template = next((e for e in entries if isinstance(e, dict)), None)
            if template is None:
                raise SystemExit(
                    f"codex 模型目录里没有可用模板（{source} 为空）：先在该 CODEX_HOME 里准备 models.json，"
                    "或把模型加进 ~/.codex/models.json")
            entry = dict(template)
            entry["slug"] = spec["name"]
            entry["display_name"] = spec["name"]
        entries.append(entry)
    entry.setdefault("display_name", spec["name"])
    # 某些模型需要额外的 codex 侧说明（如 glm-5.3 必须禁用 apply_patch：其自由文本补丁
    # 格式在上游 vLLM 上会解析失败并返回 400，实测 codex 读的是 model_messages.instructions_template）。
    suffix = (spec.get("codex_instructions_suffix") or "").strip()
    if suffix:
        note = "\n\n" + suffix
        msgs = entry.setdefault("model_messages", {})
        msgs["instructions_template"] = (msgs.get("instructions_template") or "") + note
        entry["base_instructions"] = (entry.get("base_instructions") or "") + note
    if spec.get("context"):
        entry["context_window"] = int(spec["context"])
        entry["max_context_window"] = int(spec["context"])
    (home / "models.json").write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n",
                                      encoding="utf-8")
    return home


def write_opencode_config(runs_dir, run_id, spec):
    """写 per-run opencode 配置（native 用）：以全局配置为底，只改 provider/model/limits。

    native opencode 必须保留 bash/read/edit 等原生工具；全局配置缺失时使用最小配置。
    """
    source = global_opencode_config()
    try:
        base = load_jsonc(source) if source else {}
    except (OSError, ValueError):
        base = {}
    provider = spec["provider"]
    block = base.setdefault("provider", {}).setdefault(provider, {})
    options = block.setdefault("options", {})
    options["baseURL"] = spec["openai_base"]
    options["apiKey"] = model_token(spec)
    block.setdefault("npm", "@ai-sdk/openai-compatible")
    block.setdefault("models", {})[spec["name"]] = {
        "name": spec["name"],
        # output 走注册表 `max_output`（switch_model_README §2.1 一直这么写，但此前这里
        # 硬编码 8192 没读它）。写死 8192 会让"单步思考较长"的模型每步被截断：opencode 的
        # step 以 reason="length" 收尾并**直接结束会话**，run 以 agent_finished + 0 候选告终
        # （2026-09-26 实测 claude-opus-5：8 个 run 全部死在 length，out_tok 恰好 8192；
        # 同模型走 claude 路径正常，因为 Claude Code 请求的输出上限远大于 8192）。
        "limit": {"context": int(spec.get("context") or 131072),
                  "output": int(spec.get("max_output") or 8192)},
    }
    base["provider"] = {provider: block}
    base["model"] = f"{provider}/{spec['name']}"
    base["small_model"] = f"{provider}/{spec['name']}"
    runtime_dir = Path(runs_dir) / ".opencode-runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    path = runtime_dir / f"{run_id}.json"
    write_private_text(path, json.dumps(base, ensure_ascii=False, indent=2) + "\n")
    os.chmod(path, 0o600)                    # 内含 apiKey
    return path


def split_sandbox(url):
    """把沙盒代理 URL 拆成 k-lens Backend 要的 host/port。"""
    from urllib.parse import urlsplit

    parts = urlsplit(url if "//" in url else f"//{url}")
    if not parts.hostname or not parts.port:
        raise SystemExit(f"无法从 {url!r} 解析沙盒 host/port（需要形如 http://127.0.0.1:9000）")
    return parts.hostname, parts.port


def _free_local_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _without_v1(url):
    base = expand_env(url or "").rstrip("/")
    return base[:-3].rstrip("/") if base.endswith("/v1") else base


def _with_v1(url):
    base = expand_env(url or "").rstrip("/")
    return base if base.endswith("/v1") else base + "/v1"


def _probe_url(spec, protocol):
    if protocol == "anthropic":
        return _without_v1(spec.get("anthropic_base") or spec.get("openai_base")) + "/v1/messages"
    return _with_v1(spec.get("openai_base")) + "/responses"


def _probe_endpoint(url, token, protocol, model, timeout=10):
    """Send a minimal valid request and return (status, body).

    The probe is intentionally one token: it verifies the route exists and the
    response has the expected Anthropic message / OpenAI response shape, without
    affecting the actual run budget.  404/405 or a connection failure means the
    protocol may need the local proxy.
    """
    if protocol == "anthropic":
        body = {
            "model": model,
            "max_tokens": 1,
            "messages": [{"role": "user", "content": "ping"}],
            "stream": False,
        }
    else:
        body = {
            "model": model,
            "input": "ping",
            "max_output_tokens": 1,
            "stream": False,
        }
    headers = {"Content-Type": "application/json"}
    if protocol == "anthropic":
        headers["anthropic-version"] = "2023-06-01"
    if token:
        headers["Authorization"] = "Bearer " + token
        if protocol == "anthropic":
            headers["x-api-key"] = token
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read(8192)
    except urllib.error.HTTPError as exc:
        try:
            raw = exc.read(8192)
        except OSError:
            raw = b""
        finally:
            exc.close()
        return exc.code, raw
    except (urllib.error.URLError, TimeoutError, OSError):
        return None, b""


def _probe_shape_ok(protocol, status, raw):
    if not (200 <= status < 300):
        return False
    text = raw.decode("utf-8", "replace")
    if protocol == "anthropic" and "event: message_start" in text:
        return True
    if protocol == "responses" and "event: response." in text:
        return True
    try:
        payload = json.loads(text)
    except ValueError:
        return False
    if not isinstance(payload, dict):
        return False
    if protocol == "anthropic":
        return payload.get("type") == "message" or isinstance(payload.get("content"), list)
    return payload.get("object") == "response" or isinstance(payload.get("output"), list)


def protocol_supported(spec, protocol):
    """Auto-detect native support for the protocol the selected agent speaks."""
    url = _probe_url(spec, protocol)
    status, raw = _probe_endpoint(url, model_token(spec), protocol, spec["name"])
    if status is None:
        log(f"{protocol} 协议探测: {url} 连接失败/超时，按不支持处理（将使用本地代理）")
        return False
    if status in (404, 405):
        log(f"{protocol} 协议探测: {url} -> HTTP {status}，按不支持处理（将使用本地代理）")
        return False
    if 200 <= status < 300:
        if _probe_shape_ok(protocol, status, raw):
            log(f"{protocol} 协议探测: {url} -> HTTP {status}，返回结构匹配，直连上游")
            return True
        log(f"{protocol} 协议探测: {url} -> HTTP {status}，但返回结构不匹配，使用本地代理")
        return False
    if status >= 500:
        log(f"{protocol} 协议探测: {url} -> HTTP {status}，按不支持处理（将使用本地代理）")
        return False
    # 400/401/403/422: the route exists; the request may lack credentials or a
    # required field.  Do not start a proxy that cannot fix those errors.
    log(f"{protocol} 协议探测: {url} -> HTTP {status}，接口存在，直连上游")
    return True


def start_model_proxy(spec, runs_dir, run_id, protocol, usage_token=None, native_endpoint=None):
    """Start one run-scoped protocol bridge or native-protocol usage relay.

    ``protocol`` is ``anthropic`` (Claude Code /v1/messages) or ``responses``
    (Codex /v1/responses).  Both protocols live in tools/proxy.py.  The proxy
    binds a random localhost port and exits with its parent (atexit + parent
    PID watchdog), so it never shares a port with another user's service.
    """
    upstream = expand_env(spec.get("anthropic_base" if native_endpoint == "anthropic" else "openai_base") or "")
    if not upstream:
        raise SystemExit(
            f"模型条目缺少 openai_base，无法为 {protocol} 启动协议代理: {spec.get('key') or spec.get('name')}")
    port = _free_local_port()
    log_path = per_run_dir(runs_dir, run_id) / f"{protocol}_proxy.log"
    command = [
        DEFAULT_PYTHON, str(Path(__file__).resolve().parent / "tools" / "proxy.py"),
        "--protocol", protocol,
        "--parent-pid", str(os.getpid()),
        "--host", "127.0.0.1", "--port", str(port),
        "--upstream", upstream,
        "--timeout", "3600",
    ]
    if protocol == "anthropic":
        command += ["--max-tokens", str(int(spec.get("max_output") or 16384)),
                    "--model", spec["name"]]
    elif protocol == "responses":
        if spec.get("max_output"):
            command += ["--max-output", str(int(spec["max_output"]))]
        command += ["--name", spec["name"]]
    env = os.environ.copy()
    env["PROXY_API_KEY"] = model_token(spec)
    if usage_token:
        env["NATIVE_BENCH_USAGE_TOKEN"] = usage_token
    with open(log_path, "ab", buffering=0) as log_file:
        proc = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT,
                                env=env, start_new_session=True)
    # 等端口可连接；代理启动失败时把日志路径带回给用户。
    deadline = time.time() + 10
    while time.time() < deadline:
        if proc.poll() is not None:
            raise SystemExit(
                f"{protocol} 代理启动失败（exit={proc.returncode}），日志: {log_path}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                break
        except OSError:
            time.sleep(0.1)
    else:
        proc.terminate()
        raise SystemExit(f"{protocol} 代理未在 10s 内就绪，日志: {log_path}")
    adjusted = dict(spec)
    if protocol == "passthrough":
        from urllib.parse import urlsplit
        base_path = urlsplit(upstream).path.rstrip("/")
        adjusted["openai_base"] = f"http://127.0.0.1:{port}{base_path}"
        adjusted["anthropic_base"] = f"http://127.0.0.1:{port}{base_path}"
    elif protocol == "anthropic":
        adjusted["anthropic_base"] = f"http://127.0.0.1:{port}"
    else:
        adjusted["openai_base"] = f"http://127.0.0.1:{port}/v1"
        adjusted["wire_api"] = "responses"
        adjusted["responses_ok"] = True
    log(f"{protocol} proxy: http://127.0.0.1:{port} -> {upstream}"
        f"（日志 {log_path}）")
    return proc, adjusted


def stop_model_proxy(proc):
    """Terminate a run-level model proxy; safe to call more than once."""
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def prepare_model_artifacts(catalog, agent_name, harness, runs_dir, run_id,
                           model=None, model_url=None, model_name=None, codex_template_home=None,
                           start_proxy=False):
    """解析本次 run 的模型并把 per-run 产物写盘；返回 (spec, claude_settings, extra_env, proxy)。

    四个 agent 的落地方式（都只写 run/batch 私有目录，绝不改全局配置）：
      claude   → <runs_dir>/.per-run/<run_id>/claude-settings.json（用 --settings 下发）
      codex    → <runs_dir>/.per-run/<run_id>/codex-home/（config.toml + models.json）
      opencode → <runs_dir>/.opencode-runtime/<run_id>.json
      bare     → 无需额外文件（模型写在 run config 里）
    """
    spec = resolve_model(catalog, name=model, url=model_url, raw_name=model_name)
    if spec is None:
        return None, None, {}, None
    if harness not in ("native", "bare"):
        raise SystemExit(
            f"--model 目前只支持 --harness native / bare（当前 harness={harness}）")
    write_json(per_run_dir(runs_dir, run_id) / "model.json", redact_credentials(spec))      # 可追溯
    log(f"本次 run 指定模型: {model_summary(spec)}")
    extra_env = {}
    claude_settings = None
    proxy = None
    prepared = dict(spec)
    if start_proxy and harness == "native" and agent_name in ("claude", "codex", "opencode"):
        usage_token = secrets.token_urlsafe(32)
        protocol = "passthrough"
        if agent_name == "claude" and not protocol_supported(spec, "anthropic"):
            protocol = "anthropic"
        elif agent_name == "codex" and not protocol_supported(spec, "responses"):
            protocol = "responses"
        proxy, prepared = start_model_proxy(spec, runs_dir, run_id, protocol,
            usage_token=usage_token, native_endpoint="anthropic" if agent_name == "claude" and protocol == "passthrough" else None)
        endpoint = prepared["anthropic_base" if agent_name == "claude" else "openai_base"]
        from urllib.parse import urlsplit
        parts = urlsplit(endpoint)
        extra_env["NATIVE_BENCH_USAGE_URL"] = f"{parts.scheme}://{parts.netloc}/__native_usage"
        extra_env["NATIVE_BENCH_USAGE_TOKEN"] = usage_token
    if agent_name == "claude":
        claude_settings = write_claude_settings(runs_dir, run_id, prepared)
        log(f"per-run claude settings: {claude_settings}（--settings 覆盖用户级同名 env）")
    elif agent_name == "codex":
        prepared["wire_api"] = "responses"
        home = write_codex_home(runs_dir, run_id, prepared,
                                codex_template_home or str(Path.home() / ".codex"))
        extra_env["CODEX_HOME"] = str(home)
        log(f"per-run CODEX_HOME: {home}")
    elif agent_name == "opencode":
        path = write_opencode_config(runs_dir, run_id, prepared)
        extra_env["OPENCODE_CONFIG"] = str(path)
        log(f"per-run OPENCODE_CONFIG: {path}")
    return spec, claude_settings, extra_env, proxy


def bare_run_config(catalog, resolved, model_name, run_id, budget, metadata, package_dir, spec_path,
                    profiling_measured=None, profiling_warmup=None, sandbox_url=None,
                    model_spec=None):
    """bare harness 的 run config（bare_bench/runner.py 消费）。

    与 native 的差别：agent 不是原生 CLI 而是「裸模型 + 工具池」，评测由 k-lens Backend
    承担，因此这里下发的是 Backend/客户端的连接与配额，而不是 native_bench 的
    evaluator/wrapper 字段。

    ``model_spec`` 是已经解析好的模型描述（含 --model-url / --model-name 覆盖）。
    只按注册表名重新解析会丢掉这些覆盖，把配置里的占位主机原样发下去。
    """
    bare = catalog.get("bare") or {}
    for key in ("client_root", "backend_root"):
        if not bare.get(key):
            raise SystemExit(f"op_catalog.json 缺少 bare.{key}")
    model = resolve_bare_model(catalog, model_name, spec=model_spec)
    host, port = split_sandbox(expand_env(sandbox_url or os.environ.get("SANDBOX_URL", "http://127.0.0.1:9000")))
    protocol = dict(bare.get("profiling") or {"total_runs": 8, "warmup_runs": 3, "statistic": "mean"})
    # k-lens 的 total_runs 是**含预热**的总轮次（8 = 5 测量 + 3 预热），与 bridge 的
    # --profiling-runs（测量轮数）不是同一口径；客户端会逐字校验并拒绝其它协议，
    # 所以这里永远按 catalog 的固定协议下发，用户显式传了不一致的值时只告警。
    explicit = [v for v in (profiling_measured, profiling_warmup) if v is not None]
    if profiling_measured is not None and profiling_warmup is not None \
            and profiling_measured + profiling_warmup != protocol["total_runs"]:
        log(f"warning: --profiling-runs {profiling_measured} + --warmup-runs {profiling_warmup} "
            f"= {profiling_measured + profiling_warmup}，与 bare 固定协议 {protocol} 不一致；"
            f"已按协议下发（k-lens 客户端会拒绝其它值）")
    elif explicit and profiling_measured is not None and profiling_measured != protocol["total_runs"] - protocol["warmup_runs"]:
        log(f"warning: --profiling-runs {profiling_measured} 与 bare 协议 "
            f"{protocol['total_runs']}-{protocol['warmup_runs']}={protocol['total_runs'] - protocol['warmup_runs']} 不一致；"
            f"已按协议下发")
    stop = {
        "candidate_evaluations_max": budget["candidate_evaluations_max"],
        "candidate_solutions_target": budget.get("candidate_solutions_target")
        or budget["candidate_evaluations_max"],
        "final_reprofile_count": budget.get("final_reprofile_count", 1),
        "shutdown_grace_seconds": 30,
    }
    config = {
        "run_id": run_id,
        "harness": "bare",
        "input": {
            "op_name": resolved["name"],
            "op_snake": resolved["name"],
            "source_dir": str(package_dir),
            "spec_path": str(spec_path),
            "editable_paths": metadata["editable_paths"],
            "test_paths": metadata["test_paths"],
            "ops_repo": resolved.get("ops_repo") or metadata.get("ops_repo"),
            "data_dir": expand_path(metadata.get("data_dir")),
            "test_kind": metadata.get("test_kind"),
            "test_script": (metadata["test_paths"] or [None])[0],
        },
        "agent": {"kind": "bare", "model": model},
        "stop": stop,
        "profiling": protocol,
        "backend": dict(bare.get("backend") or {}, root=expand_path(bare["backend_root"]),
                        python=bare.get("python") or DEFAULT_PYTHON),
        "client": dict(bare.get("client") or {}, root=expand_path(bare["client_root"])),
        "sandbox": {"url": sandbox_url, "host": host, "port": port},
        "metrics": {k: v for k, v in catalog.get("metrics_defaults", {}).items()
                    if k in NATIVE_METRIC_KEYS},
    }
    return config


def backend_command(catalog):
    """Expand the catalog evaluator command ({python}/{config_dir}/{env vars}).

    Both the native run config and the bridge-level smoke run must use the same
    expansion; taking ``catalog["backend"]["command"]`` verbatim leaves the
    literal ``{config_dir}`` in argv and the evaluator fails to start.
    """
    backend = catalog["backend"]
    return [expand_path(arg.replace("{python}", DEFAULT_PYTHON)
                        .replace("{config_dir}", str(Path(__file__).resolve().parent)))
            for arg in backend.get("command", [])]


def native_run_config(catalog, resolved, agent_name, run_id, budget, metadata, package_dir, spec_path,
                      profiling_measured=None, profiling_warmup=None, log_protocol=True,
                      model_spec=None, claude_settings=None):
    """把 run 预算翻译成 native schema（input/agent/evaluator/stop/metrics/isolation）。

    与 v4 的区别：没有 budget/backend/wrapper；agent 是原生 CLI（claude/codex/opencode），
    evaluator 仍是 sandbox_evaluator.py（协议两边一致）；max_model_turns、墙钟等字段
    native 不接受，会被显式忽略而不是静默写入。
    """
    native_root_path = require_native_harness(catalog)
    backend = catalog["backend"]
    command = backend_command(catalog)
    if not command or any(not arg for arg in command):
        raise SystemExit("op_catalog backend.command 为空，native 模式需要可信 evaluator 命令")
    dropped = sorted(set(budget) - {"candidate_evaluations_max", "final_reprofile_count",
                                    "candidate_solutions_target"})
    if dropped:
        log(f"native harness 无这些旧预算字段（已忽略）: {', '.join(dropped)}")
    candidate_max = budget.get("candidate_evaluations_max", 1)
    if isinstance(candidate_max, bool) or not isinstance(candidate_max, int) or candidate_max < 1:
        raise SystemExit(f"candidate_evaluations_max 必须是 >=1 的整数（native），当前 {candidate_max!r}")
    final_count = budget.get("final_reprofile_count", 1)
    if isinstance(final_count, bool) or not isinstance(final_count, int) or final_count < 0:
        raise SystemExit(f"final_reprofile_count 必须是非负整数（native），当前 {final_count!r}")
    target = budget.get("candidate_solutions_target")
    if target is not None:
        if isinstance(target, bool) or not isinstance(target, int) or target < 1:
            raise SystemExit(f"candidate_solutions_target 必须是 >=1 的整数，当前 {target!r}")
        if target > candidate_max:
            raise SystemExit(
                f"candidate_solutions_target={target} 超过 candidate_evaluations_max={candidate_max}"
                "（native 要求 1 <= 软目标 <= 硬上限）\n"
                f"  candidate_evaluations_max={candidate_max} 来自 op_catalog.budget_defaults，"
                "可用 --candidate-evaluations-max 显式覆盖。")
        if not native_supports_search(native_root_path):
            raise SystemExit(
                f"当前 native harness 不支持 search.candidate_solutions_target: {native_root_path}\n"
                "  该字段需要 v2 及以上（native_bench/config.py 认识 candidate_solutions_target）；"
                "v1 会以「未知字段」拒绝整份配置。")
        log(f"native search 软目标: candidate_solutions_target={target}（硬上限 {candidate_max}）")
    agent = {"kind": agent_name}
    agent.update(NATIVE_AGENT_DEFAULTS.get(agent_name, {}))
    inject_claude_retry(agent, log_protocol=log_protocol)
    if model_spec is not None:
        # 显式指定模型：argv 的 --model 只吃字符串；opencode 走 <provider>/<name>，
        # 端点由 per-run OPENCODE_CONFIG 提供（见 cmd_run）。
        argv_model = (f"{model_spec['provider']}/{model_spec['name']}"
                      if agent_name == "opencode" else model_spec["name"])
        agent["model"] = argv_model
        if agent_name == "claude" and claude_settings:
            # --settings 的 env 段胜过用户级 settings.json（已实测），因此可以只靠
            # 一个 per-run 文件换模型+端点，完全不碰全局配置。
            agent["extra_args"] = list(agent.get("extra_args") or []) + ["--settings", str(claude_settings)]
        log(f"native agent model: {model_summary(model_spec)} → --model {argv_model}"
            + (f"（--settings {claude_settings}）" if claude_settings else ""))
    else:
        model, source = native_agent_model(agent_name, catalog)
        if model:
            agent["model"] = model
            log(f"native agent model: {model}（{source}）")
    metrics = {key: value for key, value in (catalog.get("metrics_defaults") or {}).items()
               if key in NATIVE_METRIC_KEYS}
    measured_runs = resolve_profiling_runs(profiling_measured)
    warmup_runs = resolve_warmup_runs(profiling_warmup)
    evaluator = {
        "command": command,
        "evidence_kind": backend.get("evidence_kind", "real"),
        "per_call_timeout_seconds": backend.get("timeout_seconds", 1800),
    }
    if native_supports_profiling(native_root_path):
        # v2+ 固定协议：总轮数含预热；评测器必须原样回显，否则 harness 拒绝该次结果。
        evaluator["profiling"] = {
            "total_runs": measured_runs + warmup_runs,
            "warmup_runs": warmup_runs,
            "statistic": "mean",
        }
        if log_protocol:
            log(f"native profiling 协议: total_runs={measured_runs + warmup_runs}"
                f"（测量 {measured_runs} + 预热 {warmup_runs}）statistic=mean；"
                f"评测器会回显 raw.profiling_protocol")
    elif log_protocol:
        log(f"native profiling: 该 harness 不支持 evaluator.profiling，评测器按环境变量采样"
            f"（测量={measured_runs} 预热={warmup_runs}，共 {measured_runs + warmup_runs} 轮）")
    config = {
        "input": {
            "op_name": resolved["name"],
            "source_dir": str(package_dir),
            "spec_path": str(spec_path),
            "editable_paths": list(metadata["editable_paths"]),
            "test_paths": list(metadata["test_paths"]),
        },
        "agent": agent,
        "evaluator": evaluator,
        "stop": {
            "candidate_evaluations_max": candidate_max,
            "final_reprofile_count": final_count,
            "shutdown_grace_seconds": 30,
        },
        "metrics": metrics,
        "isolation": "process_only",
        # 便于把 native 的 uuid run 与 bridge 的 <run_id> 目录对应起来（写进初始 prompt）。
        "task_note": infra_task_note(infra_retry_policy(catalog), run_id),
    }
    if target is not None:
        config["search"] = {"candidate_solutions_target": target}
    return config


def cmd_scan(args, catalog, catalog_path):
    resolved_doc = resolve_catalog(catalog, catalog_path)
    target = Path(args.output) if args.output else Path(catalog_path).with_name("op_catalog.resolved.json")
    write_json(target, resolved_doc)
    rows = []
    for item in resolved_doc["ops"]:
        if item["status"] != "resolved":
            rows.append((item["name"], "UNRESOLVED", "-", "-", "-", "-", "; ".join(item["warnings"])))
            continue
        data = item["data"]
        data_desc = "-" if not data else f"{len(data['cases'])}c/{data['bytes'] / 1e6:.1f}MB"
        rows.append((
            item["name"],
            item["ops_repo"],
            item["bucket"],
            f"{item['agent_package']['files']}f/{item['agent_package']['bytes'] / 1e3:.0f}KB",
            item["test"]["name"] or "-",
            data_desc,
            "; ".join(item["warnings"]) or "",
        ))
    width = max(len(r[0]) for r in rows) + 2
    log(f"{'operator'.ljust(width)}{'repo':<12}{'bucket':<14}{'pkg':<14}{'test':<44}{'golden':<16}warnings")
    for row in rows:
        log("".join([row[0].ljust(width), row[1].ljust(12), row[2].ljust(14), row[3].ljust(14),
                     str(row[4])[:42].ljust(44), str(row[5]).ljust(16), row[6]]))
    log(f"\nresolved manifest: {target}")
    unresolved = [r[0] for r in rows if r[1] == "UNRESOLVED"]
    return 1 if unresolved else 0


def cmd_build(args, catalog, catalog_path):
    resolved_doc = resolve_catalog(catalog, catalog_path)
    write_json(Path(catalog_path).with_name("op_catalog.resolved.json"), resolved_doc)
    wanted = [item for item in resolved_doc["ops"]
              if args.op in (None, "all") or item["name"] in args.op]
    if not wanted:
        raise SystemExit("no matching operators")
    failures = 0
    for item in wanted:
        result = build_package(catalog, item, force=args.force)
        state = result["status"]
        if state in {"built", "exists"}:
            if state == "exists":
                metadata_for_existing(catalog, item)
            log(f"[{state}] {result['name']}: {result.get('package')}"
                + (f" ({result['files']} files, {result['bytes']} bytes)" if state == "built" else ""))
        else:
            failures += 1
            log(f"[{state}] {result['name']}: {result.get('reason', result.get('error', ''))}")
        for warning in result.get("warnings", []):
            log(f"    ! {warning}")
    return 1 if failures else 0


def cmd_validate(args, catalog, catalog_path):
    resolved_doc = resolve_catalog(catalog, catalog_path)
    wanted = [item for item in resolved_doc["ops"]
              if args.op in (None, "all") or item["name"] == args.op]
    harness = getattr(args, "harness", "native")
    failures = 0
    for item in wanted:
        result = validate_package(catalog, item, harness=harness)
        if result["status"] == "ok":
            log(f"[ok] {result['name']}: {result['files']} files, {result['bytes']} bytes"
                f" (harness={harness})")
        else:
            failures += 1
            log(f"[{result['status']}] {result['name']}: {result.get('error', '')}")
    return 1 if failures else 0


def resolve_runs_dir(catalog, args):
    """run/config 的产物根目录。

    显式 --runs-dir 优先；否则默认 <catalog.outputs.runs_dir>，并在给出 --tag 时
    自动放入 <runs_dir>/<tag>/，让同一批次的配置、产物与运行日志集中在一处。
    """
    explicit = getattr(args, "runs_dir", None)
    base = Path(expand_path(explicit)) if explicit else Path(expand_path(catalog["outputs"]["runs_dir"]))
    if not explicit and getattr(args, "tag", None):
        base = base / args.tag
    # Resolve once at runtime: per-run artifacts (CODEX_HOME, OPENCODE_CONFIG,
    # output_dir) are passed to child processes whose cwd differs from bridge/.
    base = base.resolve()
    base.mkdir(parents=True, exist_ok=True)
    return base


def resolve_agent_arg(args):
    """--agent 缺省时按 harness 推断：bare→bare，其余沿用 claude。"""
    if getattr(args, "agent", None):
        return args.agent
    harness = getattr(args, "harness", "native")
    if harness == "bare":
        return "bare"
    return "claude"


def cmd_config(args, catalog, catalog_path):
    resolved_doc = resolve_catalog(catalog, catalog_path)
    item = next((i for i in resolved_doc["ops"] if i["name"] == args.op), None)
    if item is None or item["status"] != "resolved":
        raise SystemExit(f"operator not resolved: {args.op}")
    args.agent = resolve_agent_arg(args)
    runs_dir = resolve_runs_dir(catalog, args)
    run_id = args.run_id or f"{args.op}-{args.agent}-{args.tag or 'mini'}"
    output = Path(args.output) if args.output else runs_dir / f"{run_id}.json"
    budget = {}
    for field in ("candidate_evaluations_max", "max_model_turns", "final_reprofile_count",
                  "wall_clock_seconds_max", "candidate_solutions_target"):
        value = getattr(args, field)
        if value is not None:
            budget[field] = value
    harness = getattr(args, "harness", "native")
    measured_runs = resolve_profiling_runs(getattr(args, "profiling_runs", None))
    warmup_runs = resolve_warmup_runs(getattr(args, "warmup_runs", None))
    # config 是"预览运行的配置"，同样把 per-run 模型落地，保证预览与实跑一致。
    model_spec, claude_settings, _, _ = prepare_model_artifacts(
        catalog, args.agent, harness, runs_dir, run_id,
        model=getattr(args, "model", None), model_url=getattr(args, "model_url", None),
        model_name=getattr(args, "model_name", None),
        codex_template_home=os.environ.get("CODEX_HOME") or str(Path.home() / ".codex"))
    # bare 的协议以 catalog 为准，这里传原始值（None 表示没覆盖），
    # 避免 bridge 的 measured/warm-up 默认值被误当成 k-lens 的总轮数。
    if harness == "bare":
        measured_runs = getattr(args, "profiling_runs", None)
        warmup_runs = getattr(args, "warmup_runs", None)
    make_run_config(catalog, item, args.agent, run_id, budget, output,
                    bare_model=getattr(args, "model", None) or getattr(args, "bare_model", None),
                    model_spec=model_spec, claude_settings=claude_settings,
                    sandbox_url=getattr(args, "sandbox_url", None) or os.environ.get("SANDBOX_URL"),
                    agent_timeout_seconds=args.agent_timeout_seconds, harness=harness,
                    profiling_measured=measured_runs, profiling_warmup=warmup_runs)
    log(f"harness: {harness}")
    log(f"profiling: 测量={measured_runs} 预热={warmup_runs}")
    log(str(output))
    return 0


def run_smoke(catalog, resolved, smoke_root, profiling_runs, cases, timeout):
    """One bridge-level smoke run: real compile + frozen test + msprof, no agent."""
    name = resolved["name"]
    package_dir, _, meta_path = package_layout(catalog, name)
    if not package_dir.is_dir():
        built = build_package(catalog, resolved, force=False)
        if built["status"] not in {"built", "exists"}:
            return {"op": name, "status": "no_package", "error": built.get("reason", "")}
    metadata = read_json(meta_path)
    run_dir = Path(smoke_root) / name
    artifacts = run_dir / "artifacts"
    run_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(sandbox_env(catalog, resolved, metadata, profiling_runs=profiling_runs))
    if cases:
        env["SANDBOX_TEST_DATA_CASES"] = cases
    request = {
        "schema_version": "ascend-gym-evaluator-v1",
        "evaluation_kind": "smoke",
        "source_dir": str(package_dir),
        "artifacts_dir": str(artifacts),
    }
    started = time.time()
    try:
        proc = subprocess.run(
            backend_command(catalog),
            input=json.dumps(request, ensure_ascii=False), capture_output=True, text=True,
            env=env, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"op": name, "status": "timeout", "elapsed_sec": round(time.time() - started, 1),
                "artifacts": str(artifacts), "error": f"smoke exceeded {timeout}s"}
    (run_dir / "bridge_stderr.log").write_text(proc.stderr or "", encoding="utf-8")
    elapsed = round(time.time() - started, 1)
    if proc.returncode != 0 or not (proc.stdout or "").strip():
        return {"op": name, "status": "bridge_error", "elapsed_sec": elapsed, "artifacts": str(artifacts),
                "error": (proc.stderr or proc.stdout or "")[-1200:]}
    try:
        result = json.loads(proc.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError) as exc:
        return {"op": name, "status": "bridge_error", "elapsed_sec": elapsed, "artifacts": str(artifacts),
                "error": f"unparsable bridge output: {exc}"}
    row = {
        "op": name,
        "status": result.get("status"),
        "correctness_passed": result.get("correctness_passed"),
        "duration_ms": result.get("duration_ms"),
        "elapsed_sec": elapsed,
        "ops_repo": resolved["ops_repo"],
        "artifacts": str(artifacts),
    }
    data = (result.get("raw") or {}).get("test_data")
    if data:
        row["cases_staged"] = data.get("cases_staged")
    report = result.get("profiling_report") or ""
    if report:
        row["profiling_lines"] = [line for line in report.splitlines() if line.startswith("|")]
    if result.get("status") != "passed":
        row["error"] = (result.get("diagnostic") or "")[:1200]
    write_json(run_dir / "smoke.json", row)
    return row


def cmd_smoke(args, catalog, catalog_path):
    resolved_doc = resolve_catalog(catalog, catalog_path)
    wanted = [item for item in resolved_doc["ops"]
              if args.op in (None, []) or "all" in args.op or item["name"] in args.op]
    if not wanted:
        raise SystemExit("no matching operators")
    smoke_root = Path(args.artifacts_dir) if args.artifacts_dir else \
        Path(catalog["outputs"]["runs_dir"]) / f"smoke-{time.strftime('%Y%m%d-%H%M%S')}"
    smoke_root.mkdir(parents=True, exist_ok=True)
    log(f"smoke root: {smoke_root}  (operators: {len(wanted)}, parallel: {args.parallel})")
    rows = []
    if args.parallel > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as pool:
            futures = {
                pool.submit(run_smoke, catalog, item, smoke_root, args.profiling_runs, args.cases, args.timeout): item
                for item in wanted
            }
            for future in concurrent.futures.as_completed(futures):
                row = future.result()
                rows.append(row)
                log(smoke_line(row))
    else:
        for item in wanted:
            row = run_smoke(catalog, item, smoke_root, args.profiling_runs, args.cases, args.timeout)
            rows.append(row)
            log(smoke_line(row))
    rows.sort(key=lambda r: r["op"])
    summary = {"schema_version": "ascend-gym-smoke-summary-v1",
               "root": str(smoke_root), "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               "total": len(rows), "passed": sum(1 for r in rows if r["status"] == "passed"), "rows": rows}
    write_json(smoke_root / "summary.json", summary)
    log(f"\npassed {summary['passed']}/{summary['total']}; summary: {smoke_root / 'summary.json'}")
    return 0 if summary["passed"] == summary["total"] else 1


def smoke_line(row):
    if row.get("status") == "passed":
        duration = row.get("duration_ms")
        duration_text = f"{duration * 1000:.3f}us" if isinstance(duration, (int, float)) else "n/a"
        cases = row.get("cases_staged")
        cases_text = f", cases={cases}" if cases else ""
        return (f"[passed] {row['op']}: {duration_text}, {row['elapsed_sec']}s"
                f"{cases_text} -> {row['artifacts']}")
    return f"[{row.get('status')}] {row['op']}: {row.get('error', '')[:300]}"


def cmd_run(args, catalog, catalog_path):
    resolved_doc = resolve_catalog(catalog, catalog_path)
    item = next((i for i in resolved_doc["ops"] if i["name"] == args.op), None)
    if item is None or item["status"] != "resolved":
        raise SystemExit(f"operator not resolved: {args.op}")
    runs_dir = resolve_runs_dir(catalog, args)
    args.agent = resolve_agent_arg(args)
    tag = args.tag or time.strftime("%Y%m%d-%H%M%S")
    run_id = args.run_id or f"{args.op}-{args.agent}-{tag}"
    config_path = runs_dir / f"{run_id}.json"
    budget = {}
    for field in ("candidate_evaluations_max", "max_model_turns", "final_reprofile_count",
                  "wall_clock_seconds_max", "candidate_solutions_target"):
        value = getattr(args, field, None)
        if value is not None:
            budget[field] = value
    harness = getattr(args, "harness", "native")
    measured_runs = resolve_profiling_runs(args.profiling_runs)
    warmup_runs = resolve_warmup_runs(getattr(args, "warmup_runs", None))
    if harness == "bare":
        # bare 的协议以 catalog 为准；传原始值（None=未覆盖），
        # 避免 bridge 的 measured/warm-up 默认值被误用。
        measured_runs = getattr(args, "profiling_runs", None)
        warmup_runs = getattr(args, "warmup_runs", None)
    output_dir = runs_dir / run_id
    if harness in ("native", "bare") and output_dir.exists():
        raise SystemExit(
            f"{harness} harness 要求 --output 目录必须不存在: {output_dir}\n"
            f"  换 run-id/--tag，或用 start_all_runs.py --retry-incomplete 自动换 -r2 重跑。")
    sandbox_url = expand_env(getattr(args, "sandbox_url", None) or os.environ.get("SANDBOX_URL"))

    # 本次 run 用哪个模型：--model <注册表名> / --model-url+--model-name / 环境变量；
    # 都不给则返回 None，各 agent 继续读自己的全局配置（零回归）。
    model_spec, claude_settings, model_env, model_proxy_proc = prepare_model_artifacts(
        catalog, args.agent, harness, runs_dir, run_id,
        model=getattr(args, "model", None), model_url=getattr(args, "model_url", None),
        model_name=getattr(args, "model_name", None),
        codex_template_home=os.environ.get("CODEX_HOME") or str(Path.home() / ".codex"),
        start_proxy=not args.dry_run)
    if model_proxy_proc is not None:
        import atexit
        atexit.register(stop_model_proxy, model_proxy_proc)

    config = make_run_config(catalog, item, args.agent, run_id, budget, config_path,
                             agent_timeout_seconds=args.agent_timeout_seconds, harness=harness,
                             profiling_measured=measured_runs, profiling_warmup=warmup_runs,
                             sandbox_url=sandbox_url,
                             bare_model=getattr(args, "model", None) or getattr(args, "bare_model", None),
                             model_spec=model_spec, claude_settings=claude_settings)
    log("config: " + json.dumps({
        "run_id": run_id,
        "harness": harness,
        "budget": config.get("budget") or config.get("stop"),
        "agent": redact_credentials(config["agent"]),
        "backend": config.get("backend") or config.get("evaluator"),
        "metrics": config["metrics"],
        "sandbox": config.get("sandbox"),
    }, ensure_ascii=False))
    if harness == "bare":
        model = config["agent"]["model"]
        log(f"bare harness: 裸模型={model['key']} ({model['name']} @ {model['url']})；"
            f"停止条件=" + json.dumps(
                {k: v for k, v in config["stop"].items() if k != "shutdown_grace_seconds"},
                ensure_ascii=False))
        log(f"bare harness: k-lens Backend={config['backend']['root']}；"
            f"沙盒={config['sandbox']['host']}:{config['sandbox']['port']}；"
            f"协议={json.dumps(config['profiling'], ensure_ascii=False)}")
    elif harness == "native":
        log("native harness: 无 token/墙钟/模型轮数上限；停止条件=" + json.dumps(
            {k: v for k, v in config["stop"].items() if k != "shutdown_grace_seconds"},
            ensure_ascii=False))
    else:
        log(f"agent timeout: {config['agent'].get('timeout_seconds')}s"
            f"（{'命令行' if args.agent_timeout_seconds is not None else 'op_catalog 默认值'}）")
    package_dir, _, meta_path = package_layout(catalog, item["name"])
    metadata = read_json(meta_path)
    env = os.environ.copy()
    env.update(sandbox_env(catalog, item, metadata,
                           profiling_runs=measured_runs, warmup_runs=warmup_runs))
    env.update(model_env)
    if args.agent == "codex":
        env.setdefault("CODEX_HOME", str(Path.home() / ".codex"))
    if args.agent == "opencode":
        if harness != "native":
            raise SystemExit("opencode 仅支持 --harness native")
        # native opencode 使用自己的原生工具；per-run 配置由 model_env 注入。
        model, source = native_agent_model("opencode", catalog)
        log("opencode(native) model: "
            + (f"{model}（{source}）" if model else "未固定，使用 opencode 自带配置"))
        # opencode 判定"项目目录"用的是环境变量 PWD 而不是进程真实 cwd：native harness
        # 用 cwd=<output>/workspace 拉起 CLI，却会继承我们这里的 PWD（通常是 bridge 目录），
        # 于是 workspace/.bench/ 被判成项目外目录并自动拒绝读取。这里把 PWD 对齐到
        # harness 的 workspace（native_bench 固定用 <output>/workspace）。
        env["PWD"] = str(output_dir / "workspace")
    if harness == "bare":
        command = [DEFAULT_PYTHON, "-m", "bare_bench", "run", "--config", str(config_path),
                   "--output", str(output_dir)]
        cwd = str(Path(__file__).resolve().parent)
    elif harness == "native":
        command = [DEFAULT_PYTHON, "-m", "native_bench", "run", "--config", str(config_path),
                   "--output", str(output_dir)]
        cwd = str(require_native_harness(catalog))
    else:
        raise SystemExit(
            "harness not in supported list: native, bare"
        )
    if args.dry_run:
        log("dry-run; command:")
        log("  " + " ".join(command))
        log("  cwd: " + str(cwd))
        log("environment:")
        for key in sorted(env):
            if key.startswith(("SANDBOX_", "CODEX_", "OPENCODE_")):
                log(f"  {key}={redact_credentials({key: env[key]})[key]}")
        return 0
    log("launching: " + " ".join(command))
    proc = subprocess.run(command, cwd=cwd, env=env)
    stop_model_proxy(model_proxy_proc)
    return proc.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG))
    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan", help="resolve operators and write the resolved manifest")
    scan.add_argument("--output")
    scan.set_defaults(func=cmd_scan)

    build = sub.add_parser("build", help="build text-only input packages + spec files")
    build.add_argument("--op", action="append", help="operator name, repeatable (default: all)")
    build.add_argument("--all", action="store_true")
    build.add_argument("--force", action="store_true")
    build.set_defaults(func=cmd_build)

    validate = sub.add_parser("validate", help="run the real harness preflight on built packages")
    validate.add_argument("--op", help="operator name or 'all' (default: all)")
    validate.add_argument("--harness", choices=list(HARNESS_KINDS), default="native",
                          help="native=native_bench；bare=k-lens-v2 bare_client")
    validate.set_defaults(func=cmd_validate)

    smoke = sub.add_parser("smoke", help="bridge-level smoke run on the real sandbox (no model)")
    smoke.add_argument("--op", action="append", help="operator name, repeatable, or 'all'")
    smoke.add_argument("--parallel", type=int, default=1, help="concurrent sandbox tasks (4 NPUs available)")
    smoke.add_argument("--cases", help="golden case ids for this smoke, e.g. '0,1'; default uses catalog policy")
    smoke.add_argument("--profiling-runs", type=int, default=1)
    smoke.add_argument("--timeout", type=int, default=3600)
    smoke.add_argument("--artifacts-dir", help="default <runs_dir>/smoke-<timestamp>")
    smoke.set_defaults(func=cmd_smoke)

    config = sub.add_parser("config", help="emit a harness run config")
    config.add_argument("--op", required=True)
    config.add_argument("--agent", default=None,
                        choices=["claude", "codex", "opencode", "bare"],
                        help="bare harness 的 agent 条件；缺省时按 harness 推断")
    config.add_argument("--model", dest="model",
                        help="本次 run 的模型（op_catalog.models 的键；优先级高于 catalog 默认）")
    config.add_argument("--model-url", dest="model_url",
                        help="直接指定 OpenAI 兼容端点（如 http://host:8008/v1），覆盖注册表")
    config.add_argument("--model-name", dest="model_name",
                        help="直接指定 served model name，覆盖注册表")
    config.add_argument("--bare-model", dest="bare_model",
                        help="bare harness 的裸模型名（取 op_catalog.bare.models；缺省用 default_model）")
    config.add_argument("--sandbox-url", dest="sandbox_url",
                        help="沙盒代理 URL（bare harness 需要，取 host/port 传给 k-lens-v2 Backend）")
    config.add_argument("--harness", choices=list(HARNESS_KINDS), default="native",
                        help="native=native_bench；bare=k-lens-v2 bare_client")
    config.add_argument("--run-id", help="显式 run-id（优先级高于 --tag）")
    config.add_argument("--tag", help="批次/时间戳标签；run-id = <op>-<agent>-<tag>（默认 mini）")
    config.add_argument("--output")
    config.add_argument("--runs-dir", dest="runs_dir",
                        help="产物根目录；默认 <op_catalog.outputs.runs_dir>/<tag>（不带 --tag 时为 runs_dir）")
    config.add_argument("--candidate-evaluations-max", dest="candidate_evaluations_max", type=int)
    config.add_argument("--candidate-solutions-target", dest="candidate_solutions_target", type=int,
                        help="native 软目标：希望 agent 提交的实质不同方案数（默认=硬上限；需 v2+ harness）")
    config.add_argument("--profiling-runs", type=int,
                        help="profiling 测量轮数；默认取 SANDBOX_PROFILING_RUNS，缺省 8")
    config.add_argument("--warmup-runs", type=int,
                        help="profiling 预热轮数；默认取 SANDBOX_WARMUP_RUNS，缺省 3")
    config.add_argument("--max-model-turns", dest="max_model_turns", type=int)
    config.add_argument("--final-reprofile-count", dest="final_reprofile_count", type=int)
    config.add_argument("--wall-clock-seconds", "--wall-clock-seconds-max",
                        dest="wall_clock_seconds_max", type=int, metavar="SECONDS",
                        help="搜索阶段墙钟上限（秒），覆盖 op_catalog 默认值；0=关闭限制")
    config.add_argument("--agent-timeout-seconds", dest="agent_timeout_seconds", type=int,
                        metavar="SECONDS",
                        help="所选 agent 的单轮模型调用超时（秒），覆盖 op_catalog 的 agents.<agent>.timeout_seconds")
    config.set_defaults(func=cmd_config)

    run = sub.add_parser("run", help="config + launch the harness with sandbox environment")
    run.add_argument("--op", required=True)
    run.add_argument("--agent", default=None,
                       choices=["claude", "codex", "opencode", "bare"],
                       help="bare harness 的 agent 条件；缺省时按 harness 推断")
    run.add_argument("--model", dest="model",
                        help="本次 run 的模型（op_catalog.models 的键；优先级高于 catalog 默认）")
    run.add_argument("--model-url", dest="model_url",
                        help="直接指定 OpenAI 兼容端点（如 http://host:8008/v1），覆盖注册表")
    run.add_argument("--model-name", dest="model_name",
                        help="直接指定 served model name，覆盖注册表")
    run.add_argument("--bare-model", dest="bare_model",
                       help="bare harness 的裸模型名（取 op_catalog.bare.models；缺省用 default_model）")
    run.add_argument("--sandbox-url", dest="sandbox_url",
                       help="沙盒代理 URL（bare harness 需要，取 host/port 传给 k-lens-v2 Backend）")
    run.add_argument("--harness", choices=list(HARNESS_KINDS), default="native",
                     help="native=native_bench；bare=k-lens-v2 bare_client")
    run.add_argument("--run-id", help="显式 run-id（优先级高于 --tag）")
    run.add_argument("--tag", help="时间戳/批次标签；run-id = <op>-<agent>-<tag>，默认当前时间戳")
    run.add_argument("--runs-dir", dest="runs_dir",
                     help="产物根目录；默认 <op_catalog.outputs.runs_dir>/<tag>（不带 --tag 时为 runs_dir）")
    run.add_argument("--profiling-runs", type=int,
                     help="profiling 测量轮数（不含预热）；默认取 SANDBOX_PROFILING_RUNS，缺省 8")
    run.add_argument("--warmup-runs", type=int,
                     help="profiling 预热轮数（不含在 --profiling-runs 内）；"
                          "默认取 SANDBOX_WARMUP_RUNS，缺省 3")
    run.add_argument("--candidate-evaluations-max", dest="candidate_evaluations_max", type=int)
    run.add_argument("--candidate-solutions-target", dest="candidate_solutions_target", type=int,
                     help="native 软目标：希望 agent 提交的实质不同方案数（默认=硬上限；需 v2+ harness）")
    run.add_argument("--max-model-turns", dest="max_model_turns", type=int)
    run.add_argument("--final-reprofile-count", dest="final_reprofile_count", type=int)
    run.add_argument("--wall-clock-seconds", "--wall-clock-seconds-max",
                     dest="wall_clock_seconds_max", type=int, metavar="SECONDS",
                     help="搜索阶段墙钟上限（秒），覆盖 op_catalog 默认值；0=关闭限制")
    run.add_argument("--agent-timeout-seconds", dest="agent_timeout_seconds", type=int,
                     metavar="SECONDS",
                     help="所选 agent 的单轮模型调用超时（秒），覆盖 op_catalog 的 agents.<agent>.timeout_seconds")
    run.add_argument("--dry-run", action="store_true")
    run.set_defaults(func=cmd_run)

    args = parser.parse_args()
    global CATALOG
    CATALOG = read_json(args.catalog)
    return args.func(args, CATALOG, args.catalog)


CATALOG = read_json(DEFAULT_CATALOG)


if __name__ == "__main__":
    raise SystemExit(main())
