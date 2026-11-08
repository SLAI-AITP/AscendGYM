# 算子冻结测试脚本准备指南

面向要往评测集里新增/替换算子的用户：说明 `${ASCEND_GYM_DATA_ROOT}/ops_src/<backend>/<bucket>/kernels/<op>/` 下的
冻结测试脚本必须满足哪些契约，`bridge build` 才能发现它、两个 harness 才能跑通它。

harness 侧的执行链路见
[`eval/bare_harness/client_v2/docs/07_OPERATOR_HARNESS.md`](../eval/bare_harness/client_v2/docs/07_OPERATOR_HARNESS.md)；
本文只讲**数据集这一侧**要准备什么。

<!-- toc -->
## 目录

- [1. 三类冻结测试](#1-三类冻结测试)
- [2. 契约 A：文件名与位置](#2-契约-a文件名与位置)
- [3. 契约 B：退出码与成功标志](#3-契约-b退出码与成功标志)
- [4. 契约 C：必须支持 `--runs N`](#4-契约-c必须支持---runs-n)
- [5. 契约 D：不得出现硬编码绝对路径](#5-契约-d不得出现硬编码绝对路径)
- [6. 新增/替换一个算子的完整步骤](#6-新增替换一个算子的完整步骤)
- [7. 校验清单](#7-校验清单)
- [8. 已知坑](#8-已知坑)

<!-- /toc -->

## 1. 三类冻结测试

`op_manager.py` 的 `test_kind()` 按内容判定，判定结果写进 `package.json.test_kind`：

| 类型 | 判定依据 | 典型来源 | 需要 golden |
|---|---|---|---|
| `golden-compare` | 输出含 `ALL CHECKS PASSED` 或 `All tests PASSED` | 数据集自带的 `<op>_test.py` | 是（`test_data/golden_input|golden_result/<Camel>/case_*.pt`） |
| `pytest` | 含 `import pytest` 或 `pytest.main` | `vllm_ascend` 独有算子（自带 CPU 金标） | 否 |
| `custom` | 其余 | 上游示例脚本 | 视脚本而定 |

三种都能被 build 收进 `op_speed_test/`。`test_kind` 只是一个标签，不决定能否评测；
真正决定成败判定的是脚本打印的标志行。桥接默认已识别 `All tests PASSED`、
`ALL CHECKS PASSED`、pytest 的 `N passed`、`N/M checks passed`，以及失败侧的
`REGRESSION DETECTED`、`[FAIL]`、`N failed` 等（见
[CATALOG_REFERENCE.md](CATALOG_REFERENCE.md) §5）。脚本的打印不符合这些模式时，
要么改成符合，要么通过 `SANDBOX_SUCCESS_PATTERN` / `SANDBOX_FAILURE_PATTERN` 覆盖判定正则。

## 2. 契约 A：文件名与位置

`find_test_script()`（`bridge/op_manager.py`）的解析顺序：

1. 先找 `<op_dir>/<op_name>_test.py`；
2. 否则取 `<op_dir>/*_test.py` 排序后的第一个；
3. **不递归子目录**。

`<op_dir>` 是 `catalog.source_root + search_root.path + <op_name>`，即
`${ASCEND_GYM_DATA_ROOT}/ops_src/<backend>/<bucket>/kernels/<op_name>/`。

由此得两条硬约束：

- **测试脚本必须是算子目录的直接子文件**，不能只放在 `<OpName>/` 之类的子目录里。
  子目录里的 `*_test.py` 对 build 不可见。
- **建议就用 `<op_name>_test.py` 这个名字**。依赖"排序后第一个"是脆弱的：同目录多一个
  `xxx_test.py` 就可能选错文件。

build 只按 **basename** 把它复制到包的 `op_speed_test/<file>.py`，目录结构不会带过去。
所以脚本里一切路径解析都必须相对于脚本自身或 CWD，不能假设还在原目录。

## 3. 契约 B：退出码与成功标志

- 全部用例通过：退出码 `0`，并打印 `ALL CHECKS PASSED (N cases)` 或 `All tests PASSED`。
- 有用例失败：退出码非 `0`，并打印可识别的失败标志（如 `REGRESSION DETECTED`、`[FAIL]`）。

桥接同时看**退出码**和**标志行**。只打印标志、退出码仍为 0，会被判成成功；
只返回非 0、没有任何标志行，诊断信息会很难读。

## 4. 契约 C：必须支持 `--runs N`

bare（k-lens-v2）的 profiling 链路是：

```text
msprof --application="python3 <test>.py --runs N" ...
```

`msprof` **不会**替你循环，它只是把 `--runs N` 当参数传给脚本。脚本必须自己重复完整用例集
`N` 次。Backend 的 `profile_op_summary.aggregate()` 硬性要求
`total_rows % total_loops == 0`，所以只跑一遍会让总行数不整除，直接
`baseline_failed`（报"总行数 X 不是循环次数 8 的整数倍"）。

native harness 对这个不整除是宽容的，所以**缺 `--runs` 的脚本在 native 下能跑、在 bare 下必挂**。
不要用 native 的结果推断 bare 可用。

### 标准收尾模板

不要改动 `main()` / `_main()` 本体，只把文件结尾的启动块换掉：

```python
def _run_suite(runs):
    """重复整套测试 runs 次（warmup + 测量）；单次 FAIL 不中断后续轮次。"""
    all_ok = True
    for run_idx in range(1, runs + 1):
        print(f"\n{'=' * 98}")
        print(f"Run {run_idx}/{runs}")
        print(f"{'=' * 98}")
        try:
            main()                      # 入口叫 _main() 就写 _main()
        except SystemExit as exc:
            if exc.code not in (0, None):
                all_ok = False
    return all_ok


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="repeated suite runs for warmup + measurement")
    parser.add_argument("--runs", type=int, default=1,
                        help="Number of repeated runs (each run iterates all cases)")
    args = parser.parse_args()
    raise SystemExit(0 if _run_suite(args.runs) else 1)
```

要点：

- 捕获 `SystemExit`，让某一轮 FAIL 不中断后续轮次；最终按累计结果返回退出码。
- `--runs` 默认 `1`，这样手动单跑行为不变。
- 用例顺序在每轮必须一致（Backend 按行统计，不按轮次对齐语义）。

## 5. 契约 D：不得出现硬编码绝对路径

宿主机上的绝对路径在沙盒里不存在。golden 数据由桥接在评测时注入到
`<op>/test_data/`，脚本按相对位置找：

```python
def resolve_data_root():
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "test_data"),
        os.path.join(here, OP_NAME, "test_data"),
        os.path.join(os.getcwd(), OP_NAME, "test_data"),
        os.path.join(os.getcwd(), "test_data"),
    ]
    for path in candidates:
        if os.path.isdir(os.path.join(path, "golden_result", OP_CAMEL)):
            return path
    raise FileNotFoundError("golden data not found. Tried:\n  " + "\n  ".join(candidates))
```

上游示例脚本常把 golden 路径写成常量，例如
`GOLDEN_INPUT_PATH = '<abs>/test_data/golden_input/<Camel>'`。收录进数据集时改成
环境变量可覆盖 + 脚本相对默认值：

```python
GOLDEN_INPUT_PATH = os.environ.get(
    'ASCENDGYM_GOLDEN_INPUT_PATH',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '../test_data/golden_input/<Camel>'))
```

不要用 `__file__.startswith('tmp')` 之类的分支判断沙盒——客户端临时目录命名会变。

## 6. 新增/替换一个算子的完整步骤

1. **放源码树**：把 `op_host/`、`op_kernel/`、`CMakeLists.txt`（必要时 `op_graph/`、`framework/`）
   放到 `${ASCEND_GYM_DATA_ROOT}/ops_src/<backend>/<bucket>/kernels/<op_name>/`。
2. **放测试脚本**：`<op_name>_test.py` 直接放在同一个目录下（见契约 A）。
3. **放 golden**：`test_data/golden_input/<Camel>/case_*.pt`、
   `test_data/golden_result/<Camel>/case_*.pt`；必要时补 `test_data/cases_manifest.json`。
4. **改测试脚本**：按契约 B / C / D 检查。改造前先留一份 `<file>.pre_runs` 或 `.bak` 备份，
   便于对比"只加了 `--runs`"这件事是否成立。
5. **登记**：把算子名加进 `bridge/configs/op_catalog.json` 的 `ops`（`prefer` 指向实际所在
   `search_root` 的 `id`）。
6. **验证**：按第 7 节跑一遍。
7. **更新快照哈希**：`datasets/SOURCE_MANIFEST.json` 里 `files[].sha256/bytes` 和
   `tasks[].bytes/files` 要跟着改，否则快照与磁盘不一致。

> 如果新脚本是"从源码仓库的改造版本原样搬过来"，**先确认差异只有 `--runs` 收尾块**：
> `diff -u` 应当只有一个 hunk，且只包含 `_run_suite` + argparse。差异更大时不要整文件覆盖，
> 只把 `_run_suite` 收尾块追加到数据集现有脚本上——否则可能把老的、带硬编码路径的脚本
> 带进来。

## 7. 校验清单

```bash
source "${ASCEND_GYM_WORKSPACE}/datasets/activate.sh"
cd "${ASCEND_GYM_WORKSPACE}/bridge"

# 1) 解析：确认每个算子的 test 字段指向你期望的文件，且没有 "no *_test.py" 告警
python3 op_manager.py scan

# 2) 构建：建到临时数据根下，避免覆盖已有包
#    ASCEND_GYM_DATA_ROOT 指到临时目录（其中 ops_src 软链到真实快照），再 build --all --force
python3 op_manager.py build --all --force

# 3) 包内自检：op_speed_test/<file>.py 存在、与 test_source 字节一致、含 --runs
python3 - <<'PY'
import json, pathlib
root = pathlib.Path("<inputs_dir>")
for pkg in sorted(root.glob("*/package.json")):
    meta = json.loads(pkg.read_text(encoding="utf-8"))
    test = pkg.parent / "agent" / meta["test_paths"][0]
    text = test.read_text(encoding="utf-8")
    assert "--runs" in text and "_run_suite" in text, test
    print("ok", meta["op_name"], test.name)
PY

# 4) 语法检查（不 import torch_npu，只查语法）
python3 -m py_compile $(find "${ASCEND_GYM_DATA_ROOT}/ops_src" -name "*_test.py")

# 5) 真机连通性（不调用模型）
python3 op_manager.py validate --op all --harness native
python3 op_manager.py smoke --op <op_name> --profiling-runs 1
```

只有第 5 步会真正暴露 `--runs` 问题：`smoke` 走 msprof，行数不整除会立刻报出来。

## 8. 已知坑

| 现象 | 原因 |
|---|---|
| `scan` 报 `no *_test.py found` | 脚本在 `<OpName>/` 子目录里，`glob` 不递归 |
| build 选中了错误的测试 | 同目录有多个 `*_test.py`，且没有 `<op_name>_test.py` |
| 沙盒里 `ModuleNotFoundError` | 只复制了 basename，脚本依赖的同目录模块没有一起进包 |
| bare `baseline_failed`，提示总行数不是循环次数的整数倍 | 脚本没解析 `--runs` |
| sandbox 里 `FileNotFoundError: golden data not found` | 硬编码路径，或 `OP_NAME`/`OP_CAMEL` 与实际目录不符 |
| 目录里存在 `*.pre_runs`、`*.bak` | 备份文件；build 不会收（不在 `op_speed_test/`），但会被快照哈希统计，注意别误提交进包 |
