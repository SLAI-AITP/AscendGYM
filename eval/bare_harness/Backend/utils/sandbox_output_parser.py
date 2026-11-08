"""
Shared utilities for parsing SandboxFusion server output.
Used by both sandbox_compile_test_service and ascendc_code_profiling_service.
"""
import json
import logging
import re

logger = logging.getLogger(__name__)

_ANSI_ESCAPE = re.compile(
    r'\x1b\[[0-9;]*[A-Za-z]'
    r'|\x1b\][^\x07]*\x07'
    r'|\x1b\][^\x1b]*\x1b\\'
    r'|\x1b\][^\x07\x1b\n]*'
    r'|\x1b[PX\^_].*?\x1b\\'
    r'|\x1b[^\[\]PX\^_]'
)


def strip_ansi(text: str) -> str:
    """去除 ANSI 转义码"""
    return _ANSI_ESCAPE.sub('', text)


def extract_response_json(text: str) -> dict | None:
    """从沙盒输出中提取 Full Response JSON 块。

    使用 json.JSONDecoder.raw_decode 解析，能正确处理 JSON 字符串值中
    包含的 { } 等特殊字符（简单大括号计数无法应对这种情况）。

    兼容多种 marker 格式（按优先级）：
      - 'Profiling step full server response:'  — profiling 最终响应（含 run_result）
      - 'Compile step full server response:'   — 编译步骤响应
      - 'Full server response:'                — 旧版通用 marker
      - 'Full Response JSON:'                  — 更早版本 marker

    当存在多个 marker 时，优先返回最后一个成功解析的 JSON（通常是 profiling 步骤）。
    """
    markers = [
        'Full Response JSON:',
        'Full server response:',
        'Compile step full server response:',
        'Profiling step full server response:',
    ]

    last_parsed = None
    for marker in markers:
        idx = text.find(marker)
        if idx == -1:
            continue
        json_start = text.find('{', idx)
        if json_start == -1:
            continue
        try:
            decoder = json.JSONDecoder()
            obj, _ = decoder.raw_decode(text, json_start)
            last_parsed = obj
        except (json.JSONDecodeError, ValueError):
            continue

    return last_parsed


def extract_compile_errors(compile_log: str) -> list[dict]:
    """从编译日志中提取错误信息。

    支持以下格式：
    1. 标准 GCC/Clang: file.cpp:line:col: error: ...
    2. AscendC OPC 编译器: [timestamp] [ERROR] [soc] hash file.cpp:line:col: error: ...
    3. CMake Error: CMake Error at file:line (func):\n  message\n...
    4. Python Traceback 块（构建脚本中 binding.py 等 Python 脚本异常）
    5. [ERROR] 日志行（如 binding.py 输出的 "[ERROR] There is no keyword 'OpDef'..."）
    """
    clean_log = strip_ansi(compile_log)
    lines = clean_log.splitlines()
    errors = []
    i = 0
    n = len(lines)

    file_line_pattern = re.compile(r'(?:^|[ ])[^:\s][^:]*\.\w+:\d+:\d+:')
    error_pattern = re.compile(r':\s*(fatal\s+)?error:', re.IGNORECASE)
    cmake_error_pattern = re.compile(r'^CMake Error\b', re.IGNORECASE)
    traceback_pattern = re.compile(r'^Traceback \(most recent call last\):')
    error_log_pattern = re.compile(r'[\[(]?ERROR[\])]')

    while i < n:
        line = lines[i]

        # Per-error-message truncation removed intentionally (2026-06-30):
        # the SparseAttnSharedkv Qwen3.6_27B-0630 trace showed the
        # PitfallExtractor on Client side (which keys on phrases like
        # "shadows template parameter" / "use of undeclared identifier")
        # was getting empty input because the only error surfaced was a
        # [FATAL] tail line — every real compiler diagnostic was lost
        # upstream in SandboxFusion's truncated stdout. As long as the
        # Client gets a real diagnostic block here it should reach the
        # LLM and the extractor verbatim. The full-block message is what
        # WM compile_pitfalls.recent_records keeps (capped at 200 chars
        # there for prompt embedding), but the source-of-truth message
        # itself must not be lossy.
        if cmake_error_pattern.match(line):
            end = i
            while end + 1 < n and lines[end + 1].strip() != '':
                end += 1
            block = '\n'.join(lines[i:end + 1]).strip()
            errors.append({'type': 'compile_error', 'message': block})
            i = end + 1

        elif traceback_pattern.match(line):
            tb_end = i
            for j in range(i + 1, n):
                ln = lines[j]
                if not ln.strip():
                    break
                if ln and not ln[0].isspace() and re.search(r'(Error|Exception)', ln):
                    tb_end = j
                if ln.lstrip().startswith(('+ ', 'gmake')):
                    break
            start = i
            for j in range(1, min(4, i + 1)):
                candidate = lines[i - j]
                if error_log_pattern.search(candidate) and not candidate.lstrip().startswith('+ echo'):
                    start = i - j
                else:
                    break
            block = '\n'.join(lines[start:tb_end + 1]).strip()
            if block:
                errors.append({'type': 'compile_error', 'message': block})
            i = tb_end + 1

        elif error_pattern.search(line) and not line.lstrip().startswith(('+ echo', 'gmake')):
            start = i
            for j in range(min(5, i + 1)):
                candidate = lines[i - j]
                if file_line_pattern.search(candidate):
                    start = i - j
                    break

            end = i
            while end + 1 < n:
                next_line = lines[end + 1]
                if (next_line.strip() == '' or
                        next_line.startswith(('gmake', 'make', 'CMake')) or
                        re.match(r'^\d+\s+(error|warning)s?\s+generated', next_line.strip())):
                    break
                if file_line_pattern.search(next_line) and error_pattern.search(next_line):
                    break
                end += 1

            block = '\n'.join(lines[start:end + 1]).strip()
            errors.append({'type': 'compile_error', 'message': block})
            i = end + 1

        elif error_log_pattern.search(line) and not line.lstrip().startswith(('+ echo', 'gmake')):
            if i + 1 < n and traceback_pattern.match(lines[i + 1]):
                i += 1
            else:
                errors.append({'type': 'compile_error', 'message': line.strip()})
                i += 1

        elif '[FATAL]' in line and not line.lstrip().startswith('+ echo'):
            errors.append({'type': 'compile_error', 'message': line.strip()})
            i += 1

        else:
            i += 1

    if not errors:
        errors.append({'type': 'compile_error', 'message': '编译失败，请查看完整日志'})
    return errors


def extract_run_errors(run_stdout: str, run_stderr: str) -> list[dict]:
    """从运行日志中提取运行时错误信息。

    提取策略（优先级由高到低）：
    1. Python Traceback 块：从 "Traceback (most recent call last):" 到最后一个
       异常行（RuntimeError/Exception 等），整块作为一条 error，最多保留 1 块。
    2. NPU 核心错误描述：含 "errorStr:" 的行（描述具体硬件错误类型），最多 1 条。
    3. 兜底：若以上均未提取到，返回通用信息。
    """
    stderr_text = strip_ansi(run_stderr or '')
    stdout_text = strip_ansi(run_stdout or '')
    errors = []

    tb_pattern = re.compile(r'^Traceback \(most recent call last\):', re.MULTILINE)
    tb_match = tb_pattern.search(stderr_text)
    if tb_match:
        tb_start = tb_match.start()
        lines_after = stderr_text[tb_start:].splitlines()
        tb_end_idx = 0
        for idx, ln in enumerate(lines_after):
            if idx == 0:
                continue
            if ln and not ln[0].isspace() and re.search(r'(Error|Exception)', ln):
                tb_end_idx = idx
        tb_block = '\n'.join(lines_after[:tb_end_idx + 1]).strip()
        if tb_block:
            # Untruncated for the same reason as extract_compile_errors —
            # downstream LLM / PitfallExtractor needs the full diagnostic.
            errors.append({'type': 'runtime_error', 'message': tb_block})

    for text in (stderr_text, stdout_text):
        for line in text.splitlines():
            if 'errorStr:' in line:
                errors.append({'type': 'runtime_error', 'message': line.strip()})
                break
        else:
            continue
        break

    if not errors:
        errors.append({'type': 'runtime_error', 'message': '运行时异常，请查看完整日志'})
    return errors
