"""
AscendC Profiling 服务
功能：执行沙盒 profiling，编译测试验证和性能数据采集在一次调用中完成
"""
import logging
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from config.settings import SANDBOX_CONFIG, RUS_OP_PATH, RUS_OP_PATH_VERSION, PROFILING_CONFIG
from services.profile_op_summary import profile_op_summary
from utils.backup import snapshot_to_op_backup, update_best_snapshot, get_op_backup_version_info, ensure_op_in_version
from utils.naming import to_snake_case_compact
from utils.sandbox_output_parser import (
    strip_ansi,
    extract_response_json,
    extract_compile_errors,
    extract_run_errors,
)

logger = logging.getLogger(__name__)



def _to_camel_case(text: str) -> str:
    """将下划线分隔名转为驼峰，例如 sparse_attn_sharedkv -> SparseAttnSharedkv"""
    return ''.join(word.capitalize() for word in text.split('_'))


class AscendCCodeProfilingService:
    """Profiling 采集服务：编译测试验证 + 性能数据采集合二为一"""

    def __init__(self):
        self.rus_op_path = RUS_OP_PATH
        self.rus_op_path_version = RUS_OP_PATH_VERSION
        self.profiling_config = dict(PROFILING_CONFIG)
        total_runs = self.profiling_config['profiling_runs']
        warmup_runs = self.profiling_config['warmup_runs']
        statistic = self.profiling_config.get('profiling_statistic', 'mean')
        if type(total_runs) is not int or total_runs < 1:
            raise ValueError(f"profiling_runs must be a positive integer, got {total_runs!r}")
        if type(warmup_runs) is not int or not 0 <= warmup_runs < total_runs:
            raise ValueError(
                f"warmup_runs must be in [0, profiling_runs), got "
                f"warmup={warmup_runs!r} total={total_runs!r}"
            )
        if statistic != 'mean':
            raise ValueError(f"only profiling_statistic='mean' is supported, got {statistic!r}")
        self.profiling_protocol = {
            'total_runs': total_runs,
            'warmup_runs': warmup_runs,
            'statistic': statistic,
        }
        logger.info(
            "AscendCCodeProfilingService initialized, rus_op_path=%s, rus_op_path_version=%s, "
            "ascend_c_path=%s, profiling_protocol=%s",
            self.rus_op_path,
            self.rus_op_path_version,
            self.profiling_config['ascend_c_path'],
            self.profiling_protocol,
        )

    def _resolve_op_root(self, op_name: str):
        """
        按优先级查找算子代码根目录（即同时包含 op_host 和 op_kernel 的子目录）：
        1. rus_op_path_version 下 <op_name>/<op_snake>/ 是否存在
        2. rus_op_path 下 <op_name>/<op_snake>/ 是否存在
        3. 以上均不存在，返回 (None, None, 'code_not_exist')。

        新布局约定（无 _custom 后缀）：
            <root>/<op_name>/<to_snake_case_compact(op_name)>/
                ├── op_host/
                └── op_kernel/

        Returns:
            (op_root: Path, source: str, error_status: str | None)
            error_status 为 None 表示找到了有效目录。
        """
        op_snake = to_snake_case_compact(op_name)

        # ── 优先：版本路径 ──
        version_base = Path(self.rus_op_path_version)
        version_op_root = version_base / op_name / op_snake
        if version_op_root.is_dir():
            logger.info("[%s] 在 version 路径下找到算子目录: %s", op_name, version_op_root)
            return version_op_root, 'version', None

        # ── 次优：原始 op_path ──
        base = Path(self.rus_op_path)
        op_root = base / op_name / op_snake
        if op_root.is_dir():
            logger.info("[%s] 在 op_path 下找到算子目录: %s", op_name, op_root)
            return op_root, 'op_path', None

        # ── 未找到 ──
        logger.warning(
            "[%s] 未在 version_path=%s 或 op_path=%s 下找到算子目录 <%s>/<%s>",
            op_name, version_base, base, op_name, op_snake,
        )
        return None, None, 'code_not_exist'

    def execute(self, op_name: str) -> Dict[str, Any]:
        """
        执行 profiling（编译测试验证 + 性能数据采集合二为一）。

        Args:
            op_name: 算子名称（如 'SELU_Phase1_Version1'）

        Returns:
            dict: 结果字典
        """
        if not op_name:
            return {
                'status': 'failed',
                'debug_options': {'op_name': op_name, 'op_type': '2d'},
                'message': '参数错误: op_name 为空',
                'errors': [{'type': 'validation_error', 'message': 'op_name 不能为空'}],
            }

        start_time = datetime.now()

        # ── Step 1: 按优先级解析算子目录 ────────────────────────────────
        op_root, source, error_status = self._resolve_op_root(op_name)
        if error_status == 'code_not_exist':
            return {
                'status': 'code_not_exist',
                'debug_options': {'op_name': op_name, 'op_type': '2d'},
                'message': (
                    f'未找到算子 "{op_name}" 的任何代码版本。'
                    f'已查找路径: version_path={self.rus_op_path_version}, op_path={self.rus_op_path}。'
                    '请先生成或上传算子代码后重试。'
                ),
                'errors': [{'type': 'code_not_exist', 'message': f'算子 {op_name} 在两个路径下均未找到'}],
            }

        logger.info("[%s] 使用算子目录 (source=%s): %s", op_name, source, op_root)

        # 日志统一存储在 op-path-version/profiling/<op_name>/ 下
        log_dir = Path(self.rus_op_path_version) / 'profiling' / op_name
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # ── Step 2: 推导 profiling 所需路径 ──────────────────────────────
        if not op_root.is_dir():
            return {
                'status': 'failed',
                'debug_options': {'op_name': op_name, 'op_type': '2d'},
                'message': f'算子目录不存在: {op_root}',
                'errors': [{'type': 'file_not_found', 'message': f'算子目录不存在: {op_root}'}],
            }

        op_folder_path = str(op_root)
        kernel_name = _to_camel_case(op_root.name)

        # 查找 .py 执行脚本（在算子目录 op_root.parent 下，与 op_snake 子目录同级）
        # 命名约定：首选 {op_snake}_test.py（与 ascend 标准生成结果一致）；
        # 找不到时回退到任意 *.py 取第一个，兼容历史/手写测试脚本。
        op_dir = op_root.parent
        op_snake = to_snake_case_compact(op_name)
        preferred_test = op_dir / f'{op_snake}_test.py'
        if preferred_test.exists():
            exec_code_path = str(preferred_test)
        else:
            py_files = sorted(op_dir.glob('*.py'))
            if not py_files:
                return {
                    'status': 'failed',
                    'debug_options': {'op_name': op_name, 'op_type': '2d'},
                    'message': (
                        f'在 {op_dir} 中既未找到首选测试脚本 {op_snake}_test.py，'
                        f'也未找到任何 .py 文件'
                    ),
                    'errors': [{
                        'type': 'file_not_found',
                        'message': (
                            f'未找到测试脚本: {preferred_test} ；'
                            f'回退 glob 在 {op_dir} 下也没匹配到 .py'
                        ),
                    }],
                }
            exec_code_path = str(py_files[0])
            logger.warning(
                "[%s] 首选测试脚本 %s 不存在，回退到 %s",
                op_name, preferred_test.name, py_files[0].name,
            )

        # profiling 输出目录：统一存储在 op-path-version/profiling/<op_name>/ 下
        profiling_output_dir = str(log_dir / f'profiling_results_{kernel_name}_{timestamp}')
        os.makedirs(profiling_output_dir, exist_ok=True)

        logger.info(
            "[%s] op_folder=%s, kernel_name=%s, exec_code=%s, profiling_dir=%s",
            op_name, op_folder_path, kernel_name, exec_code_path, profiling_output_dir,
        )

        # ── Step 3: 执行 profiling（编译测试验证 + 性能采集一次完成） ───
        profiling_result = self._run_profiling(
            op_name=op_name,
            op_folder_path=op_folder_path,
            kernel_name=kernel_name,
            exec_code_path=exec_code_path,
            profiling_output_dir=profiling_output_dir,
            log_dir=log_dir,
            timestamp=timestamp,
        )

        # 性能采集成功后，附加整体耗时字段、更新 best 快照、注入 version_info
        if profiling_result.get('status') == 'success':
            elapsed = (datetime.now() - start_time).total_seconds()

            profiling_result['op_source'] = source
            profiling_result['op_root'] = str(op_root)
            logger.info("[%s] Profiling 全流程完成，耗时 %.3fs", op_name, elapsed)

            # 若本次使用的是 op_path（原始路径），说明是首次 profiling 且 version 目录还未初始化。
            # 在 profiling 成功后将算子迁移到 version 目录，确保后续所有写操作都在 version 路径进行，
            # 并让 best 快照、version_info 写入正确位置。
            if source == 'op_path':
                try:
                    ensure_op_in_version(op_name, self.rus_op_path, self.rus_op_path_version)
                    logger.info("[%s] 首次 profiling 成功，已将算子迁移到 version 目录", op_name)
                except Exception as e:
                    logger.warning("[%s] 迁移到 version 目录失败（不影响本次结果）: %s", op_name, e)
                # 切换 op_root 到 version 目录，后续 operator_path 从这里找
                op_root = Path(self.rus_op_path_version) / op_name

            # 成功 profiling 后快照整个算子文件夹到 op_backup，并更新 best 记录
            try:
                snapshot_info = snapshot_to_op_backup(op_name, self.rus_op_path_version)
                profiling_result['op_backup'] = snapshot_info
                logger.info("[%s] op_backup snapshot: v%d -> %s", op_name, snapshot_info['version'], snapshot_info['snapshot_dir'])

                exec_time_us = profiling_result.get('exec_time_us')
                if exec_time_us is not None:
                    eps = float(self.profiling_config.get('min_improvement_epsilon', 0.0))
                    best_result = update_best_snapshot(
                        op_name,
                        self.rus_op_path_version,
                        exec_time_us,
                        snapshot_info['version'],
                        min_improvement_epsilon=eps,
                    )
                    profiling_result['best_update'] = best_result
                    logger.info(
                        "[%s] best update: updated=%s, best_exec_time_us=%.2f (eps=%.4f)",
                        op_name, best_result['updated'],
                        best_result['best_exec_time_us'] or 0, eps,
                    )
            except Exception as e:
                logger.warning("[%s] op_backup snapshot 失败（不影响本次结果）: %s", op_name, e)

            profiling_result['version_info'] = get_op_backup_version_info(op_name, self.rus_op_path_version)
        else:
            # 非成功状态也注入 version_info
            profiling_result['version_info'] = get_op_backup_version_info(op_name, self.rus_op_path_version)

        profiling_result.setdefault('profiling_protocol', dict(self.profiling_protocol))
        profiling_result.setdefault(
            'measured_runs',
            self.profiling_protocol['total_runs'] - self.profiling_protocol['warmup_runs'],
        )
        return profiling_result

    # ──────────────────────────────────────────────────────────────────────
    # 内部方法
    # ──────────────────────────────────────────────────────────────────────

    def _run_profiling(
        self,
        op_name: str,
        op_folder_path: str,
        kernel_name: str,
        exec_code_path: str,
        profiling_output_dir: str,
        log_dir: Path = None,
        timestamp: str = None,
    ) -> Dict[str, Any]:
        """执行 profiling 命令，验证编译测试结果，并生成 Markdown 报告。"""
        cfg = self.profiling_config
        ascend_c_path = cfg['ascend_c_path']

        profiling_cmd = (
            f"set -euo pipefail\n"
            f"cd {ascend_c_path}\n"
            f"python {cfg['main_py_path']} \\\n"
            f"--custom-op-dir {op_folder_path} \\\n"
            f"--exec-code-path {exec_code_path} \\\n"
            f"--host {cfg['host']} \\\n"
            f"--port {cfg['port']} \\\n"
            f"--compile-timeout {cfg['compile_timeout']} \\\n"
            f"--run-timeout {cfg['run_timeout']} \\\n"
            f"--profile \\\n"
            f"--profiling-runs {cfg['profiling_runs']} \\\n"
            f"--profiling-output-dir {profiling_output_dir}\n"
        )

        logger.info("[%s] 执行 profiling 命令:\n%s", op_name, profiling_cmd)

        timeout_sec = int(cfg['run_timeout']) / 1000 + 60
        profiling_log_path = None

        try:
            proc = subprocess.run(
                profiling_cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired:
            logger.error("[%s] Profiling 超时", op_name)
            return {
                'status': 'profiling_failed',
                'debug_status': 'profiling_failed',
                'debug_options': {'op_name': op_name, 'op_type': '2d'},
                'message': f'Profiling 执行超时 ({timeout_sec:.0f}s)',
                'errors': [{'type': 'timeout', 'message': f'超时 {timeout_sec:.0f}s'}],
            }
        except Exception as e:
            logger.error("[%s] Profiling 执行异常: %s", op_name, str(e))
            return {
                'status': 'profiling_failed',
                'debug_status': 'profiling_failed',
                'debug_options': {'op_name': op_name, 'op_type': '2d'},
                'message': f'Profiling 执行异常: {str(e)}',
                'errors': [{'type': 'system_error', 'message': str(e)}],
            }

        logger.info("[%s] Profiling 命令退出码: %d", op_name, proc.returncode)

        # 保存 profiling 日志（stdout + stderr）
        if log_dir is not None:
            ts = timestamp or datetime.now().strftime('%Y%m%d_%H%M%S')
            profiling_log_path = log_dir / f'profiling_{ts}.log'
            try:
                log_content = f"=== STDOUT ===\n{proc.stdout or ''}\n=== STDERR ===\n{proc.stderr or ''}"
                profiling_log_path.write_text(log_content, encoding='utf-8')
                logger.info("[%s] Profiling 日志已保存: %s", op_name, profiling_log_path)
            except Exception as e:
                logger.warning("[%s] Profiling 日志保存失败: %s", op_name, e)
                profiling_log_path = None

        # 验证编译测试结果（从 profiling 的同一份服务器响应中）
        validation = self._validate_profiling_response(op_name, proc.stdout, proc.stderr)

        if validation['status'] != 'success':
            logger.warning("[%s] Profiling 验证失败: %s", op_name, validation['status'])
            return {
                'status': validation['status'],
                'debug_status': validation['status'],
                'debug_options': {'op_name': op_name, 'op_type': '2d'},
                'message': validation['message'],
                'errors': validation['errors'],
                'debug_output': validation.get('output', ''),
                'compile_detail': validation.get('compile_detail'),
                'run_detail': validation.get('run_detail'),
                'profiling_stdout': proc.stdout[:2000] if proc.stdout else '',
                'profiling_stderr': proc.stderr[:2000] if proc.stderr else '',
                'profiling_log_path': str(profiling_log_path) if profiling_log_path else None,
                'op_source': None,
            }

        # 提取 CSV 并生成 Markdown 报告
        result = self._extract_and_build_report(op_name, profiling_output_dir)
        if profiling_log_path:
            result['profiling_log_path'] = str(profiling_log_path)
        return result

    def _validate_profiling_response(self, op_name: str, stdout: str, stderr: str) -> Dict[str, Any]:
        """从 profiling 命令的服务器响应中验证编译测试结果。

        验证逻辑与 SandboxCompileTestService._parse_output 一致：
        1. HTTP 状态码检查
        2. 编译结果检查（return_code + 错误提取）
        3. 运行结果检查（return_code + 错误提取）
        4. 精度检查（PASS/FAIL 模式匹配）
        """
        clean_output = strip_ansi(stdout or '')

        # 检查 HTTP 状态码
        status_code_match = re.search(r'Status Code:\s*(\d+)', clean_output)
        if status_code_match:
            http_status = int(status_code_match.group(1))
            if http_status != 200:
                return {
                    'status': 'failed',
                    'message': f'沙盒服务返回 HTTP {http_status}，可能服务繁忙或超时',
                    'errors': [{'type': 'sandbox_http_error', 'message': f'沙盒服务返回 HTTP {http_status}'}],
                    'output': clean_output[:2000],
                }

        # 提取服务器响应 JSON
        response_json = extract_response_json(clean_output)
        if response_json is None:
            # 把 stdout 尾部 3000 字符塞进 errors.message，方便 LLM 看到真实输出
            # 而不是只有一句"无法解析"——sandbox 各种异常都会卡在这一步。
            tail = (clean_output[-3000:]).strip()
            return {
                'status': 'failed',
                'message': '无法解析沙盒输出',
                'errors': [{
                    'type': 'parse_error',
                    'message': '无法从沙盒输出中解析 JSON 响应\n' + tail,
                }],
                'output': clean_output[:2000],
            }

        # SandboxError 兜底：sandbox 自身崩溃（不是用户代码错误）时，
        # response_json 会带 status=SandboxError 且 compile_result 为 null。
        # 不做这层区分的话会被后续逻辑当成"无运行结果"汇报为 compile_failed，
        # 让 LLM 误以为是它的代码 bug。这里把 sandbox message 抽出来透传，
        # 让上层能看出是基础设施问题。
        sandbox_status = response_json.get('status', '')
        sandbox_message = response_json.get('message', '')
        if sandbox_status == 'SandboxError' and response_json.get('compile_result') is None:
            error_msg = sandbox_message or 'SandboxFusion 内部异常，未产生编译结果'
            prefix = 'exception on running code '
            if error_msg.startswith(prefix):
                colon_pos = error_msg.find(': ', len(prefix))
                error_msg = (
                    error_msg[colon_pos + 2:].strip() if colon_pos != -1
                    else error_msg[len(prefix):].strip()
                )
            error_msg = strip_ansi(error_msg)[:2000]
            return {
                'status': 'compile_failed',
                'message': f'SandboxFusion 内部异常: {error_msg[:200]}',
                'errors': [{'type': 'sandbox_error', 'message': error_msg}],
                'compile_detail': {'return_code': -1, 'execution_time': None},
                'run_detail': None,
                'output': clean_output[:2000],
            }

        compile_result = response_json.get('compile_result') or {}
        run_result = response_json.get('run_result')

        # Triton 模式下 compile_result 为 null，跳过编译检查。其余情况
        # （AscendC 编译阶段产物存在）走标准 return_code 判定。
        # 注意：早期版本曾在 else 分支末尾错误地多写了一条 fallback return
        # 'compile_failed'，导致编译成功 + 有运行结果时也被误判为编译失败
        # （SparseAttnSharedkv Qwen3.6-27B 2026-06-25 trace 命中此 bug）。
        # 这里改写为线性 fall-through：检查编译失败 → 返回；否则继续走
        # run_result 判定。
        if not compile_result and run_result is not None:
            compile_detail = {'return_code': None, 'execution_time': None}
            logger.info("[%s] Triton 模式：compile_result 为空，跳过编译检查", op_name)
        else:
            compile_return_code = compile_result.get('return_code', -1)
            compile_stdout = compile_result.get('stdout', '')
            compile_stderr = compile_result.get('stderr', '')
            compile_detail = {
                'return_code': compile_return_code,
                'execution_time': compile_result.get('execution_time'),
            }
            # ========== 编译失败 ==========
            if compile_return_code != 0:
                errors = extract_compile_errors(compile_stdout + '\n' + compile_stderr)
                return {
                    'status': 'compile_failed',
                    'message': '编译失败',
                    'errors': errors,
                    'compile_detail': compile_detail,
                    'output': clean_output[:2000],
                }
            # 编译成功，fall through 到下面统一的 run_result 判定

        # ========== 编译成功但无运行结果 ==========
        # 适用于 AscendC 编译成功后 Step 2 profiling 未产生输出，
        # 以及 Triton 分支理论上的 run_result=None（虽然该分支判定时
        # 已要求 run_result is not None，但防御性保留此分支）。
        if run_result is None:
            return {
                'status': 'compile_failed',
                'message': '编译成功但无运行结果',
                'errors': [{'type': 'no_run_result', 'message': '编译成功但无运行结果'}],
                'compile_detail': compile_detail,
                'output': clean_output[:2000],
            }

        run_return_code = run_result.get('return_code', -1)
        run_stdout = run_result.get('stdout', '')
        run_stderr = run_result.get('stderr', '')

        run_detail = {
            'return_code': run_return_code,
            'execution_time': run_result.get('execution_time'),
            'stdout': run_stdout[:2000],
            'stderr': run_stderr[:2000] if run_stderr else '',
        }

        # ========== 运行时崩溃 ==========
        if run_return_code != 0:
            errors = extract_run_errors(run_stdout, run_stderr)
            return {
                'status': 'test_failed',
                'message': '运行时异常',
                'errors': errors,
                'compile_detail': compile_detail,
                'run_detail': run_detail,
                'output': clean_output[:2000],
            }

        # ========== 精度检查 ==========
        has_pass = bool(re.search(r'PASS', run_stdout))
        has_fail = bool(re.search(r'FAIL', run_stdout))
        # 完成标记有两族写法，都必须认：
        #   (1) "All tests PASSED" / "ALL CHECKS PASSED (10 cases)"（rms_norm 等）
        #   (2) "N/M checks passed"（hc_pre_sinkhorn / fused_gdn_gating / hc_post）
        # 只认 (1) 时，(2) 族算子即使 40/40 全过也会掉进下面的 "has_pass and not has_fail"
        # 分支被判 test_failed（"测试未完整运行"）——2026-09-26 定位：bare 批里这三个算子
        # 100% 假失败、0 候选；native 侧评测器（sandbox_evaluator）认 "checks passed"，
        # 所以同样算子在 claude/codex 下正常。N/M 形式要求每个匹配都 N==M，
        # 保留"跑了一部分用例后崩溃"的检出。
        _nm_markers = re.findall(r'(\d+)\s*/\s*(\d+)\s+checks passed', run_stdout, re.IGNORECASE)
        has_completion_marker = bool(re.search(
            r'(?:All tests|ALL CHECKS)\s+PASSED(?:\s*\(\d+\s+cases\))?',
            run_stdout,
            re.IGNORECASE,
        )) or (bool(_nm_markers) and all(a == b for a, b in _nm_markers))

        if has_completion_marker and not has_fail:
            return {
                'status': 'success',
                'message': '编译测试全部通过',
                'errors': [],
                'compile_detail': compile_detail,
                'run_detail': run_detail,
                'output': clean_output[:2000],
            }

        if has_pass and not has_fail:
            return {
                'status': 'test_failed',
                'message': '测试未完整运行',
                'errors': [{
                    'type': 'incomplete_test_run',
                    'message': (
                        '测试输出包含部分 PASS，但缺少最终完成标记；'
                        '可能在后续用例中发生运行时崩溃'
                    ),
                }],
                'compile_detail': compile_detail,
                'run_detail': run_detail,
                'output': clean_output[:2000],
            }

        # ========== 精度不对齐 ==========
        # Full run_stdout: LLM needs to see which tensor / which element /
        # what tolerance was exceeded to fix the precision mismatch. The
        # legacy [:500] cut tail-off many cases where the actual numeric
        # diff lines appeared after a long header.
        errors = [{'type': 'precision_mismatch', 'message': f'精度校验失败: {run_stdout.strip()}'}]
        return {
            'status': 'test_mismatch',
            'message': '精度不对齐',
            'errors': errors,
            'compile_detail': compile_detail,
            'run_detail': run_detail,
            'output': clean_output[:2000],
        }

    def _extract_and_build_report(self, op_name: str, profiling_output_dir: str) -> Dict[str, Any]:
        """从 profiling 输出目录中定位 op_summary CSV，聚合多循环数据并生成 Markdown 报告。"""
        if not os.path.isdir(profiling_output_dir):
            return {
                'status': 'profiling_failed',
                'message': f'Profiling 输出目录不存在: {profiling_output_dir}',
                'errors': [{'type': 'file_not_found', 'message': f'目录不存在: {profiling_output_dir}'}],
            }

        try:
            result = profile_op_summary(
                profiling_output_dir,
                output_name='profiling_report.md',
                write=True,
                verbose=True,
                op_name=op_name,
                total_runs=self.profiling_protocol['total_runs'],
                warmup_runs=self.profiling_protocol['warmup_runs'],
                statistic=self.profiling_protocol['statistic'],
            )

            logger.info("[%s] Profiling 报告生成成功: %s", op_name, result.report_path)
            exec_time_us = _extract_exec_time_us(result)          # 旧口径：按 (Op Type,shape,case) 分组求和
            aligned = _extract_aligned_duration(result)           # 与 native 对齐的口径
            return {
                'status': 'success',
                'message': 'Profiling 报告生成成功',
                'profiling_report_path': result.report_path,
                'profiling_report': result.report,
                # 与 native 评测器同口径：按算子名分组、组内取均值、跨算子求和（单位 us）。
                # 同一算子的多次 launch / 多 shape 只计一次，可直接与 native 的 duration_ms 比较。
                'exec_time_us': aligned,
                'duration_ms': (aligned / 1000.0) if aligned is not None else None,
                # 旧口径（按 launch/shape/case 分组后直接求和）保留，便于和 2026-09-24 之前的历史数据对照。
                'exec_time_us_legacy': exec_time_us,
                'profiling_protocol': dict(result.profiling_protocol),
                'measured_runs': result.kept_loops,
            }

        except (NotADirectoryError, FileNotFoundError, ValueError, RuntimeError) as e:
            logger.error("[%s] Profiling 数据处理失败: %s", op_name, str(e))
            return {
                'status': 'profiling_failed',
                'message': f'Profiling 数据处理失败: {str(e)}',
                'errors': [{'type': 'extract_error', 'message': str(e)}],
            }
        except Exception as e:
            logger.error("[%s] 提取 profiling 字段时异常: %s", op_name, str(e))
            return {
                'status': 'profiling_failed',
                'message': f'提取 profiling 字段时发生异常: {str(e)}',
                'errors': [{'type': 'extract_error', 'message': str(e)}],
            }


def _extract_aligned_duration(profile_result) -> float | None:
    """与 native 评测器同口径的耗时（微秒）：按算子名分组 → 组内按样本数加权平均 → 跨算子求和。

    对照 native 侧 `sandbox_evaluator.parse_profiling`（:419-491）：
      * native 用 Op Name（缺失退回 Op Type / Kernel Name）作为分组标签，把一个算子的所有
        采样（多个 launch、多 shape、多轮）池化成一个均值，然后对所有算子求和，再 /1000 得 ms；
      * 本模块的 groups 是按 (Op Type, shapes, case_idx) 分的，直接求和会把同一算子的
        多次 launch/多 shape 重复累加（历史上 bare 的数因此比 native 大一个量级），
        这里按 op_name（缺失退回 op_type）重新聚合，得到与 native 可比的数。
    """
    groups = getattr(profile_result, 'groups', None)
    if not groups:
        return None
    per_op: dict[str, list[float]] = {}
    for g in groups.values():
        value = g.mean('Task Duration (us)')
        if value != value:                      # NaN：该组没有可用采样
            continue
        label = getattr(g, 'op_name', '') or getattr(g, 'op_type', '') or 'unknown'
        weight = max(1, int(getattr(g, 'samples', 1) or 1))
        acc = per_op.setdefault(label, [0.0, 0.0])
        acc[0] += value * weight
        acc[1] += weight
    if not per_op:
        return None
    return sum(total / count for total, count in per_op.values())


def _extract_exec_time_us(profile_result) -> float | None:
    """【旧口径，保留供对照】单轮端到端耗时（微秒）= 一次完整算子调用中所有 launch 的 Task Duration 之和。

    每个 group 已经是"某个 launch 在循环间的均值"，把它们求和等于"一个循环内
    完整算子调用的耗时"。对不切 launch 的算子（基线常见）等于该 launch 的均值；
    对切多个 launch 的算子（如 grid 越过 65535 上限）则正确累加。
    """
    if profile_result is None:
        return None
    groups = getattr(profile_result, 'groups', None)
    if not groups:
        return None
    durations = [
        g.mean('Task Duration (us)')
        for g in groups.values()
        if g.mean('Task Duration (us)') == g.mean('Task Duration (us)')
    ]
    return sum(durations) if durations else None
