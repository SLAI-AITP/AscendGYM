"""
沙盒编译测试服务
功能：通过 SandboxFusion 沙盒环境编译并测试算子（支持 2D 及其他类型）
调用方式：本地执行 bash <sandbox_path>/<op_name>_test.sh
"""
import logging
import os
import re
import subprocess

from config.settings import SANDBOX_CONFIG, RUS_OP_PATH
from utils.sandbox_output_parser import (
    strip_ansi,
    extract_response_json,
    extract_compile_errors,
    extract_run_errors,
)

logger = logging.getLogger(__name__)


class SandboxCompileTestService:
    """沙盒编译测试服务"""

    def __init__(self):
        self.sandbox_path = SANDBOX_CONFIG['sandbox_path']
        self.rus_op_path = RUS_OP_PATH
        self.timeout = SANDBOX_CONFIG['timeout']
        self.conda_env = SANDBOX_CONFIG.get('conda_env', '')
        logger.info("SandboxCompileTestService initialized, sandbox_path=%s, rus_op_path=%s, conda_env=%s",
                     self.sandbox_path, self.rus_op_path, self.conda_env)

    def execute(self, op_name):
        """
        执行沙盒编译测试。

        Args:
            op_name: 算子名称（如 'SELU_Phase1_Version1'），对应 <op_name>_test.sh

        Returns:
            dict: 与 ascendc_code_debug_service 返回结构对齐
        """
        if not op_name:
            return {
                'debug_status': 'failed',
                'debug_options': {'op_name': op_name, 'op_type': '2d'},
                'output': '',
                'errors': [{'type': 'validation_error', 'message': 'op_name 不能为空'}],
                'message': '参数错误: op_name 为空',
            }

        # 测试脚本位于 {RUS_OP_PATH}/{op_name}/{op_name}_test.sh
        script_path = os.path.join(self.rus_op_path, op_name, f'{op_name}_test.sh')
        if not os.path.isfile(script_path):
            return {
                'debug_status': 'failed',
                'debug_options': {'op_name': op_name, 'op_type': '2d'},
                'output': '',
                'errors': [{'type': 'file_not_found', 'message': f'测试脚本不存在: {script_path}'}],
                'message': f'测试脚本不存在: {script_path}',
            }

        try:
            raw_output = self._run_script(script_path)
        except subprocess.TimeoutExpired:
            return {
                'debug_status': 'test_failed',
                'debug_options': {'op_name': op_name, 'op_type': '2d'},
                'output': '',
                'errors': [{'type': 'timeout', 'message': f'沙盒执行超时 ({self.timeout}s)'}],
                'message': f'沙盒执行超时 ({self.timeout}s)',
            }
        except Exception as e:
            logger.error("Sandbox execution error: %s", str(e))
            return {
                'debug_status': 'failed',
                'debug_options': {'op_name': op_name, 'op_type': '2d'},
                'output': '',
                'errors': [{'type': 'system_error', 'message': str(e)}],
                'message': f'沙盒执行异常: {str(e)}',
            }

        return self._parse_output(op_name, raw_output)

    def _run_script(self, script_path):
        """本地执行沙盒测试脚本，返回合并的 stdout+stderr 文本。

        沙盒脚本内的 python 需要 sandbox-npu-runtime conda 环境（Python 3.11），
        而 Flask 服务可能运行在其他 conda 环境中。通过 `conda run` 确保使用正确环境。

        注意：沙盒脚本使用 set -euo pipefail，当内部 python 返回非零退出码时
        脚本会提前退出。我们仍然需要捕获已产生的输出用于解析。
        """
        logger.info("Running sandbox script: %s (conda_env=%s)", script_path, self.conda_env)

        if self.conda_env:
            # 通过 conda run 激活正确环境后执行脚本
            cmd = ['conda', 'run', '--no-capture-output', '-n', self.conda_env,
                   'bash', script_path]
        else:
            cmd = ['bash', script_path]

        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # 合并 stderr 到 stdout，保持输出顺序
            text=True,
            timeout=self.timeout,
        )
        logger.info("Sandbox script exited with code %d", proc.returncode)
        return proc.stdout or ''

    def _parse_output(self, op_name, raw_output):
        """
        解析沙盒脚本输出，判断编译测试状态。

        沙盒输出格式:
          === Testing <name> ===
          Server URL: ...
          Status Code: 200
          Full Response JSON:
          { ... }

          --- COMPILATION DETAILS ---
          ...
          --- EXECUTION DETAILS (if any) ---
          ...
        """
        clean_output = strip_ansi(raw_output)

        # 检查 HTTP 状态码（沙盒脚本输出中包含 "Status Code: xxx"）
        status_code_match = re.search(r'Status Code:\s*(\d+)', clean_output)
        if status_code_match:
            http_status = int(status_code_match.group(1))
            if http_status != 200:
                return {
                    'debug_status': 'failed',
                    'debug_options': {'op_name': op_name, 'op_type': '2d'},
                    'output': clean_output[:2000],
                    'errors': [{'type': 'sandbox_http_error', 'message': f'沙盒服务返回 HTTP {http_status}'}],
                    'message': f'沙盒服务返回 HTTP {http_status}，可能服务繁忙或超时',
                }

        # 提取 JSON 响应块
        response_json = extract_response_json(clean_output)

        if response_json is None:
            return {
                'debug_status': 'failed',
                'debug_options': {'op_name': op_name, 'op_type': '2d'},
                'output': clean_output[:2000],
                'errors': [{'type': 'parse_error', 'message': '无法从沙盒输出中解析 JSON 响应'}],
                'message': '无法解析沙盒输出',
            }

        status = response_json.get('status', '')
        compile_result = response_json.get('compile_result') or {}
        run_result = response_json.get('run_result')

        compile_return_code = compile_result.get('return_code', -1)
        compile_stdout = compile_result.get('stdout', '')
        compile_stderr = compile_result.get('stderr', '')

        # ========== 判断编译是否失败 ==========
        if compile_return_code != 0:
            errors = extract_compile_errors(compile_stdout + '\n' + compile_stderr)
            return {
                'debug_status': 'compile_failed',
                'debug_options': {'op_name': op_name, 'op_type': '2d'},
                'output': clean_output[:2000],
                'errors': errors,
                'message': '编译失败',
                'compile_detail': {
                    'return_code': compile_return_code,
                    'execution_time': compile_result.get('execution_time'),
                },
            }

        # ========== 编译成功，检查运行结果 ==========
        if run_result is None:
            # 编译成功但无运行结果（不应发生，但防御性处理）
            return {
                'debug_status': 'compile_failed',
                'debug_options': {'op_name': op_name, 'op_type': '2d'},
                'output': clean_output[:2000],
                'errors': [{'type': 'no_run_result', 'message': '编译成功但无运行结果'}],
                'message': '编译成功但无运行结果',
                'compile_detail': {
                    'return_code': compile_return_code,
                    'execution_time': compile_result.get('execution_time'),
                },
            }

        run_return_code = run_result.get('return_code', -1)
        run_stdout = run_result.get('stdout', '')
        run_stderr = run_result.get('stderr', '')

        compile_detail = {
            'return_code': compile_return_code,
            'execution_time': compile_result.get('execution_time'),
        }
        run_detail = {
            'return_code': run_return_code,
            'execution_time': run_result.get('execution_time'),
            'stdout': run_stdout[:2000],
            'stderr': run_stderr[:2000] if run_stderr else '',
        }

        # 运行时崩溃（return_code != 0）
        if run_return_code != 0:
            errors = extract_run_errors(run_stdout, run_stderr)
            return {
                'debug_status': 'test_failed',
                'debug_options': {'op_name': op_name, 'op_type': '2d'},
                'output': clean_output[:2000],
                'errors': errors,
                'message': '运行时异常',
                'compile_detail': compile_detail,
                'run_detail': run_detail,
            }

        # 运行完成，检查精度
        has_pass = bool(re.search(r'All tests PASSED|PASS', run_stdout))
        has_fail = bool(re.search(r'FAIL', run_stdout))

        if has_pass and not has_fail:
            return {
                'debug_status': 'success',
                'debug_options': {'op_name': op_name, 'op_type': '2d'},
                'output': clean_output[:2000],
                'errors': [],
                'message': '编译测试全部通过',
                'compile_detail': compile_detail,
                'run_detail': run_detail,
            }

        # 精度不对齐
        errors = [{'type': 'precision_mismatch', 'message': f'精度校验失败: {run_stdout.strip()[:500]}'}]
        return {
            'debug_status': 'test_mismatch',
            'debug_options': {'op_name': op_name, 'op_type': '2d'},
            'output': clean_output[:2000],
            'errors': errors,
            'message': '精度不对齐',
            'compile_detail': compile_detail,
            'run_detail': run_detail,
        }
