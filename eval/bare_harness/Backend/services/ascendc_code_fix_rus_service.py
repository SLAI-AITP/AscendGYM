"""
2D tiling 代码修复服务（RUS）
支持 op_host 和 op_kernel 板块，view 和 fix 两种模式。

fix 模式写入策略：
- 若 code_item 包含有效的 function_name 且在文件中能找到对应函数，则进行函数粒度替换，文件其余内容不变。
- 若 function_name 有效但函数未找到，则不做任何写入，返回错误提示。
- 若 function_name 未传或为空，则进行文件级覆盖写入。

版本管理：
- 首次 fix 时将 op-path 中的算子整体拷贝到 op-path-version，生成 v1 原始快照。
- 后续所有读写均在 op-path-version 工作目录中进行。
- 每次 fix 前备份涉及文件到 _backups/v{N}/。
- 返回 version_info 供模型了解当前版本状态。
"""
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.settings import RUS_OP_PATH, RUS_OP_PATH_VERSION
from utils.naming import to_snake_case_compact
from utils.backup import ensure_op_in_version, get_op_backup_version_info

logger = logging.getLogger(__name__)


class AscendCCodeFixRusService:
    """2D tiling 代码修复服务"""

    def __init__(self):
        self.work_dir = Path(RUS_OP_PATH)
        self.op_path = RUS_OP_PATH
        self.op_path_version = RUS_OP_PATH_VERSION
        logger.info("AscendCCodeFixRusService initialized, work_dir=%s", self.work_dir)

    # ------------------------------------------------------------------
    # 路径辅助
    # ------------------------------------------------------------------

    def _get_operator_path(self, operator_name: str, base_dir: Optional[Path] = None) -> Path:
        root = base_dir if base_dir is not None else self.work_dir
        op_snake = to_snake_case_compact(operator_name)
        return root / operator_name / op_snake

    def _get_board_path(self, operator_name: str, board: str,
                        base_dir: Optional[Path] = None) -> Path:
        op_path = self._get_operator_path(operator_name, base_dir)
        if board == 'triton':
            return op_path
        return op_path / board

    def _version_base(self) -> Path:
        return Path(self.op_path_version)

    # ------------------------------------------------------------------
    # 公共入口
    # ------------------------------------------------------------------

    def process(self, mode: str, operator_name: str,
                code_items: List[Dict[str, Any]],
                context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        logger.info("2D fix: mode=%s, operator=%s", mode, operator_name)
        try:
            if mode == 'view':
                return self._view(operator_name, code_items)
            else:
                return self._fix(operator_name, code_items, context)
        except Exception as e:
            logger.error("2D fix failed: %s", str(e), exc_info=True)
            return {
                'success': False,
                'message': f'处理过程出现异常: {str(e)}',
                'mode': mode,
                'operator_name': operator_name
            }

    # ------------------------------------------------------------------
    # view 模式：优先读 version 目录，回退到 op_path
    # ------------------------------------------------------------------

    def _view(self, operator_name: str, code_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 选择读取目录
        if (self._version_base() / operator_name).exists():
            base_dir = self._version_base()
        else:
            base_dir = Path(self.op_path)

        operator_path = self._get_operator_path(operator_name, base_dir)
        results = []
        seen_files = set()

        for item in code_items:
            board = item.get('board', '')
            file_path_str = item.get('file_path', '')
            target = self._get_board_path(operator_name, board, base_dir) / file_path_str

            key = str(target)
            if key in seen_files:
                continue
            seen_files.add(key)

            if not target.exists():
                results.append({
                    'board': board, 'file_path': file_path_str,
                    'success': False, 'error': f'文件不存在: {target}'
                })
                continue

            try:
                content = target.read_text(encoding='utf-8', errors='ignore')
                results.append({
                    'board': board, 'file_path': file_path_str,
                    'success': True, 'content': content,
                    'line_count': len(content.splitlines())
                })
            except Exception as e:
                results.append({
                    'board': board, 'file_path': file_path_str,
                    'success': False, 'error': str(e)
                })

        success_count = sum(1 for r in results if r.get('success'))
        return {
            'success': success_count == len(results),
            'message': f'成功查看 {success_count}/{len(results)} 个文件',
            'mode': 'view', 'operator_name': operator_name, 'results': results,
            'version_info': get_op_backup_version_info(operator_name, self.op_path_version),
        }

    # ------------------------------------------------------------------
    # fix 模式
    # ------------------------------------------------------------------

    def _replace_cpp_function(self, content: str, function_name: str, new_code: str) -> tuple:
        """替换文件中的指定 C++ 函数，返回 (更新后的内容, 是否成功替换)"""
        lines = content.splitlines(keepends=True)
        found_functions = []
        function_start = -1
        brace_count = 0
        in_function = False
        looking_for_brace = False

        for i, line in enumerate(lines):
            if not in_function and not looking_for_brace:
                if function_name in line and '(' in line:
                    stripped = line.strip()
                    if stripped.endswith(';') or ');' in stripped:
                        continue
                    before_paren = line.split('(')[0]
                    tokens = re.findall(r'\w+', before_paren)
                    if tokens and tokens[-1] == function_name:
                        before_function = before_paren.strip()
                        if before_function == function_name:
                            continue
                        function_start = i
                        if '{' in line:
                            in_function = True
                            brace_count = line.count('{') - line.count('}')
                            if brace_count == 0:
                                function_end = i
                                found_functions.append({'start': function_start, 'end': function_end})
                                in_function = False
                                function_start = -1
                                brace_count = 0
                        else:
                            looking_for_brace = True
            elif looking_for_brace:
                if '{' in line:
                    in_function = True
                    brace_count = line.count('{') - line.count('}')
                    looking_for_brace = False
                    if brace_count == 0:
                        function_end = i
                        found_functions.append({'start': function_start, 'end': function_end})
                        in_function = False
                        function_start = -1
                        brace_count = 0
            elif in_function:
                brace_count += line.count('{') - line.count('}')
                if brace_count == 0:
                    function_end = i
                    found_functions.append({'start': function_start, 'end': function_end})
                    in_function = False
                    function_start = -1
                    brace_count = 0

        if not found_functions:
            return content, False

        target_function = found_functions[0]
        if not new_code.endswith('\n'):
            new_code += '\n'
        new_lines = (lines[:target_function['start']] +
                     [new_code] +
                     lines[target_function['end'] + 1:])
        return ''.join(new_lines), True

    def _replace_python_function(self, content: str, function_name: str, new_code: str) -> tuple:
        """替换 Python 函数，返回 (更新后的内容, 是否成功替换)"""
        lines = content.splitlines(keepends=True)
        function_start = -1
        function_end = -1
        base_indent = None
        in_function = False

        for i, line in enumerate(lines):
            stripped = line.lstrip()
            if not in_function:
                if stripped.startswith('def ') and '(' in stripped:
                    match = re.match(r'def\s+(\w+)\s*\(', stripped)
                    if match and match.group(1) == function_name:
                        function_start = i
                        base_indent = len(line) - len(stripped)
                        in_function = True
            elif in_function:
                if stripped and not stripped.startswith('#'):
                    current_indent = len(line) - len(stripped)
                    if current_indent <= base_indent:
                        function_end = i - 1
                        break

        if in_function and function_end == -1:
            function_end = len(lines) - 1

        if function_start != -1 and function_end != -1:
            if not new_code.endswith('\n'):
                new_code += '\n'
            new_lines = lines[:function_start] + [new_code] + lines[function_end + 1:]
            return ''.join(new_lines), True

        return content, False

    def _fix(self, operator_name: str, code_items: List[Dict[str, Any]],
             context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        # Step 1: 确保算子已初始化到 op-path-version
        init_result = ensure_op_in_version(operator_name, self.op_path, self.op_path_version)
        if init_result.get('initialized'):
            logger.info("fix: initialized op %s into version path", operator_name)

        base_dir = self._version_base()
        operator_path = self._get_operator_path(operator_name, base_dir)

        if not operator_path.parent.exists():
            return {
                'success': False,
                'message': f'算子目录不存在: {operator_path.parent}',
                'mode': 'fix', 'operator_name': operator_name
            }

        # Step 2: 写入
        results = []

        for item in code_items:
            board = item.get('board', '')
            file_path_str = item.get('file_path', '')
            code = item.get('code', '')
            function_name = (item.get('function_name') or '').strip()

            target = self._get_board_path(operator_name, board, base_dir) / file_path_str

            if not target.exists():
                results.append({
                    'board': board, 'file_path': file_path_str,
                    'success': False, 'error': f'目标文件不存在: {target}'
                })
                continue

            try:
                if function_name:
                    content = target.read_text(encoding='utf-8', errors='ignore')
                    if target.suffix == '.py':
                        new_content, replaced = self._replace_python_function(content, function_name, code)
                    else:
                        new_content, replaced = self._replace_cpp_function(content, function_name, code)

                    if replaced:
                        target.write_text(new_content, encoding='utf-8')
                        logger.info("2D fix: function-level replace for %s::%s", target, function_name)
                        results.append({
                            'board': board, 'file_path': file_path_str,
                            'function_name': function_name, 'success': True,
                            'write_mode': 'function',
                            'message': f'函数级替换成功: {function_name}',
                            'lines_written': len(new_content.splitlines())
                        })
                    else:
                        logger.warning("2D fix: function '%s' not found in %s", function_name, target)
                        results.append({
                            'board': board, 'file_path': file_path_str,
                            'function_name': function_name, 'success': False,
                            'error': f'未找到函数 {function_name}，请检查函数名。如需覆盖整个文件，不传 function_name 即可'
                        })
                else:
                    target.write_text(code, encoding='utf-8')
                    logger.info("2D fix: file-level write for %s", target)
                    results.append({
                        'board': board, 'file_path': file_path_str,
                        'success': True, 'write_mode': 'file',
                        'message': '修复成功（文件级）',
                        'lines_written': len(code.splitlines())
                    })
            except Exception as e:
                logger.error("2D fix write failed for %s: %s", target, e)
                results.append({
                    'board': board, 'file_path': file_path_str,
                    'success': False, 'error': str(e)
                })

        success_count = sum(1 for r in results if r.get('success'))
        return {
            'success': success_count == len(code_items),
            'message': f'成功修复 {success_count}/{len(code_items)} 个文件',
            'mode': 'fix', 'operator_name': operator_name,
            'total_items': len(code_items),
            'success_count': success_count,
            'failed_count': len(code_items) - success_count,
            'version_info': get_op_backup_version_info(operator_name, self.op_path_version),
            'results': results
        }
