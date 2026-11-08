"""
2D tiling 代码回填服务
以文件粒度直接覆盖写入，code_items 每项只需 file_path 和 code 字段。
支持 op_host 和 op_kernel 板块。

写操作流程：
1. 首次调用时将 op-path 中的算子目录整体拷贝到 op-path-version，并生成 v1 原始快照。
2. 后续所有读写均在 op-path-version 的工作目录中进行。
3. 每次写入前备份涉及的文件到 _backups/v{N}/。
4. 返回结果包含 version_info，供模型了解当前版本状态。
"""
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.settings import CODE_BACKFILL_CONFIG, RUS_OP_PATH, RUS_OP_PATH_VERSION
from utils.naming import to_snake_case_compact
from utils.backup import ensure_op_in_version, get_op_backup_version_info

logger = logging.getLogger(__name__)


class CodeBackfill2DService:
    """2D tiling 代码回填服务（文件粒度直接覆盖写入）"""

    def __init__(self):
        self.work_dir = Path(CODE_BACKFILL_CONFIG['2d_work_dir'])
        self.op_path = RUS_OP_PATH
        self.op_path_version = RUS_OP_PATH_VERSION
        logger.info("CodeBackfill2DService initialized")

    def _get_operator_path(self, operator_name: str, base_dir: Optional[Path] = None) -> Path:
        root = base_dir if base_dir is not None else self.work_dir
        return root / operator_name / to_snake_case_compact(operator_name)

    def process(self, mode: str, board: str, operator_name: str,
                context: Optional[Dict[str, Any]] = None,
                code_items: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        logger.info(f"2D backfill: mode={mode}, board={board}, operator={to_snake_case_compact(operator_name)}")

        try:
            if mode == 'view':
                return self._view(board, operator_name)
            else:
                return self._backfill(board, operator_name, context, code_items)
        except Exception as e:
            logger.error(f"2D backfill failed: {e}", exc_info=True)
            return {
                'success': False,
                'message': f'处理过程出现异常: {str(e)}',
                'mode': mode,
                'board': board,
                'operator_name': operator_name
            }

    def _view(self, board: str, operator_name: str) -> Dict[str, Any]:
        """查看模式：优先从 op-path-version 读取，回退到 op-path。"""
        # 优先读 version 目录
        version_base = Path(self.op_path_version)
        if (version_base / operator_name).exists():
            operator_path = self._get_operator_path(operator_name, version_base)
        else:
            operator_path = self._get_operator_path(operator_name, Path(self.op_path))

        # triton 时直接查看 custom 目录，不走 board 子目录
        if board == 'triton':
            board_path = operator_path
        else:
            board_path = operator_path / board

        if not operator_path.exists():
            return {
                'success': False,
                'message': f'算子目录不存在: {operator_path}',
                'mode': 'view', 'board': board, 'operator_name': operator_name
            }
        if not board_path.exists():
            return {
                'success': False,
                'message': f'板块路径不存在: {board_path}',
                'mode': 'view', 'board': board, 'operator_name': operator_name
            }

        code_extensions = {'.cpp', '.h', '.hpp', '.cc', '.c', '.cu', '.cuh', '.py'}
        files_info = []

        for fp in sorted(board_path.rglob('*')):
            if fp.is_file() and fp.suffix.lower() in code_extensions:
                try:
                    content = fp.read_text(encoding='utf-8', errors='ignore')
                    files_info.append({
                        'file_path': str(fp.relative_to(board_path)),
                        'file_type': fp.suffix,
                        'line_count': len(content.splitlines()),
                        'content': content
                    })
                except Exception as e:
                    files_info.append({'file_path': str(fp.relative_to(board_path)), 'error': str(e)})

        return {
            'success': True,
            'message': f'成功查看板块 {board}，找到 {len(files_info)} 个文件',
            'mode': 'view', 'board': board, 'operator_name': operator_name,
            'operator_path': str(operator_path),
            'file_count': len(files_info),
            'files': files_info,
            'version_info': get_op_backup_version_info(operator_name, self.op_path_version),
        }

    def _backfill(self, board: str, operator_name: str,
                  context: Optional[Dict[str, Any]],
                  code_items: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """回填模式：按 file_path 直接覆盖写入整个文件内容（文件粒度）"""
        if not code_items:
            return {
                'success': False,
                'message': 'code_items 不能为空',
                'mode': 'backfill', 'board': board, 'operator_name': operator_name
            }

        # Step 1: 确保算子已初始化到 op-path-version
        init_result = ensure_op_in_version(operator_name, self.op_path, self.op_path_version)
        if init_result.get('initialized'):
            logger.info("backfill: initialized op %s into version path", operator_name)

        version_base = Path(self.op_path_version)
        operator_path = self._get_operator_path(operator_name, version_base)

        # triton 时直接写入 custom 目录，不走 board 子目录
        if board == 'triton':
            board_path = operator_path
        else:
            board_path = operator_path / board

        if not operator_path.exists():
            return {
                'success': False,
                'message': f'算子目录不存在: {operator_path}',
                'mode': 'backfill', 'board': board, 'operator_name': operator_name
            }
        if not board_path.exists():
            return {
                'success': False,
                'message': f'板块路径不存在: {board_path}',
                'mode': 'backfill', 'board': board, 'operator_name': operator_name
            }

        # Step 2: 写入
        results = []
        for item in code_items:
            file_path_str = item.get('file_path', '')
            code = item.get('code', '')
            target = board_path / file_path_str

            try:
                if not target.exists():
                    results.append({
                        'file_path': file_path_str,
                        'success': False,
                        'error': f'目标文件不存在: {target}'
                    })
                    continue

                target.write_text(code, encoding='utf-8')
                logger.info(f"2D backfill wrote {target}")
                results.append({
                    'file_path': file_path_str,
                    'success': True,
                    'message': '回填成功',
                    'lines_written': len(code.splitlines())
                })
            except Exception as e:
                logger.error(f"2D backfill write failed for {target}: {e}")
                results.append({
                    'file_path': file_path_str,
                    'success': False,
                    'error': str(e)
                })

        success_count = sum(1 for r in results if r.get('success'))
        return {
            'success': success_count == len(code_items),
            'message': f'成功回填 {success_count}/{len(code_items)} 个文件',
            'mode': 'backfill', 'board': board, 'operator_name': operator_name,
            'operator_path': str(operator_path),
            'total_items': len(code_items),
            'success_count': success_count,
            'failed_count': len(code_items) - success_count,
            'context': {
                'dtype': context.get('dtype', 'N/A') if context else 'N/A',
                'soc': context.get('soc', 'N/A') if context else 'N/A',
                'functionality': context.get('functionality', 'N/A') if context else 'N/A'
            },
            'version_info': get_op_backup_version_info(operator_name, self.op_path_version),
            'results': results
        }
