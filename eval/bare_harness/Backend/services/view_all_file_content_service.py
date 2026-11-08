import logging
from pathlib import Path
from typing import Dict, Any
from config.settings import RUS_OP_PATH, RUS_OP_PATH_VERSION
from utils.naming import to_snake_case_compact

logger = logging.getLogger(__name__)


class ViewAllFileContentService:
    """查看文件完整内容服务（2D tiling）"""

    def __init__(self):
        self.rus_op_path = Path(RUS_OP_PATH)
        self.rus_op_path_version = Path(RUS_OP_PATH_VERSION)
        logger.info("ViewAllFileContentService initialized, rus_op_path=%s", self.rus_op_path)

    def _resolve_base(self, op_name: str) -> Path:
        """优先使用 version 目录，回退到原始 op_path。"""
        if (self.rus_op_path_version / op_name).exists():
            return self.rus_op_path_version
        return self.rus_op_path

    def view_file_content(self, op_name: str, board: str = 'triton', file_name: str = '', tiling: str = '2d') -> Dict[str, Any]:
        op_snake = to_snake_case_compact(op_name)
        base = self._resolve_base(op_name)

        # op_speed_test：测试脚本是算子目录下的 {op_snake}_test.py，不是子目录
        if board == 'op_speed_test':
            test_file = base / op_name / f"{op_snake}_test.py"
            logger.info("Viewing op_speed_test: op=%s, path=%s", op_name, test_file)
            if not test_file.exists():
                return {
                    'success': False,
                    'message': f'测试脚本不存在: {test_file}',
                    'op_name': op_name,
                    'board': board,
                    'file_name': file_name
                }
            file_path = test_file
        elif board == 'triton':
            custom_dir = base / op_name / op_snake
            logger.info("Viewing triton file: op=%s, file=%s, path=%s", op_name, file_name, custom_dir)

            file_path = None
            if custom_dir.exists():
                direct = custom_dir / file_name
                if direct.exists() and direct.is_file():
                    file_path = direct
                else:
                    file_path = next((fp for fp in custom_dir.rglob('*')
                                      if fp.is_file() and fp.name == file_name), None)

            if file_path is None:
                return {
                    'success': False,
                    'message': f'文件不存在: {file_name}',
                    'op_name': op_name,
                    'board': board,
                    'file_name': file_name
                }
        else:
            custom_dir = base / op_name / op_snake
            board_path = custom_dir / board
            logger.info("Viewing file: op=%s, board=%s, file=%s, path=%s", op_name, board, file_name, board_path)

            # 先按标准 {custom}/{board}/{file} 路径查找
            file_path = None
            if board_path.exists():
                direct = board_path / file_name
                if direct.exists() and direct.is_file():
                    file_path = direct
                else:
                    file_path = next((fp for fp in board_path.rglob('*')
                                      if fp.is_file() and fp.name == file_name), None)

            # 回退：board 子目录不存在时，在 {custom}/ 下直接查找
            if file_path is None and custom_dir.exists():
                direct = custom_dir / file_name
                if direct.exists() and direct.is_file():
                    file_path = direct
                else:
                    file_path = next((fp for fp in custom_dir.rglob('*')
                                      if fp.is_file() and fp.name == file_name), None)

            if file_path is None:
                return {
                    'success': False,
                    'message': f'文件不存在: {file_name}',
                    'op_name': op_name,
                    'board': board,
                    'file_name': file_name
                }

        file_size = file_path.stat().st_size
        if file_size > 10 * 1024 * 1024:
            return {
                'success': False,
                'message': f'文件过大 ({file_size / 1024 / 1024:.2f}MB)，超过限制 10MB',
                'op_name': op_name,
                'board': board,
                'file_name': file_name
            }

        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
        except Exception as e:
            return {
                'success': False,
                'message': f'读取文件失败: {str(e)}',
                'op_name': op_name,
                'board': board,
                'file_name': file_name
            }

        line_count = len(content.splitlines())
        return {
            'success': True,
            'message': f'成功读取文件 {file_path.name}，共 {line_count} 行',
            'op_name': op_name,
            'board': board,
            'file_info': {
                'file_name': file_path.name,
                'file_type': file_path.suffix,
                'line_count': line_count,
            },
            'content': content
        }
