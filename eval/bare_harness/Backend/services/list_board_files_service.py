import logging
from pathlib import Path
from typing import Dict, Any
from config.settings import RUS_OP_PATH, RUS_OP_PATH_VERSION
from utils.naming import to_snake_case_compact

logger = logging.getLogger(__name__)


class BoardFilesListService:
    """板块文件列表服务（2D tiling）"""

    def __init__(self):
        self.rus_op_path = Path(RUS_OP_PATH)
        self.rus_op_path_version = Path(RUS_OP_PATH_VERSION)
        logger.info("BoardFilesListService initialized, rus_op_path=%s", self.rus_op_path)

    def _resolve_base(self, op_name: str) -> Path:
        """优先使用 version 目录，回退到原始 op_path。"""
        if (self.rus_op_path_version / op_name).exists():
            return self.rus_op_path_version
        return self.rus_op_path

    def list_board_files(self, op_name: str, board: str = 'triton', tiling: str = '2d') -> Dict[str, Any]:
        op_snake = to_snake_case_compact(op_name)
        base = self._resolve_base(op_name)
        custom_dir = base / op_name / op_snake

        # triton 时，列出 {custom} 目录下一级的所有文件和子目录
        if board == 'triton':
            if not custom_dir.exists():
                return {
                    'success': False,
                    'message': f'算子目录不存在: {custom_dir}',
                    'op_name': op_name,
                    'board': board,
                    'files': []
                }
            files_info = []
            for fp in sorted(custom_dir.iterdir()):
                is_dir = fp.is_dir()
                files_info.append({
                    'file_name': fp.name,
                    'type': 'directory' if is_dir else 'file'
                })
            return {
                'success': True,
                'message': f'成功列出算子 {op_name} 下一级内容，共 {len(files_info)} 项',
                'op_name': op_name,
                'board': board,
                'files': files_info
            }

        # op_speed_test 是算子目录下的 {op_snake}_test.py，不是子目录
        if board == 'op_speed_test':
            test_file = base / op_name / f"{op_snake}_test.py"
            logger.info("Listing op_speed_test: op=%s, path=%s", op_name, test_file)
            if not test_file.exists():
                return {
                    'success': False,
                    'message': f'测试脚本不存在: {test_file}',
                    'op_name': op_name,
                    'board': board,
                    'files': []
                }
            return {
                'success': True,
                'message': f'成功列出 op_speed_test，共 1 个文件',
                'op_name': op_name,
                'board': board,
                'files': [{'file_name': test_file.name, 'file_path': str(test_file)}]
            }

        board_path = base / op_name / op_snake / board
        logger.info("Listing files: op=%s, board=%s, path=%s", op_name, board, board_path)

        if not board_path.exists():
            return {
                'success': False,
                'message': f'板块目录不存在: {board_path}',
                'op_name': op_name,
                'board': board,
                'files': []
            }

        files_info = []
        for fp in sorted(board_path.rglob('*')):
            if fp.is_file():
                files_info.append({'file_name': fp.name})

        return {
            'success': True,
            'message': f'成功列出板块 {board} 的文件，共 {len(files_info)} 个文件',
            'op_name': op_name,
            'board': board,
            'files': files_info
        }
