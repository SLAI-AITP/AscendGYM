"""
代码回填服务（2D tiling，文件粒度直接覆盖写入）
"""
import logging
from typing import Any, Dict, List, Optional

from services.code_backfill_2d_service import CodeBackfill2DService

logger = logging.getLogger(__name__)


class CodeBackfillService:

    def __init__(self):
        self._service_2d = CodeBackfill2DService()
        logger.info("CodeBackfillService initialized")

    def process(self, mode: str, board: str, operator_name: str,
                tiling_type: Optional[str] = None,
                context: Optional[Dict[str, Any]] = None,
                code_items: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        return self._service_2d.process(mode, board, operator_name, context, code_items)
