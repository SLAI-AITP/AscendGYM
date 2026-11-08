import filecmp
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from services.get_op_version_service import GetOpVersionService
from utils.naming import to_snake_case_compact
from utils.backup import ensure_op_in_version, get_op_backup_version_info

logger = logging.getLogger(__name__)

_CODE_EXTENSIONS = {'.cpp', '.h', '.hpp', '.cc', '.c', '.cu', '.cuh', '.py'}
_WRITABLE_BOARDS = {'op_host', 'op_kernel'}


class RestoreOpVersionService:
    """Restore a snapshot into the mutable operator workspace.

    This is the write-side companion to get_op_version. It intentionally
    returns only metadata, not file contents, so LLM rollback does not push
    full source trees into the conversation.
    """

    def __init__(self, version_service: Optional[GetOpVersionService] = None):
        self.version_service = version_service or GetOpVersionService()
        logger.info("RestoreOpVersionService initialized")

    def restore(self, op_name: str, version: str) -> Dict[str, Any]:
        if version.strip().lower() == 'current':
            return {
                'success': False,
                'op_name': op_name,
                'version': version,
                'message': "restore_op_version does not accept version='current'",
            }

        ensure_op_in_version(
            op_name,
            self.version_service.op_path,
            self.version_service.op_path_version,
        )
        version_result = self.version_service.get_version(op_name, version)
        if not version_result.get('success'):
            return {
                'success': False,
                'op_name': op_name,
                'version': version,
                'message': version_result.get('message', 'failed to resolve version'),
                'available_snapshots': version_result.get('available_snapshots', []),
            }

        workspace = (
            Path(self.version_service.op_path_version)
            / op_name
            / to_snake_case_compact(op_name)
        )
        if not workspace.is_dir():
            return {
                'success': False,
                'op_name': op_name,
                'version': version_result.get('version', version),
                'message': f'workspace does not exist: {workspace}',
            }

        source_dir = self._snapshot_source_dir(op_name, version_result)
        if source_dir is None or not source_dir.is_dir():
            return {
                'success': False,
                'op_name': op_name,
                'version': version_result.get('version', version),
                'message': f"resolved version has no usable source tree: {version_result.get('version_dir')}",
            }

        restored, deleted, failed = self._sync_code_tree(source_dir, workspace)
        success = not failed
        return {
            'success': success,
            'op_name': op_name,
            'requested_version': version,
            'restored_version': version_result.get('version', version),
            'version_dir': version_result.get('version_dir'),
            'best_exec_time_us': version_result.get('best_exec_time_us'),
            'best_version': version_result.get('best_version'),
            'workspace_dir': str(workspace),
            'restored_count': len(restored),
            'deleted_count': len(deleted),
            'failed_count': len(failed),
            'restored_files': restored,
            'deleted_files': deleted,
            'failed_files': failed,
            'version_info': get_op_backup_version_info(op_name, self.version_service.op_path_version),
            'message': (
                f"restored {len(restored)} file(s), deleted {len(deleted)} stale file(s) "
                f"from {version_result.get('version', version)}"
            ),
        }

    @staticmethod
    def _snapshot_source_dir(op_name: str, version_result: Dict[str, Any]) -> Optional[Path]:
        version_dir = version_result.get('version_dir')
        if not isinstance(version_dir, str) or not version_dir:
            return None
        source_dir = Path(version_dir) / to_snake_case_compact(op_name)
        return source_dir if source_dir.is_dir() else None

    def _sync_code_tree(self, source_dir: Path, workspace: Path):
        restored = []
        deleted = []
        failed = []

        source_files = {
            str(p.relative_to(source_dir)): p
            for p in sorted(source_dir.rglob('*'))
            if self._is_writable_code_file(source_dir, p)
        }
        workspace_files = {
            str(p.relative_to(workspace)): p
            for p in sorted(workspace.rglob('*'))
            if self._is_writable_code_file(workspace, p)
        }

        for rel, src in source_files.items():
            dst = workspace / rel
            try:
                dst.parent.mkdir(parents=True, exist_ok=True)
                changed = not dst.exists() or not filecmp.cmp(str(src), str(dst), shallow=False)
                content = src.read_bytes()
                dst.write_bytes(content)
                restored.append({
                    'file_path': rel,
                    'changed': changed,
                    'bytes_written': len(content),
                })
            except Exception as e:
                logger.error('restore_op_version write failed for %s: %s', dst, e)
                failed.append({'file_path': rel, 'error': str(e)})

        for rel, dst in workspace_files.items():
            if rel in source_files:
                continue
            try:
                dst.unlink()
                deleted.append({'file_path': rel})
            except Exception as e:
                logger.error('restore_op_version delete failed for %s: %s', dst, e)
                failed.append({'file_path': rel, 'error': str(e)})

        return restored, deleted, failed

    @staticmethod
    def _is_writable_code_file(root: Path, path: Path) -> bool:
        if not path.is_file() or path.suffix.lower() not in _CODE_EXTENSIONS:
            return False
        rel_parts = path.relative_to(root).parts
        if not rel_parts:
            return False
        if rel_parts[0] == 'op_speed_test':
            return False
        return rel_parts[0] in _WRITABLE_BOARDS or len(rel_parts) == 1
