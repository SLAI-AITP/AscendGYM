"""
算子版本代码获取服务

从 op_backup/ 目录中读取指定版本或最新版本的全量快照代码。
支持：
  - version="current"：获取当前工作目录中相对 baseline 有改动的文件
  - version="best"：获取最优版本相对 *当前工作目录* 有差异的文件
  - version="latest"：获取最新快照相对 *当前工作目录* 有差异的文件
  - version="v{N}" / version="{op_name}_version{N}"：指定快照相对 *当前工作目录* 有差异的文件

设计说明：
  - `current` 走 vs-baseline diff（语义是"算子被改了哪些文件"）。
  - 其他三种走 vs-current-workspace diff（语义是"要把这些文件写回工作目录就能恢复到该快照"）。
  - 这样 orchestrator / LLM 拿到 files 后,直接逐个 code_backfill 即可把 workspace 还原到目标版本,无需自己拼 diff。
"""
import filecmp
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from config.settings import RUS_OP_PATH, RUS_OP_PATH_VERSION
from utils.naming import to_snake_case_compact
from utils.backup import _OP_BACKUP_DIR, get_best_snapshot

logger = logging.getLogger(__name__)

_CODE_EXTENSIONS = {'.cpp', '.h', '.hpp', '.cc', '.c', '.cu', '.cuh', '.py'}

_VERSION_SHORTHAND_RE = re.compile(r'^v(\d+)$', re.IGNORECASE)


class GetOpVersionService:
    """算子版本代码获取服务"""

    def __init__(self):
        self.op_path = RUS_OP_PATH
        self.op_path_version = RUS_OP_PATH_VERSION
        logger.info("GetOpVersionService initialized")

    def _resolve_operator_path(self, op_name: str) -> Optional[Path]:
        """找到算子代码子目录（含 op_host/op_kernel），优先 version 路径。

        新布局：<root>/<op_name>/<to_snake_case_compact(op_name)>/
        """
        op_snake = to_snake_case_compact(op_name)
        for base in (Path(self.op_path_version), Path(self.op_path)):
            code_dir = base / op_name / op_snake
            if code_dir.is_dir():
                return code_dir
        return None

    def _read_dir_files(self, root: Path) -> List[Dict[str, Any]]:
        """递归读取目录下所有代码文件内容。"""
        return self._read_dir_files_filtered(root, changed_files=None)

    def _read_dir_files_filtered(self, root: Path, changed_files: Optional[Set[str]] = None) -> List[Dict[str, Any]]:
        """递归读取目录下代码文件。

        Args:
            root: 要扫描的根目录
            changed_files: 与基线不同的文件相对路径集合。
                           None 表示全部返回（旧行为），
                           非 None 则只返回集合内的文件（unchanged 文件排除）。
        """
        files = []
        if not root.exists():
            return files
        for fp in sorted(root.rglob('*')):
            if fp.is_file() and fp.suffix.lower() in _CODE_EXTENSIONS:
                rel = str(fp.relative_to(root.parent))
                is_changed = changed_files is None or rel in changed_files
                if not is_changed:
                    continue
                try:
                    content = fp.read_text(encoding='utf-8', errors='ignore')
                    files.append({
                        'file_path': rel,
                        'content': content,
                        'line_count': len(content.splitlines()),
                    })
                except Exception as e:
                    files.append({'file_path': rel, 'error': str(e)})
        return files

    def _diff_against_dir(
        self, src_dir: Path, ref_dir: Path, op_name: str, label: str,
    ) -> Set[str]:
        """对比 src_dir 与 ref_dir，返回内容不同的文件相对路径集合（基于代码子目录的父目录）。"""
        changed: Set[str] = set()
        ref_files = {f for f in ref_dir.rglob('*')
                     if f.is_file() and f.suffix.lower() in _CODE_EXTENSIONS}
        src_files = {f for f in src_dir.rglob('*')
                     if f.is_file() and f.suffix.lower() in _CODE_EXTENSIONS}

        ref_rels = {str(f.relative_to(ref_dir.parent)) for f in ref_files}
        src_rels = {str(f.relative_to(src_dir.parent)) for f in src_files}

        for rel in src_rels - ref_rels:
            changed.add(rel)
        for rel in ref_rels - src_rels:
            changed.add(rel)

        for rel in src_rels & ref_rels:
            src_file = src_dir.parent / rel
            ref_file = ref_dir.parent / rel
            if not filecmp.cmp(str(src_file), str(ref_file), shallow=False):
                changed.add(rel)

        logger.info(
            "[%s] %s diff: %d changed / %d unchanged out of %d total files",
            op_name, label, len(changed), len(src_rels) - len(changed), len(src_rels),
        )
        return changed

    def _diff_against_baseline(self, custom_dir: Path, op_name: str) -> Set[str]:
        """对比 custom_dir 与 op_backup version1，返回内容不同的文件相对路径集合。

        若 version1 不存在，返回 None（表示全部视为 changed，回退旧行为）。
        """
        baseline_dir = self._find_baseline_custom_dir(op_name)
        if baseline_dir is None:
            return None
        return self._diff_against_dir(custom_dir, baseline_dir, op_name, "baseline")

    def _diff_against_current(self, snapshot_custom_dir: Path, op_name: str) -> Set[str]:
        """对比快照目录与当前 workspace。

        返回"需要从快照写回 workspace 才能让两者一致"的文件路径集合：
          - 快照里有、workspace 里没有 → 要补
          - workspace 里有、快照里没有 → 要补（用快照里"没有"的状态去覆盖；
            实际工作流中 workspace 不会比快照少文件，这条防御）
          - 两边都有但内容不同 → 要覆盖
        若当前 workspace 找不到（算子尚未生成）则返回 None。
        """
        workspace_dir = self._resolve_operator_path(op_name)
        if workspace_dir is None:
            return None
        return self._diff_against_dir(snapshot_custom_dir, workspace_dir, op_name, "vs current workspace")

    def _find_baseline_custom_dir(self, op_name: str) -> Optional[Path]:
        """找到 op_backup/{op_name}_version1 下的算子代码子目录。"""
        snapshot_dir = Path(self.op_path_version) / _OP_BACKUP_DIR / f"{op_name}_version1"
        if not snapshot_dir.is_dir():
            return None
        return self._find_custom_dir_in_snapshot(snapshot_dir, op_name)

    def _list_op_snapshots(self, op_name: str) -> List[str]:
        """列出 op_backup 下该算子的所有快照目录名。"""
        op_backup_base = Path(self.op_path_version) / _OP_BACKUP_DIR
        if not op_backup_base.exists():
            return []
        return sorted(
            d.name for d in op_backup_base.iterdir()
            if d.is_dir() and d.name.startswith(op_name + "_version")
        )

    def _find_custom_dir_in_snapshot(self, snapshot_dir: Path, op_name: str) -> Optional[Path]:
        """在全量快照目录中定位算子代码子目录。

        新布局：快照目录是 <op_path_version>/<op_name>/ 的完整拷贝，
        其下子目录名严格等于 to_snake_case_compact(op_name)。
        """
        code_dir = snapshot_dir / to_snake_case_compact(op_name)
        if code_dir.is_dir():
            return code_dir
        return None

    def get_version(self, op_name: str, version: str) -> Dict[str, Any]:
        """
        获取指定版本的代码。

        Args:
            op_name:  算子名称（驼峰，如 'WindMatmulV2'）
            version:  'current' | 'best' | 'latest' | '{op_name}_version{N}'

        Returns:
            {
                success: bool,
                op_name: str,
                version: str,
                version_dir: str,
                best_exec_time_us: float | None,  # 仅 best 版本有
                files: [{file_path, content, line_count}, ...]
            }
        """
        # ── 归一化 'v{N}' 简写到完整快照名 '{op_name}_version{N}' ──────────
        m = _VERSION_SHORTHAND_RE.match(version.strip()) if isinstance(version, str) else None
        if m:
            version = f"{op_name}_version{m.group(1)}"

        # ── current：直接读工作目录 ──────────────────────────────────────
        if version == 'current':
            operator_path = self._resolve_operator_path(op_name)
            if operator_path is None:
                return {
                    'success': False,
                    'op_name': op_name,
                    'version': version,
                    'message': f'未找到算子 {op_name} 的代码目录',
                    'files': [],
                }
            changed = self._diff_against_baseline(operator_path, op_name)
            if changed is None:
                return {
                    'success': False,
                    'op_name': op_name,
                    'version': version,
                    'message': f'算子 {op_name} 尚无基线版本（version1），代码未被修改过',
                    'files': [],
                }
            files = self._read_dir_files_filtered(operator_path, changed_files=changed)
            return {
                'success': True,
                'op_name': op_name,
                'version': 'current',
                'version_dir': str(operator_path),
                'files': files,
                'message': f'当前工作代码，共 {len(files)} 个文件与基线不同',
            }

        op_backup_base = Path(self.op_path_version) / _OP_BACKUP_DIR

        # ── best：读 best.txt 指向的最优快照（vs current workspace） ──────────
        if version == 'best':
            best = get_best_snapshot(op_name, self.op_path_version)
            if not best:
                return {
                    'success': False,
                    'op_name': op_name,
                    'version': 'best',
                    'message': '尚无最优版本记录，请先完成至少一次 profiling',
                    'files': [],
                }
            snapshot_dir = Path(best['snapshot_dir'])
            if not snapshot_dir.is_dir():
                return {
                    'success': False,
                    'op_name': op_name,
                    'version': 'best',
                    'message': f'最优快照目录不存在: {snapshot_dir}',
                    'files': [],
                }
            custom_dir = self._find_custom_dir_in_snapshot(snapshot_dir, op_name)
            changed = self._diff_against_current(custom_dir, op_name) if custom_dir else None
            if changed is None:
                return {
                    'success': False,
                    'op_name': op_name,
                    'version': 'best',
                    'message': f'未找到算子 {op_name} 的当前工作目录，无法比对',
                    'files': [],
                }
            files = self._read_dir_files_filtered(custom_dir, changed_files=changed) if custom_dir else []
            return {
                'success': True,
                'op_name': op_name,
                'version': f"{op_name}_version{best['version']}",
                'version_dir': str(snapshot_dir),
                'best_exec_time_us': best['exec_time_us'],
                'best_version': best['version'],
                'files': files,
                'message': f'最优版本代码（exec_time={best["exec_time_us"]} us），共 {len(files)} 个文件与当前工作目录不同',
            }

        # ── latest：读 op_backup 下最新的全量快照（vs current workspace） ────
        if version == 'latest':
            snapshots = self._list_op_snapshots(op_name)
            if not snapshots:
                return {
                    'success': False,
                    'op_name': op_name,
                    'version': 'latest',
                    'message': '尚无快照记录，请先完成至少一次 profiling',
                    'files': [],
                }
            latest = snapshots[-1]
            snapshot_dir = op_backup_base / latest
            custom_dir = self._find_custom_dir_in_snapshot(snapshot_dir, op_name)
            changed = self._diff_against_current(custom_dir, op_name) if custom_dir else None
            if changed is None:
                return {
                    'success': False,
                    'op_name': op_name,
                    'version': 'latest',
                    'message': f'未找到算子 {op_name} 的当前工作目录，无法比对',
                    'files': [],
                    'available_snapshots': snapshots,
                }
            files = self._read_dir_files_filtered(custom_dir, changed_files=changed) if custom_dir else []
            return {
                'success': True,
                'op_name': op_name,
                'version': latest,
                'version_dir': str(snapshot_dir),
                'files': files,
                'message': f'最新快照 {latest}，共 {len(files)} 个文件与当前工作目录不同',
                'available_snapshots': snapshots,
            }

        # ── {op_name}_version{N}：读指定全量快照（vs current workspace） ─────
        snapshot_dir = op_backup_base / version
        if not snapshot_dir.is_dir():
            available = self._list_op_snapshots(op_name)
            return {
                'success': False,
                'op_name': op_name,
                'version': version,
                'message': f'快照 {version} 不存在，可用快照: {available}',
                'available_snapshots': available,
                'files': [],
            }

        custom_dir = self._find_custom_dir_in_snapshot(snapshot_dir, op_name)
        changed = self._diff_against_current(custom_dir, op_name) if custom_dir else None
        if changed is None:
            return {
                'success': False,
                'op_name': op_name,
                'version': version,
                'message': f'未找到算子 {op_name} 的当前工作目录，无法比对',
                'files': [],
            }
        files = self._read_dir_files_filtered(custom_dir, changed_files=changed) if custom_dir else []
        return {
            'success': True,
            'op_name': op_name,
            'version': version,
            'version_dir': str(snapshot_dir),
            'files': files,
            'message': f'快照 {version}，共 {len(files)} 个文件与当前工作目录不同',
        }
