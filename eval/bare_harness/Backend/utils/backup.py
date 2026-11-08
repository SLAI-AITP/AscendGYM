"""
算子代码备份与版本管理工具

全量快照机制（op_backup）：
  op-path-version/op_backup/
  ├── WindMatmulV2_version1/
  ├── WindMatmulV2_version2/
  └── Cat_version1/

每次 profiling 成功后将算子整个文件夹拷贝到 op_backup，每个算子独立维护版本号。
"""
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

_OP_BACKUP_DIR = "op_backup"
_BEST_FILE = "best.txt"


# ──────────────────────────────────────────────────────────────────────────────
# 内部辅助
# ──────────────────────────────────────────────────────────────────────────────

def _find_custom_dir(op_dir: Path) -> Optional[Path]:
    """在算子目录下找到算子代码子目录（即同时包含 op_host 和 op_kernel 的那个）。

    新布局下子目录名严格等于 to_snake_case_compact(op_name)，不再带 _custom 后缀。
    保留函数名以避免现有 import 雪崩；语义改为按"含 op_host"判据来定位，
    对子目录命名零依赖。
    """
    if not op_dir.is_dir():
        return None
    for d in op_dir.iterdir():
        if d.is_dir() and (d / "op_host").is_dir():
            return d
    return None


def _read_op_backup_version(op_name: str, op_path_version: str) -> int:
    """读取算子在 op_backup 中的当前版本号（不递增）。"""
    ver_file = Path(op_path_version) / _OP_BACKUP_DIR / f".{op_name}_version.txt"
    if ver_file.exists():
        try:
            return int(ver_file.read_text(encoding="utf-8").strip())
        except (ValueError, OSError):
            pass
    return 0


# ──────────────────────────────────────────────────────────────────────────────
# 公共 API
# ──────────────────────────────────────────────────────────────────────────────

def ensure_op_in_version(op_name: str, op_path: str, op_path_version: str) -> Dict:
    """
    确保算子已初始化到 op-path-version 目录。

    首次调用时将 op-path/<op_name>/ 整体拷贝到 op-path-version/<op_name>/。
    若目标已存在则直接返回。

    Returns:
        {"initialized": bool, "op_root": str}
    """
    src_dir = Path(op_path) / op_name
    dst_dir = Path(op_path_version) / op_name

    if dst_dir.exists():
        return {"initialized": False, "op_root": str(dst_dir)}

    if not src_dir.exists():
        raise FileNotFoundError(f"源算子目录不存在: {src_dir}")

    shutil.copytree(str(src_dir), str(dst_dir))
    logger.info("ensure_op_in_version: copied %s -> %s", src_dir, dst_dir)
    return {"initialized": True, "op_root": str(dst_dir)}


def snapshot_to_op_backup(op_name: str, op_path_version: str) -> dict:
    """
    将算子整个文件夹快照到 op-path-version/op_backup/ 下，每个算子独立维护版本号。

    Returns:
        {"version": int, "snapshot_dir": str}
    """
    op_backup_base = Path(op_path_version) / _OP_BACKUP_DIR
    op_backup_base.mkdir(parents=True, exist_ok=True)

    current = _read_op_backup_version(op_name, op_path_version)
    next_ver = current + 1
    ver_file = op_backup_base / f".{op_name}_version.txt"
    ver_file.write_text(str(next_ver), encoding="utf-8")

    snapshot_name = f"{op_name}_version{next_ver}"
    snapshot_dir = op_backup_base / snapshot_name

    src_dir = Path(op_path_version) / op_name
    if not src_dir.exists():
        logger.warning("snapshot_to_op_backup: source not found %s", src_dir)
        return {"version": next_ver, "snapshot_dir": str(snapshot_dir), "error": f"source not found: {src_dir}"}

    if snapshot_dir.exists():
        shutil.rmtree(snapshot_dir)
    shutil.copytree(str(src_dir), str(snapshot_dir))
    logger.info("snapshot_to_op_backup: copied %s -> %s", src_dir, snapshot_dir)
    return {"version": next_ver, "snapshot_dir": str(snapshot_dir)}


def update_best_snapshot(
    op_name: str,
    op_path_version: str,
    exec_time_us: float,
    snapshot_version: int,
    min_improvement_epsilon: float = 0.0,
) -> dict:
    """
    更新 best.txt，记录最优性能对应的快照版本。
    首次直接写入，后续仅在性能"显著更优"时覆盖。

    显著更优的判定：exec_time_us <= prev_best * (1 - min_improvement_epsilon)。
    eps=0 时退化为严格 < prev_best（保留旧行为兼容性）。eps>0 时形成与
    Client 侧 orchestrator.min_improvement_epsilon 一致的噪音容差——避免
    一个抖动 0.5% 的伪 best 被 LLM 通过 get_op_version("best") 当作真 best
    回滚到。

    best.txt 内容格式：JSON 单行 {"version": N, "exec_time_us": float}

    Args:
        op_name: 算子名。
        op_path_version: version 根目录（写入 op_backup/.{op}_best.txt）。
        exec_time_us: 本次 profiling 的 exec_time_us。
        snapshot_version: 本次 snapshot_to_op_backup 返回的版本号。
        min_improvement_epsilon: 接受为新 best 所需的相对提升（默认 0.0）。

    Returns:
        {"updated": bool, "best_version": int, "best_exec_time_us": float, "best_snapshot_dir": str}
    """
    op_backup_base = Path(op_path_version) / _OP_BACKUP_DIR
    op_backup_base.mkdir(parents=True, exist_ok=True)
    best_file = op_backup_base / f".{op_name}_best.txt"

    prev_best = float("inf")
    prev_version = 0
    if best_file.exists():
        try:
            meta = json.loads(best_file.read_text(encoding="utf-8"))
            prev_best = float(meta.get("exec_time_us", float("inf")))
            prev_version = int(meta.get("version", 0))
        except (ValueError, json.JSONDecodeError):
            pass

    # eps 截断保护：负值无意义，按 0 处理（即原 >= 严格判定）。
    eps = max(0.0, float(min_improvement_epsilon))
    # 判定逻辑：
    #   - eps == 0：保持原行为 —— exec_time_us >= prev_best 不接受。
    #   - eps  > 0：必须 ≤ prev_best * (1 - eps) 才接受，否则视为噪音区拒绝。
    # 首次（prev_version==0）跳过判定直接写入。
    updated = True
    if prev_version > 0:
        if eps == 0.0:
            reject = exec_time_us >= prev_best
            reason_msg = "current %.2f us >= best %.2f us (v%d), no update"
            reason_args = (exec_time_us, prev_best, prev_version)
        else:
            threshold = prev_best * (1.0 - eps)
            reject = exec_time_us > threshold
            reason_msg = (
                "current %.2f us > threshold %.2f us (= best %.2f us * "
                "(1-%.4f), v%d), no update"
            )
            reason_args = (exec_time_us, threshold, prev_best, eps, prev_version)
        if reject:
            updated = False
            logger.info("update_best_snapshot: " + reason_msg, *reason_args)

    if updated:
        best_file.write_text(
            json.dumps({"version": snapshot_version, "exec_time_us": exec_time_us}, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info(
            "update_best_snapshot: new best %.2f us at snapshot v%d "
            "(prev %.2f us at v%d, eps=%.4f)",
            exec_time_us, snapshot_version, prev_best, prev_version, eps,
        )

    best_version = snapshot_version if updated else prev_version
    best_exec = exec_time_us if updated else prev_best
    best_snapshot_dir = str(op_backup_base / f"{op_name}_version{best_version}")
    return {
        "updated": updated,
        "best_version": best_version,
        "best_exec_time_us": best_exec,
        "best_snapshot_dir": best_snapshot_dir,
    }


def get_best_snapshot(op_name: str, op_path_version: str) -> dict:
    """
    读取 best.txt，返回最优快照信息。

    Returns:
        {"version": int, "exec_time_us": float, "snapshot_dir": str} 或空 dict（无记录时）
    """
    best_file = Path(op_path_version) / _OP_BACKUP_DIR / f".{op_name}_best.txt"
    if not best_file.exists():
        return {}
    try:
        meta = json.loads(best_file.read_text(encoding="utf-8"))
        version = int(meta.get("version", 0))
        return {
            "version": version,
            "exec_time_us": float(meta.get("exec_time_us", 0)),
            "snapshot_dir": str(Path(op_path_version) / _OP_BACKUP_DIR / f"{op_name}_version{version}"),
        }
    except (ValueError, json.JSONDecodeError):
        return {}


def get_op_backup_version_info(op_name: str, op_path_version: str) -> dict:
    """
    从 op_backup 目录读取算子的快照版本信息。

    Returns:
        {
            "current_version": int,         # 最新版本号（0 = 尚无快照）
            "latest_snapshot": str,         # 最新快照目录路径
            "best_version": int,            # 最优版本号（0 = 尚无）
            "best_exec_time_us": float,     # 最优执行时间（None = 尚无）
            "best_snapshot_dir": str,       # 最优快照目录路径
            "op_backup_dir": str,           # op_backup 基目录
        }
    """
    op_backup_base = Path(op_path_version) / _OP_BACKUP_DIR
    current = _read_op_backup_version(op_name, op_path_version)
    latest_snapshot = str(op_backup_base / f"{op_name}_version{current}") if current > 0 else ""

    best = get_best_snapshot(op_name, op_path_version)
    return {
        "current_version": current,
        "latest_snapshot": latest_snapshot,
        "best_version": best.get("version", 0),
        "best_exec_time_us": best.get("exec_time_us"),
        "best_snapshot_dir": best.get("snapshot_dir", ""),
        "op_backup_dir": str(op_backup_base),
    }
