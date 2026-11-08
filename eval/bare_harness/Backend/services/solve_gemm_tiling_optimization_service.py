"""
GEMM Tiling 优化求解服务
功能：根据硬件参数和矩阵乘维度，求解最优的 tiling 方案
"""
import logging
import os
from typing import Any, Dict, List, Optional

import yaml

# gemm 模块内的 imports 使用相对导入，需先将 gemm 目录加入 path
import sys
from pathlib import Path

_gemm_dir = str(Path(__file__).parent / 'gemm')
if _gemm_dir not in sys.path:
    sys.path.insert(0, _gemm_dir)

from gemm_model_base import HardwareParam
from gemm_model_objs import evaluate_tiling_key
from gemm_existing_operator import GemmGts
from gemm_solver import solve_problem
from gemm_problem import Problem

logger = logging.getLogger(__name__)


class SolveGemmTilingOptimizationService:
    """GEMM Tiling 优化求解服务"""

    def __init__(self):
        """初始化服务，加载硬件配置"""
        self._load_hardware_config()
        self._readme = self._load_readme()
        logger.info("SolveGemmTilingOptimizationService initialized")

    def _load_hardware_config(self) -> None:
        """从配置文件加载硬件参数"""
        config_path = os.path.join(
            os.path.dirname(__file__), 'gemm', 'hardware.yml'
        )
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            logger.info("Loaded hardware config from %s", config_path)
        except FileNotFoundError:
            logger.warning(
                "hardware.yml not found at %s, using fallback config", config_path
            )
            self.config = {}

    def _load_readme(self) -> str:
        readme_path = os.path.join(
            os.path.dirname(__file__), 'gemm', 'README.md'
        )
        try:
            with open(readme_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            logger.warning("gemm/README.md not found at %s", readme_path)
            return ''

    def execute(
        self,
        mkn: Optional[List[int]] = None,
        data_b: Optional[List[int]] = None,
        isnd_abc: Optional[List[bool]] = None,
        bias: Optional[bool] = None,
        reuse_l0l1: Optional[List[bool]] = None,
    ) -> Dict[str, Any]:
        """
        执行 GEMM tiling 优化求解

        Args:
            mkn: 矩阵乘维度 [M, K, N]，默认为 [8192, 4096, 2048]
            data_b: 输入输出数据位宽(字节)，默认 [2, 4]
            isnd_abc: ABC矩阵是否为ND格式，默认 [True, False, True]
            bias: 是否需要加bias，默认 False
            reuse_l0l1: L0和L1是否数据复用，默认 [True, True]

        Returns:
            dict: 包含求解结果的字典
        """
        # 参数校验
        if mkn is not None:
            if not isinstance(mkn, (list, tuple)) or len(mkn) != 3:
                return {
                    'success': False,
                    'message': 'mkn 必须是包含3个元素的列表 [M, K, N]',
                }
            if not all(isinstance(x, (int, float)) and x > 0 for x in mkn):
                return {
                    'success': False,
                    'message': 'mkn 中的元素必须是正数',
                }

        try:
            # 构建 Problem 对象
            problem = self._build_problem(
                mkn=mkn or [8192, 4096, 2048],
                data_b=data_b or [2, 4],
                isnd_abc=isnd_abc or [True, False, True],
                bias=bias if bias is not None else False,
                reuse_l0l1=reuse_l0l1 or [True, True],
            )

            # 执行求解
            tiling_key, model_performance, solver_status = solve_problem(
                problem, solver=GemmGts, time_limit=120
            )

            # 用原始硬件参数对求解结果做精确性能评估
            predict_performance = None
            if tiling_key is not None:
                predict_performance = evaluate_tiling_key(problem.hardware_param_orig, tiling_key)

            # 格式化输出
            result = self._format_result(
                tiling_key, model_performance, solver_status, predict_performance
            )
            perf_obj = float(model_performance.obj) if hasattr(model_performance, 'obj') else 0.0
            logger.info(
                "Tiling optimization completed: status=%s, performance=%.4f",
                solver_status,
                perf_obj,
            )
            return result

        except FileNotFoundError as e:
            logger.error("Config file not found: %s", e)
            return {
                'success': False,
                'message': f'配置文件未找到: {e}',
            }
        except Exception as e:
            logger.error("Tiling optimization failed: %s", str(e), exc_info=True)
            return {
                'success': False,
                'message': f'求解过程异常: {str(e)}',
            }

    def _build_problem(
        self,
        mkn: List[int],
        data_b: List[int],
        isnd_abc: List[bool],
        bias: bool,
        reuse_l0l1: List[bool],
    ) -> Problem:
        """构建 GEMM 求解问题对象"""
        param_dict = self.config.get('910B3_modify', {})
        hardware_param = HardwareParam(param_dict)

        problem = Problem(hardware_param=hardware_param, MKN_orig=mkn)
        problem.dataB = data_b
        problem.isnd_abc = isnd_abc
        problem.bias = bias
        problem.reuse_l0l1 = reuse_l0l1
        problem.ub_l1_num = 4  # 910B3上L1缓存块多于4个时MTE2时间变长

        # 固定 L1 只缓存 K 维；先切分核数据再核内tiling；buffer/loop_order 全部作为决策变量
        problem.fixed_solution.m1k1n1_tilde = [1, None, 1]
        problem.fixed_solution.core_loop = 1

        return problem

    def _format_result(
        self,
        tiling_key,
        model_performance,
        solver_status,
        predict_performance=None,
    ) -> Dict[str, Any]:
        """格式化求解结果为统一返回格式"""
        # 转换 solver_status 为字符串
        status_str = str(solver_status).split('.')[-1].strip("'")
        is_optimal = status_str.lower() == 'optimal'

        # 提取 model_performance 中的目标值
        perf_value = None
        if hasattr(model_performance, 'obj'):
            perf_value = float(model_performance.obj)

        # 提取 tiling 参数（附字段说明）
        _tiling_key_descriptions = {
            'mckcnc_num': '每个维度上切分的核数（mDim, kDim, nDim）',
            'mckcnc': '单核单次迭代的数据的尺寸',
            'm0k0n0': 'L0 tiling size（baseM, baseK, baseN）',
            'm1k1n1_tilde': 'L1缓存块个数（第[1]个元素对应l1CopyBufferNum）',
            'm2n2_tilde': 'L2 swizzle count, L2内存不够提高L2命中率',
            'orders_l0l1l2': 'L0, L1, L2的遍历顺序，例："KNM" 表示先K再N再M',
            'buffer_l0abc': 'L0的数据复用数(CacheCnt)和double buffer(BufferNum)',
            'buffer_l1ab': 'L1的数据复用数(CacheCnt)和double buffer(BufferNum)',
            'reuse_l0l1': 'L0和L1是否数据复用',
            'core_num': '核数',
            'core_loop': '单核的迭代次数，1表示先切分核数据再核内tiling',
        }
        tiling_params = {}
        if hasattr(tiling_key, '__dict__'):
            for k, v in tiling_key.__dict__.items():
                if k.startswith('_'):
                    continue
                entry = {'value': v.tolist() if hasattr(v, 'tolist') else v}
                if k in _tiling_key_descriptions:
                    entry['description'] = _tiling_key_descriptions[k]
                tiling_params[k] = entry

        # 精确性能评估（evaluate_tiling_key 结果）
        predict_perf_detail = {}
        if predict_performance is not None:
            unit_time = predict_performance.unit_time
            predict_perf_detail = {
                'inner_us': dict(unit_time.inner_us),
                'total_us': dict(unit_time.total_us),
                'data': unit_time.data.to_dict(),
            }

        # GTS wind_matmul_v2 白名单参数对应表
        white_list_params = {}
        if tiling_key is not None:
            try:
                white_list_params = {
                    'baseM': int(tiling_key.m0k0n0[0]),
                    'baseK': int(tiling_key.m0k0n0[1]),
                    'baseN': int(tiling_key.m0k0n0[2]),
                    'mDim': int(tiling_key.mckcnc_num[0]),
                    'nDim': int(tiling_key.mckcnc_num[2]),
                    'l1CopyBufferNum': int(tiling_key.m1k1n1_tilde[1]),
                }
            except (TypeError, IndexError):
                white_list_params = {}

        return {
            'success': is_optimal,
            'message': f'求解完成，状态: {status_str}',
            'solver_status': status_str,
            'model_performance': perf_value,
            'predict_performance_detail': predict_perf_detail,
            'tiling_params': tiling_params,
            'white_list_params': white_list_params,
            'readme': self._readme,
        }