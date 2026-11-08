# -*- coding: utf-8 -*-
# linziwei l00624134
import copy
import numpy as np

from gemm_mip import GemmMipBottleneck
from gemm_problem import Problem
import pyomo.environ as pyo


def modify_problem(problem:Problem):
    problem.reuse = [False, True]
    problem.lb['buffer_l1ab'] = [0, 0]
    problem.fixed_solution.m1k1n1_tilde[0] = 1  # L1只存K维
    problem.fixed_solution.m1k1n1_tilde[2] = 1  # L1只存K维
    problem.fixed_solution.buffer_l0abc = [2, 2, 1]
    problem.fixed_solution.buffer_l1ab[1] = 0
    problem.fixed_solution.loop_order = ['KNM', None, None]  # 固定计算顺序
    if problem.MKN_orig[0] > problem.hardware_param.core_num * 128 and problem.MKN_orig[2] > 256:
        problem.fixed_solution.core_loop = 1
    return problem


class GemmGts(GemmMipBottleneck):
    def __init__(self, problem: Problem):
        super().__init__(problem)
        self.problem = modify_problem(problem)

    def _add_constraints(self):
        super()._add_constraints()
        model, problem = self.model, self.problem
        ub = problem.ub
        m1_num, _, n1_num = [model.m1k1n1_num[i] for i in range(3)]
        M, K, N = problem.MKN

        model.m1n1_num_total = pyo.Var(pyo.RangeSet(0, 1), within=pyo.NonNegativeIntegers, bounds=lambda m, i: (0, [M, N][i]))
        self._add_equal('m1n1_num_total', 0, 'm1k1n1_num', 0, 'mckcnc_num', 0)
        self._add_equal('m1n1_num_total', 1, 'm1k1n1_num', 2, 'mckcnc_num', 2)

        # 计算顺序
        model.constr_order.add(expr=model.m1n1_num_total[0] >= model.m1n1_num_total[1] - (1 - model.order_l2[0]) * N * 2)  # swizzle_0
        model.constr_order.add(expr=model.m1n1_num_total[1] >= model.m1n1_num_total[0] + 1 - (1 - model.order_l2[1]) * M * 2)  # swizzle_1

        # 先切后分或先分后切，无混合模式
        model.I_cut_tile = pyo.Var(within=pyo.Binary)
        model.constr_cover.add(expr=model.core_loop <= 1 + ub['core_loop'] * (1 - model.I_cut_tile))
        model.constr_cover.add(expr=m1_num <= 1 + ub['m1k1n1_num'][0] * model.I_cut_tile)
        model.constr_cover.add(expr=n1_num <= 1 + ub['m1k1n1_num'][2] * model.I_cut_tile)
        # M维切分过少或者N维一块时tiling无尾块
        if problem.MKN_orig[0] <= problem.hardware_param.core_num * 128 or problem.MKN_orig[2] <= 256:
            model.constr_cover.add(expr=model.m1k1n1[0] * model.m1k1n1_num[0] == model.mckcnc[0])
            model.constr_cover.add(expr=model.m1k1n1[2] * model.m1k1n1_num[2] == model.mckcnc[2])
        else:
            model.constr_cover.add(expr=model.core_loop == 1)
        # l2
        if hasattr(model, 'order_gm'):
            model.constr_order.add(expr=model.order_gm[0] == model.order_l2[0])
            model.constr_order.add(expr=model.m2n2_tilde[0] <= 1 + (1 - model.order_l2[0]) * ub['m2n2_tilde'][0])
            model.constr_order.add(expr=model.order_gm[1] == model.order_l2[1])
            model.constr_order.add(expr=model.m2n2_tilde[1] <= 1 + (1 - model.order_l2[1]) * ub['m2n2_tilde'][1])

    def _add_constraints_cachel1(self):
        model, problem = self.model, self.problem
        if not hasattr(model, 'constr_cache'):
            model.constr_cache = pyo.ConstraintList()
        scale_a, scale_b = problem.l1l2ab_scale
        if hasattr(model, 'buffer_l1ab'):
            bfl1a, bfl1b = model.buffer_l1ab[0], model.buffer_l1ab[1]
        else:
            bfl1a, bfl1b = problem.fixed_solution.buffer_l1ab
        model.constr_cache.add(expr=bfl1a <= model.a1b1c1_num[0])
        model.constr_cache.add(expr=bfl1b <= model.a1b1c1_num[1])
        m1_num = model.m1k1n1_num[0]
        mc_num = model.mckcnc_num[0]
        model.Im1_num = pyo.Var(within=pyo.Binary)  # bfl1a > 0 时 m1_num * mc_num == 1
        model.constr_cache.add(expr=m1_num <= 1 + problem.ub['m1k1n1_num'][0] * (1 - model.Im1_num))
        model.constr_cache.add(expr=mc_num <= 1 + problem.ub['m1k1n1_num'][0] * (1 - model.Im1_num))
        model.constr_cache.add(expr=bfl1a <= problem.ub['buffer_l1ab'][0] * model.Im1_num)
        model.constr_cache.add(expr=bfl1a <= problem.ub['buffer_l1ab'][0] * model.Im1_num)
        model.constr_cache.add(expr=bfl1a <= model.m1k1n1_num[1])
        a1, b1, cache_l1 = model.a1b1c1[0], model.a1b1c1[1], problem.cache_l1
        if problem.bias:
            model.constr_cache.add(expr=scale_a * a1 * (bfl1a + 2) + scale_b * b1 * (bfl1b + 2) + scale_b * model.m1k1n1[2] <= cache_l1)
        else:
            model.constr_cache.add(expr=scale_a * a1 * (bfl1a + 2) + scale_b * b1 * (bfl1b + 2) <= cache_l1)

    def solve(self, time_limit=120):
        tiling_key, performance, solver_status = super().solve(time_limit)
        #tiling_key.buffer_l1ab[0] += 2
        #tiling_key.buffer_l1ab[1] += 2
        return tiling_key, performance, solver_status


def get_tiling_str(tiling_key):
    if tiling_key is None:
        return None
    m0, k0, n0 = tiling_key.m0k0n0
    mc_num, _, nc_num = tiling_key.mckcnc_num
    if mc_num * nc_num != tiling_key.core_num:
        min_cn = min(mc_num, nc_num, int(np.sqrt(tiling_key.core_num)))
        for i in range(min_cn, 0, -1):
            if tiling_key.core_num % i == 0:
                min_cn = i
                break
        other_cn = int(tiling_key.core_num // min_cn)
        mc_num, nc_num = (min_cn, other_cn) if mc_num <= nc_num else (other_cn, min_cn)
    _, k1_tilde, _ = tiling_key.m1k1n1_tilde
    tiling = [m0, k0, n0, mc_num, nc_num, k1_tilde]
    tiling = ', '.join(list(map(str, tiling)))
    return tiling


def get_gts_tiling_key(problem: Problem, tiling_str: str):
    problem.initialize()
    problem = modify_problem(problem)
    tk = problem.get_fix_solution()
    m0, k0, n0, mc_num, nc_num, k1_tilde = list(map(int, tiling_str.split(',')))
    tk.m0k0n0 = [m0, k0, n0]
    tk.m1k1n1_tilde[1] = k1_tilde
    tk.core_num = mc_num * nc_num
    # 获取实现方式
    if True:
        tk.core_loop = 1
        tk.mckcnc_num = [mc_num, 1, nc_num]
        #tk.mckcnc = [np.ceil(tk.MKN[0] / mc_num / m0) * m0, tk.MKN[1], np.ceil(tk.MKN[2] / nc_num / n0) * n0]
    else:
        tk.mckcnc[0] = tk.m0k0n0[0]
        tk.mckcnc[2] = tk.m0k0n0[2]
    tk.update()
    # 获取顺序
    tk.loop_order[0] = 'KNM'
    if tk.m1k1n1_num[0] * tk.mckcnc_num[0] >= tk.m1k1n1_num[2] * tk.mckcnc_num[2]:
        tk.loop_order[1] = 'KNM'
        tk.loop_order[2] = 'NM'
    else:
        tk.loop_order[1] = 'KMN'
        tk.loop_order[2] = 'MN'
    # 获取buffer
    tk.buffer_l1ab = [0, 0]
    if tk.m1k1n1_num[0] * tk.mckcnc_num[0] == 1:
        l1 = problem.hardware_param.cacheB_l1 / tk.dataB[0] - tk.m1k1n1[1] * tk.m1k1n1[2] * 2
        tk.buffer_l1ab[0] = l1 / tk.m1k1n1[0] / tk.m1k1n1[1] - 2
        tk.buffer_l1ab[0] = max(tk.buffer_l1ab[0], 0)
        tk.buffer_l1ab[0] = min(tk.buffer_l1ab[0], tk.m1k1n1_num[0])
    #tk.buffer_l1ab[0] += 2
    #tk.buffer_l1ab[1] += 2
    # 获取m2
    if not problem.l2_sufficient():
        l2 = problem.hardware_param.cacheB_l2 / tk.dataB[0] / tk.MKN[1]
        if tk.loop_order[2] == 'NM':
            m2_tilde = 1
            n2_tilde = int((l2 - min(tk.m1k1n1[0] * tk.mckcnc_num[0], tk.MKN[0])) / tk.mckcnc_num[2] / tk.m1k1n1[2])
            n2_tilde = min(n2_tilde, tk.m1k1n1_num[2])
        else:
            n2_tilde = 1
            m2_tilde = int((l2 - min(tk.m1k1n1[2] * tk.mckcnc_num[2], tk.MKN[2])) / tk.mckcnc_num[0] / tk.m1k1n1[0])
            m2_tilde = min(m2_tilde, tk.m1k1n1_num[0])
        tk.m2n2_tilde = [m2_tilde, n2_tilde]
    tk.update()
    return tk