# -*- coding: utf-8 -*-
# linziwei l00624134
import numpy as np

from gemm_mip import GemmMipBottleneck
from gemm_problem import Problem
import pyomo.environ as pyo


def modify_problem(problem:Problem):
    problem.reuse_l0, problem.reuse_l1 = False, True
    problem.fixed_solution.buffer_num_l0abc = [2, 2, 1]
    problem.fixed_solution.reuse_l0 = False  # L0不数据复用
    problem.fixed_solution.reuse_num_l0ab = [0, 0]
    problem.fixed_solution.buffer_num_l1ab[1] = 2
    problem.fixed_solution.reuse_l1 = True
    problem.fixed_solution.reuse_num_l1ab[1] = 0  # L1B无cache
    problem.fixed_solution.m1k1n1_tilde[0] = 1  # L1只存K维
    problem.fixed_solution.m1k1n1_tilde[2] = 1  # L1只存K维
    problem.fixed_solution.orders_l0l1l2 = ['KNM', None, None]  # 固定计算顺序
    return problem

def mckcnc_align(problem):
    return problem.MKN_orig[0] <= problem.hardware_param.core_num * 128 or problem.MKN_orig[2] <= 256


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
        model.constr_order.add(expr=model.m1n1_num_total[0] >= model.m1n1_num_total[1] - (1 - model.order_l1[0]) * N * 2)  # swizzle_0
        model.constr_order.add(expr=model.m1n1_num_total[1] >= model.m1n1_num_total[0] + 1 - (1 - model.order_l1[1]) * M * 2)  # swizzle_1

        # M维切分过少或者N维一块时tiling无尾块
        if mckcnc_align(self.problem):
            model.constr_cover.add(expr=model.m1k1n1[0] * model.m1k1n1_num[0] == model.mckcnc[0])
            model.constr_cover.add(expr=model.m1k1n1[2] * model.m1k1n1_num[2] == model.mckcnc[2])
        else:
            model.constr_cover.add(expr=model.core_loop == 1)
        # l2
        if hasattr(model, 'order_l2'):  # 当前算子没有为提高L2命中率的swizzle
            model.constr_order.add(expr=model.order_l2[0] == model.order_l1[0])
            model.constr_order.add(expr=model.order_l2[1] == model.order_l1[1])
            model.m2n2_full = pyo.Var(pyo.RangeSet(0, 1), within=pyo.Binary)
            model.constr_order.add(expr=model.m2n2_num[0] <= 1 + (1 - model.m2n2_full[0]) * ub['m2n2_num'][0])
            model.constr_order.add(expr=model.m2n2_num[1] <= 1 + (1 - model.m2n2_full[1]) * ub['m2n2_num'][1])
            model.constr_order.add(expr=model.m2n2_tilde[0] <= 1 + (1 - model.order_l1[0] + model.m2n2_full[1]) * ub['m2n2_tilde'][0])
            model.constr_order.add(expr=model.m2n2_tilde[1] <= 1 + (1 - model.order_l1[1] + model.m2n2_full[0]) * ub['m2n2_tilde'][1])

    def _add_constraints_reuse(self):
        model, problem = self.model, self.problem
        if hasattr(model, 'reuse_num_l1ab'):
            reuse_a, reuse_b = model.reuse_num_l1ab[0], model.reuse_num_l1ab[1]
            if hasattr(model, 'buffer_num_l1ab'):
                bfl1a, bfl1b = model.buffer_num_l1ab[0], model.buffer_num_l1ab[1]
            else:
                bfl1a, bfl1b = problem.fixed_solution.buffer_num_l1ab
            model.constr_cache.add(expr=reuse_a <= model.a1b1c1_num[0])
            model.constr_cache.add(expr=reuse_b <= model.a1b1c1_num[1])
            model.constr_cache.add(expr=reuse_a + 2 <= bfl1a)
            # model.constr_cache.add(expr=reuse_a <= bfl1a)
            model.constr_cache.add(expr=reuse_b + 2 <= bfl1b)
            m1_num = model.m1k1n1_num[0]
            mc_num = model.mckcnc_num[0]
            model.Im1_num = pyo.Var(within=pyo.Binary)  # model.reuse_num_l1ab[0] > 0 时 m1_num * mc_num == 1
            model.constr_cache.add(expr=m1_num <= 1 + problem.ub['m1k1n1_num'][0] * (1 - model.Im1_num))
            model.constr_cache.add(expr=mc_num <= 1 + problem.ub['m1k1n1_num'][0] * (1 - model.Im1_num))
            model.constr_cache.add(expr=reuse_a <= problem.ub['reuse_num_l1ab'][0] * model.Im1_num)
            model.constr_cache.add(expr=reuse_a <= problem.ub['reuse_num_l1ab'][0] * model.Im1_num)
            model.constr_cache.add(expr=reuse_a <= model.m1k1n1_num[1])


def get_gts_tiling_key(problem: Problem, tiling_str: str, operator_type: int):
    problem.initialize()
    problem = modify_problem(problem)
    tk = problem.get_fix_solution()
    m0, k0, n0, mc_num, nc_num, k1_tilde = list(map(int, tiling_str.split(',')))
    tk.m0k0n0 = [m0, k0, n0]
    tk.m1k1n1_tilde[1] = k1_tilde
    tk.core_num = mc_num * nc_num
    # 获取实现方式
    if operator_type == 0: #'cut_first':
        tk.core_loop = 1
        tk.mckcnc_num = [mc_num, 1, nc_num]
    elif operator_type == 1: #'tiling_first':
        tk.mckcnc[0] = tk.m0k0n0[0]
        tk.mckcnc[2] = tk.m0k0n0[2]
    else:
        return False
    tk.update()
    if mckcnc_align(problem):
        tk.mckcnc = (np.ceil(tk.mckcnc / tk.m0k0n0) * tk.m0k0n0).astype(int)
    # 获取顺序
    tk.orders_l0l1l2[0] = 'KNM'
    if tk.m1k1n1_num[0] * tk.mckcnc_num[0] >= tk.m1k1n1_num[2] * tk.mckcnc_num[2]:
        tk.orders_l0l1l2[1] = 'KNM'
        tk.orders_l0l1l2[2] = 'NM'
    else:
        tk.orders_l0l1l2[1] = 'KMN'
        tk.orders_l0l1l2[2] = 'MN'
    # 获取buffer
    tk.reuse_num_l1ab = [0, 0]
    if tk.m1k1n1_num[0] * tk.mckcnc_num[0] == 1:
        l1 = problem.hardware_param.cacheB_l1 / tk.dataB[0] - tk.m1k1n1[1] * tk.m1k1n1[2] * 2
        tk.reuse_num_l1ab[0] = int(l1 / tk.m1k1n1[0] / tk.m1k1n1[1] - 2)
        tk.reuse_num_l1ab[0] = max(tk.reuse_num_l1ab[0], 0)
        tk.reuse_num_l1ab[0] = int(min(tk.reuse_num_l1ab[0], tk.m1k1n1_num[0] * tk.m1k1n1_num[1]))
    tk.buffer_num_l1ab = np.array(tk.reuse_num_l1ab) + 2
    # 获取m2
    if not problem.l2_sufficient():
        m1k1n1_total = np.minimum(tk.m1k1n1 * tk.mckcnc_num, tk.MKN)
        l2 = problem.hardware_param.cacheB_l2 / tk.dataB[0] / tk.MKN[1]
        if tk.orders_l0l1l2[2] == 'NM':
            n2_tilde = int((l2 - min(m1k1n1_total[0], tk.MKN[0])) / tk.mckcnc_num[2] / tk.m1k1n1[2])
            n2_tilde = min(n2_tilde, tk.m1k1n1_num[2])
            m2_tilde = int((l2 - min(m1k1n1_total[2] * n2_tilde, tk.MKN[2])) / tk.mckcnc_num[0] / tk.m1k1n1[0])
            m2_tilde = min(m2_tilde, tk.m1k1n1_num[0])
        else:
            m2_tilde = int((l2 - min(m1k1n1_total[2], tk.MKN[2])) / tk.mckcnc_num[0] / tk.m1k1n1[0])
            m2_tilde = min(m2_tilde, tk.m1k1n1_num[0])
            n2_tilde = int((l2 - min(m1k1n1_total[0] * m2_tilde, tk.MKN[0])) / tk.mckcnc_num[2] / tk.m1k1n1[2])
            n2_tilde = min(n2_tilde, tk.m1k1n1_num[2])
        tk.m2n2_tilde = [m2_tilde, n2_tilde]
    tk.update()
    return tk
