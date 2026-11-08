# -*- coding: utf-8 -*-
# linziwei l00624134
import copy
from enum import Enum
from itertools import chain

import numpy as np
import pyomo.environ as pyo
from typing import Optional, Tuple
from loguru import logger
from pyomo.opt import SolverFactory

from gemm_model_base import TilingPerformance, TilingKey
from gemm_problem import Problem


class GemmMipBase:
    def __init__(self, problem: Problem):
        self.problem: Problem = problem
        self.problem.initialize()
        self.print_process = False
        self.model_name = '_'.join(map(str, map(int, self.problem.MKN_orig)))
        # 用于存储模型对象
        self.model: Optional[pyo.Model] = None
        self.solver: Optional[SolverFactory] = None

    """决策变量"""
    def _add_var(self, var_name, num, var_type):
        lb, ub = self.problem.lb[var_name], self.problem.ub[var_name]
        if num > 1:
            var = pyo.Var(pyo.RangeSet(0, num - 1), within=var_type, bounds=lambda m, i: (lb[i], ub[i]))
        else:
            var = pyo.Var(within=var_type, bounds=(lb, ub))
        setattr(self.model, var_name, var)

    def _add_variables(self):
        for var_name, num in self.problem.var_names_integer.items():
            self._add_var(var_name, num, pyo.NonNegativeIntegers)
        for var_name, num in self.problem.var_names_continuous.items():
            self._add_var(var_name, num, pyo.NonNegativeReals)
        model = self.model
        model.constr_order = pyo.ConstraintList()
        if self.problem.reuse_l0:
            model.order_l0 = pyo.Var(pyo.RangeSet(0, 5), within=pyo.Binary)
            model.constr_order.add(expr=pyo.summation(model.order_l0) == 1)
        if self.problem.reuse_l1:
            model.order_l1 = pyo.Var(pyo.RangeSet(0, 1), within=pyo.Binary)
            model.constr_order.add(expr=pyo.summation(model.order_l1) == 1)
        model.Ik = pyo.Var(within=pyo.Binary)
        k1_num = model.m1k1n1_num[1]
        model.constr_order.add(expr=k1_num + model.Ik >= 2)
        model.constr_order.add(expr=k1_num - 1 <= (1 - model.Ik) * self.problem.ub['m1k1n1_num'][1])

    def _get_var(self, var_name, idx=None):
        if type(var_name) is str:
            return getattr(self.model, var_name) if idx is None else getattr(self.model, var_name)[idx]
        else:
            return var_name if idx is None else var_name[idx]

    """约束条件"""
    def _add_constraints(self):
        self._add_constraints_shape()
        self._add_constraints_data_coverage()
        self._add_constraints_cachel0()
        self._add_constraints_cachel1()
        self._add_constraints_reuse()
        self._add_constraints_powers2()
        self._add_constraints_align()
        self._add_constraints_mte2_512B()
        self._add_constraints_fixpipe_512B()
        self._add_constraints_l2_not_sufficient()

    def _add_equal(self, z_name, z_id, x_name, x_id, y_name, y_id):
        # z = xy
        z, x, y = self._get_var(z_name, z_id), self._get_var(x_name, x_id), self._get_var(y_name, y_id)
        self.model.constr_shape.add(expr=z == x * y)

    def _add_constraints_shape(self):
        model = self.model
        model.constr_shape = pyo.ConstraintList()
        idxes = self.problem.len2area([0, 1, 2])
        for i in range(3):
            xid, yid = idxes[i][0], idxes[i][1]
            for abc, mkn in zip(['a0b0c0', 'a0b0c0_num', 'a1b1c1', 'a1b1c1_tilde', 'a1b1c1_num', 'acbccc', 'acbccc_num'],
                                ['m0k0n0', 'm0k0n0_num', 'm1k1n1', 'm1k1n1_tilde', 'm1k1n1_num', 'mckcnc', 'mckcnc_num']):
                if hasattr(model, abc):
                    self._add_equal(abc, i, mkn, xid, mkn, yid)
            self._add_equal('m1k1n1', i, 'm1k1n1_tilde', i, 'm0k0n0', i)
        for w, abc, mkn in zip(['w0', 'w0_num', 'w1_tilde', 'w1_num', 'w1', 'wc', 'wc_num'],
                               ['a0b0c0', 'a0b0c0_num', 'a1b1c1_tilde', 'a1b1c1_num', 'a1b1c1', 'acbccc', 'acbccc_num'],
                               ['m0k0n0', 'm0k0n0_num', 'm1k1n1_tilde', 'm1k1n1_num', 'm1k1n1', 'mckcnc', 'mckcnc_num']):
            if hasattr(model, w):
                self._add_equal(w, None, abc, 2, mkn, 1)

    def _add_ceil(self, z_name, z_id, x_name, x_id, y_name, y_id):
        # z = ceil(x / y)
        z, x, y = self._get_var(z_name, z_id), self._get_var(x_name, x_id), self._get_var(y_name, y_id)
        self.model.constr_cover.add(expr=z <= x)
        self.model.constr_cover.add(expr=y <= x)
        self.model.constr_cover.add(expr=z * y >= x)
        self.model.constr_cover.add(expr=z * y - y <= x - 1)

    def _add_constraints_data_coverage(self):
        model, problem = self.model, self.problem
        model.constr_cover = pyo.ConstraintList()
        for i in range(3):
            self._add_ceil('m0k0n0_num', i, 'mckcnc', i, 'm0k0n0', i)
            self._add_ceil('m1k1n1_num', i, 'm0k0n0_num', i, 'm1k1n1_tilde', i)
            self._add_ceil('mckcnc_num', i, self.problem.MKN, i, 'mckcnc', i)
        # loop
        #model.I_core_loop = pyo.Var(within=pyo.Binary)
        #model.constr_cover.add(expr=model.core_loop <= 1 + problem.ub['core_loop'] * (1 - model.I_core_loop))
        #model.constr_cover.add(expr=model.core_loop >= 2 - model.I_core_loop)
        cl = model.core_loop if problem.fixed_solution.core_loop is None else problem.fixed_solution.core_loop
        if problem.split_k_core:
            model.constr_cover.add(expr=model.core_num * cl >= model.wc_num)
            model.constr_cover.add(expr=model.core_num <= model.wc_num)
        else:
            model.constr_cover.add(expr=model.core_num * cl >= model.acbccc_num[2])
            model.constr_cover.add(expr=model.core_num <= model.acbccc_num[2])
        # 先切后分或先分后切，无混合模式
        model.I_cut_tile = pyo.Var(within=pyo.Binary)
        model.constr_cover.add(expr=model.core_loop <= 1 + problem.ub['core_loop'] * (1 - model.I_cut_tile))
        model.constr_cover.add(expr=model.m1k1n1_num[0] <= 1 + problem.ub['m1k1n1_num'][0] * model.I_cut_tile)
        model.constr_cover.add(expr=model.m1k1n1_num[2] <= 1 + problem.ub['m1k1n1_num'][2] * model.I_cut_tile)

    def _add_constraints_cachel0(self):
        model, problem = self.model, self.problem
        model.constr_cache = pyo.ConstraintList()
        a0, b0, c0 = [model.a0b0c0[i] for i in range(3)]
        l0a, l0b, l0c = [problem.cache_l0abc[i] for i in range(3)]
        # L0AB
        if hasattr(model, 'buffer_num_l0abc'):
            bfl0a, bfl0b = model.buffer_num_l0abc[0], model.buffer_num_l0abc[1]
        else:
            bfl0a, bfl0b = problem.fixed_solution.buffer_num_l0abc[:2]
        model.constr_cache.add(expr=bfl0a * a0 <= l0a)
        model.constr_cache.add(expr=bfl0b * b0 <= l0b)
        # L0C
        ub_c0, ub_c1 = l0c // problem.fixed_solution.buffer_num_l0abc[2], problem.ub['a1b1c1'][2]
        c1 = model.a1b1c1[2]
        model.constr_cache.add(expr=c0 <= ub_c0)
        model.constr_cache.add(expr=c1 <= ub_c0 + model.Ik * ub_c1)
        if problem.reuse_l0:
            order = model.order_l0
            m1_tilde, _, n1_tilde = [model.m1k1n1_tilde[i] for i in range(3)]
            model.constr_cache.add(expr=c0 * m1_tilde <= ub_c0 + (1 - order[3]) * ub_c1)
            model.constr_cache.add(expr=c0 * n1_tilde <= ub_c0 + (1 - order[2]) * ub_c1)
            model.constr_cache.add(expr=c1 <= ub_c0 + (1 - order[4] - order[5]) * ub_c1)

    def _add_constraints_cachel1(self):
        model, problem = self.model, self.problem
        scale_a, scale_b = problem.l1l2ab_scale
        if hasattr(model, 'buffer_num_l1ab'):
            bfl1a, bfl1b = model.buffer_num_l1ab[0], model.buffer_num_l1ab[1]
        else:
            bfl1a, bfl1b = problem.fixed_solution.buffer_num_l1ab
        a1, b1, cache_l1 = model.a1b1c1[0], model.a1b1c1[1], problem.cache_l1
        if problem.bias:
            model.constr_cache.add(expr=scale_a * a1 * bfl1a + scale_b * b1 * bfl1b + scale_b * model.m1k1n1[2] <= cache_l1)
        else:
            model.constr_cache.add(expr=scale_a * a1 * bfl1a + scale_b * b1 * bfl1b <= cache_l1)

    def _add_constraints_reuse(self):
        model, problem = self.model, self.problem
        for reuse, buffer, abc_num in zip(['reuse_num_l0ab', 'reuse_num_l1ab'],
                                      ['buffer_num_l0abc', 'buffer_num_l1ab'],
                                      [model.a0b0c0_num, model.a1b1c1_num]):
            if hasattr(model, reuse):
                reuse_a, reuse_b = getattr(model, reuse)[0], getattr(model, reuse)[1]
                if hasattr(model, buffer):
                    bfl1a, bfl1b = getattr(model, buffer)[0], getattr(model, buffer)[1]
                else:
                    sol = problem.fixed_solution
                    bfl1a, bfl1b = getattr(sol, buffer)[0], getattr(sol, buffer)[1]
                model.constr_cache.add(expr=reuse_a <= abc_num[0])
                model.constr_cache.add(expr=reuse_b <= abc_num[1])
                model.constr_cache.add(expr=reuse_a <= bfl1a)
                model.constr_cache.add(expr=reuse_b <= bfl1b)

    def _add_constraints_powers2(self):
        model = self.model
        m0k0n0_powers2_ub = [int(np.log2(self.problem.ub['m0k0n0'][i])) for i in range(3)]
        model.constr_powers2 = pyo.ConstraintList()
        var_names = ['m0_binary', 'k0_binary', 'n0_binary']
        model.Ipower2 = pyo.Var(pyo.RangeSet(0, 2), within=pyo.Binary)
        for i, flag in enumerate(self.problem.powers2_m0k0n0):
            if flag or i == 2:
                ub = m0k0n0_powers2_ub[i]
                var = pyo.Var(pyo.RangeSet(0, ub), within=pyo.Binary)
                setattr(model, var_names[i], var)
                model.constr_powers2.add(expr=model.m0k0n0[i] == sum([(2 ** j) * var[j] for j in range(ub + 1)]))
            if flag:
                model.constr_powers2.add(expr=sum([var[j] for j in range(ub + 1)]) == 1)
            if i == 2:
                model.constr_powers2.add(expr=sum([var[j] for j in range(ub + 1)]) <= 1 + (1 - model.Ipower2[i]) * ub)
                model.constr_powers2.add(expr=sum([var[j] for j in range(ub + 1)]) >= 2 - model.Ipower2[i])

    def _add_constraints_align(self):
        model, problem = self.model, self.problem
        for var_name in ['m0k0n0', 'm1k1n1']:
            align_name = 'align_' + var_name
            align = (getattr(problem, align_name) / problem.mkn_min).astype(int)
            if np.any(align > 1):
                lb = np.ceil(problem.lb[var_name] / align).astype(int)
                ub = np.floor(problem.ub[var_name] / align).astype(int)
                align_var = pyo.Var(pyo.RangeSet(0, 2), within=pyo.NonNegativeIntegers, bounds=lambda m, i: (lb[i], ub[i]))
                setattr(model, align_name, align_var)
                for i in range(3):
                    if align[i] > 1:
                        self._add_equal(var_name, i, align, i, align_name, i)

    def _add_constraints_fixpipe_512B(self):
        if not self.problem.isnd_abc[2]:
            return
        vint = pyo.NonNegativeIntegers
        model, problem = self.model, self.problem
        model.constr_fixpipe_512B = pyo.ConstraintList()
        constr = model.constr_fixpipe_512B
        scale_fixpipe = int(512 / problem.dataB[1] / problem.mkn_min[2])
        # n0补齐
        lb_n = np.ceil(problem.lb['m0k0n0'][2] / scale_fixpipe).astype(int)
        ub_n = np.ceil(problem.ub['m1k1n1'][2] / scale_fixpipe).astype(int)
        lb_nc = np.ceil(problem.lb['mckcnc'][2] / scale_fixpipe).astype(int)
        ub_nc = problem.ub['m0k0n0_num'][2] * ub_n
        model.nout_512B = pyo.Var(within=vint, bounds=(lb_n, ub_n))
        model.nout_last_512B = pyo.Var(within=vint, bounds=(lb_n, ub_n))
        model.nc_512B = pyo.Var(within=vint, bounds=(lb_nc, ub_nc))
        n0, n0_num = model.m0k0n0[2], model.m0k0n0_num[2]
        constr.add(expr=scale_fixpipe * model.nout_512B >= n0)
        constr.add(expr=scale_fixpipe * model.nout_last_512B >= model.mckcnc[2] - n0 * (n0_num - 1))
        constr.add(expr=model.nc_512B >= model.nout_512B * (n0_num - 1) + model.nout_last_512B)
        # C0, Cc补齐
        lb_c0 = problem.lb['a0b0c0'][2]
        ub_c0 = problem.ub['m0k0n0'][0] * ub_n * scale_fixpipe
        lb_cc = problem.lb['acbccc'][2]
        ub_cc = problem.ub['mckcnc'][0] * ub_nc * scale_fixpipe
        model.cout_512B = pyo.Var(within=vint, bounds=(lb_c0, ub_c0))
        model.cc_512B = pyo.Var(within=vint, bounds=(lb_cc, ub_cc))
        constr.add(expr=model.cout_512B == scale_fixpipe * model.nout_512B * model.m0k0n0[0])
        constr.add(expr=model.cc_512B == scale_fixpipe * model.nc_512B * model.mckcnc[0])

    def _add_constraints_mte2_512B(self):
        if not self.problem.isnd_abc[0] and not self.problem.isnd_abc[1]:
            return
        vint = pyo.NonNegativeIntegers
        model, problem = self.model, self.problem
        model.constr_mte2_512B = pyo.ConstraintList()
        constr = model.constr_mte2_512B
        scale_mte2 = int(512 / problem.dataB[0] / problem.mkn_min[1])
        # k1补齐
        lb_k1 = np.ceil(problem.lb['m1k1n1'][1] / scale_mte2).astype(int)
        ub_k1 = np.ceil(problem.ub['m1k1n1'][1] / scale_mte2).astype(int)
        lb_kc = np.ceil(problem.lb['mckcnc'][1] / scale_mte2).astype(int)
        ub_kc = problem.ub['m1k1n1_num'][1] * ub_k1
        model.k1_512B = pyo.Var(within=vint, bounds=(lb_k1, ub_k1))
        model.k1_last_512B = pyo.Var(within=vint, bounds=(lb_k1, ub_k1))
        model.kc_512B = pyo.Var(within=vint, bounds=(lb_kc, ub_kc))
        k1, k1_num = model.m1k1n1[1], model.m1k1n1_num[1]
        constr.add(expr=scale_mte2 * model.k1_512B >= k1)
        constr.add(expr=scale_mte2 * model.k1_last_512B >= model.mckcnc[1] - k1 * (k1_num - 1))
        constr.add(expr=model.kc_512B >= model.k1_512B * (k1_num - 1) + model.k1_last_512B)
        # A1B1, ACBC补齐
        lb_a1b1 = problem.lb['a1b1c1'][:2]
        ub_a1b1 = problem.ub['m1k1n1'][[0, 2]] * ub_k1 * scale_mte2
        lb_acbc = problem.lb['acbccc'][:2]
        ub_acbc = problem.ub['mckcnc'][[0, 2]] * ub_kc * scale_mte2
        model.a1b1_512B = pyo.Var(pyo.RangeSet(0, 1), within=vint, bounds=lambda m, i: (lb_a1b1[i], ub_a1b1[i]))
        model.acbc_512B = pyo.Var(pyo.RangeSet(0, 1), within=vint, bounds=lambda m, i: (lb_acbc[i], ub_acbc[i]))
        constr.add(expr=model.a1b1_512B[0] == scale_mte2 * model.k1_512B * model.m1k1n1[0])
        constr.add(expr=model.acbc_512B[0] == scale_mte2 * model.kc_512B * model.mckcnc[0])
        constr.add(expr=model.a1b1_512B[1] == scale_mte2 * model.k1_512B * model.m1k1n1[2])
        constr.add(expr=model.acbc_512B[1] == scale_mte2 * model.kc_512B * model.mckcnc[2])
        self.scale_mte2 = scale_mte2

    def _add_constraints_l2_not_sufficient(self):
        if self.problem.l2_sufficient():
            return
        model, problem = self.model, self.problem
        model.order_l2 = pyo.Var(pyo.RangeSet(0, 1), within=pyo.Binary)
        model.constr_order.add(expr=pyo.summation(model.order_l2) == 1)
        scale_a, scale_b = problem.l1l2ab_scale
        l2 = problem.cache_l2 // problem.MKN[1]
        # 先分核再切块，core_loop==1
        self._add_equal('m2n2', 0, 'm2n2_tilde', 0, 'm1k1n1', 0)
        self._add_equal('m2n2', 1, 'm2n2_tilde', 1, 'm1k1n1', 2)
        model.constr_cover.add(expr=model.m2n2_num[0] * model.m2n2_tilde[0] >= model.m1k1n1_num[0])
        model.constr_cover.add(expr=model.m2n2_num[1] * model.m2n2_tilde[1] >= model.m1k1n1_num[2])
        model.constr_cache.add(expr=scale_a * model.m2n2[0] * model.mckcnc_num[0] + scale_b * model.m2n2[1] * model.mckcnc_num[2] <= l2)

    """
    求解
    """
    def _fix_solution(self):
        def do_fix(var0, value0):
            if value0 is not None:
                var0.fix(value0)
        model, sol = self.model, self.fixed_solution
        if sol is None:
            return
        for var_name, num in self.problem.var_names_integer.items():
            if hasattr(sol, var_name) and getattr(sol, var_name) is not None:
                value, var = getattr(sol, var_name), getattr(model, var_name)
                if num == 1:
                    do_fix(var, value)
                else:
                    for i in range(num):
                        do_fix(var[i], value[i])
        if hasattr(sol, 'Ik') and sol.Ik is not None:
            do_fix(model.Ik, sol.Ik)
        if hasattr(sol, 'orders_l0l1l2') and sol.orders_l0l1l2 is not None:
            if sol.orders_l0l1l2[0] is not None and hasattr(model, 'order_l0'):
                idx = self.problem.orders_l0.index(sol.orders_l0l1l2[0])
                do_fix(model.order_l0[idx], 1)
            if sol.orders_l0l1l2[1] is not None and hasattr(model, 'order_l1'):
                idx = self.problem.orders_l1.index(sol.orders_l0l1l2[1])
                do_fix(model.order_l1[idx], 1)
            if sol.orders_l0l1l2[2] is not None and hasattr(model, 'order_l2'):
                idx = self.problem.orders_l2.index(sol.orders_l0l1l2[2])
                do_fix(model.order_l2[idx], 1)
        if hasattr(sol, 'core_loop') or self.problem.ub['core_loop'] == 1:
            cl = 1 if self.problem.ub['core_loop'] == 1 else sol.core_loop
            do_fix(model.core_loop, cl)

    def _get_value(self, var):
        value = pyo.value(var, exception=False)
        if value is not None:
            return np.round(value, 1).item()
        else:
            return 0

    def _get_solution_pyomo(self):
        model = self.model
        tiling_key = copy.deepcopy(self.fixed_solution)
        for var_name, num in chain(self.problem.var_names_integer.items(), self.problem.var_names_continuous.items()):
            if not hasattr(model, var_name):
                continue
            var = getattr(model, var_name)
            if num == 1:
                setattr(tiling_key, var_name, self._get_value(var))
            else:
                setattr(tiling_key, var_name, [self._get_value(var[i]) for i in range(num)])
        if hasattr(model, 'order_l0'):
            tiling_key.order_l0 = [self._get_value(model.order_l0[i]) for i in range(6)]
            tiling_key.orders_l0l1l2[0] = self.problem.orders_l0[np.argmax(tiling_key.order_l0)]
        if hasattr(model, 'order_l1'):
            tiling_key.order_l1 = [self._get_value(model.order_l1[i]) for i in range(2)]
            tiling_key.orders_l0l1l2[1] = self.problem.orders_l1[np.argmax(tiling_key.order_l1)]
        if hasattr(model, 'order_l2'):
            tiling_key.order_l2 = [self._get_value(model.order_l2[i]) for i in range(2)]
            tiling_key.orders_l0l1l2[2] = self.problem.orders_l2[np.argmax(tiling_key.order_l2)]
        # 性能
        performance = TilingPerformance()
        return tiling_key, performance

    def _add_objective(self):
        self.model.obj = pyo.Objective(expr=self.model.m0k0n0[0] * self.model.m0k0n0[1] * self.model.m0k0n0[2], sense=pyo.maximize)

    def _solve_pyomo(self, time_limit):
        self.solver = SolverFactory('scip')
        self.solver.options['limits/time'] = time_limit
        #self.solver.options['limits/gap'] = 0
        self.solver.options['display/verblevel'] = 4
        self.solver.options['display/freq'] = 100000
        #self.solver.options['numerics/numericsfocus'] = 1
        self.solver.options['presolving/maxrounds'] = 0 #if np.prod(self.problem.MKN) > 1e9 else 0  # 0:出错概率小， -1:出误差概率小
        self.solver.options['emphasis/numerics'] = True
        #self.solver.options['limits/bignumber'] = 1e9
        try:
            results = self.solver.solve(self.model, tee=self.print_process)
        except Exception:
            try:
                self.solver.options['presolving/maxrounds'] = - self.solver.options['presolving/maxrounds'] - 1
                results = self.solver.solve(self.model, tee=self.print_process)
            except Exception:
                #self.model.write(f'log/{self.model_name}.nl', io_options={"symbolic_solver_labels": True})
                logger.error(f'MKN = {self.problem.MKN_orig}, solver error')
                return None, None, None
        if results.solver.status != pyo.SolverStatus.ok:
            #self.model.write(f'log/{self.model_name}.nl', io_options = {"symbolic_solver_labels":True})
            logger.error(f'MKN = {self.problem.MKN_orig}, solver failed: {results.solver.status}')
            return None, None, None
        tiling_key, performance = self._get_solution_pyomo()
        performance.obj = self.problem.hardware_param.cycle2us(self._get_value(self.model.obj))
        return tiling_key, performance, results.solver.termination_condition

    def solve(self, time_limit=120) -> Tuple[TilingKey, TilingPerformance, Enum]:
        self.model = pyo.ConcreteModel()
        self._add_variables()
        self._add_constraints()
        self._add_objective()
        self.fixed_solution = self.problem.orig_tiling_key2model_tiling_key(self.problem.fixed_solution)
        self._fix_solution()
        model_tiling_key, performance, solver_status = self._solve_pyomo(time_limit)
        #print(model_tiling_key)
        #model_tiling_key.update()
        #print(model_tiling_key)
        if model_tiling_key is not None:
            tiling_key = self.problem.model_tiling_key2orig_tiling_key(model_tiling_key)
            tiling_key.update()
        else:
            tiling_key = None
        return tiling_key, performance, solver_status


class GemmMipBottleneck(GemmMipBase):
    def __init__(self, problem):
        super().__init__(problem)
        self.problem.var_names_integer['w0'] = 1
        self.problem.var_names_integer['w0_num'] = 1
        self.problem.var_names_integer['w1'] = 1
        self.problem.var_names_integer['w1_num'] = 1
        self.problem.var_names_integer['a1b1c1_tilde'] = 3
        self.problem.var_names_integer['w1_tilde'] = 1
        self.problem.var_names_integer['acbccc'] = 3
        self.problem.var_names_integer['wc'] = 1
        self.tvar_type = pyo.NonNegativeReals
        self.nvar_type = pyo.NonNegativeReals

    def _add_objective_mte1(self):
        model, problem = self.model, self.problem
        model.mte1_constr = pyo.ConstraintList()
        constr = model.mte1_constr
        ub_w0_num = problem.ub['w0_num']
        ub_a = problem.ub['acbccc'][0] * problem.ub['m0k0n0_num'][2]
        ub_b = problem.ub['acbccc'][1] * problem.ub['m0k0n0_num'][0]
        model.mte1a_count = pyo.Var(within=self.nvar_type, bounds=(1, ub_w0_num))
        model.mte1b_count = pyo.Var(within=self.nvar_type, bounds=(1, ub_w0_num))
        model.mte1a_num = pyo.Var(within=self.nvar_type, bounds=(1, ub_a))
        model.mte1b_num = pyo.Var(within=self.nvar_type, bounds=(1, ub_b))
        w0_num = model.w0_num
        m0_num, k0_num, n0_num = [model.m0k0n0_num[i] for i in range(3)]
        a0, b0, c0 = [model.a0b0c0[i] for i in range(3)]
        ac, bc = [model.acbccc[i] for i in range(2)]
        if not problem.reuse_l0:
            constr.add(expr=model.mte1a_count >= w0_num)
            constr.add(expr=model.mte1b_count >= w0_num)
            constr.add(expr=model.mte1a_num >= ac * n0_num)
            constr.add(expr=model.mte1b_num >= bc * m0_num)
        else:
            a0_num, b0_num, c0_num = [model.a0b0c0_num[i] for i in range(3)]
            m1_num, k1_num, n1_num = [model.m1k1n1_num[i] for i in range(3)]
            a1_num, b1_num, c1_num = [model.a1b1c1_num[i] for i in range(3)]
            w1_num = model.w1_num
            od = model.order_l0
            if hasattr(model, 'reuse_num_l0ab'):
                bfa, bfb = model.reuse_num_l0ab[0], model.reuse_num_l0ab[1]
            else:
                bfa, bfb = problem.fixed_solution.reuse_num_l0ab
            constr.add(expr=model.mte1a_count >= a0_num)
            constr.add(expr=model.mte1b_count >= b0_num)
            constr.add(expr=model.mte1a_num >= ac)
            constr.add(expr=model.mte1b_num >= bc)
            for name in ['save1', 'save2', 'save3']:
                setattr(model, 'mte1a_' + name, pyo.Var(within=self.nvar_type, bounds=(0, ub_a)))
                setattr(model, 'mte1b_' + name, pyo.Var(within=self.nvar_type, bounds=(0, ub_b)))
            # count
            constr.add(expr=model.mte1a_save1 <= bfa * a1_num * n0_num - bfa * w1_num)
            constr.add(expr=model.mte1a_count >= w0_num - model.mte1a_save1 - (1 - od[1] - od[3]) * ub_w0_num)
            constr.add(expr=model.mte1b_save1 <= bfb * b1_num * m0_num - bfb * w1_num)
            constr.add(expr=model.mte1b_count >= w0_num - model.mte1b_save1 - (1 - od[0] - od[2]) * ub_w0_num)
            constr.add(expr=model.mte1a_save2 <= bfa * c0_num * k1_num - bfa * b1_num * m0_num)
            constr.add(expr=model.mte1a_count >= w0_num - model.mte1a_save2 - (1 - od[0]) * ub_w0_num)
            constr.add(expr=model.mte1b_save2 <= bfb * c0_num * k1_num - bfb * a1_num * n0_num)
            constr.add(expr=model.mte1b_count >= w0_num - model.mte1b_save2 - (1 - od[1]) * ub_w0_num)
            constr.add(expr=model.mte1a_count >= a0_num * n1_num - (1 - od[2] - od[4]) * ub_w0_num)
            constr.add(expr=model.mte1b_count >= b0_num * m1_num - (1 - od[3] - od[5]) * ub_w0_num)
            constr.add(expr=model.mte1a_save3 <= bfa * b0_num * m1_num - bfa * c1_num * k0_num)
            constr.add(expr=model.mte1a_count >= w0_num - model.mte1a_save3 - (1 - od[5]) * ub_w0_num)
            constr.add(expr=model.mte1b_save3 <= bfb * a0_num * n1_num - bfb * c1_num * k0_num)
            constr.add(expr=model.mte1b_count >= w0_num - model.mte1b_save3 - (1 - od[4]) * ub_w0_num)
            # num
            constr.add(expr=model.mte1a_num >= ac * n0_num - model.mte1a_save1 * a0 - (1 - od[1] - od[3]) * ub_a)
            constr.add(expr=model.mte1b_num >= bc * m0_num - model.mte1b_save1 * b0 - (1 - od[0] - od[2]) * ub_b)
            constr.add(expr=model.mte1a_num >= ac * n0_num - model.mte1a_save2 * a0 - (1 - od[0]) * ub_a)
            constr.add(expr=model.mte1b_num >= bc * m0_num - model.mte1b_save2 * b0 - (1 - od[1]) * ub_b)
            constr.add(expr=model.mte1a_num >= ac * n1_num - (1 - od[2] - od[4]) * ub_a)
            constr.add(expr=model.mte1b_num >= bc * m1_num - (1 - od[3] - od[5]) * ub_b)
            constr.add(expr=model.mte1a_num >= ac * n0_num - model.mte1a_save3 * a0 - (1 - od[5]) * ub_a)
            constr.add(expr=model.mte1b_num >= bc * m0_num - model.mte1b_save3 * b0 - (1 - od[4]) * ub_b)
        # time
        ta_fix = tb_fix = problem.hardware_param.fix_time['mte1']
        tl1_fix = problem.hardware_param.fix_time['mte1_l1']
        ta = problem.abc_minB[0] / problem.hardware_param.bandwidth['mte1a']
        tb = problem.abc_minB[1] / problem.hardware_param.bandwidth['mte1b']
        model.mte1a_t = pyo.Var(within=self.tvar_type, bounds=(0, problem.ub_time))
        model.mte1b_t = pyo.Var(within=self.tvar_type, bounds=(0, problem.ub_time))
        model.mte1_t = pyo.Var(within=self.tvar_type, bounds=(0, problem.ub_time))
        constr.add(expr=model.mte1a_t >= ta_fix * model.mte1a_count + ta * model.mte1a_num + tl1_fix * model.w1_num)
        constr.add(expr=model.mte1b_t >= tb_fix * model.mte1b_count + tb * model.mte1b_num + tl1_fix * model.w1_num)
        constr.add(expr=model.mte1_t >= model.mte1a_t + model.mte1b_t)
        # L0 L1 (未考虑复用)
        model.mte1_t0 = pyo.Var(within=self.tvar_type, bounds=(0, problem.ub_time))
        constr.add(expr=model.mte1_t0 >= ta_fix + ta * a0 + tb_fix + tb * b0)
        model.mte1_t1 = pyo.Var(within=self.tvar_type, bounds=(0, problem.ub_time))
        a1, b1 = [model.a1b1c1[i] for i in range(2)]
        m1_tilde, _, n1_tilde = [model.m1k1n1_tilde[i] for i in range(3)]
        constr.add(expr=model.mte1_t1 >= (ta_fix + tb_fix) * model.w1_tilde + ta * a1 * n1_tilde + tb * b1 * m1_tilde)

    def _add_objective_mte2pre(self):
        model, problem, hp = self.model, self.problem, self.problem.hardware_param
        model.mte2pre_constr = pyo.ConstraintList()
        constr = model.mte2pre_constr
        model.mte2pre_t = pyo.Var(within=self.tvar_type, bounds=(0, problem.ub_time))
        model.mte2pre_a_t = pyo.Var(within=self.tvar_type, bounds=(0, problem.ub_time))
        model.mte2pre_b_t = pyo.Var(within=self.tvar_type, bounds=(0, problem.ub_time))
        ta, tb = [problem.abc_minB[i] / hp.bandwidth['mte2nd_pre']
                          if problem.isnd_abc[i] and hp.bandwidth['mte2nd_pre'] > 0 else 0 for i in range(2)]
        M, K, N = problem.MKN
        constr.add(expr=model.mte2pre_a_t * model.core_num >= ta * M * K)
        constr.add(expr=model.mte2pre_b_t * model.core_num >= tb * N * K)
        constr.add(expr=model.mte2pre_t >= model.mte2pre_a_t + model.mte2pre_b_t)

    def _add_objective_mte2gm(self):
        model, problem, hp = self.model, self.problem, self.problem.hardware_param
        model.mte2gm_t = pyo.Var(within=self.tvar_type, bounds=(0, problem.ub_time))
        if problem.l2_sufficient():
            return
        model.mte2gm_constr = pyo.ConstraintList()
        constr = model.mte2gm_constr
        M, K, N = problem.MKN
        cl = model.core_loop if problem.fixed_solution.core_loop is None else problem.fixed_solution.core_loop
        mc, kc, nc = [model.mckcnc[i] for i in range(3)]
        m2_num, n2_num = [model.m2n2_num[i] for i in range(2)]
        m2, n2 = [model.m2n2[i] for i in range(2)]
        model.mte2gm_a_num = pyo.Var(within=self.tvar_type, bounds=(0, M * N))
        model.mte2gm_b_num = pyo.Var(within=self.tvar_type, bounds=(0, M * N))
        constr.add(expr=model.mte2gm_a_num >= mc * cl - m2 + (model.order_l2[0] - 1) * M * cl)
        constr.add(expr=model.mte2gm_b_num >= m2_num * (nc * cl - n2) + (model.order_l2[0] - 1) * M * N)
        constr.add(expr=model.mte2gm_a_num >= n2_num * (mc * cl - m2) + (model.order_l2[1] - 1) * M * N)
        constr.add(expr=model.mte2gm_b_num >= nc * cl - n2 + (model.order_l2[1] - 1) * N * cl)
        #constr.add(expr=model.mte2gm_a_num >= mc + (model.order_l2[0] - 1) * M * N)
        #constr.add(expr=model.mte2gm_b_num >= m2_num * nc + (model.order_l2[0] - 1) * M * N)
        #constr.add(expr=model.mte2gm_a_num >= n2_num * mc + (model.order_l2[1] - 1) * M * N)
        #constr.add(expr=model.mte2gm_b_num >= nc + (model.order_l2[1] - 1) * M * N)
        ta, tb = [problem.abc_minB[i] / hp.bandwidth['mte2gm'] for i in range(2)]
        kca, kcb = [kc if problem.isnd_abc[i] else self.scale_mte2 * model.kc_512B for i in range(2)]
        constr.add(expr=model.mte2gm_t >= ta * model.mte2gm_a_num * kca + tb * model.mte2gm_b_num * kcb)

    def _add_objective_mte2(self):
        model, problem, hp = self.model, self.problem, self.problem.hardware_param
        model.mte2_constr = pyo.ConstraintList()
        constr = model.mte2_constr
        ub_w1_num = problem.ub['w1_num']
        ub_kc = problem.ub['m1k1n1'][1] * problem.ub['m1k1n1_num'][1]
        ub_a = problem.ub['mckcnc'][0] * ub_kc * problem.ub['m1k1n1_num'][2]
        ub_b = problem.ub['mckcnc'][2] * ub_kc * problem.ub['m1k1n1_num'][0]
        model.mte2a_count = pyo.Var(within=self.nvar_type, bounds=(1, ub_w1_num))
        model.mte2b_count = pyo.Var(within=self.nvar_type, bounds=(1, ub_w1_num))
        model.mte2a_num = pyo.Var(within=self.nvar_type, bounds=(1, ub_a))
        model.mte2b_num = pyo.Var(within=self.nvar_type, bounds=(1, ub_b))
        w1_num = model.w1_num
        m1_num, _, n1_num = [model.m1k1n1_num[i] for i in range(3)]
        a1 = model.a1b1_512B[0] if problem.isnd_abc[0] else model.a1b1c1[0]
        b1 = model.a1b1_512B[1] if problem.isnd_abc[1] else model.a1b1c1[1]
        ac = model.acbc_512B[0] if problem.isnd_abc[0] else model.acbccc[0]
        bc = model.acbc_512B[1] if problem.isnd_abc[1] else model.acbccc[1]
        if not problem.reuse_l1:
            constr.add(expr=model.mte2a_count >= w1_num)
            constr.add(expr=model.mte2b_count >= w1_num)
            constr.add(expr=model.mte2a_num >= ac * n1_num)
            constr.add(expr=model.mte2b_num >= bc * m1_num)
        else:
            a1_num, b1_num, c1_num = [model.a1b1c1_num[i] for i in range(3)]
            if hasattr(model, 'reuse_num_l1ab'):
                bfa, bfb = model.reuse_num_l1ab[0], model.reuse_num_l1ab[1]
            else:
                bfa, bfb = problem.fixed_solution.reuse_num_l1ab
            constr.add(expr=model.mte2a_count >= a1_num)
            constr.add(expr=model.mte2b_count >= b1_num)
            constr.add(expr=model.mte2a_num >= ac)
            constr.add(expr=model.mte2b_num >= bc)
            # KNM
            order = model.order_l1[0]
            constr.add(expr=model.mte2a_count >= w1_num - bfa * c1_num + bfa * m1_num - ub_w1_num * (1 - order))
            constr.add(expr=model.mte2b_count >= w1_num - bfb * m1_num + bfb - ub_w1_num * (1 - order))
            constr.add(expr=model.mte2a_num >= ac * n1_num - bfa * a1 * c1_num + bfa * a1 * m1_num - ub_a * (1 - order))
            constr.add(expr=model.mte2b_num >= bc * m1_num - bfb * b1 * m1_num + bfb * b1 - ub_b * (1 - order))
            # KMN
            order = model.order_l1[1]
            constr.add(expr=model.mte2a_count >= w1_num - bfa * n1_num + bfa - ub_w1_num * (1 - order))
            constr.add(expr=model.mte2b_count >= w1_num - bfb * c1_num + bfb * n1_num - ub_w1_num * (1 - order))
            constr.add(expr=model.mte2a_num >= ac * n1_num - bfa * a1 * n1_num + bfa * a1 - ub_a * (1 - order))
            constr.add(expr=model.mte2b_num >= bc * m1_num - bfb * b1 * c1_num + bfb * b1 * n1_num - ub_b * (1 - order))
        # time
        ta_fix, tb_fix = [hp.fix_time['mte2nd' if problem.isnd_abc[i] else 'mte2nz'] for i in range(2)]
        ta, tb = [problem.abc_minB[i] / hp.bandwidth['mte2nd'if problem.isnd_abc[i] else 'mte2nz'] for i in range(2)]
        model.mte2a_t = pyo.Var(within=self.tvar_type, bounds=(0, problem.ub_time))
        model.mte2b_t = pyo.Var(within=self.tvar_type, bounds=(0, problem.ub_time))
        model.mte2_t = pyo.Var(within=self.tvar_type, bounds=(0, problem.ub_time))
        constr.add(expr=model.mte2a_t >= ta_fix * model.mte2a_count + ta * model.mte2a_num)
        constr.add(expr=model.mte2b_t >= tb_fix * model.mte2b_count + tb * model.mte2b_num)
        # all
        constr.add(expr=model.mte2_t >= model.mte2a_t + model.mte2b_t)
        model.mte2_t1 = pyo.Var(within=self.tvar_type, bounds=(0, problem.ub_time))
        constr.add(expr=model.mte2_t1 >= ta_fix + ta * a1 + tb_fix + tb * b1)

    def _add_objective_fixpipe(self):
        model, problem, ub_t, hp = self.model, self.problem, self.problem.ub_time, self.problem.hardware_param
        model.fixpipe_constr = pyo.ConstraintList()
        constr = model.fixpipe_constr
        ub_c0_num = problem.ub['a0b0c0_num'][2]
        model.fixpipe_count = pyo.Var(within=self.nvar_type, bounds=(0, ub_c0_num))
        count = model.fixpipe_count
        constr.add(expr=count >= model.a1b1c1_num[2] - model.Ik * ub_c0_num)
        if not hasattr(model, 'order_l0'):
            constr.add(expr=count >= model.a0b0c0_num[2] + (model.Ik - 1) * ub_c0_num)
        else:
            m0_num, _, n0_num = [model.m0k0n0_num[i] for i in range(3)]
            m1_num, _, n1_num = [model.m1k1n1_num[i] for i in range(3)]
            od = model.order_l0
            constr.add(expr=count >= model.a0b0c0_num[2] + (model.Ik + od[0] + od[1] - 2) * ub_c0_num)
            constr.add(expr=count >= model.a1b1c1_num[2] + (od[4] + od[5] - 1) * ub_c0_num)
            constr.add(expr=count >= m0_num * n1_num + (model.Ik + od[2] - 2) * ub_c0_num)
            constr.add(expr=count >= m1_num * n0_num + (model.Ik + od[3] - 2) * ub_c0_num)
        model.fixpipe_t = pyo.Var(within=self.tvar_type, bounds=(0, ub_t))
        model.fixpipe_t0 = pyo.Var(within=self.tvar_type, bounds=(0, ub_t))
        model.fixpipe_t1 = pyo.Var(within=self.tvar_type, bounds=(0, ub_t))
        if problem.isnd_abc[2]:
            ## 1024B对齐的额外收益
            model.n1024 = pyo.Var(within=pyo.Binary)
            n1024 = 1024 / problem.dataB[1] / problem.mkn_min[2]
            n_ub = max(n1024, problem.MKN[2])
            constr.add(expr=model.m0k0n0[2] >= n1024 - model.n1024 * n_ub)
            constr.add(expr=model.m0k0n0[2] <= n1024 + model.n1024 * n_ub)

            tfix = hp.fix_time['fixpipe_nd']
            tmove = problem.abc_minB[2] / hp.bandwidth['fixpipe_nd']
            tscale_powe2 = hp.tscale['fixpipe_not_power2']
            tscale_1024B = hp.tscale['fixpipe_1024B']

            tmp = (tscale_powe2 - (tscale_powe2 - 1) * model.Ipower2[2]) * (tscale_1024B + (1 - tscale_1024B) * model.n1024)
            constr.add(expr=model.fixpipe_t >= tfix * count + tmove * model.cc_512B * tmp)
            constr.add(expr=model.fixpipe_t0 >= tfix + tmove * model.cout_512B * tmp)
        else:
            tfix = hp.fix_time['fixpipe_nz']
            tmove = problem.abc_minB[2] / hp.bandwidth['fixpipe_nz']
            constr.add(expr=model.fixpipe_t >= tfix * count + tmove * model.acbccc[2])
            constr.add(expr=model.fixpipe_t0 >= tfix + tmove * model.a0b0c0[2])
        constr.add(expr=model.fixpipe_t1 >= model.fixpipe_t0 * model.a1b1c1_tilde[2])

    def _add_objective_mac(self):
        model, problem = self.model, self.problem
        tfix = problem.hardware_param.fix_time['mac']
        tcal = np.prod(problem.mkn_min) * problem.dataB[0] / problem.hardware_param.bandwidth['mac_cal']
        tk = problem.mkn_min[1] / problem.hardware_param.bandwidth['mac_k']
        m0_num, _, n0_num = [model.m0k0n0_num[i] for i in range(3)]
        model.mac_t = pyo.Var(within=self.tvar_type, bounds=(0, problem.ub_time))
        model.mac_constr = pyo.ConstraintList()
        model.mac_constr.add(expr=model.mac_t >= tfix * model.w0_num + tcal * model.wc + tk * model.mckcnc[1] * model.a0b0c0_num[2])
        model.mac_t0 = pyo.Var(within=self.tvar_type, bounds=(0, problem.ub_time))
        model.mac_constr.add(expr=model.mac_t0 >= tfix + tcal * model.w0 + tk * model.m0k0n0[1])
        model.mac_t1 = pyo.Var(within=self.tvar_type, bounds=(0, problem.ub_time))
        model.mac_constr.add(expr=model.mac_t1 >= model.mac_t0 * model.w1_tilde)

    def _add_objective_scalar(self):
        model, problem = self.model, self.problem
        ts = problem.hardware_param.scalar_time
        tfix, tl0, tl0c, tl1 = ts['fix'], ts['l0num'], ts['l0cnum'], ts['l1num']
        model.scalar_t = pyo.Var(within=self.tvar_type, bounds=(0, problem.ub_time))
        model.scalar_t1 = pyo.Var(within=self.tvar_type, bounds=(0, problem.ub_time))
        model.scalar_constr = pyo.ConstraintList()
        model.scalar_constr.add(expr=model.scalar_t >= tfix + tl0 * model.w0_num + tl0c * model.a0b0c0_num[2] + tl1 * model.w1_num)
        model.scalar_constr.add(expr=model.scalar_t1 >= tl0 * model.w1_tilde + tl0c * model.a1b1c1_tilde[2] + tl1)

    def _add_objective_core(self):
        core_t = self.problem.hardware_param.fix_time['core']
        self.model.core_t = pyo.Var(within=self.tvar_type, bounds=(0, self.problem.ub_time))
        self.model.constr_cover.add(expr=self.model.core_t >= core_t * self.model.core_num)

    def _add_objective_t(self):
        model, problem = self.model, self.problem
        model.constr = pyo.ConstraintList()
        model.inner = pyo.Var(within=self.tvar_type, bounds=(0, problem.ub_time))
        model.t = pyo.Var(within=self.tvar_type, bounds=(0, problem.ub_time))
        self.all_unit = []
        self.all_total = []
        self.block_unit = []
        for unit in ['mte1', 'mte2', 'fixpipe', 'mac', 'scalar']:
            # total
            var = pyo.Var(within=self.tvar_type, bounds=(0, problem.ub_time))
            setattr(model, unit + '_total', var)
            model.constr.add(expr=var <= model.t)
            # block
            var_block = pyo.Var(within=self.tvar_type, bounds=(0, problem.ub_time))
            setattr(model, unit + '_block', var_block)
            model.constr.add(expr=var_block >= var - 0.85 * model.t)
            self.all_unit.append(getattr(model, unit + '_t'))
            if self.problem.fixed_solution.buffer_num_l0abc[2] == 1 and unit == 'fixpipe':
                continue
            self.block_unit.append(var_block)
            self.all_total.append(var)
        cl = model.core_loop if problem.fixed_solution.core_loop is None else problem.fixed_solution.core_loop
        model.constr.add(expr=model.inner >= model.mte1_t1 + model.mac_t0 + model.fixpipe_t0)
        model.constr.add(expr=model.inner >= model.mte1_t0 + model.mac_t1 + model.fixpipe_t0)
        model.constr.add(expr=model.inner >= model.mte1_t0 + model.mac_t0 + model.fixpipe_t1)
        model.constr.add(expr=model.inner >= model.scalar_t1 + model.fixpipe_t0)
        model.constr.add(expr=model.mte2_total >= model.mte2gm_t + cl * model.mte2_t + model.inner)
        model.constr.add(expr=model.mte1_total >= model.mte2_t1 + cl * model.mte1_t + model.mac_t0 + model.fixpipe_t0)
        if problem.fixed_solution.buffer_num_l0abc[2] == 1:
            model.constr.add(expr=model.mac_total >= model.mte2_t1 + model.mte1_t0 + cl * model.mac_t + cl * model.fixpipe_t)
            model.constr.add(expr=model.fixpipe_total >= model.mte2_t1 + model.mte1_t0 + cl * model.mac_t + cl * model.fixpipe_t)
        else:
            model.constr.add(expr=model.mac_total >= model.mte2_t1 + model.mte1_t0 + cl * model.mac_t + model.fixpipe_t0)
            model.constr.add(expr=model.fixpipe_total >= model.mte2_t1 + model.mte1_t0 + model.mac_t0 + cl * model.fixpipe_t)
        model.constr.add(expr=model.scalar_total >= cl * model.scalar_t + model.fixpipe_t0)

    def _add_objective_final(self):
        model, problem = self.model, self.problem
        #model.obj = pyo.Objective(expr=model.mte2pre_t + model.t + 0.2 * model.mte2_t, sense=pyo.minimize)
        model.obj = pyo.Objective(expr=model.core_t + model.mte2pre_t + model.t + 0.2 * sum(self.block_unit), sense=pyo.minimize)
        #model.obj = pyo.Objective(expr=model.core_t + model.mte2pre_t + model.t + 0.2 * sum(self.block_unit) + 0.01 * sum(self.all_total), sense=pyo.minimize)

    def _add_objective(self):
        self._add_objective_mte2pre()
        self._add_objective_mte2gm()
        self._add_objective_mte2()
        self._add_objective_mte1()
        self._add_objective_mac()
        self._add_objective_fixpipe()
        self._add_objective_scalar()
        self._add_objective_core()
        self._add_objective_t()
        self._add_objective_final()

    def _get_solution_pyomo(self):
        model, problem = self.model, self.problem
        tiling_key, performance = super()._get_solution_pyomo()
        data = performance.unit_time.data
        cl = self._get_value(model.core_loop) if problem.fixed_solution.core_loop is None else problem.fixed_solution.core_loop
        for unit in ['mte2', 'mte2a', 'mte2b', 'mte1', 'mte1a', 'mte1b', 'mac', 'fixpipe', 'scalar', 'mte2pre', 'mte2pre_a', 'mte2pre_b', 'mte2gm', 'mte2gm_a', 'mte2gm_b']:
            for name, col in zip(['_count', '_num', '_t', '_t0', '_t1'], ['count', 'num', 'time(cycle)', 'l0us', 'l1us']):
                cl0 = 1 if name in ['_t0', '_t1'] or unit in ['mte2pre', 'mte2pre_a', 'mte2pre_b', 'mte2gm', 'mte2gm_a', 'mte2gm_b'] else cl
                if hasattr(model, unit + name):
                    data.loc[unit, col] = self._get_value(getattr(model, unit + name)) * cl0
            data.loc[unit, 'time(us)'] = problem.hardware_param.cycle2us(data.loc[unit, 'time(cycle)'])
            data.loc[unit, 'l0us'] = problem.hardware_param.cycle2us(data.loc[unit, 'l0us'])
            data.loc[unit, 'l1us'] = problem.hardware_param.cycle2us(data.loc[unit, 'l1us'])
            if hasattr(model, unit + '_total'):
                var = getattr(model, unit + '_total')
                performance.unit_time.total_us[unit] = problem.hardware_param.cycle2us(self._get_value(var))
        data.loc['mac', 'count'] = self._get_value(model.w0_num)
        data.loc['mac', 'num'] = self._get_value(model.wc)
        data.loc['fixpipe', 'num'] = self._get_value(model.cc_512B if problem.isnd_abc[2] else model.acbccc[2])
        performance.unit_time.total_us['final'] = problem.hardware_param.cycle2us(self._get_value(model.t) + self._get_value(model.mte2pre_t))
        performance.unit_time.inner_us['final'] = problem.hardware_param.cycle2us(self._get_value(model.inner))
        return tiling_key, performance
