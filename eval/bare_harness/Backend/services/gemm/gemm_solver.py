# -*- coding: utf-8 -*-
# linziwei l00624134
import copy
import multiprocessing
from enum import Enum
import numpy as np
import itertools
import time
from typing import List, Tuple
from loguru import logger

from gemm_problem import Problem
from gemm_model_base import TilingKey, TilingPerformance, WhiteList


def get_problem_list(problem: Problem, iterate_solve=False) -> List[Problem]:
    if problem.fixed_solution.buffer_num_l0abc[2] is not None:
        problem_list = [problem]
    else:
        problem1, problem2 = copy.deepcopy(problem), copy.deepcopy(problem)
        problem1.fixed_solution.buffer_num_l0abc[2] = 1
        problem2.fixed_solution.buffer_num_l0abc[2] = 2
        problem_list = [problem1, problem2]
    if not iterate_solve:
        return problem_list
    new_list = []
    for problem in problem_list:
        orders = [problem.orders_l0 if problem.reuse_l0 else None,
                  problem.orders_l1 if problem.reuse_l1 else None,
                  problem.orders_l2 if not problem.l2_sufficient() else None]
        fix_order = problem.fixed_solution.orders_l0l1l2
        all_orders = [[i, j] for i, j in itertools.product(orders)]
        for order in all_orders:
            if fix_order[0] is not None and order[0] is not None and fix_order[0] != order[0]:
                continue
            if fix_order[1] is not None and order[1] is not None and fix_order[1] != order[1]:
                continue
            if fix_order[2] is not None and order[2] is not None and fix_order[2] != order[2]:
                continue
            new_problem = copy.deepcopy(problem)
            new_problem.fixed_solution.orders_l0l1l2 = order
            new_list.append(new_problem)
    return new_list


def solve_problem_once(args):
    ts = time.time()
    problem, solver, time_limit, print_process, print_log = args
    solver = solver(problem)
    solver.print_process = print_process
    tiling_key, performance, solver_status = solver.solve(time_limit=time_limit)
    if print_log:
        logger.info(f"buffer_num_l0c={problem.fixed_solution.buffer_num_l0abc[2]}, orders={problem.fixed_solution.orders_l0l1l2}, "
                   f"opt={performance.obj if performance is not None else None}, "
                   f"time_used={time.time()-ts:.2f},"
                   f"solver_status={solver_status}.")
    return tiling_key, performance, solver_status


def solve_problem(problem, solver, time_limit=120, work_num=1, iterate_solve=False, print_process=False, print_log=False) \
        -> Tuple[TilingKey, TilingPerformance, Enum]:
    problem_list = get_problem_list(problem, iterate_solve=iterate_solve)
    t_limit = time_limit / len(problem_list) if work_num == 1 else time_limit
    all_params = [(p, solver, t_limit, print_process, print_log) for p in problem_list]
    work_num = min(work_num, len(all_params))
    if work_num == 1:
        results = []
        for i in range(len(all_params)):
            results.append(solve_problem_once(all_params[i]))
    else:
        with multiprocessing.Pool(work_num) as pool:
            results = pool.map(solve_problem_once, all_params)
    best_idx = np.argmin([y.obj if y is not None else problem.ub_time for (_, y, _) in results])
    return results[best_idx]


def tiling_key2white_list(tiling_key: TilingKey) -> str:
    m0, k0, n0 = tiling_key.m0k0n0
    mc_num, kc_num, nc_num = tiling_key.mckcnc_num
    _, k1_tilde, _ = tiling_key.m1k1n1_tilde
    if mc_num * nc_num != tiling_key.core_num:
        min_cn = min(mc_num, nc_num, int(np.sqrt(tiling_key.core_num)))
        for i in range(min_cn, 0, -1):
            if tiling_key.core_num % i == 0:
                min_cn = i
                break
        other_cn = int(tiling_key.core_num // min_cn)
        mc_num, nc_num = (min_cn, other_cn) if mc_num <= nc_num else (other_cn, min_cn)
    wl_str = list(map(int, [m0, k0, n0, mc_num, nc_num, k1_tilde]))
    wl_str = ', '.join(list(map(str, wl_str)))
    return wl_str


def solve_white_list_m(param):
    row_idx, problem, m, m_sep, solver, time_limit, tk2str = param
    new_problem = copy.deepcopy(problem)
    new_problem.MKN_orig = [m, problem.MKN_orig[1], problem.MKN_orig[2]]
    tk, performance, solver_status = solve_problem(new_problem, solver, time_limit=time_limit)
    if tk is not None:
        output = tk2str(tk)
        logger.info(f"MKN = {new_problem.MKN_orig}, params = {output}.")
        return [row_idx, m, output]
    return [row_idx, m, None]


def solve_white_list(problem, solver, m_sep=16, time_limit=600, work_num=1, tk2str=None) -> WhiteList:
    tk2str = tiling_key2white_list if tk2str is None else tk2str
    t_start = time.time()
    MKN = copy.copy(problem.MKN_orig)
    logger.info(f'start shapes = {MKN}')
    M = int(np.ceil(MKN[0] / m_sep) * m_sep)
    sol_list = []

    def get_final_wl(sl):
        ii = 0
        while ii < len(sl) - 1:
            if sl[ii][1] is None or sl[ii][1] == sl[ii + 1][1]:
                sl.pop(ii)
            else:
                ii += 1
        wl = []
        prev_mm = 1
        for curr_mm, curr_vall in sl:
            wl.append(f'{prev_mm}, {curr_mm}, {curr_vall}')
            prev_mm = curr_mm + 1
        return wl

    def get_param(row_idx, m):
        t_limit = max(min(int(time_limit - time.time() + t_start), time_limit // 10), 30)
        return [row_idx, problem, m, m_sep, solver, t_limit, tk2str]

    max_m_sep = m_sep * 8
    m_limit = 6400
    while time.time() - t_start < time_limit:
        param_list = []
        if len(sol_list) == 0:
            #m_max = min(m_limit, M)
            #mr = int(np.ceil(max(max_m_sep, int((m_max - 1) / work_num)) / m_sep) * m_sep)
            #for i in range(m_sep, m_max, mr):
            #    param_list.append(get_param(0, i))
            param_list.append(get_param(0, m_sep))
            param_list.append(get_param(0, M))
        else:
            for i in range(1, len(sol_list)):
                prev_m, prev_val = sol_list[i - 1]
                curr_m, curr_val = sol_list[i]
                if prev_m >= m_limit or abs(curr_m - prev_m) <= m_sep:
                    continue
                if prev_val != curr_val or abs(curr_m - prev_m) >= max_m_sep:
                    mid_m = int(np.ceil((prev_m + curr_m) / m_sep / 2) * m_sep)
                    param_list.append(get_param(i, mid_m))
        if len(param_list) == 0:
            break
        wn = min(work_num, len(param_list))
        if wn == 1:
            results = []
            for p in param_list:
                result = solve_white_list_m(p)
                results.append(result)
        else:
            with multiprocessing.Pool(wn) as pool:
                results = pool.map(solve_white_list_m, param_list)
        for result in results[::-1]:
            row_idx, mid_m, mid_val = result
            sol_list.insert(row_idx, [mid_m, mid_val])

    white_list = get_final_wl(sol_list)
    logger.info(f'finish shapes = {MKN}')
    return WhiteList(hardware_name=problem.hardware_param_orig.name, K=MKN[1], N=MKN[2], wl=white_list)
