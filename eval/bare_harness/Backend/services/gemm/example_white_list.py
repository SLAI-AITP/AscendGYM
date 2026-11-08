# -*- coding: utf-8 -*-
# linziwei l00624134
import yaml
import argparse

from gemm_model_base import HardwareParam
from gemm_mip import GemmMipBottleneck
from gemm_solver import solve_white_list
from gemm_problem import Problem
from gemm_existing_operator import GemmGts


def main(args):
    with open('hardware.yml', 'r', encoding='utf-8') as f:
        param_dict = yaml.safe_load(f)
    hardware_param = HardwareParam(param_dict["910B3_modify"])

    problem = Problem(hardware_param=hardware_param, MKN_orig=list(map(int, args.mkn.split(','))))  # 硬件参数和矩阵乘维度
    problem.dataB = [2, 4]  # 输入（AB矩阵）和输出（C矩阵）数据的位宽，单位Byte
    problem.isnd_abc = [v == 'nd' for v in args.ndnz.split(',')]  # ABC矩阵是否为ND格式
    problem.bias = args.bias  # 是否需要加bias
    problem.reuse_l0, problem.reuse_l1 = list(map(bool, args.reuse.split(',')))

    solver = GemmGts if args.solver == 'gts' else GemmMipBottleneck

    white_list = solve_white_list(problem=problem, solver=solver, m_sep=16, time_limit=600, work_num=8, tk2str=None)
    white_list.write(r'white_list.txt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mkn', type=str, help='逗号分割，引号包裹', default='1,4096,62080')
    parser.add_argument('--ndnz', type=str, help='逗号分割，引号包裹', default='nd,nz,nd')
    parser.add_argument('--bias', type=bool, default=False)
    parser.add_argument('--reuse', type=str, default='True,False')
    parser.add_argument('--solver', type=str, default='gts')
    args = parser.parse_args()
    main(args)
