# -*- coding: utf-8 -*-
# linziwei l00624134
import copy
from typing import Optional
import math
import numpy as np

from gemm_model_base import HardwareParam, TilingKey


class Problem:
    def __init__(self, hardware_param: HardwareParam, MKN_orig):
        self.hardware_param_orig: HardwareParam = hardware_param
        self.MKN_orig = MKN_orig  # 原始输入尺寸
        # 参数设置
        self.dataB = [2, 4]  # INPUT, OUTPUT的位宽
        self.isnd_abc = [True, False, True]  # ABC矩阵
        self.bias = False  # bias尺寸为N*1，数据类型与矩阵B相同
        self.reuse_l0 = True  # l0是否数据复用
        self.reuse_l1 = True  # l1是否数据复用
        self.split_k_core = False  # K维不分核
        self.powers2_m0k0n0 = [False, False, False]  # m0k0n0是否需要是2的幂次方
        self.align_m0k0n0 = [1, 1, 1]   # m0k0n0必须为给定数值的整数倍
        self.align_m1k1n1 = [1, 1, 1]  # m1k1n1必须为给定数值的整数倍
        self.nd_alignB_mte2 = 256  # B
        self.orders_l0 = ['KNM', 'KMN', 'NKM', 'MKN', 'NMK', 'MNK']  # L0遍历顺序
        self.orders_l1 = ['KNM', 'KMN']  # L1遍历顺序
        self.orders_l2 = ['NM', 'MN']  # L2遍历顺序
        self.fixed_solution: Optional[TilingKey] = self.fillin_tiling_key(TilingKey())
        self.time_scale = 1e-2
        # 范围设置
        self.ub_user = {}
        self.lb_user = {}
        self.ub_l1_num = 6  # L1缓存块上限
        self.ub_time = 1e8

    def _get_ub_time(self):
        if self.fixed_solution.buffer_num_l0abc[2] is None:
            return int(1e8 * self.time_scale)
        # 算子最长执行时间(cycle)，假设核利用率1%
        hp = self.hardware_param_orig
        cn = hp.core_num
        t_cal = np.prod(self.MKN_orig) / hp.bandwidth['mac_cal'] / cn
        mn_num = np.ceil(np.ceil(self.MKN_orig[0] * self.MKN_orig[1] / (hp.cacheB_l0abc[2] / self.fixed_solution.buffer_num_l0abc[2] / self.dataB[1])) / cn)
        t_fix = mn_num * self.MKN_orig[1] * hp.fix_time['mac']
        t_k = mn_num * self.MKN_orig[1] / hp.bandwidth['mac_k']
        t = t_fix + t_cal + t_k
        return int(max(t / 0.1, 1e8) * self.time_scale)

    def initialize(self):
        self.hardware_param: HardwareParam = self.hardware_param_orig.get_time_scaled_hp(self.time_scale)
        self.mkn_min = np.array([16, int(32 / self.dataB[0]), 16])  # 最小计算单元或分形尺寸
        for attr in ["mkn_min", "MKN_orig"]:
            setattr(self, attr, np.array(getattr(self, attr), dtype=int))
        # 辅助变量
        self.abc_min = np.prod(self.len2area(self.mkn_min), axis=1).astype(int)
        self.abc_minB = copy.copy(self.abc_min)
        self.abc_minB[:2] = self.abc_minB[:2] * self.dataB[0]
        self.abc_minB[2] = self.abc_minB[2] * self.dataB[1]
        # ND处理
        self.align_m0k0n0 = np.array([math.lcm(u, v) for u, v in zip(self.align_m0k0n0, self.mkn_min)])
        self.align_m1k1n1 = np.array([math.lcm(u, v) for u, v in zip(self.align_m1k1n1, self.mkn_min)])
        self._deal_nd()
        # 模型尺寸转换
        self.MKN = np.ceil(self.MKN_orig / self.mkn_min).astype(int)
        self._initialize_cache()
        self.fixed_solution = self.fillin_tiling_key(self.fixed_solution)
        # 变量名
        self.var_names_integer = {'m0k0n0': 3, 'm0k0n0_num': 3, 'a0b0c0': 3, 'a0b0c0_num': 3,
                                  'm1k1n1_tilde': 3, 'm1k1n1': 3, 'm1k1n1_num': 3, 'a1b1c1': 3, 'a1b1c1_num': 3,
                                  'mckcnc': 3, 'mckcnc_num': 3, 'acbccc_num': 3,
                                  'core_num': 1, 'core_loop': 1}
        self.var_names_continuous = {}
        if self.reuse_l0:
            if None in self.fixed_solution.buffer_num_l0abc[:2]:
                self.var_names_integer['buffer_num_l0abc'] = 3
            if None in self.fixed_solution.reuse_num_l0ab:
                self.var_names_integer['reuse_num_l0ab'] = 2
        else:
            self.fixed_solution.buffer_num_l0abc[0] = self.fixed_solution.buffer_num_l0abc[1] = 2
            self.fixed_solution.reuse_num_l0ab = [0, 0]
        if self.reuse_l1:
            if None in self.fixed_solution.buffer_num_l1ab:
                self.var_names_integer['buffer_num_l1ab'] = 2
            if None in self.fixed_solution.reuse_num_l1ab:
                self.var_names_integer['reuse_num_l1ab'] = 2
        else:
            self.fixed_solution.buffer_num_l1ab = [2, 2]
            self.fixed_solution.reuse_num_l1ab = [0, 0]
        if self.split_k_core:
            self.var_names_integer['wc_num'] = 1
        if not self.l2_sufficient():
            #self.var_names_integer['m2n2_tilde'] = 2
            #self.var_names_integer['m2n2'] = 2
            self.var_names_continuous['m2n2_tilde'] = 2
            self.var_names_continuous['m2n2'] = 2
            self.var_names_integer['m2n2_num'] = 2
            self.fixed_solution.core_loop = 1
        self._initialize_bounds()
        self.ub_time = self._get_ub_time()

    def _deal_nd(self):
        if self.isnd_abc[0] or self.isnd_abc[1]:
            align_mte2 = self.nd_alignB_mte2 // self.dataB[0]
            if self.MKN_orig[1] >= align_mte2:
                self.align_m1k1n1[1] = math.lcm(self.align_m1k1n1[1], align_mte2)

    def _initialize_cache(self):
        self.cache_l0abc = np.floor(self.hardware_param.cacheB_l0abc / self.abc_minB)
        self.l1l2ab_scale = self.abc_minB[:2]
        scale = self._gcd(self.l1l2ab_scale[0], self.l1l2ab_scale[1])
        self.l1l2ab_scale = (self.l1l2ab_scale / scale).astype(int)
        self.cache_l1 = int(self.hardware_param.cacheB_l1 // scale)
        self.cache_l2 = int(self.hardware_param.cacheB_l2 // scale)

    def _initialize_bounds(self):
        def get_w_bounds(level, tt):
            abc = f'a{level}b{level}c{level}' + tt
            mkn = f'm{level}k{level}n{level}' + tt
            return int(self.lb[abc][2] * self.lb[mkn][1]), int(self.ub[abc][2] * self.ub[mkn][1])

        self.lb = {}
        self.ub = {}
        MKN = self.MKN
        # L0
        self.lb['a0b0c0'] = np.array([1, 1, 1])
        self.ub['a0b0c0'] = np.minimum(np.prod(self.len2area(MKN), axis=1), self.cache_l0abc).astype(int)
        self.lb['m0k0n0'] = np.array([1, 1, 1])
        self.ub['m0k0n0'] = np.minimum(MKN, np.min(self.area2len(self.ub['a0b0c0']), axis=1)).astype(int)
        self.lb['m0k0n0_num'] = np.array([1, 1, 1])
        self.ub['m0k0n0_num'] = copy.copy(MKN)
        self.lb['a0b0c0_num'] = np.prod(self.len2area(self.lb['m0k0n0_num']), axis=1)
        self.ub['a0b0c0_num'] = np.prod(self.len2area(self.ub['m0k0n0_num']), axis=1)
        self.lb['w0'], self.ub['w0'] = get_w_bounds('0', '')
        self.lb['w0_num'], self.ub['w0_num'] = get_w_bounds('0', '_num')
        # L1
        l1a, l1b = np.floor(self.hardware_param.cacheB_l1 / self.abc_minB[:2]).astype(int)
        ub_l1_num = self.ub_l1_num
        self.lb['a1b1c1'] = np.array([1, 1, 1])
        self.ub['a1b1c1'] = np.minimum(np.prod(self.len2area(MKN), axis=1), np.array([l1a, l1b, l1a * l1b]))
        self.lb['a1b1c1_tilde'] = np.array([1, 1, 1])
        self.ub['a1b1c1_tilde'] = np.minimum(self.ub['a1b1c1'] / self.lb['a0b0c0'], [ub_l1_num, ub_l1_num, ub_l1_num ** 2]).astype(int)
        self.lb['m1k1n1'] = np.array([1, 1, 1])
        self.ub['m1k1n1'] = np.minimum(MKN * 2, np.min(self.area2len(self.ub['a1b1c1']), axis=1)).astype(int)
        self.lb['m1k1n1_tilde'] = np.min(self.area2len(self.lb['a1b1c1_tilde']), axis=1)
        self.ub['m1k1n1_tilde'] = np.min(self.area2len(self.ub['a1b1c1_tilde']), axis=1)
        self.lb['m1k1n1_num'] = np.array([1, 1, 1])
        self.ub['m1k1n1_num'] = copy.copy(MKN)
        self.lb['a1b1c1_num'] = np.prod(self.len2area(self.lb['m1k1n1_num']), axis=1)
        self.ub['a1b1c1_num'] = np.prod(self.len2area(self.ub['m1k1n1_num']), axis=1)
        self.lb['w1'], self.ub['w1'] = get_w_bounds('1', '')
        self.lb['w1_tilde'], self.ub['w1_tilde'] = get_w_bounds('1', '_tilde')
        self.lb['w1_num'], self.ub['w1_num'] = get_w_bounds('1', '_num')
        # core
        self.lb['mckcnc'] = np.array([1, 1, 1])
        self.ub['mckcnc'] = copy.copy(MKN)
        self.lb['mckcnc_num'] = np.array([1, 1, 1])
        self.ub['mckcnc_num'] = copy.copy(MKN)
        self.lb['acbccc'] = np.prod(self.len2area(self.lb['mckcnc']), axis=1)
        self.ub['acbccc'] = np.prod(self.len2area(self.ub['mckcnc']), axis=1)
        self.lb['acbccc_num'] = np.prod(self.len2area(self.lb['mckcnc_num']), axis=1)
        self.ub['acbccc_num'] = np.prod(self.len2area(self.ub['mckcnc_num']), axis=1)
        self.lb['wc'], self.ub['wc'] = get_w_bounds('c', '')
        self.lb['wc_num'], self.ub['wc_num'] = get_w_bounds('c', '_num')
        # core loop num
        self.lb['core_loop'] = 1
        self.ub['core_loop'] = int(np.ceil(self.MKN[0] * self.MKN[2] / self.hardware_param.core_num))
        self.lb['core_num'] = 1
        self.ub['core_num'] = self.hardware_param.core_num
        # L2命中率
        MN = np.array([MKN[0], MKN[2]])
        l2a, l2b = np.floor(self.hardware_param.cacheB_l2 / self.abc_minB[:2]).astype(int)
        self.lb['m2n2'] = np.array([1, 1])
        self.ub['m2n2'] = np.minimum(2 * MN, [l2a, l2b]).astype(int)
        self.lb['m2n2_tilde'] = np.array([1, 1])
        self.ub['m2n2_tilde'] = (self.ub['m2n2'] / np.array([self.lb['m1k1n1'][0], self.lb['m1k1n1'][2]])).astype(int)
        self.lb['m2n2_num'] = np.array([1, 1])
        self.ub['m2n2_num'] = copy.copy(MN)
        # buffer
        self.lb['buffer_num_l0abc'] = [2, 2, 1]
        self.ub['buffer_num_l0abc'] = [50, 50, 2]
        self.lb['buffer_num_l1ab'] = [2, 2]
        self.ub['buffer_num_l1ab'] = [50, 50]
        self.lb['reuse_num_l0ab'] = [0, 0]
        self.ub['reuse_num_l0ab'] = [50, 50]
        self.lb['reuse_num_l1ab'] = [0, 0]
        self.ub['reuse_num_l1ab'] = [50, 50]
        # 与自定义汇总
        for key in set(self.ub.keys()).intersection(set(self.ub_user.keys())):
            self.ub_user[key] = np.array(self.ub_user[key]).astype(float)
            self.ub_user[key][np.isnan(self.ub_user[key])] = 1e15
            self.ub[key] = np.minimum(self.ub_user[key], self.ub[key]).astype(int)
        for key in set(self.lb.keys()).intersection(set(self.lb_user.keys())):
            self.lb_user[key] = np.array(self.lb_user[key]).astype(float)
            self.lb_user[key][np.isnan(self.lb_user[key])] = 1
            self.lb[key] = np.maximum(self.lb_user[key], self.lb[key]).astype(int)

    def l2_sufficient(self):
        return (self.hardware_param.l2_sufficient(self.MKN_orig, self.dataB)
                or (self.mkn_min[0] + self.mkn_min[2]) * self.MKN_orig[1] * self.dataB[0] > self.hardware_param.cacheB_l2)

    @staticmethod
    def len2area(mkn):
        return np.array([[mkn[0], mkn[1]], [mkn[1], mkn[2]], [mkn[2], mkn[0]]])

    @staticmethod
    def area2len(abc):
        return np.array([[abc[0], abc[2]], [abc[0], abc[1]], [abc[1], abc[2]]])

    @staticmethod
    def _gcd(a, b):
        (a, b) = (b, a) if b > a else (a, b)
        r = a % b
        while r != 0:
            a, b = b, r
            r = a % b
        return b

    def fillin_tiling_key(self, tiling_key) -> TilingKey:
        setattr(tiling_key, 'MKN', copy.copy(getattr(self, 'MKN_orig')))
        for attr in ['dataB', 'isnd_abc', 'bias', 'reuse_l0', 'reuse_l1']:
            setattr(tiling_key, attr, copy.copy(getattr(self, attr)))
        if not self.split_k_core:
            if hasattr(tiling_key, 'mckcnc') and tiling_key.mckcnc is not None:
                tiling_key.mckcnc[1] = self.MKN_orig[1]
            else:
                tiling_key.mckcnc = [None, self.MKN_orig[1], None]
            if hasattr(tiling_key, 'mckcnc_num') and tiling_key.mckcnc_num is not None:
                tiling_key.mckcnc_num[1] = 1
            else:
                tiling_key.mckcnc_num = [None, 1, None]
        return tiling_key

    def orig_tiling_key2model_tiling_key(self, tiling_key: TilingKey) -> TilingKey:
        return tiling_key.scale_tiling_key(1.0 / np.array(self.mkn_min), is_orig=False)

    def model_tiling_key2orig_tiling_key(self, tiling_key: TilingKey) -> TilingKey:
        return tiling_key.scale_tiling_key(self.mkn_min, is_orig=True)

    def get_fix_solution(self) -> TilingKey:
        return copy.deepcopy(self.fixed_solution)
