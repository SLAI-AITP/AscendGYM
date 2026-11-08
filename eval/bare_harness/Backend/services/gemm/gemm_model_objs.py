# -*- coding: utf-8 -*-
# linziwei l00624134
import numpy as np

from gemm_model_base import HardwareParam, TilingKey, TilingPerformance


class UnitTimeCalculator:
    """搬运单元分析"""
    def __init__(self, hardware_param: HardwareParam,
                 tiling_key: TilingKey,
                 performance: TilingPerformance):
        self.hardware_param: HardwareParam = hardware_param
        self.tiling_key: TilingKey = tiling_key
        self.performance: TilingPerformance = performance
        self.data = self.performance.unit_time.data

    def _update_mac(self):
        tk, hp, data, unit = self.tiling_key, self.hardware_param, self.data, 'mac'
        dataB = tk.dataB[0]
        data.loc[unit, 'count'] = int(np.prod(tk.m0k0n0_num))
        data.loc[unit, 'num'] = int(np.prod(tk.mckcnc))
        data.loc[unit, 'dataB'] = dataB * data.loc[unit, 'num']
        m0_num, _, n0_num = tk.m0k0n0_num
        tfix, bw_cal, bw_k = hp.fix_time[unit], hp.bandwidth['mac_cal'], hp.bandwidth['mac_k']
        data.loc[unit, 'time(cycle)'] = tfix * data.loc[unit, 'count'] + data.loc[unit, 'dataB'] / bw_cal + tk.mckcnc[1] * n0_num * m0_num / bw_k
        data.loc[unit, 'time(us)'] = hp.cycle2us(data.loc[unit, 'time(cycle)'])
        data.loc[unit, 'l0us'] = hp.cycle2us(tfix + np.prod(tk.m0k0n0) * dataB / bw_cal + tk.m0k0n0[1] / bw_k)
        data.loc[unit, 'l1us'] = data.loc[unit, 'l0us'] * np.prod(tk.m1k1n1_tilde)

    def _update_scalar(self):
        tk, hp, data, unit = self.tiling_key, self.hardware_param, self.data, 'scalar'
        tfix, tl0, tl0c, tl1 = hp.scalar_time['fix'], hp.scalar_time['l0num'], hp.scalar_time['l0cnum'], hp.scalar_time['l1num']
        data.loc[unit, 'time(cycle)'] = tl0 * np.prod(tk.m0k0n0_num) + tl1 * np.prod(tk.m1k1n1_num) + tl0c * tk.a0b0c0_num[2] + tfix
        data.loc[unit, 'time(us)'] = hp.cycle2us(data.loc[unit, 'time(cycle)'])
        data.loc[unit, 'l1us'] = hp.cycle2us(tl0 * np.prod(tk.m1k1n1_tilde) + tl1 + tl0c * tk.a1b1c1_tilde[2])

    def _update_fixpipe(self):
        tk, hp, data, unit = self.tiling_key, self.hardware_param, self.data, 'fixpipe'
        dataB = tk.dataB[1]
        m0, _, n0 = tk.m0k0n0
        m0_num, _, n0_num = tk.m0k0n0_num
        m1, _, n1 = tk.m1k1n1
        m1_num, k1_num, n1_num = tk.m1k1n1_num
        mc, _, nc = tk.mckcnc
        inner_order = tk.orders_l0l1l2[0]
        if k1_num >= 2:
            data.loc[unit, 'count'] = int(m1_num * n1_num)
            n, n_num, m = n1, n1_num, m1
        elif inner_order is None or inner_order == 'KNM' or inner_order == 'KMN':
            data.loc[unit, 'count'] = int(m0_num * n0_num)
            n, n_num, m = n0, n0_num, m0
        elif inner_order == 'NKM':
            data.loc[unit, 'count'] = int(m0_num * n1_num)
            n, n_num, m = n1, n1_num, m0
        elif inner_order == 'MKN':
            data.loc[unit, 'count'] = int(m1_num * n0_num)
            n, n_num, m = n0, n0_num, m1
        else:
            data.loc[unit, 'count'] = int(m1_num * n1_num)
            n, n_num, m = n1, n1_num, m1
        if tk.isnd_abc[2]:
            tfix, bw = hp.fix_time['fixpipe_nd'], hp.bandwidth['fixpipe_nd']
            if abs(np.log2(n) - np.round(np.log2(n))) > 0.01:
                bw /= hp.tscale['fixpipe_not_power2']
            if (n * dataB) % 1024 == 0:
                bw /= hp.tscale['fixpipe_1024B']
            n_512B = np.ceil(n * dataB / 512) * 512 / dataB
            n_last_512B = np.ceil((nc - n * (n_num - 1)) * dataB / 512) * 512 / dataB
            n, nc = n_512B, n_512B * (n_num - 1) + n_last_512B
        else:
            tfix, bw = hp.fix_time['fixpipe_nz'], hp.bandwidth['fixpipe_nz']
        data.loc[unit, 'num'] = int(mc * nc)
        data.loc[unit, 'dataB'] = dataB * data.loc[unit, 'num']
        data.loc[unit, 'time(cycle)'] = tfix * data.loc[unit, 'count'] + data.loc[unit, 'dataB'] / bw
        data.loc[unit, 'time(us)'] = hp.cycle2us(data.loc[unit, 'time(cycle)'])
        data.loc[unit, 'l0us'] = hp.cycle2us(tfix + m * n * dataB / bw)
        data.loc[unit, 'l1us'] = data.loc[unit, 'l0us'] * tk.m1k1n1_tilde[0] * tk.m1k1n1_tilde[2]

    def _update_mte1(self):
        tk, hp, data, uta, utb = self.tiling_key, self.hardware_param, self.data, 'mte1a', 'mte1b'
        inner_order = tk.orders_l0l1l2[0]
        m0, k0, n0 = tk.m0k0n0
        m0_num, k0_num, n0_num = tk.m0k0n0_num
        m1_num, k1_num, n1_num = tk.m1k1n1_num
        mc, kc, nc = tk.mckcnc
        a0, b0, _ = tk.a0b0c0
        ac, bc, cc = tk.acbccc
        mkn0_num = m0_num * k0_num * n0_num
        dataB = tk.dataB[0]
        reuse_a, reuse_b = tk.reuse_num_l0ab
        if not tk.reuse_l0:
            data.loc[uta, 'count'] = mkn0_num
            data.loc[uta, 'num'] = ac * n0_num
            data.loc[utb, 'count'] = mkn0_num
            data.loc[utb, 'num'] = bc * m0_num
        elif inner_order == 'KNM':
            data.loc[uta, 'count'] = mkn0_num - reuse_a * m0_num * (n0_num - n1_num) * k1_num
            data.loc[uta, 'num'] = ac * n0_num - reuse_a * mc * (n0_num - n1_num) * k1_num * k0
            data.loc[utb, 'count'] = mkn0_num - reuse_b * (m0_num - m1_num) * n1_num * k1_num
            data.loc[utb, 'num'] = bc * m0_num - reuse_b * (m0_num - m1_num) * n1_num * n0 * k1_num * k0
        elif inner_order == 'KMN':
            data.loc[uta, 'count'] = mkn0_num - reuse_a * m1_num * (n0_num - n1_num) * k1_num
            data.loc[uta, 'num'] = ac * n0_num - reuse_a * m1_num * m0 * (n0_num - n1_num) * k1_num * k0
            data.loc[utb, 'count'] = mkn0_num - reuse_b * (m0_num - m1_num) * n0_num * k1_num
            data.loc[utb, 'num'] = bc * m0_num - reuse_b * (m0_num - m1_num) * nc * k1_num * k0
        elif inner_order == 'NKM':
            data.loc[uta, 'count'] = m0_num * n1_num * k0_num
            data.loc[uta, 'num'] = ac * n1_num
            data.loc[utb, 'count'] = mkn0_num - reuse_b * (m0_num - m1_num) * n1_num * k1_num
            data.loc[utb, 'num'] = bc * m0_num - reuse_b * (m0_num - m1_num) * n1_num * n0 * k1_num * k0
        elif inner_order == 'MKN':
            data.loc[uta, 'count'] = mkn0_num - reuse_a * m1_num * (n0_num - n1_num) * k1_num
            data.loc[uta, 'num'] = ac * n0_num - reuse_a * m1_num * m0 * (n0_num - n1_num) * k1_num * k0
            data.loc[utb, 'count'] = m1_num * n0_num * k0_num
            data.loc[utb, 'num'] = bc * m1_num
        elif inner_order == 'NMK':
            data.loc[uta, 'count'] = m0_num * n1_num * k0_num
            data.loc[uta, 'num'] = ac * n1_num
            data.loc[utb, 'count'] = mkn0_num - reuse_b * (m0_num - m1_num) * n1_num * k0_num
            data.loc[utb, 'num'] = bc * m0_num - reuse_b * (m0_num - m1_num) * n1_num * n0 * kc
        elif inner_order == 'MNK':
            data.loc[uta, 'count'] = mkn0_num - reuse_a * m1_num * (n0_num - n1_num) * k0_num
            data.loc[uta, 'num'] = ac * n0_num - reuse_a * m1_num * m0 * (n0_num - n1_num) * kc
            data.loc[utb, 'count'] = m1_num * n0_num * k0_num
            data.loc[utb, 'num'] = bc * m1_num
        data.loc[uta, 'count'] = max(data.loc[uta, 'count'], m0_num * k0_num)
        data.loc[uta, 'num'] = max(data.loc[uta, 'num'], ac)
        data.loc[utb, 'count'] = max(data.loc[utb, 'count'], n0_num * k0_num)
        data.loc[utb, 'num'] = max(data.loc[utb, 'num'], bc)
        for ut, mn0 in zip([uta, utb], [m0, n0]):
            data.loc[ut, 'dataB'] = data.loc[ut, 'num'] * dataB
            tl1fix, tfix, bw = hp.fix_time['mte1_l1'], hp.fix_time['mte1'], hp.bandwidth[ut]
            data.loc[ut, 'time(cycle)'] = tfix * data.loc[ut, 'count'] + data.loc[ut, 'dataB'] / bw + tl1fix * m1_num * n1_num * k1_num
            data.loc[ut, 'time(us)'] = hp.cycle2us(data.loc[ut, 'time(cycle)'])
            data.loc[ut, 'l0us'] = hp.cycle2us(tfix + mn0 * k0 * dataB / bw)
            data.loc[ut, 'l1us'] = data.loc[ut, 'l0us'] * np.prod(tk.m1k1n1_tilde)
        data.loc['mte1'] = np.sum(data.loc[[uta, utb], :], axis=0)
        data.loc['mte1', 'time(us)'] = hp.cycle2us(data.loc['mte1', 'time(cycle)'])

    def _update_mte2pre(self):
        tk, hp, data = self.tiling_key, self.hardware_param, self.data
        dataB = tk.dataB[0]
        M, K, N = tk.MKN
        data.loc['mte2pre_a', 'num'] = M * K / np.prod(tk.core_num) if tk.isnd_abc[0] else 0
        data.loc['mte2pre_b', 'num'] = N * K / np.prod(tk.core_num) if tk.isnd_abc[1] else 0
        for ut, abc_idx in zip(['mte2pre_a', 'mte2pre_b'], [0, 1]):
            data.loc[ut, 'dataB'] = data.loc[ut, 'num'] * dataB
            ut_name = 'mte2nd_pre' if tk.isnd_abc[abc_idx] else 'mte2nz_pre'
            tfix, bw = hp.fix_time.get(ut_name, 0), hp.bandwidth.get(ut_name, np.inf)
            bw = np.inf if bw == 0 else bw
            data.loc[ut, 'time(cycle)'] = tfix * data.loc[ut, 'count'] + data.loc[ut, 'dataB'] / bw
            data.loc[ut, 'time(us)'] = hp.cycle2us(data.loc[ut, 'time(cycle)'])
        data.loc['mte2pre'] = np.sum(data.loc[['mte2pre_a', 'mte2pre_b'], :], axis=0)

    def _update_mte2gm(self):
        tk, hp, data = self.tiling_key, self.hardware_param, self.data
        if hp.l2_sufficient(tk.MKN, tk.dataB) or None in tk.m2n2_tilde:
            return
        dataB = tk.dataB[0]
        cl = tk.core_loop
        mc, kc, nc = tk.mckcnc
        m1, k1, n1 = tk.m1k1n1
        m1_num, k1_num, n1_num = tk.m1k1n1_num
        k1_512B = np.ceil(k1 * dataB / 512) * 512 / dataB
        k1_last_512B = np.ceil((kc - k1 * (k1_num - 1)) * dataB / 512) * 512 / dataB
        kc_512B = k1_512B * (k1_num - 1) + k1_last_512B
        kc_a = kc_512B if tk.isnd_abc[0] else kc
        kc_b = kc_512B if tk.isnd_abc[1] else kc
        m2, n2 = tk.m2n2
        gm_order = tk.orders_l0l1l2[2]
        m2_num, n2_num = tk.m2n2_num
        data.loc['mte2gm_a', 'num'] = (mc * cl - m2) * kc_a if gm_order == 'NM' else (mc * cl - m2) * kc_a * n2_num
        data.loc['mte2gm_b', 'num'] = (nc * cl - n2) * kc_b if gm_order == 'MN' else (nc * cl - n2) * kc_b * m2_num
        for ut in ['mte2gm_a', 'mte2gm_b']:
            data.loc[ut, 'num'] = max(0, data.loc[ut, 'num'])
            data.loc[ut, 'dataB'] = data.loc[ut, 'num'] * dataB
            tfix, bw = hp.fix_time.get('mte2gm', 0), hp.bandwidth.get('mte2gm', np.inf)
            bw = np.inf if bw == 0 else bw
            data.loc[ut, 'time(cycle)'] = tfix * data.loc[ut, 'count'] + data.loc[ut, 'dataB'] / bw
            data.loc[ut, 'time(us)'] = hp.cycle2us(data.loc[ut, 'time(cycle)'])
        data.loc['mte2gm'] = np.sum(data.loc[['mte2gm_a', 'mte2gm_b'], :], axis=0)

    def _update_mte2(self):
        tk, hp, data = self.tiling_key, self.hardware_param, self.data
        uta, utb, utbias = 'mte2a', 'mte2b', 'mte2bias'
        outer_order = tk.orders_l0l1l2[1]
        m1, k1, n1 = tk.m1k1n1
        m1_num, k1_num, n1_num = tk.m1k1n1_num
        mc, kc, nc = tk.mckcnc
        dataB = tk.dataB[0]
        k1_512B = np.ceil(k1 * dataB / 512) * 512 / dataB
        k1_last_512B = np.ceil((kc - k1 * (k1_num - 1)) * dataB / 512) * 512 / dataB
        kc_512B = k1_512B * (k1_num - 1) + k1_last_512B
        a1 = m1 * k1_512B if tk.isnd_abc[0] else tk.a1b1c1[0]
        ac = mc * kc_512B if tk.isnd_abc[0] else tk.acbccc[0]
        b1 = n1 * k1_512B if tk.isnd_abc[1] else tk.a1b1c1[1]
        bc = nc * kc_512B if tk.isnd_abc[1] else tk.acbccc[1]
        mkn1_num = m1_num * k1_num * n1_num
        # 个数
        reuse_a, reuse_b = tk.reuse_num_l1ab
        if not tk.reuse_l1:
            data.loc[uta, 'count'] = mkn1_num
            data.loc[uta, 'num'] = ac * n1_num
            data.loc[utb, 'count'] = mkn1_num
            data.loc[utb, 'num'] = bc * m1_num
            data.loc[utbias, 'count'] = m1_num * n1_num if tk.bias else 0
            data.loc[utbias, 'num'] = m1_num * nc if tk.bias else 0
        elif outer_order == 'KNM':
            data.loc[uta, 'count'] = mkn1_num - reuse_a * m1_num * (n1_num - 1)
            data.loc[uta, 'num'] = ac * n1_num - reuse_a * a1 * m1_num * (n1_num - 1)
            data.loc[utb, 'count'] = mkn1_num - reuse_b * (m1_num - 1)
            data.loc[utb, 'num'] = bc * m1_num - reuse_b * b1 * (m1_num - 1)
            data.loc[utbias, 'count'] = m1_num * n1_num - (m1_num - 1) if tk.bias else 0
            data.loc[utbias, 'num'] = m1_num * nc - n1 * (m1_num - 1) if tk.bias else 0
        elif outer_order == 'KMN':
            data.loc[uta, 'count'] = mkn1_num - reuse_a * (n1_num - 1)
            data.loc[uta, 'num'] = ac * n1_num - reuse_a * a1 * (n1_num - 1)
            data.loc[utb, 'count'] = mkn1_num - reuse_b * n1_num * (m1_num - 1)
            data.loc[utb, 'num'] = bc * m1_num - reuse_b * b1 * n1_num * (m1_num - 1)
            data.loc[utbias, 'count'] = n1_num if tk.bias else 0
            data.loc[utbias, 'num'] = nc if tk.bias else 0
        data.loc[uta, 'count'] = max(data.loc[uta, 'count'], m1_num * k1_num)
        data.loc[uta, 'num'] = max(data.loc[uta, 'num'], ac)
        data.loc[utb, 'count'] = max(data.loc[utb, 'count'], n1_num * k1_num)
        data.loc[utb, 'num'] = max(data.loc[utb, 'num'], bc)
        for ut, abc_idx in zip([uta, utb, utbias], [0, 1, 1]):
            data.loc[ut, 'dataB'] = data.loc[ut, 'num'] * dataB
            ut_name = 'mte2nd' if tk.isnd_abc[abc_idx] else 'mte2nz'
            tfix, bw = hp.fix_time.get(ut_name, 0), hp.bandwidth.get(ut_name, np.inf)
            data.loc[ut, 'time(cycle)'] = tfix * data.loc[ut, 'count'] + data.loc[ut, 'dataB'] / bw
            if ut in [uta, utb]:
                data.loc[ut, 'l1us'] = tfix + (a1 if 'mte2a' in ut else b1) * dataB / bw
                data.loc[ut, 'l1us'] = hp.cycle2us(data.loc[ut, 'l1us'])
            data.loc[ut, 'time(us)'] = hp.cycle2us(data.loc[ut, 'time(cycle)'])
        data.loc['mte2'] = np.sum(data.loc[[uta, utb, utbias], :], axis=0)

    """汇总"""
    def update_performance(self):
        self.tiling_key.update()
        self._update_mte2pre()
        self._update_mte2gm()
        self._update_mte2()
        self._update_mte1()
        self._update_mac()
        self._update_fixpipe()
        self._update_scalar()
        tk, hp, data = self.tiling_key, self.hardware_param, self.data
        for unit in ['mte2', 'mte2a', 'mte2b', 'mte1', 'mte1a', 'mte1b', 'mac', 'fixpipe', 'scalar']:
            for col in ['count', 'num', 'dataB', 'time(cycle)', 'time(us)']:
                data.loc[unit, col] *= tk.core_loop
        inner_us = self.performance.unit_time.inner_us
        inner_us['mte1'] = data.loc['mte1', 'l1us'] + data.loc['mac', 'l0us'] + data.loc['fixpipe', 'l0us']
        inner_us['mac'] = data.loc['mte1', 'l0us'] + data.loc['mac', 'l1us'] + data.loc['fixpipe', 'l0us']
        inner_us['fixpipe'] = data.loc['mte1', 'l0us'] + data.loc['mac', 'l0us'] + data.loc['fixpipe', 'l1us']
        inner_us['scalar'] = data.loc['scalar', 'l1us'] + data.loc['fixpipe', 'l0us']
        inner_us['final'] = max(inner_us.values())
        tt = self.performance.unit_time.total_us
        mte2_l1 = data.loc['mte2', 'l1us']
        tt['mte2'] = data.loc['mte2', 'time(us)'] + inner_us['final'] + data.loc['mte2gm', 'time(us)']
        tt['mte1'] = mte2_l1 + data.loc['mte1', 'time(us)'] + data.loc['mac', 'l0us'] + data.loc['fixpipe', 'l0us']
        if tk.buffer_num_l0abc[2] == 1:
            tt['mac'] = mte2_l1 + data.loc['mte1', 'l0us'] + data.loc['mac', 'time(us)'] + data.loc['fixpipe', 'time(us)']
            tt['fixpipe'] = mte2_l1 + data.loc['mte1', 'l0us'] + data.loc['mac', 'time(us)'] + data.loc['fixpipe', 'time(us)']
        else:
            tt['mac'] = mte2_l1 + data.loc['mte1', 'l0us'] + data.loc['mac', 'time(us)'] + data.loc['fixpipe', 'l0us']
            tt['fixpipe'] = mte2_l1 + data.loc['mte1', 'l0us'] + data.loc['mac', 'l0us'] + data.loc['fixpipe', 'time(us)']
        tt['scalar'] = data.loc['scalar', 'time(us)'] + data.loc['fixpipe', 'l0us']
        tt['final'] =  data.loc['mte2pre', 'time(us)'] + max(tt.values())
        data['util'] = data['time(us)'] / tt['final']


def evaluate_tiling_key(hardware_param: HardwareParam, tiling_key: TilingKey) -> TilingPerformance:
    tiling_key.feasibility_check(hardware_param)
    performance = TilingPerformance()
    performance.core_loop_num = int(np.ceil(np.prod(tiling_key.mckcnc_num) / hardware_param.core_num))
    performance.util_cache_l0abc = tiling_key.cacheB_l0abc / hardware_param.cacheB_l0abc
    performance.util_cache_l1 = tiling_key.cacheB_l1 / hardware_param.cacheB_l1
    performance.util_cache_l2 = tiling_key.cacheB_l2 / hardware_param.cacheB_l2
    max_cal = np.prod(tiling_key.mckcnc) * performance.core_loop_num * hardware_param.core_num
    performance.util_core = np.prod(tiling_key.MKN) / max_cal
    UnitTimeCalculator(hardware_param, tiling_key, performance).update_performance()
    return performance
