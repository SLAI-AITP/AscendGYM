# -*- coding: utf-8 -*-
# linziwei l00624134
import copy
import numpy as np
import pandas as pd
from loguru import logger


class HardwareParam:
    def __init__(self, param_dict):
        self.name = ''
        self.core_num = 0  # 个
        self.cacheB_l0abc = [0, 0, 0]  # Byte
        self.cacheB_l1 = 0  # Byte
        self.cacheB_l2 = 0  # Byte
        self.clock_speed = 0  # GHz
        self.bandwidth = {'mac': 0.0}  # Byte/Cycle
        self.fix_time = {'mac': 0.0, 'core': 0.0}  # Cycle
        self.tscale = {'fixpipe_not_power2': 0.0}
        self.scalar_time = {'l0num': 0.0, 'l0cnum': 0.0, 'l1': 0.0}  # Cycle/num
        for key, value in param_dict.items():
            setattr(self, key, value)

    def get_time_scaled_hp(self, time_scale=1e-3):
        new_one = copy.deepcopy(self)
        new_one.clock_speed *= time_scale
        for key, value in self.bandwidth.items():
            new_one.bandwidth[key] = value / time_scale
        for key, value in self.fix_time.items():
            new_one.fix_time[key] = value * time_scale
        for key, value in self.scalar_time.items():
            new_one.scalar_time[key] = value * time_scale
        return new_one

    def l2_sufficient(self, mkn, dataB):
        m, k, n = mkn
        return (m + n) * k * dataB[0] <= self.cacheB_l2

    def cycle2us(self, cycle):
        return cycle / self.clock_speed / 1000

    def us2cycle(self, us):
        return us * self.clock_speed * 1000


class TilingKey:
    def __init__(self):
        self.MKN = [None, None, None]  # 矩阵尺寸， MK * KN = MN
        self.dataB = [2, 4]  # INPUT, OUTPUT单位Byte
        self.isnd_abc = [True, False, True]
        self.bias = False
        self.buffer_num_l0abc = [None, None, None]  # l0ab >= 2, l0c =1, 2
        self.reuse_l0 = True
        self.reuse_num_l0ab = [None, None, None]  # 复用缓存数
        self.buffer_num_l1ab = [None, None]  # l1ab >= 2
        self.reuse_l1 = True
        self.reuse_num_l1ab = [None, None]  # 复用缓存数
        self.orders_l0l1l2 = [None, None, None]  # 0: L0遍历顺序， KMN, KNM, MKN, NKM, MNK, NMK, 1: L1遍历顺序， KMN, KNM, 2: L2遍历顺序，NM, MN
        self.mckcnc = [None, None, None]  # 单核数据尺寸
        self.mckcnc_num = [None, None, None]  # 核数
        self.m0k0n0 = [None, None, None]  # L0 tiling 尺寸
        self.m1k1n1_tilde = [None, None, None]  # L1 cache 尺寸，单位为L0 tiling
        self.m2n2_tilde = [None, None]  # swizzle count
        self.core_num = None
        self.core_loop = None

    @staticmethod
    def ceil(value1, value2):
        return np.ceil(np.array(value1) / np.array(value2)).astype(int)

    def _attr_valid(self, attr):
        if not hasattr(self, attr):
            return False
        value = getattr(self, attr)
        return value is not None and not (None in value)

    def update(self, is_orig=True):
        def length2area(lens):
            return np.array([lens[0] * lens[1], lens[1] * lens[2], lens[2] * lens[0]]).astype(int)

        for attr in ['buffer_num_l0abc', 'buffer_num_l1ab', 'reuse_num_l0ab', 'reuse_num_l1ab',
                     'MKN', 'mckcnc_num', 'mckcnc', 'm0k0n0', 'm1k1n1_tilde', 'm2n2_tilde']:
            if self._attr_valid(attr):
                setattr(self, attr, np.array(getattr(self, attr), dtype=int))
        for attr in ['core_loop', 'core_num']:
            if hasattr(self, attr) and getattr(self, attr) is not None:
                setattr(self, attr, int(getattr(self, attr)))
        if self._attr_valid('mckcnc'):
            self.mckcnc_num = self.ceil(self.MKN, self.mckcnc)
        elif self._attr_valid('mckcnc_num'):
            if not is_orig:
                self.mckcnc = np.ceil(np.array(self.MKN) / np.array(self.mckcnc_num)).astype(int)
            else:
                scale = np.array([16, 32 / self.dataB[0], 16])
                self.mckcnc = (np.ceil(np.array(self.MKN) / np.array(self.mckcnc_num) / scale) * scale).astype(int)
        else:
            return
        if self._attr_valid('mckcnc_num'):
            if self.core_num is None and self.core_loop is not None:
                self.core_num = np.ceil(np.prod(self.mckcnc_num) / self.core_loop).astype(int)
            elif self.core_num is not None and self.core_loop is None:
                self.core_loop = np.ceil(np.prod(self.mckcnc_num) / self.core_num).astype(int)
        else:
            return
        if not self._attr_valid('m0k0n0') or not self._attr_valid('m1k1n1_tilde'):
            return
        self.m0k0n0_num = self.ceil(self.mckcnc, self.m0k0n0)
        self.m0k0n0_tail = np.mod(self.mckcnc, self.m0k0n0)
        self.a0b0c0 = length2area(self.m0k0n0)
        self.a0b0c0_num = length2area(self.m0k0n0_num)
        self.m1k1n1 = self.m0k0n0 * self.m1k1n1_tilde
        self.m1k1n1_num = self.ceil(self.m0k0n0_num, self.m1k1n1_tilde)
        self.m1k1n1_tail = np.mod(self.mckcnc, self.m1k1n1)
        self.a1b1c1_tilde = length2area(self.m1k1n1_tilde)
        self.a1b1c1 = length2area(self.m1k1n1)
        self.a1b1c1_align = copy.copy(self.a1b1c1)
        if self.isnd_abc[0]:
            self.a1b1c1_align[0] = self.m1k1n1[0] * np.ceil(self.m1k1n1[1] * self.dataB[0] / 512) * 512 / self.dataB[0]
        if self.isnd_abc[1]:
            self.a1b1c1_align[1] = self.m1k1n1[2] * np.ceil(self.m1k1n1[1] * self.dataB[0] / 512) * 512 / self.dataB[0]
        if self.isnd_abc[2]:
            self.a1b1c1_align[2] = self.m1k1n1[0] * np.ceil(self.m1k1n1[2] * self.dataB[1] / 512) * 512 / self.dataB[1]
        self.mckcnc_tail = np.mod(self.MKN, self.mckcnc)
        self.a1b1c1_num = length2area(self.m1k1n1_num)
        self.acbccc = length2area(self.mckcnc)
        self.acbccc_num = length2area(self.mckcnc_num)
        self.m2n2_tilde = np.minimum(self.m1k1n1_num[[0, 2]], self.m2n2_tilde) if self._attr_valid('m2n2_tilde') else self.m1k1n1_num[[0, 2]]
        self.m2n2 = self.m1k1n1[[0, 2]] * self.m2n2_tilde
        self.m2n2_num = np.ceil(self.mckcnc[[0, 2]]/ self.m2n2).astype(int)
        # 计算顺序
        if self.orders_l0l1l2[0] is None:
            self.orders_l0l1l2[0] = 'KMN'
        if self.orders_l0l1l2[1] is None:
            self.orders_l0l1l2[1] = 'KMN'
        if self.orders_l0l1l2[2] is None:
            self.orders_l0l1l2[2] = 'MN'
        # 内存统计
        if not self._attr_valid('buffer_num_l0abc') or not self._attr_valid('buffer_num_l1ab'):
            return
        self.cache_l0abc = copy.copy(self.a0b0c0)
        self.cache_l0abc[-1] *= self._get_l0c_temp_num()
        self.cacheB_l0abc = self.cache_l0abc * self.buffer_num_l0abc
        self.cacheB_l0abc[:2] = self.cacheB_l0abc[:2] * self.dataB[0]
        self.cacheB_l0abc[2] = self.cacheB_l0abc[2] * self.dataB[1]
        self.cacheB_l1 = np.sum(self.a1b1c1[:2] * self.dataB[0] * self.buffer_num_l1ab)
        if self.bias:
            self.cacheB_l1 += self.m1k1n1[2] * self.dataB[0] * self.buffer_num_l1ab[1]
        if self._attr_valid('m2n2'):
            self.cacheB_l2 = sum(np.minimum(self.m2n2 * self.mckcnc_num[[0, 2]], self.MKN[[0, 2]])) * self.MKN[1] * self.dataB[0]
        else:
            self.cacheB_l2 = 0

    def scale_tiling_key(self, mkn_scale, is_orig):
        scaled_one = copy.deepcopy(self)
        mkn_scale = np.array(mkn_scale)
        for attr in ['MKN', 'mckcnc', 'm0k0n0']:
            if hasattr(self, attr):
                orig = getattr(self, attr)
                new_values = [None if (None in [v, s]) else int(np.ceil(v * s)) for v, s in zip(orig, mkn_scale)]
                setattr(scaled_one, attr, new_values)
        scaled_one.update(is_orig)
        return scaled_one

    def feasibility_check(self, hardware_param: HardwareParam=None):
        self.update()
        flag = True
        for attr1, attr2 in [('mckcnc', 'MKN'), ('m0k0n0', 'mckcnc'), ('m0k0n0', 'm1k1n1'),
                             ('m1k1n1_tilde', 'm0k0n0_num')]:
            if np.any(getattr(self, attr1) > getattr(self, attr2)):
                logger.warning(f"warning: {attr1} > {attr2}")
                flag = False
        if self.reuse_l0 and self.orders_l0l1l2[0] not in ['KNM', 'KMN', 'NKM', 'MKN', 'NMK', 'MNK']:
            logger.warning("warning: wrong L0 loop order")
            flag = False
        if self.reuse_l1 and self.orders_l0l1l2[1] not in ['KNM', 'KMN']:
            logger.warning("warning: wrong L1 loop order")
            flag = False
        if hardware_param is None:
            logger.warning("warning: hardware parameter is None, cache feasibility cannot be checked")
            return flag
        if self._attr_valid('cacheB_l0abc') and (self.cacheB_l0abc > np.array(hardware_param.cacheB_l0abc)).any():
            logger.warning("warning: L0 memory overflow")
            flag = False
        if self._attr_valid('cacheB_l1ab') and self.cacheB_l1 > hardware_param.cacheB_l1:
            logger.warning("warning: L1 memory overflow")
            flag = False
        return flag

    def _get_l0c_temp_num(self):
        m1_tilde, _, n1_tilde = self.m1k1n1_tilde
        order_l0 = self.orders_l0l1l2[0]
        if self.m1k1n1_num[1] >= 2 or (self.reuse_l0 and order_l0 in ['MNK', 'NMK']):
            return n1_tilde * m1_tilde
        if not self.reuse_l0 or order_l0 in ['KMN', 'KNM']:
            return 1
        if order_l0 == 'NKM':
            return n1_tilde
        if order_l0 == 'MKN':
            return m1_tilde
        return None

    def get_details(self):
        text = '\n========TILING KEY========'
        for key, value in self.__dict__.items():
            text += f"\n{key.ljust(16)}: {value.tolist() if isinstance(value, np.ndarray) else value}"
        return text

    def __str__(self):
        key_info = ['MKN', 'dataB', 'isnd_abc', 'bias',
                    'buffer_num_l0abc', 'reuse_l0', 'reuse_num_l0ab',
                    'buffer_num_l1ab', 'reuse_l1', 'reuse_num_l1ab', 'orders_l0l1l2',
                    'mckcnc_num', 'mckcnc', 'm0k0n0', 'm1k1n1_tilde', 'm1k1n1_num', 'm2n2_tilde', 'm2n2_num',
                    'core_num', 'core_loop']
        names = ['MKN', 'dataB', 'isnd_abc', 'bias',
                 'buffer_num_l0abc', 'reuse_l0', 'reuse_num_l0ab',
                 'buffer_num_l1ab', 'reuse_l1', 'reuse_num_l1ab','orders_l0l1l2',
                 'mckcnc_num', 'mckcnc', 'm0k0n0', 'm1k1n1_tilde', 'm1k1n1_num', 'm2n2_tilde', 'm2n2_num',
                 'core_num', 'core_loop']
        text = '\n========TILING KEY========'
        for key, name in zip(key_info, names):
            if key == 'MKN':
                text += '\n----problem info----'
            if key == 'buffer_num_l0abc':
                text += '\n----loop and cache info----'
            if key == 'mckcnc_num':
                text += '\n----tiling info----'
            if key == 'core_num':
                text += '\n----core info----'
            value = self.__dict__[key]
            text += f"\n{name.ljust(16)}: {value.tolist() if isinstance(value, np.ndarray) else value}"
        return text


class UnitTime:
    def __init__(self):
        self.index_list = ['mte2pre', 'mte2gm', 'mte2', 'mte1', 'mac', 'fixpipe', 'scalar', 'mte2pre_a', 'mte2pre_b', 'mte2gm_a', 'mte2gm_b', 'mte2a', 'mte2b', 'mte2bias', 'mte1a', 'mte1b']
        self.columns_list = ['count', 'num', 'dataB', 'time(cycle)', 'time(us)', 'util', 'l0us', 'l1us']
        self.data = pd.DataFrame(0.0, index=self.index_list, columns=self.columns_list, dtype=float)
        self.inner_us = {'final': 0.0, 'mte1': 0.0, 'mac': 0.0, 'fixpipe': 0.0}
        self.total_us = {'final': 0.0, 'mte2': 0.0, 'mte1': 0.0, 'mac': 0.0, 'fixpipe': 0.0}

    def __str__(self):
        spaces = [6, 12, 8, 12, 10, 8, 8, 8]
        names = ['count', 'num', 'dataMB', 'time(cycle)', 'time(us)', 'util', 'l0us', 'l1us']
        decs = [0, 0, 2, 0, 2, 1, 2, 2]
        text = '\n----unit info----\n'
        text += f"{'unit'.ljust(10)}: "  + ', '.join([f'{name.rjust(space)}' for name, space in zip(names, spaces)])
        for i, row in self.data.iterrows():
            if i == 'mte2pre_a':
                text += "\n----"
            text += f"\n{i.ljust(10)}: "
            for name, space, dec in zip(names, spaces, decs):
                if name == 'dataMB':
                    text += f"{row['dataB'] / 1024 / 1024:{space}.{dec}f}, "
                elif name == 'util':
                    text += f"{row['util'] * 100:{space - 1}.{dec}f}%, "
                else:
                    text += f'{row[name]:{space}.{dec}f}, '
        text += '\n---\ninner_us: '
        for u, v in self.inner_us.items():
            text += f'{u}: {v:8.2f},  '
        text += '\ntotal_us: '
        for u, v in self.total_us.items():
            text += f'{u}: {v:8.2f},  '
        return text


class TilingPerformance:
    def __init__(self):
        self.unit_time = UnitTime()
        self.core_loop_num = 0
        self.util_cache_l0abc = [0, 0, 0]
        self.util_cache_l1 = 0
        self.util_cache_l2 = 0
        self.util_core = 0
        self.obj = 0

    def __str__(self):
        text = '\n========PERFORMANCE========'
        cache_l0abc = ', '.join([f'{v * 100:.1f}%' for v in self.util_cache_l0abc])
        text += f'\nutilization_cache_l0abc: [{cache_l0abc}]'
        text += f'\nutilization_cache_l1   : {self.util_cache_l1 * 100:.1f}%'
        text += f'\nutilization_cache_l2   : {self.util_cache_l2 * 100:.1f}%'
        text += f'\nutilization_core       : {self.util_core * 100:.1f}%'
        text += f'\ncore_loop_num          : {self.core_loop_num:.0f}'
        text += f'\nobj                    : {self.obj:.2f}'
        text += self.unit_time.__str__()
        return text


class WhiteList:
    def __init__(self, hardware_name, K, N, wl):
        self.hardware_name = hardware_name
        self.K = K
        self.N = N
        self.wl = wl

    def __str__(self):
        text = f'{self.K}_{self.N}_{self.hardware_name}:\n'
        for val in self.wl:
            text += f"{val}\n"
        return text

    def write(self, file_name='white_list.txt'):
        wl_name = f'{self.K}_{self.N}_{self.hardware_name}'
        with open(file_name, 'a', encoding='utf-8') as f:
            f.write(f'\n    static std::vector<CustomTingParams> WHITE_LIST_{wl_name} = {{\n')
            for val in self.wl:
                f.write(f"            {{{val}}},\n")  # 记得加 \n 换行符
            f.write('    };\n')
        logger.info(f'white list {wl_name} is written in {file_name}.')
