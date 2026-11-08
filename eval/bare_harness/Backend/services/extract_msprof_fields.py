#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
华为昇腾(Ascend) NPU 性能数据字段提取脚本
纯粹将CSV中的字段搬运出来，不做任何主观判断

参考文档: https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/devaids/Profiling/atlasprofiling_16_0067.html

支持的CSV文件（对应文档中的10个表格）:
- 表1: 公共字段说明 -> OpBasicInfo.csv
- 表2: 字段说明（PipeUtilization） -> PipeUtilization.csv
- 表3: 字段说明（ArithmeticUtilization） -> ArithmeticUtilization.csv
- 表4: 字段说明（Memory） -> Memory.csv
- 表5: 字段说明（MemoryL0） -> MemoryL0.csv
- 表6: 字段说明（MemoryUB） -> MemoryUB.csv
- 表7: 字段说明（ResourceConflictRatio） -> ResourceConflictRatio.csv
- 表8: 字段说明（MemoryAccess） -> MemoryAccess.csv
- 表9: 字段说明（L2Cache） -> L2Cache.csv
- 表10: 字段说明（PipelineExecuteUtilization） -> PipelineExecuteUtilization.csv
"""

import pandas as pd
import json
import os
import warnings
from pathlib import Path
from typing import Dict, Any, Optional


class MSProfFieldExtractor:
    """MSProf性能数据字段提取器"""

    # CSV文件映射：文件名 -> 数据类型键
    CSV_FILES = {
        'op_basic': 'OpBasicInfo.csv',
        'pipe_util': 'PipeUtilization.csv',
        'arith_util': 'ArithmeticUtilization.csv',
        'memory': 'Memory.csv',
        'memory_l0': 'MemoryL0.csv',
        'memory_ub': 'MemoryUB.csv',
        'l2_cache': 'L2Cache.csv',
        'resource': 'ResourceConflictRatio.csv',
        'memory_access': 'MemoryAccess.csv',
        'pipeline_execute': 'PipelineExecuteUtilization.csv',
    }

    # 显示名称映射（用于输出）
    DISPLAY_NAMES = {
        'op_basic': 'OpBasicInfo（算子基本信息）',
        'pipe_util': 'PipeUtilization（Pipeline利用率）',
        'arith_util': 'ArithmeticUtilization（算术利用率）',
        'memory': 'Memory（内存带宽）',
        'memory_l0': 'MemoryL0（L0内存）',
        'memory_ub': 'MemoryUB（UB内存）',
        'l2_cache': 'L2Cache（L2缓存）',
        'resource': 'ResourceConflictRatio（资源冲突）',
        'memory_access': 'MemoryAccess（内存访问量）',
        'pipeline_execute': 'PipelineExecuteUtilization（Pipeline执行利用率）',
    }

    # 字段说明映射（参考官方文档）
    # key: CSV文件类型, value: {字段名: 字段说明}
    # 字段名使用小写下划线格式，以便匹配CSV列名转换后的结果
    FIELD_DESCRIPTIONS = {
        'op_basic': {
            'device_id': '设备ID',
            'op_name': '算子名称',
            'op_type': '算子类型',
            'task_duration': 'Task耗时，包含调度到加速器的时间、加速器上的执行时间以及结束响应时间，单位us',
            'block_dim': 'Task运行切分数量，对应Task运行时核数',
            'mix_block_dim': '从加速器的Block Dim',
            'pid': '进程ID',
            'current_freq': '当前频率',
            'rated_freq': '额定频率',
        },
        'pipe_util': {
            'block_id': 'Block ID',
            'sub_block_id': '子Block ID',
            'aic_time': 'AI Cube Core耗时，单位us',
            'aic_total_cycles': 'AI Cube Core执行的cycle总数',
            'aic_cube_time': 'Cube类型指令耗时，单位us',
            'aic_cube_ratio': 'Cube类型指令的cycle数在total cycle数中的占用比',
            'aic_scalar_time': 'Scalar类型指令耗时，单位us',
            'aic_mte1_time': 'MTE1类型指令（L1->L0A/L0B搬运）耗时，单位us',
            'aic_mte1_ratio': 'MTE1类型指令的cycle数在total cycle数中的占用比',
            'aic_mte2_time': 'MTE2类型指令（DDR->AICORE搬运）耗时，单位us',
            'aic_mte2_ratio': 'MTE2类型指令的cycle数在total cycle数中的占用比',
            'aic_mte3_time': 'MTE3类型指令（AICORE->DDR搬运）耗时，单位us',
            'aic_mte3_ratio': 'MTE3类型指令的cycle数在total cycle数中的占用比',
            'aic_mte3_active_bw': 'MTE3激活带宽，单位GB/s',
            'aic_fixpipe_time': 'Fixpipe类型指令耗时，单位us',
            'aic_fixpipe_active_bw': 'Fixpipe激活带宽，单位GB/s',
            'aic_icache_miss_rate': '指令缓存缺失率',
            'aiv_time': 'AI Vector Core耗时，单位us',
            'aiv_total_cycles': 'AI Vector Core执行的cycle总数',
            'aiv_vec_time': 'Vector类型指令耗时，单位us',
            'aiv_vec_ratio': 'Vector类型指令的cycle数在total cycle数中的占用比',
            'aiv_scalar_time': 'Scalar类型指令耗时，单位us',
            'aiv_scalar_ratio': 'Scalar类型指令的cycle数在total cycle数中的占用比',
            'aiv_mte2_time': 'MTE2类型指令耗时，单位us',
            'aiv_mte2_ratio': 'MTE2类型指令的cycle数在total cycle数中的占用比',
            'aiv_mte2_active_bw': 'MTE2激活带宽，单位GB/s',
            'aiv_mte3_time': 'MTE3类型指令耗时，单位us',
            'aiv_mte3_ratio': 'MTE3类型指令的cycle数在total cycle数中的占用比',
            'aiv_mte3_active_bw': 'MTE3激活带宽，单位GB/s',
            'aiv_icache_miss_rate': '指令缓存缺失率',
        },
        'arith_util': {
            'block_id': 'Block ID',
            'sub_block_id': '子Block ID',
            'aic_time': 'AI Cube Core耗时，单位us',
            'aic_total_cycles': 'AI Cube Core执行的cycle总数',
            'aic_cube_ratio': 'Cube类型指令的cycle数在total cycle数中的占用比',
            'aic_cube_fp16_ratio': 'Cube FP16类型指令的cycle数在total cycle数中的占用比',
            'aic_cube_int8_ratio': 'Cube INT8类型指令的cycle数在total cycle数中的占用比',
            'aic_cube_fops': 'Cube类型的浮点运算数（FLOPs）',
            'aic_cube_total_instr_number': 'Cube指令总数',
            'aic_cube_fp_instr_number': 'Cube FP指令数',
            'aic_cube_int_instr_number': 'Cube INT指令数',
            'aiv_time': 'AI Vector Core耗时，单位us',
            'aiv_total_cycles': 'AI Vector Core执行的cycle总数',
            'aiv_vec_ratio': 'Vector类型指令的cycle数在total cycle数中的占用比',
            'aiv_vec_fp32_ratio': 'Vector FP32类型指令的cycle数在total cycle数中的占用比',
            'aiv_vec_fp16_ratio': 'Vector FP16类型指令的cycle数在total cycle数中的占用比',
            'aiv_vec_int32_ratio': 'Vector INT32类型指令的cycle数在total cycle数中的占用比',
            'aiv_vec_int16_ratio': 'Vector INT16类型指令的cycle数在total cycle数中的占用比',
            'aiv_vec_misc_ratio': 'Vector Misc类型指令的cycle数在total cycle数中的占用比',
            'aiv_vec_fops': 'Vector类型的浮点运算数（FLOPs）',
        },
        'memory': {
            'block_id': 'Block ID',
            'sub_block_id': '子Block ID',
            'aic_time': 'AI Cube Core耗时，单位us',
            'aic_total_cycles': 'AI Cube Core执行的cycle总数',
            'aic_l1_read_bw': 'L1读带宽速率，单位GB/s',
            'aic_l1_write_bw': 'L1写带宽速率，单位GB/s',
            'aic_main_mem_read_bw': '主存储器读带宽速率，单位GB/s',
            'aic_main_mem_write_bw': '主存储器写带宽速率，单位GB/s',
            'aic_mte1_instructions': 'MTE1指令数',
            'aic_mte1_ratio': 'MTE1类型指令的cycle数在total cycle数中的占用比',
            'aic_mte2_instructions': 'MTE2指令数',
            'aic_mte2_ratio': 'MTE2类型指令的cycle数在total cycle数中的占用比',
            'aic_mte3_instructions': 'MTE3指令数',
            'aic_mte3_ratio': 'MTE3类型指令的cycle数在total cycle数中的占用比',
            'aiv_time': 'AI Vector Core耗时，单位us',
            'aiv_total_cycles': 'AI Vector Core执行的cycle总数',
            'aiv_ub_to_gm_bw': 'UB到GM带宽速率，单位GB/s',
            'aiv_gm_to_ub_bw': 'GM到UB带宽速率，单位GB/s',
            'aiv_main_mem_read_bw': '主存储器读带宽速率，单位GB/s',
            'aiv_main_mem_write_bw': '主存储器写带宽速率，单位GB/s',
            'aiv_mte2_instructions': 'MTE2指令数',
            'aiv_mte2_ratio': 'MTE2类型指令的cycle数在total cycle数中的占用比',
            'aiv_mte3_instructions': 'MTE3指令数',
            'aiv_mte3_ratio': 'MTE3类型指令的cycle数在total cycle数中的占用比',
            'read_main_memory_datas': '对片上内存读的数据量，单位KB',
            'write_main_memory_datas': '对片上内存写的数据量，单位KB',
            'gm_to_l1_datas': 'GM到L1的数据搬运量，单位KB',
            'gm_to_l1_bw_usage_rate': 'GM到L1带宽使用率',
            'l1_to_gm_datas': 'L1到GM的数据搬运量，单位KB（估计值）',
            'l1_to_gm_bw_usage_rate': 'L1到GM带宽使用率（估计值）',
            'l0c_to_l1_datas': 'L0C到L1的数据搬运量，单位KB',
            'l0c_to_l1_bw_usage_rate': 'L0C到L1带宽使用率',
            'l0c_to_gm_datas': 'L0C到GM的数据搬运量，单位KB',
            'l0c_to_gm_bw_usage_rate': 'L0C到GM带宽使用率',
            'gm_to_ub_datas': 'GM到UB的数据搬运量，单位KB',
            'gm_to_ub_bw_usage_rate': 'GM到UB带宽使用率',
            'ub_to_gm_datas': 'UB到GM的数据搬运量，单位KB',
            'ub_to_gm_bw_usage_rate': 'UB到GM带宽使用率',
        },
        'memory_l0': {
            'block_id': 'Block ID',
            'sub_block_id': '子Block ID',
            'aic_time': 'AI Cube Core耗时，单位us',
            'aic_total_cycles': 'AI Cube Core执行的cycle总数',
            'aic_l0a_read_bw': 'L0A读带宽速率，单位GB/s',
            'aic_l0a_write_bw': 'L0A写带宽速率，单位GB/s',
            'aic_l0b_read_bw': 'L0B读带宽速率，单位GB/s',
            'aic_l0b_write_bw': 'L0B写带宽速率，单位GB/s',
            'aic_l0c_read_bw_cube': 'Cube从L0C读带宽速率，单位GB/s',
            'aic_l0c_write_bw_cube': 'Cube向L0C写带宽速率，单位GB/s',
            'aiv_time': 'AI Vector Core耗时，单位us',
            'aiv_total_cycles': 'AI Vector Core执行的cycle总数',
        },
        'memory_ub': {
            'block_id': 'Block ID',
            'sub_block_id': '子Block ID',
            'aic_time': 'AI Cube Core耗时，单位us',
            'aic_total_cycles': 'AI Cube Core执行的cycle总数',
            'aiv_time': 'AI Vector Core耗时，单位us',
            'aiv_total_cycles': 'AI Vector Core执行的cycle总数',
            'aiv_ub_read_bw_vector': 'Vector从UB读带宽速率，单位GB/s',
            'aiv_ub_write_bw_vector': 'Vector向UB写带宽速率，单位GB/s',
            'aiv_ub_read_bw_scalar': 'Scalar从UB读带宽速率，单位GB/s',
            'aiv_ub_write_bw_scalar': 'Scalar向UB写带宽速率，单位GB/s',
        },
        'l2_cache': {
            'block_id': 'Block ID',
            'sub_block_id': '子Block ID',
            'aic_time': 'AI Cube Core耗时，单位us',
            'aic_total_cycles': 'AI Cube Core执行的cycle总数',
            'aic_write_cache_hit': '写缓存命中次数',
            'aic_write_cache_miss_allocate': '写缓存缺失后重新分配的次数',
            'aic_r0_read_cache_hit': '读R0通道缓存命中次数',
            'aic_r0_read_cache_miss_allocate': '读R0通道缓存缺失后重新分配的次数',
            'aic_r1_read_cache_hit': '读R1通道缓存命中次数',
            'aic_r1_read_cache_miss_allocate': '读R1通道缓存缺失后重新分配的次数',
            'aic_write_hit_rate': '写缓存命中率',
            'aic_read_hit_rate': '读缓存命中率',
            'aic_total_hit_rate': '总缓存命中率',
            'aiv_time': 'AI Vector Core耗时，单位us',
            'aiv_total_cycles': 'AI Vector Core执行的cycle总数',
            'aiv_write_cache_hit': '写缓存命中次数',
            'aiv_write_cache_miss_allocate': '写缓存缺失后重新分配的次数',
            'aiv_r0_read_cache_hit': '读R0通道缓存命中次数',
            'aiv_r0_read_cache_miss_allocate': '读R0通道缓存缺失后重新分配的次数',
            'aiv_r1_read_cache_hit': '读R1通道缓存命中次数',
            'aiv_r1_read_cache_miss_allocate': '读R1通道缓存缺失后重新分配的次数',
            'aiv_write_hit_rate': '写缓存命中率',
            'aiv_read_hit_rate': '读缓存命中率',
            'aiv_total_hit_rate': '总缓存命中率',
        },
        'resource': {
            'block_id': 'Block ID',
            'sub_block_id': '子Block ID',
            'aic_time': 'AI Cube Core耗时，单位us',
            'aic_total_cycles': 'AI Cube Core执行的cycle总数',
            'aic_cube_wait_ratio': 'Cube等待比例',
            'aic_mte1_wait_ratio': 'MTE1等待比例',
            'aic_mte2_wait_ratio': 'MTE2等待比例',
            'aic_mte3_wait_ratio': 'MTE3等待比例',
            'aiv_time': 'AI Vector Core耗时，单位us',
            'aiv_total_cycles': 'AI Vector Core执行的cycle总数',
            'aiv_vec_total_cflt_ratio': 'Vector总冲突比例',
            'aiv_vec_bankgroup_cflt_ratio': 'Vector Bank Group冲突比例',
            'aiv_vec_bank_cflt_ratio': 'Vector Bank冲突比例',
            'aiv_vec_resc_cflt_ratio': 'Vector Resource冲突比例',
            'aiv_vec_mte_cflt_ratio': 'Vector MTE冲突比例',
            'aiv_vec_wait_ratio': 'Vector等待比例',
            'aiv_mte1_wait_ratio': 'MTE1等待比例',
            'aiv_mte2_wait_ratio': 'MTE2等待比例',
            'aiv_mte3_wait_ratio': 'MTE3等待比例',
        },
    }

    def load_available_csv_files(self, root_path: str) -> Dict[str, Optional[pd.DataFrame]]:
        """
        加载所有存在的CSV文件，跳过不存在的文件

        Args:
            root_path: MSProf数据根目录路径

        Returns:
            Dict: 包含所有CSV文件数据的字典，不存在的文件值为None
        """
        data_files = {}

        for key, filename in self.CSV_FILES.items():
            file_path = os.path.join(root_path, filename)
            if os.path.exists(file_path):
                try:
                    data_files[key] = pd.read_csv(file_path)
                except Exception as e:
                    warnings.warn(f"无法读取文件 {filename}: {e}")
                    data_files[key] = None
            else:
                data_files[key] = None

        return data_files

    def extract_fields_from_dataframe(self, df: pd.DataFrame, data_type: str) -> Optional[Dict[str, str]]:
        """
        从DataFrame中提取所有字段，并添加字段说明

        Args:
            df: pandas DataFrame
            data_type: 数据类型键（用于查找字段说明）

        Returns:
            Dict: 包含所有字段和值的字典，如果df为None或空则返回None
        """
        if df is None or df.empty:
            return None

        # 获取第一行数据（通常只有一行）
        row = df.iloc[0]

        fields = {}
        descriptions = self.FIELD_DESCRIPTIONS.get(data_type, {})

        # 提取所有列
        for col in df.columns:
            value = row.get(col, 'N/A')
            # 处理NaN值
            if pd.isna(value):
                value = 'N/A'
            else:
                value = str(value)

            # 获取原始字段名（去掉括号部分）
            original_field_name = self._extract_field_name(col)

            # 获取字段说明
            description = descriptions.get(original_field_name, original_field_name)

            # 格式化为：字段名(说明)
            key = f"{original_field_name}({description})"
            fields[key] = value

        return fields

    def _extract_field_name(self, col_name: str) -> str:
        """
        从列名中提取原始字段名
        例如：'Op Name(Op Name)' -> 'op_name'
              'aic_time(us)(aic_time(us))' -> 'aic_time'
        """
        # 先去掉括号及后面的内容（第一层括号）
        if '(' in col_name:
            field_name = col_name.split('(')[0].strip()
        else:
            field_name = col_name

        # 转换为小写并替换空格为下划线，用于匹配FIELD_DESCRIPTIONS中的键
        # 例如：'Op Name' -> 'op_name'
        field_name = field_name.lower().replace(' ', '_')

        # 去掉单位后缀
        for suffix in ['_us', '_%_', '_gb/s_', '_kb_', '(us)', '(%)', '(gb/s)', '(kb)', '_active_bw', '_bw']:
            if field_name.endswith(suffix.lower()):
                field_name = field_name[:-len(suffix)]

        return field_name

    def extract_all_fields(self, root_path: str) -> Dict[str, Any]:
        """
        提取所有CSV文件中的所有字段

        Args:
            root_path: MSProf数据根目录路径

        Returns:
            Dict: 包含所有字段的数据字典，只包含已存在的文件
        """
        data_files = self.load_available_csv_files(root_path)

        result = {}

        for key, filename in self.CSV_FILES.items():
            df = data_files.get(key)
            if df is not None and not df.empty:
                display_name = self.DISPLAY_NAMES.get(key, key)
                result[display_name] = self.extract_fields_from_dataframe(df, key)

        return result

    def save_to_json(self, root_path: str, output_path: str):
        """
        将提取的字段保存为JSON文件

        Args:
            root_path: MSProf数据根目录路径
            output_path: 输出文件路径
        """
        data = self.extract_all_fields(root_path)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


# def main():
#     """主函数"""
#     import sys

#     if len(sys.argv) != 3:
#         print("用法: python extract_msprof_fields.py <profiling_data_path> <output_json_path>")
#         print("\n示例:")
#         print("  python extract_msprof_fields.py D:\\profiling_results_NewGeluCustom output.json")
#         print("\n支持的CSV文件（对应官方文档10个表格）:")
#         print("  1. OpBasicInfo.csv - 算子基本信息")
#         print("  2. PipeUtilization.csv - Pipeline利用率")
#         print("  3. ArithmeticUtilization.csv - 算术利用率")
#         print("  4. Memory.csv - 内存带宽")
#         print("  5. MemoryL0.csv - L0内存")
#         print("  6. MemoryUB.csv - UB内存")
#         print("  7. L2Cache.csv - L2缓存")
#         print("  8. ResourceConflictRatio.csv - 资源冲突")
#         print("  9. MemoryAccess.csv - 内存访问量")
#         print("  10. PipelineExecuteUtilization.csv - Pipeline执行利用率")
#         sys.exit(1)

#     root_path = sys.argv[1]
#     output_path = sys.argv[2]

#     extractor = MSProfFieldExtractor()
#     extractor.save_to_json(root_path, output_path)


# if __name__ == '__main__':
#     main()


