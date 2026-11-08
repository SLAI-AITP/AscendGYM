"""
命名转换工具

提供两种驼峰→蛇形转换：
- to_snake_case:          标准蛇形，数字与字母之间插入下划线（1d 算子用）
- to_snake_case_compact:  紧凑蛇形，数字与字母之间不拆开（2d 算子用）
"""
import re


def to_snake_case(name):
    """
    标准蛇形命名转换（1d 算子使用）。

    数字与字母之间会插入下划线：
        AddCustomTest   -> add_custom_test
        AvgPool2D       -> avg_pool_2d
        Conv2DBackprop  -> conv_2d_backprop
        ABCDef          -> abc_def
        SELU            -> selu
    """
    # 在小写字母与数字之间插入下划线: Pool2 -> Pool_2
    s1 = re.sub('([a-z])([0-9])', r'\1_\2', name)
    # 在"数字+大写序列"与"大写+小写"之间插入下划线: 2DBackprop -> 2D_Backprop
    s2 = re.sub('([0-9][A-Z]+)([A-Z][a-z])', r'\1_\2', s1)
    # 在连续大写与大写+小写之间插入下划线: ABCDef -> ABC_Def
    s3 = re.sub('([A-Z]+)([A-Z][a-z])', r'\1_\2', s2)
    # 在小写字母与大写字母之间插入下划线: avgPool -> avg_Pool
    s4 = re.sub('([a-z])([A-Z])', r'\1_\2', s3)
    return s4.lower()


def to_snake_case_compact(name):
    """
    紧凑蛇形命名转换（2d 算子使用）。

    数字与字母之间不拆开，保持紧凑：
        AvgPool2D       -> avg_pool2d
        Conv2DSquare    -> conv2d_square
        SELU            -> selu
        AddCustomTest   -> add_custom_test
        HardSigmoid     -> hard_sigmoid
    """
    # 在连续大写与大写+小写之间插入下划线: ABCDef -> ABC_Def (但 SELU 这种纯大写不拆)
    s1 = re.sub('([A-Z]+)([A-Z][a-z])', r'\1_\2', name)
    # 在小写字母与大写字母之间插入下划线: avgPool -> avg_Pool
    s2 = re.sub('([a-z])([A-Z])', r'\1_\2', s1)
    # 在数字与大写+小写序列之间插入下划线: 2DSquare -> 2D_Square（但不拆 pool2d）
    s3 = re.sub('([0-9])([A-Z][a-z])', r'\1_\2', s2)
    return s3.lower()
