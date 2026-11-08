"""
验证工具类
提供各种参数验证功能
"""


def validate_required_fields(data, required_fields):
    """
    验证必需字段是否存在

    参数:
        data: 要验证的数据字典
        required_fields: 必需字段列表

    返回:
        bool: 是否所有必需字段都存在
    """
    if not isinstance(data, dict):
        return False

    for field in required_fields:
        if field not in data or data[field] is None:
            return False

    return True


def validate_code_format(code):
    """
    验证代码格式

    参数:
        code: 代码字符串

    返回:
        bool: 代码格式是否有效
    """
    if not code or not isinstance(code, str):
        return False

    # 基本验证：代码不能为空
    if len(code.strip()) == 0:
        return False

    # TODO: 添加更多代码格式验证逻辑
    # - 检查是否包含基本的C++语法结构
    # - 检查是否有明显的语法错误

    return True


def validate_shape(shape):
    """
    验证张量形状格式

    参数:
        shape: 形状（可以是列表、元组或字符串）

    返回:
        bool: 形状格式是否有效
    """
    if isinstance(shape, (list, tuple)):
        # 检查是否所有元素都是正整数
        return all(isinstance(dim, int) and dim > 0 for dim in shape)
    elif isinstance(shape, str):
        # 支持字符串格式，如 "1,3,224,224"
        try:
            dims = [int(d.strip()) for d in shape.split(',')]
            return all(dim > 0 for dim in dims)
        except:
            return False
    return False


def validate_dtype(dtype):
    """
    验证数据类型

    参数:
        dtype: 数据类型字符串

    返回:
        bool: 数据类型是否支持
    """
    supported_dtypes = [
        'float16', 'float32', 'float64',
        'int8', 'int16', 'int32', 'int64',
        'uint8', 'uint16', 'uint32', 'uint64',
        'bool'
    ]
    return dtype in supported_dtypes


def validate_json_format(data):
    """
    验证JSON格式

    参数:
        data: 要验证的数据

    返回:
        bool: 是否为有效的JSON格式
    """
    import json

    if isinstance(data, dict):
        return True

    if isinstance(data, str):
        try:
            json.loads(data)
            return True
        except:
            return False

    return False


def validate_operator_name(name):
    """
    验证算子名称

    参数:
        name: 算子名称

    返回:
        bool: 名称是否有效
    """
    if not name or not isinstance(name, str):
        return False

    # 算子名称应该只包含字母、数字和下划线
    import re
    pattern = r'^[a-zA-Z_][a-zA-Z0-9_]*$'
    return bool(re.match(pattern, name))


def validate_test_cases(test_cases):
    """
    验证测试用例格式

    参数:
        test_cases: 测试用例列表

    返回:
        bool: 测试用例格式是否有效
    """
    if not isinstance(test_cases, list):
        return False

    if len(test_cases) == 0:
        return False

    # 每个测试用例应该是一个字典
    for test_case in test_cases:
        if not isinstance(test_case, dict):
            return False

        # 基本字段验证
        if 'name' not in test_case:
            return False

    return True