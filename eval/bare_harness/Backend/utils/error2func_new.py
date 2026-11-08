import re
import os

# 定义C++关键字集合，用于过滤无效函数名
CPP_KEYWORDS = {
    'if', 'else', 'for', 'while', 'switch', 'case', 'return', 'break', 'continue',
    'try', 'catch', 'throw', 'class', 'struct', 'union', 'enum', 'namespace', 'template',
    'typename', 'static', 'const', 'virtual', 'inline', 'explicit', 'friend', 'public',
    'private', 'protected', 'new', 'delete', 'operator', 'sizeof', 'typedef', 'typeid',
    'using', 'alignas', 'alignof', 'and', 'and_eq', 'asm', 'auto', 'bitand', 'bitor',
    'bool', 'char', 'char8_t', 'char16_t', 'char32_t', 'compl', 'concept', 'const_cast',
    'constexpr', 'decltype', 'default', 'do', 'double', 'dynamic_cast', 'false', 'float',
    'goto', 'int', 'long', 'mutable', 'noexcept', 'not', 'not_eq', 'nullptr', 'or', 'or_eq',
    'register', 'reinterpret_cast', 'requires', 'short', 'signed', 'static_assert',
    'static_cast', 'thread_local', 'true', 'typedef', 'typeid', 'typename', 'union',
    'unsigned', 'void', 'volatile', 'wchar_t', 'xor', 'xor_eq'
}

def find_cpp_function_name_at_line(file_content, error_line_num):
    """
    从C++源代码中查找指定行号所在的函数名
    特别处理类成员函数的识别
    
    Args:
        file_content: 文件的完整内容
        error_line_num: 出错的行号（1-based）
    
    Returns:
        函数名，如果找不到返回None
    """
    lines = file_content.split('\n')
    
    # 1. 首先验证行号是否有效
    if error_line_num < 1 or error_line_num > len(lines):
        print(f"警告：行号 {error_line_num} 超出文件范围 (1-{len(lines)})")
        return None
    
    # 2. 从错误行向上搜索，寻找函数定义
    current_line_idx = error_line_num - 1  # 转换为0-based索引
    
    # 3. 改进的函数签名识别逻辑
    # 先构建一个包含错误行附近上下文的代码段
    context_start = max(0, current_line_idx - 100)  # 向上搜索100行
    context_lines = lines[context_start:current_line_idx + 10]  # 包含错误行后10行以便识别函数结束
    
    # 尝试1：精确匹配类成员函数定义（处理多行情况）
    # 构建完整的上下文字符串
    context_text = '\n'.join(context_lines)
    
    # 模式1：匹配类成员函数定义（返回类型 类名::函数名(...) {...}）
    # 这个模式更宽松，可以处理跨多行的函数签名
    member_func_pattern = r'([\w:<>]+\s+[\w\d_:~]+\s*::\s*[\w\d_:~]+)\s*\([^)]*\)\s*(?:const\s*)?(?:->\s*[\w:<>]+)?\s*\{'
    matches = list(re.finditer(member_func_pattern, context_text, re.DOTALL))
    
    if matches:
        # 取最接近错误行的匹配
        best_match = None
        best_distance = float('inf')
        
        for match in matches:
            # 计算匹配位置到错误行的距离
            match_line_num = context_start + context_text[:match.start()].count('\n') + 1
            distance = abs(error_line_num - match_line_num)
            
            if distance < best_distance:
                best_distance = distance
                best_match = match
        
        if best_match:
            full_func_name = best_match.group(1).strip()
            # 提取函数名部分（最后一个::之后的部分）
            if '::' in full_func_name and not full_func_name.startswith('~'):
                func_name = full_func_name.split('::')[-1]
                # 验证函数名是否有效
                if func_name and func_name not in CPP_KEYWORDS and len(func_name) > 1:
                    return func_name
    
    # 模式2：匹配普通函数定义
    func_pattern = r'([\w:<>]+\s+[\w\d_:~]+)\s*\([^)]*\)\s*(?:const\s*)?(?:->\s*[\w:<>]+)?\s*\{'
    matches = list(re.finditer(func_pattern, context_text, re.DOTALL))
    
    if matches:
        # 取最接近错误行的匹配
        best_match = None
        best_distance = float('inf')
        
        for match in matches:
            # 计算匹配位置到错误行的距离
            match_line_num = context_start + context_text[:match.start()].count('\n') + 1
            distance = abs(error_line_num - match_line_num)
            
            if distance < best_distance:
                best_distance = distance
                best_match = match
        
        if best_match:
            full_func_name = best_match.group(1).strip()
            # 验证函数名是否有效
            if full_func_name and full_func_name not in CPP_KEYWORDS and len(full_func_name) > 1:
                return full_func_name
    
    # 尝试3：精确的逐行搜索，更灵活的模式
    search_limit = max(0, current_line_idx - 200)
    best_candidate = None
    best_distance = float('inf')
    
    for line_idx in range(current_line_idx, search_limit - 1, -1):
        line = lines[line_idx].strip()
        
        # 跳过空行和注释
        if not line or line.startswith('//') or line.startswith('/*') or line.startswith('*') or line.startswith('#'):
            continue
        
        # 跳过明显不是函数定义的控制结构
        control_keywords = ['if', 'else', 'for', 'while', 'switch', 'case', 'return', 'break', 'continue', 'try', 'catch']
        if any(line.startswith(kw) for kw in control_keywords):
            continue
        
        # 检查是否可能是函数定义
        # 模式1：类成员函数
        member_match = re.search(r'([\w:<>]+\s+)?([\w\d_:~]+::[\w\d_:~]+)\s*\([^)]*\)\s*(?:const\s*)?(?:->\s*[\w:<>]+)?\s*\{?', line)
        if member_match:
            full_name = member_match.group(2).strip()
            if '::' in full_name and not full_name.startswith('~'):
                func_name = full_name.split('::')[-1]
            else:
                func_name = full_name
            
            if func_name and func_name not in CPP_KEYWORDS and len(func_name) > 1:
                distance = current_line_idx - line_idx
                if distance < best_distance:
                    best_distance = distance
                    best_candidate = func_name
        
        # 模式2：普通函数
        func_match = re.search(r'([\w:<>]+\s+)?([\w\d_:~]+)\s*\([^)]*\)\s*(?:const\s*)?(?:->\s*[\w:<>]+)?\s*\{?', line)
        if func_match:
            func_name = func_match.group(2).strip()
            if func_name and func_name not in CPP_KEYWORDS and len(func_name) > 1:
                distance = current_line_idx - line_idx
                if distance < best_distance:
                    best_distance = distance
                    best_candidate = func_name
    
    if best_candidate:
        return best_candidate
    
    # 尝试4：最后的尝试 - 检查整个文件内容，寻找函数定义
    full_text = file_content
    # 查找错误行号附近的函数开始位置
    error_pos = 0
    for i in range(error_line_num - 1):
        error_pos = full_text.find('\n', error_pos) + 1
    
    # 向上搜索，寻找最近的函数开始
    search_pos = error_pos - 1
    function_start = None
    function_name = None
    
    while search_pos > 0:
        # 寻找可能的函数定义
        match = re.search(r'([\w:<>]+\s+)?([\w\d_:~]+::[\w\d_:~]+|\w+)\s*\([^)]*\)\s*(?:const\s*)?(?:->\s*[\w:<>]+)?\s*\{', 
                         full_text[max(0, search_pos-200):search_pos+200])
        if match:
            if '::' in match.group(2) and not match.group(2).startswith('~'):
                function_name = match.group(2).split('::')[-1]
            else:
                function_name = match.group(2)
            
            if function_name and function_name not in CPP_KEYWORDS and len(function_name) > 1:
                return function_name
        
        search_pos -= 1
    
    return None

def locate_cpp_error_function(error_detail):
    """
    从C++错误详情中定位出错的函数
    
    Args:
        error_detail: 包含错误信息的字符串
    
    Returns:
        (file_name, function_name) 元组，或 (None, None) 如果无法定位
    """
    # 1. 从错误信息中提取文件路径和行号
    file_match = re.search(r'([\w\d_./\\-]+\.cpp):(\d+):', error_detail)
    if not file_match:
        print("无法从错误信息中提取文件路径和行号")
        return None, None
    
    file_path = file_match.group(1)
    line_number = int(file_match.group(2))
    
    file_name = os.path.basename(file_path)
    
    # 2. 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"警告：文件不存在: {file_path}")
        # 尝试在当前目录或常见位置查找
        possible_paths = [
            file_name,
            os.path.join('op_host', file_name),
            os.path.join('src', file_name),
            os.path.join('src', 'op_host', file_name)
        ]
        
        found_path = None
        for path in possible_paths:
            if os.path.exists(path):
                found_path = path
                break
        
        if found_path:
            file_path = found_path
            print(f"在替代位置找到文件: {file_path}")
        else:
            print("无法找到源代码文件，无法定位函数")
            return file_name, None
    
    # 3. 读取文件内容
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            file_content = f.read()
    except Exception as e:
        print(f"读取文件失败: {e}")
        return file_name, None
    
    # 4. 查找函数名
    function_name = find_cpp_function_name_at_line(file_content, line_number)
    
    # 5. 后处理：对函数名进行清理
    if function_name:
        # 移除可能的析构函数标记
        function_name = re.sub(r'^~', '', function_name)
        
        # 移除可能的参数列表残留
        function_name = re.sub(r'\s*\(.*$', '', function_name)
        
        # 移除可能的返回类型残留
        function_name = function_name.split()[-1]
    
    if function_name:
        return file_name, function_name
    else:
        print("无法定位到具体函数，使用默认名称")
        return file_name, "unknown_function"

if __name__ == "__main__":
    # 测试用例
    error_msg1 = "${ASCEND_GYM_DATA_ROOT}/models/pangu7b_new/ModelLink/ModelLink/user/GTSOpsPlus_0122/GTSOpsPlus/op-contrib/ops/built-in/ascendc/add_test_custom_test_new/op_host/add_test_custom_test_new_tiling_calculator.cpp:36:26: error: 'GetTypeSize' was not declared in this scope; did you mean 'GetShapeSize'?\n   36 |     uint32_t dtypeSize = GetTypeSize(inputDesc0->GetDataType());\n      |                          ^~~~~~~~~~~\n      |                          GetShapeSize"
    error_msg2 = "${ASCEND_GYM_DATA_ROOT}/models/pangu7b_new/ModelLink/ModelLink/user/GTSOpsPlus_0122/GTSOpsPlus/op-contrib/ops/built-in/ascendc/add_test_custom_test_new/op_host/add_test_custom_test_new_tiling_calculator.cpp:20:36: error: 'class optiling::AddTestCustomTestNewTilingCalculator' has no member named 'GetInputDesc'; did you mean 'GetInputTensor'?\n   20 |     const auto* inputDesc0 = this->GetInputDesc(0);\n      |                                    ^~~~~~~~~~~~\n      |                                    GetInputTensor"
    
    # 测试error_msg1
    file1, func1 = locate_cpp_error_function(error_msg1)
    print("\nerror_msg1结果:")
    print(f"文件名: {file1}")
    print(f"函数名: {func1}")
    
    # 测试error_msg2
    file2, func2 = locate_cpp_error_function(error_msg2)
    print("\nerror_msg2结果:")
    print(f"文件名: {file2}")
    print(f"函数名: {func2}")