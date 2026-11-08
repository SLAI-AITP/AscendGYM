"""
JSON工具函数
功能：从Markdown渲染的文本中提取和解析JSON
"""
import json
import re
import logging

logger = logging.getLogger(__name__)


def loads_json_from_markdown_rendered(text: str) -> dict:
    """
    从Markdown渲染的文本中提取JSON

    支持的格式：
    1. ```json
       {...}
```
    2. ```
       {...}
```
    3. 直接的JSON文本（无Markdown标记）

    参数:
        text: 包含JSON的文本，可能被Markdown代码块包裹

    返回:
        dict: 解析后的JSON对象

    异常:
        ValueError: 如果无法提取或解析JSON
    """
    if not text or not isinstance(text, str):
        raise ValueError("Input text must be a non-empty string")

    # 尝试多种提取方法
    json_str = None

    # 方法1: 尝试提取Markdown代码块中的JSON (```json ... ```)
    json_match = re.search(r'```json\s*\n(.*?)\n```', text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1).strip()
        logger.debug("Extracted JSON from ```json block")

    # 方法2: 尝试提取普通代码块中的JSON (``` ... ```)
    if not json_str:
        code_match = re.search(r'```\s*\n(.*?)\n```', text, re.DOTALL)
        if code_match:
            json_str = code_match.group(1).strip()
            logger.debug("Extracted JSON from ``` block")

    # 方法3: 尝试查找JSON对象 (以 { 开始，以 } 结束)
    if not json_str:
        # 查找第一个 { 和最后一个 }
        first_brace = text.find('{')
        last_brace = text.rfind('}')
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            json_str = text[first_brace:last_brace + 1]
            logger.debug("Extracted JSON by finding braces")

    # 方法4: 尝试查找JSON数组 (以 [ 开始，以 ] 结束)
    if not json_str:
        first_bracket = text.find('[')
        last_bracket = text.rfind(']')
        if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
            json_str = text[first_bracket:last_bracket + 1]
            logger.debug("Extracted JSON array by finding brackets")

    # 如果所有方法都失败，尝试直接解析整个文本
    if not json_str:
        json_str = text.strip()
        logger.debug("Using entire text as JSON")

    # 尝试解析JSON
    try:
        result = json.loads(json_str)
        logger.debug("Successfully parsed JSON")
        return result
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {str(e)}")
        logger.debug(f"Attempted to parse: {json_str[:200]}...")
        raise ValueError(f"Invalid JSON format: {str(e)}")


def loads_json_from_text(text: str, strict: bool = True) -> dict:
    """
    从文本中提取JSON（更宽松的版本）

    参数:
        text: 包含JSON的文本
        strict: 是否使用严格模式（默认True）

    返回:
        dict: 解析后的JSON对象
    """
    try:
        return loads_json_from_markdown_rendered(text)
    except ValueError as e:
        if strict:
            raise
        else:
            logger.warning(f"Failed to parse JSON: {str(e)}, returning empty dict")
            return {}


def extract_json_blocks(text: str) -> list:
    """
    从文本中提取所有JSON块

    参数:
        text: 包含多个JSON块的文本

    返回:
        list: JSON对象列表
    """
    json_blocks = []

    # 提取所有Markdown JSON代码块
    pattern = r'```json\s*\n(.*?)\n```'
    matches = re.finditer(pattern, text, re.DOTALL)

    for match in matches:
        json_str = match.group(1).strip()
        try:
            json_obj = json.loads(json_str)
            json_blocks.append(json_obj)
            logger.debug(f"Extracted JSON block: {len(json_str)} chars")
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON block: {str(e)}")

    return json_blocks


def clean_json_string(text: str) -> str:
    """
    清理JSON字符串，移除常见的格式问题

    参数:
        text: 原始JSON字符串

    返回:
        str: 清理后的JSON字符串
    """
    # 移除Markdown代码块标记
    text = re.sub(r'```json\s*\n?', '', text)
    text = re.sub(r'```\s*\n?', '', text)

    # 移除前后空白
    text = text.strip()

    # 移除可能的BOM标记
    if text.startswith('\ufeff'):
        text = text[1:]

    return text


def is_valid_json(text: str) -> bool:
    """
    检查文本是否是有效的JSON

    参数:
        text: 要检查的文本

    返回:
        bool: 是否是有效的JSON
    """
    try:
        json.loads(text)
        return True
    except (json.JSONDecodeError, TypeError, ValueError):
        return False


def safe_json_loads(text: str, default=None):
    """
    安全地解析JSON，失败时返回默认值

    参数:
        text: JSON文本
        default: 解析失败时的默认返回值

    返回:
        解析后的对象或默认值
    """
    try:
        return loads_json_from_markdown_rendered(text)
    except Exception as e:
        logger.warning(f"Failed to parse JSON: {str(e)}, returning default value")
        return default if default is not None else {}


def pretty_json(obj: dict, indent: int = 2) -> str:
    """
    将JSON对象格式化为易读的字符串

    参数:
        obj: JSON对象
        indent: 缩进空格数

    返回:
        str: 格式化后的JSON字符串
    """
    try:
        return json.dumps(obj, indent=indent, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Failed to format JSON: {str(e)}")
        return str(obj)


# 别名，保持向后兼容
load_json_from_markdown = loads_json_from_markdown_rendered


# ---------------------------------------------------------------------------
# 工具调用 JSON 修复工具集
# ---------------------------------------------------------------------------

_OPEN = {'{': '}', '[': ']', '(': ')'}
_CLOSE = {'}', ']', ')'}
_PAIR = {'}': '{', ']': '[', ')': '('}


def _in_string(s: str, idx: int) -> bool:
    """判断 s[idx] 是否处于 JSON 字符串字面量内部（考虑转义）。"""
    inside = False
    escape = False
    for i in range(idx):
        c = s[i]
        if escape:
            escape = False
            continue
        if c == '\\' and inside:
            escape = True
            continue
        if c == '"':
            inside = not inside
    return inside


def fix_brackets(raw: str) -> str:
    """
    修复括号不匹配：删除孤立闭括号，末尾追加未闭合开括号的对应闭括号。
    返回修复后的字符串。
    """
    chars = list(raw)
    stack = []  # (open_ch, idx)

    for i, c in enumerate(chars):
        if _in_string(raw, i):
            continue
        if c in _OPEN:
            stack.append((c, i))
        elif c in _CLOSE:
            if stack and stack[-1][0] == _PAIR[c]:
                stack.pop()
            else:
                chars[i] = ''

    appended = [_OPEN[ch] for ch, _ in reversed(stack)]
    return ''.join(chars) + ''.join(appended)


def _fix_node(node, path: str, fixes: list) -> None:
    """
    递归遍历已解析的 Python 对象，原地修复过度转义的字符串值和 dict key。

    过度转义的三种表现：
      - dict key 带多余引号：  {"\"board\"": ...}    -> {"board": ...}
      - 值被多余引号包裹：     "\"hello\""           -> "hello"
      - 值本身是序列化 JSON：  "{\"k\": \"v\"}"      -> {"k": "v"}
    """
    if isinstance(node, dict):
        # 修复带多余引号的 key（需先收集再重建，避免迭代中修改 dict）
        keys_to_fix = {}
        for k in list(node.keys()):
            if isinstance(k, str):
                fixed_k = _unescape_string_value(k)
                # 只接受 str 结果（key 不应被解析成 dict/list）
                if isinstance(fixed_k, str) and fixed_k != k:
                    keys_to_fix[k] = fixed_k
        for old_k, new_k in keys_to_fix.items():
            child_path = f"{path}.{old_k}"
            fixes.append({'path': child_path + '[key]', 'before': old_k, 'after': new_k})
            node[new_k] = node.pop(old_k)

        for k, v in node.items():
            child_path = f"{path}.{k}"
            if isinstance(v, str):
                fixed = _unescape_string_value(v)
                if fixed is not v:
                    fixes.append({'path': child_path, 'before': v, 'after': fixed})
                    node[k] = fixed
            else:
                _fix_node(v, child_path, fixes)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            child_path = f"{path}[{i}]"
            if isinstance(v, str):
                fixed = _unescape_string_value(v)
                if fixed is not v:
                    fixes.append({'path': child_path, 'before': v, 'after': fixed})
                    node[i] = fixed
            else:
                _fix_node(v, child_path, fixes)


def _unescape_string_value(v: str):
    """
    对单个字符串值尝试过度转义修复，返回修复后对象（dict/list/str）。
    若无需修复，返回原对象本身（is 检测可区分）。
    """
    s = v.strip()

    # 情形1：被多余引号包裹，如 "\"hello\""
    if s.startswith('"') and s.endswith('"') and len(s) >= 2:
        inner = s[1:-1]
        # inner 不含未转义引号说明是合法的字符串内容，去掉外层引号
        try:
            candidate = json.loads(s)          # 把整个带引号字符串当 JSON 字符串解析
            if isinstance(candidate, str) and candidate != v:
                # 再看 candidate 是否本身是 JSON 对象/数组
                nested = _try_parse_json(candidate)
                return nested if nested is not None else candidate
        except (json.JSONDecodeError, ValueError):
            pass

    # 情形2：字符串本身是序列化 JSON，如 "{\"k\": \"v\"}"
    nested = _try_parse_json(s)
    if nested is not None:
        return nested

    return v  # 无需修复，返回原对象


def _try_parse_json(s: str):
    """尝试把字符串当 JSON 解析，成功返回解析结果，否则返回 None。

    修复流水线（逐步尝试）：
      1. 直接 json.loads
      2. 括号修复后 json.loads
      3. 针对数组字符串：修复对象间缺失逗号（`}}{` 或 `}}\\n{` -> `},{`）后 json.loads
         同时处理字符串值内部的字面控制字符（转义为 \\n 等）
    """
    s = s.strip()
    if not s or s[0] not in ('{', '['):
        return None

    # 阶段1：直接解析
    try:
        return json.loads(s)
    except (json.JSONDecodeError, ValueError):
        pass

    # 阶段2：括号修复
    try:
        fixed = fix_brackets(s)
        return json.loads(fixed)
    except (json.JSONDecodeError, ValueError):
        pass

    # 阶段3：数组专项修复
    # 3a. 先做括号修复
    candidate = fix_brackets(s)
    # 3b. 修复对象间缺失逗号：`}\s*\n\s*{` 或 `}}{` -> `},\n{`
    candidate = re.sub(r'\}\s*\n\s*\{', r'},\n{', candidate)
    candidate = re.sub(r'\}\}\s*\{', r'},{', candidate)
    # 3c. 将字符串值内部的字面控制字符转义（用状态机区分结构字符和字符串内容）
    candidate = _escape_ctrl_in_strings(candidate)
    try:
        return json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        pass

    # 阶段4：修复字符串值内未转义的字面量引号和控制字符（模型输出代码块时常见）
    try:
        candidate = _fix_unescaped_in_strings(s)
        return json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        return None


def _escape_ctrl_in_strings(s: str) -> str:
    """将 JSON 字符串值内部的字面控制字符转义为合法 JSON 转义序列。

    仅处理字符串字面量内部的控制字符，不影响 JSON 结构字符。
    """
    _CTRL = {'\n': '\\n', '\r': '\\r', '\t': '\\t', '\b': '\\b', '\f': '\\f'}
    result = []
    in_str = False
    escape = False
    for c in s:
        if escape:
            result.append(c)
            escape = False
            continue
        if c == '\\' and in_str:
            result.append(c)
            escape = True
            continue
        if c == '"':
            in_str = not in_str
            result.append(c)
            continue
        if in_str and c in _CTRL:
            result.append(_CTRL[c])
        else:
            result.append(c)
    return ''.join(result)


def _fix_unescaped_in_strings(s: str) -> str:
    """修复 JSON 字符串值内未转义的字面量引号和控制字符。

    适用场景：模型输出的代码字符串中包含真实换行和未转义的双引号，
    例如 {"code": "#include "kernel.h"\n\nint main() {}"} 。

    算法：在字符串内遇到 " 时，向前看下一个非空白字符；
    若该字符是 JSON 结构延续符（: , } ]）或 EOF，则认为是字符串终止，
    否则认为是字面量引号并转义为 \" 。
    """
    AFTER_STRING = set(':,}]')
    out = []
    i = 0
    n = len(s)
    in_str = False
    escape = False
    while i < n:
        c = s[i]
        if not in_str:
            out.append(c)
            if c == '"':
                in_str = True
            i += 1
            continue
        if escape:
            escape = False
            out.append(c)
            i += 1
            continue
        if c == '\\':
            escape = True
            out.append(c)
            i += 1
            continue
        # 控制字符转义
        if c == '\n':
            out.append('\\n'); i += 1; continue
        if c == '\r':
            out.append('\\r'); i += 1; continue
        if c == '\t':
            out.append('\\t'); i += 1; continue
        if c == '\b':
            out.append('\\b'); i += 1; continue
        if c == '\f':
            out.append('\\f'); i += 1; continue
        if c == '"':
            # 向前看下一个非空白字符，判断是字符串终止还是字面量引号
            j = i + 1
            while j < n and s[j] in ' \t\r\n':
                j += 1
            next_char = s[j] if j < n else ''
            if next_char in AFTER_STRING or next_char == '':
                out.append('"')
                in_str = False
            else:
                out.append('\\"')
            i += 1
            continue
        out.append(c)
        i += 1
    return ''.join(out)


def fix_over_escape(raw: str) -> tuple:
    """
    检测并修复 JSON 字符串中的过度转义问题。

    过度转义的两种表现：
      - 字符串值被多余引号包裹：  "\"hello\""        -> "hello"
      - 字符串值是转义的JSON：    "{\"k\": \"v\"}"   -> {"k": "v"}

    参数:
        raw (str): 原始 JSON 字符串

    返回:
        ok      (bool)  : 修复后能否被 json.loads 成功加载
        parsed  (object): 修复后解析得到的 Python 对象（失败时为 None）
        fixed   (str)   : 修复后重新序列化的 JSON 字符串（失败时为原字符串）
        report  (dict)  : {
                            'has_over_escape': bool,   # 是否检测到过度转义
                            'fixes': list,             # 每条修复记录
                            'load_ok': bool,           # 原始字符串能否直接加载
                            'fixed_ok': bool,          # 修复后能否加载
                          }
    """
    report = {
        'has_over_escape': False,
        'fixes': [],
        'load_ok': False,
        'fixed_ok': False,
    }

    # 先看原始是否能直接加载
    try:
        parsed_raw = json.loads(raw)
        report['load_ok'] = True
    except json.JSONDecodeError:
        parsed_raw = None

    if parsed_raw is None:
        # 原始不合法，无法进行值层面修复（先让调用方做括号修复）
        report['fixed_ok'] = False
        return False, None, raw, report

    # 在解析后的对象上递归修复
    fixes = []
    _fix_node(parsed_raw, 'root', fixes)

    if fixes:
        report['has_over_escape'] = True
        report['fixes'] = fixes
        fixed_str = json.dumps(parsed_raw, ensure_ascii=False, indent=2)
        try:
            re_parsed = json.loads(fixed_str)
            report['fixed_ok'] = True
            return True, re_parsed, fixed_str, report
        except json.JSONDecodeError:
            report['fixed_ok'] = False
            return False, None, raw, report
    else:
        # 无过度转义
        report['fixed_ok'] = report['load_ok']
        fixed_str = json.dumps(parsed_raw, ensure_ascii=False, indent=2)
        return report['load_ok'], parsed_raw, fixed_str, report


def repair_tool_call_json(raw: str) -> tuple:
    """
    统一入口：对工具调用中提取出的原始 JSON 字符串依次尝试修复。

    修复流水线：
      1. 直接 json.loads
         1a. 若结果是 dict/list  -> 成功，stage='direct'
         1b. 若结果是 str（整体双重编码）-> 对该字符串再做一次 json.loads
             成功且为 dict/list  -> stage='double_encoded'
      2. 过度转义修复（fix_over_escape）：原始能 loads 但字段值有过度转义
         -> stage='over_escape'
      3. 括号修复（fix_brackets）+ json.loads
         -> stage='brackets'
      4. 括号修复后再做过度转义修复
         -> stage='brackets+over_escape'
      5. 未转义引号/控制字符修复（_fix_unescaped_in_strings）
         适用于字符串值内含字面量 " 和真实换行的场景（如代码块）
         -> stage='unescaped_strings'
      6. 全部失败 -> stage='failed'

    返回:
        ok      (bool)  : 最终是否能成功加载
        parsed  (object): 解析后的 Python 对象（失败时为 None）
        stage   (str)   : 成功所在阶段
        err_msg (str)   : 失败时的错误描述（成功时为空字符串）
    """
    # 阶段1：直接加载
    try:
        parsed = json.loads(raw, strict=False)
        # 1a：结果已是结构体
        if isinstance(parsed, (dict, list)):
            # 仍然过一遍过度转义修复（字段值可能是转义 JSON 字符串）
            ok, fixed_parsed, _, report = fix_over_escape(raw)
            if ok and report['has_over_escape']:
                logger.info("[repair_tool_call_json] direct+over_escape, fixes=%s", report['fixes'])
                return True, fixed_parsed, 'over_escape', ''
            return True, parsed, 'direct', ''
        # 1b：结果是字符串 -> 整体被双重编码
        if isinstance(parsed, str):
            try:
                inner = json.loads(parsed, strict=False)
                if isinstance(inner, (dict, list)):
                    logger.warning("[repair_tool_call_json] double-encoded JSON unwrapped")
                    return True, inner, 'double_encoded', ''
            except json.JSONDecodeError:
                pass
    except json.JSONDecodeError as e1:
        first_err = str(e1)
    else:
        # json.loads 成功但结果既不是结构体也不是可再解析的字符串，视为失败
        first_err = "top-level value is not a JSON object or array"

    # 阶段2：过度转义修复（原始语法合法但字段值有转义问题）
    ok, parsed, _, report = fix_over_escape(raw)
    if ok and report['has_over_escape']:
        logger.info("[repair_tool_call_json] fixed via over_escape, fixes=%s", report['fixes'])
        return True, parsed, 'over_escape', ''

    # 阶段3：括号修复
    fixed = fix_brackets(raw)
    try:
        parsed = json.loads(fixed, strict=False)
        if isinstance(parsed, (dict, list)):
            logger.info("[repair_tool_call_json] fixed via bracket repair")
            return True, parsed, 'brackets', ''
    except json.JSONDecodeError:
        pass

    # 阶段4：括号修复后再做过度转义修复
    ok, parsed, _, report = fix_over_escape(fixed)
    if ok and isinstance(parsed, (dict, list)):
        logger.info("[repair_tool_call_json] fixed via brackets+over_escape, fixes=%s", report['fixes'])
        return True, parsed, 'brackets+over_escape', ''

    # 阶段5：修复字符串值内未转义的字面量引号和控制字符
    try:
        candidate = _fix_unescaped_in_strings(raw)
        parsed = json.loads(candidate, strict=False)
        if isinstance(parsed, (dict, list)):
            logger.info("[repair_tool_call_json] fixed via unescaped_in_strings repair")
            return True, parsed, 'unescaped_strings', ''
    except (json.JSONDecodeError, ValueError):
        pass

    # 全部失败
    try:
        json.loads(fixed, strict=False)
        last_err = first_err
    except json.JSONDecodeError as e_final:
        last_err = str(e_final)

    logger.warning(
        "[repair_tool_call_json] all repair attempts failed. original_err=%s final_err=%s",
        first_err, last_err,
    )
    return False, None, 'failed', last_err


_XML_TAG_PAT = re.compile(r'</\w[\w:.-]*\s*>|<\w[\w:.-]*\s*/>')

def has_xml_tag_outside_strings(s: str) -> bool:
    """Return True only if an XML closing/self-closing tag exists outside JSON string literals."""
    in_str = False
    escape = False
    struct_chars: list[str] = []
    for c in s:
        if escape:
            escape = False
            struct_chars.append('\x00')
            continue
        if c == '\\' and in_str:
            escape = True
            struct_chars.append('\x00')
            continue
        if c == '"':
            in_str = not in_str
            struct_chars.append(c)
            continue
        if in_str:
            struct_chars.append('\x00')
        else:
            struct_chars.append(c)
    return bool(_XML_TAG_PAT.search(''.join(struct_chars)))


def strip_xml_tags_outside_strings(s: str) -> str:
    """删除 JSON 字符串字面量外部的 XML 闭合/自闭合标签（及其前后空白行）。

    字符串值内的内容（如 C++ 模板、HTML 片段）不受影响。
    """
    in_str = False
    escape = False
    struct_chars: list[str] = []
    for c in s:
        if escape:
            escape = False
            struct_chars.append('\x00')
            continue
        if c == '\\' and in_str:
            escape = True
            struct_chars.append('\x00')
            continue
        if c == '"':
            in_str = not in_str
            struct_chars.append(c)
            continue
        if in_str:
            struct_chars.append('\x00')
        else:
            struct_chars.append(c)

    struct_str = ''.join(struct_chars)
    # 找到所有结构外的 XML tag 在原字符串中的位置并删除
    result = list(s)
    for m in _XML_TAG_PAT.finditer(struct_str):
        for i in range(m.start(), m.end()):
            result[i] = ''
    return re.sub(r'\n{3,}', '\n\n', ''.join(result)).strip()