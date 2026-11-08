import json
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import os
from services.search_from_windrag_service import WindRAGSearchService
from services.code_backfill_service import CodeBackfillService
from services.ascendc_code_fix_rus_service import AscendCCodeFixRusService
from services.list_board_files_service import BoardFilesListService
from services.view_all_file_content_service import ViewAllFileContentService
from services.load_cann_skills_service import LoadCannSkillsService
from services.sandbox_compile_test_service import SandboxCompileTestService
from services.ascendc_code_profiling_service import AscendCCodeProfilingService
from utils.json_utils import repair_tool_call_json, has_xml_tag_outside_strings, strip_xml_tags_outside_strings
from services.get_op_version_service import GetOpVersionService
from services.restore_op_version_service import RestoreOpVersionService
from services.finish_task_service import FinishTaskService
from services.solve_gemm_tiling_optimization_service import SolveGemmTilingOptimizationService
from utils.validators import validate_required_fields, validate_code_format
from utils.exceptions import ValidationError, ServiceError
from config.settings import CODE_BACKFILL_CONFIG
from pathlib import Path

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_NATIVE_TOOL_MODE = False

# 原生工具调用格式：通过环境变量 NATIVE_TOOL=1 启用（gunicorn 模式）
if os.environ.get('NATIVE_TOOL', '0') == '1':
    _NATIVE_TOOL_MODE = True
    logger.info("Native tool call mode enabled via NATIVE_TOOL env var")

# 初始化服务实例
windrag_search_service = WindRAGSearchService()
code_backfill_service = CodeBackfillService()
ascendc_code_fix_rus_service = AscendCCodeFixRusService()
board_files_list_service = BoardFilesListService()
view_all_file_content_service = ViewAllFileContentService()
load_cann_skills_service = LoadCannSkillsService()
sandbox_compile_test_service = SandboxCompileTestService()
ascendc_code_profiling_service = AscendCCodeProfilingService()
get_op_version_service = GetOpVersionService()
restore_op_version_service = RestoreOpVersionService(get_op_version_service)
finish_task_service = FinishTaskService()
solve_gemm_tiling_service = SolveGemmTilingOptimizationService()
# 工具列表
tool_list = [
    "code_backfill",
    "search_from_windrag",
    "search_from_windrag_error",
    "ascend_code_fix",
    "list_board_files",
    "view_all_file_content",
    "load_cann_skills",
    "ascend_code_profiling",
    "get_op_version",
    "restore_op_version",
    "finish_task",
    "skill_summary",
    "solve_gemm_tiling_optimization",
]

def parse_native_tool_call(text: str) -> str:
    """将 Qwen3.6 原生工具调用格式转换为 OpenAI JSON 格式。

    原生格式示例：
        <tool_call>
        <function=my_tool>
        <parameter=param1>
        value1
        </parameter>
        <parameter=param2>
        value2
        </parameter>
        </function>
        </tool_call>

    转换后格式（供 check_tool_format 消费）：
        <tool_call>
        {"name": "my_tool", "parameters": {"param1": "value1", "param2": "value2"}}
        </tool_call>

    如果文本中没有原生格式标记（<function=...>），则原样返回，由后续逻辑处理。
    """
    # 匹配 <tool_call>...</tool_call> 块（允许多行）
    tool_call_pattern = re.compile(r'<tool_call>(.*?)</tool_call>', re.DOTALL)
    tc_match = tool_call_pattern.search(text)
    if not tc_match:
        return text

    inner = tc_match.group(1).strip()

    # 检测是否是原生格式（含 <function=...>）
    func_pattern = re.compile(r'^<function=([^>]+)>(.*)</function>$', re.DOTALL)
    func_match = func_pattern.match(inner)
    if not func_match:
        # 不是原生格式，不做转换
        return text

    tool_name = func_match.group(1).strip()
    params_block = func_match.group(2)

    # 提取所有参数：<parameter=name>\nvalue\n</parameter>
    param_pattern = re.compile(r'<parameter=([^>]+)>\n?(.*?)\n?</parameter>', re.DOTALL)
    parameters = {}
    for pm in param_pattern.finditer(params_block):
        param_name = pm.group(1).strip()
        param_value_raw = pm.group(2).strip()
        try:
            param_value = json.loads(param_value_raw)
        except (json.JSONDecodeError, ValueError):
            param_value = param_value_raw
        parameters[param_name] = param_value

    converted = json.dumps({"name": tool_name, "parameters": parameters}, ensure_ascii=False)
    # 替换原始 <tool_call> 块为 OpenAI JSON 格式
    converted_block = f"<tool_call>\n{converted}\n</tool_call>"
    result = text[:tc_match.start()] + converted_block + text[tc_match.end():]
    logger.info("[parse_native_tool_call] converted native tool call: function=%s params_keys=%s",
                tool_name, list(parameters.keys()))
    return result

def check_tool_format(data):
    """抽取ground truth"""
    pattern = r'<ground_truth>(.*?)</ground_truth>'
    matches = re.search(pattern, data, re.DOTALL)
    ground = matches.group(1).strip() if matches else None

    """检查工具调用格式"""
    pattern = r'<tool_call>\n(.*?)\n</tool_call>'
    matches = re.search(pattern, data, re.DOTALL)

    # fallback: 模型可能输出 ```json {...} ``` 或裸 {"name":...} 而没用 <tool_call> 标签
    if not matches:
        # 尝试匹配 ```json\n{...}\n``` 格式
        fb = re.search(r'```json\s*\n(\{.*?\})\s*\n```', data, re.DOTALL)
        if not fb:
            # 尝试匹配裸 JSON：{"name": "工具名", "parameters": ...}
            fb = re.search(r'(\{\s*"name"\s*:\s*"[^"]+"\s*,\s*"parameters"\s*:\s*\{.*?\}(?:\s*\})?)', data, re.DOTALL)
        if fb:
            matches = fb
            logger.warning(f"[check_tool_format] <tool_call> tag missing, tool call parsed via fallback")
        else:
            # 尝试从 payload 中提取 agent 侧传入的 no_tool_call_count
            no_tool_count = 0
            count_match = re.search(r'no_tool_call_count[=:](\d+)', data or '')
            if count_match:
                no_tool_count = int(count_match.group(1))
            logger.warning(
                "[NoToolCall] No tool call recognized, no_tool_call_count=%d payload_preview=%s",
                no_tool_count,
                (data or '')[:100],
            )
            return jsonify({
                'success': False,
                'error_type': 'NoToolCall',
                'tool_call_result': {
                    'message': '未识别到工具调用。请务必使用 <tool_call>\\n{...}\\n</tool_call> 格式包裹工具调用JSON，不要使用```json```代码块格式。示例：\n<tool_call>\n{"name": "工具名", "parameters": {"参数名": "参数值"}}\n</tool_call>\n生成可编译测试通过AscendC算子的任务尚未完成，请继续尝试。如果未能找到修复方向，请重新生成框架并填充'
                }
            })

    raw = matches.group(1).strip()

    # 检测 XML 关闭标签（</parameter>、</function> 等）或自闭合标签（<parameters/>）
    # 使用状态机跳过 JSON 字符串值内部，避免误判 C++ 模板语法（如 LocalTensor<half>）
    had_xml_tags = False
    if has_xml_tag_outside_strings(raw):
        had_xml_tags = True
        raw = strip_xml_tags_outside_strings(raw)
        logger.warning("[check_tool_format] XML tags stripped from tool_call, retrying JSON parse")

    ok, data, stage, err_msg = repair_tool_call_json(raw)
    if not ok and had_xml_tags:
        return jsonify({
            'success': False,
            'error_type': 'XMLTagError',
            'tool_call_result': "<tool_call></tool_call> 之间仅支持可加载的JSON字符串，不要出现任何XML tag形式（如 </parameter>、</function> 等），请将工具调用修正为标准JSON格式后重试"
        })
    if not ok:
        return jsonify({
            'success': False,
            'error_type': 'JSONDecodeError',
            'tool_call_result': f"工具调用不是可加载的JSON标准格式: {err_msg}"
        })
    if stage != 'direct':
        logger.warning("[check_tool_format] JSON repaired via stage=%s", stage)

    try:
        tool_name = data.get("name", "")
        tool_para = data.get("parameters")

        if not tool_name:
            return jsonify({
                'success': False,
                'error_type': 'FormatError',
                'tool_call_result': "无法解析<tool_call>中的工具名及参数，请检查工具调用格式是否正确"
            })

        if tool_para is None:
            return jsonify({
                'success': False,
                'error_type': 'FormatError',
                'tool_call_result': f"工具调用缺少 parameters 字段，无法解析工具调用参数，请补充 parameters 后重试"
            })

        if not tool_para:
            return jsonify({
                'success': False,
                'error_type': 'FormatError',
                'tool_call_result': f"工具调用的 parameters 字段为空，无法解析工具调用参数，请补充 parameters 后重试"
            })

        if tool_name not in tool_list:
            return jsonify({
                'success': False,
                'error_type': 'ToolNotFound',
                'tool_name': tool_name,
                'tool_call_result': f"工具名称 '{tool_name}' 不在支持列表中，支持的工具: {', '.join(tool_list)}"
            })

        # code_items 后置检查：repair_tool_call_json 的 over_escape 阶段应已将字符串还原为 list，
        # 若仍为字符串说明修复失败，返回明确提示。
        tools_with_code_items = {"ascend_code_fix", "code_backfill"}
        if tool_name in tools_with_code_items and isinstance(tool_para, dict):
            code_items = tool_para.get("code_items")
            if isinstance(code_items, str):
                return jsonify({
                    'success': False,
                    'error_type': 'JSONDecodeError',
                    'tool_call_result': f"工具调用中 code_items 参数应为列表(list)格式，而非字符串，请检查并修正"
                })

        return tool_name, tool_para, ground
    except Exception as e:
        logger.error(f"Error parsing tool call: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error_type': 'ParseError',
            'tool_call_result': f"解析工具调用时发生错误: {str(e)}"
        })

def check_tool_para(data, required_fields, specific_value=None):
    """检查工具参数，缺失或格式错误时直接抛出 ValidationError"""
    require_tag = validate_required_fields(data, required_fields)
    if not require_tag:
        raise ValidationError(f"请检查必选参数是否齐全，需要: {', '.join(required_fields)}")

    if specific_value:
        code_tag = validate_code_format(specific_value)
        if not code_tag:
            raise ValidationError("请检查代码参数格式是否正确")


@app.errorhandler(ValidationError)
def handle_validation_error(error):
    """处理验证错误"""
    return jsonify({
        'success': False,
        'error': 'Validation Error',
        'message': str(error)
    }), 400


@app.errorhandler(ServiceError)
def handle_service_error(error):
    """处理服务错误"""
    return jsonify({
        'success': False,
        'error': 'Service Error',
        'message': str(error)
    }), 500


@app.errorhandler(Exception)
def handle_general_error(error):
    """处理通用错误"""
    logger.error(f"Unhandled error: {str(error)}", exc_info=True)
    return jsonify({
        'success': False,
        'error': 'Internal Server Error',
        'message': 'An unexpected error occurred'
    }), 500

def extract_json_output(sequence, name):
    """提取JSON输出"""
    if not isinstance(sequence, dict):
        if isinstance(sequence, str):
            try:
                res = json.loads(sequence)
                return True, res
            except Exception as e:
                pattern = r'```json\n(.*?)\n```'
                matches = re.search(pattern, sequence, re.DOTALL)
                if matches:
                    json_content = matches.group(1).strip()
                    try:
                        res = json.loads(json_content)
                        return True, res
                    except Exception as e:
                        print(f"{name}参数格式不正确")
                        return False, jsonify({
                            'success': False,
                            'tool_call_result': f"请检查{name}参数格式是否正确"
                        })
    return True, sequence


def search_from_windrag(data):
    """
    RAG知识库检索工具
    参数:
        - queries: 问题列表（必需，1-5个字符串）
        - top_k: 每个问题返回的文档数量（可选，默认5，范围1-10）
    """
    try:
        # 检查必需参数
        required_fields = ['queries']
        check_tool_para(data, required_fields)

        # 验证queries是否为列表
        queries = data['queries']
        if not isinstance(queries, list):
            raise ValidationError("Parameter 'queries' must be a list")

        if len(queries) < 1 or len(queries) > 10:
            raise ValidationError("Parameter 'queries' must contain 1-10 items")

        # 验证每个query是否为字符串
        for i, query in enumerate(queries):
            if not isinstance(query, str) or not query.strip():
                raise ValidationError(f"Query at index {i} must be a non-empty string")

        # 验证top_k（如果提供）
        top_k = data.get('top_k', 15)
        if not isinstance(top_k, int) or top_k < 5 or top_k > 15:
            raise ValidationError("Parameter 'top_k' must be an integer between 5 and 15")

        # 调用服务
        result = windrag_search_service.search(
            queries=queries,
            top_k=top_k
        )

        return jsonify({
            'success': result.get('success', False),
            'tool_name': 'search_from_windrag',
            'tool_call_result': result
        })

    except ValidationError as e:
        raise e
    except Exception as e:
        logger.error(f"Error in search_from_windrag: {str(e)}")
        raise ServiceError(f"Failed to search from WindRAG: {str(e)}")


def code_backfill(data):
    """
    代码回填工具，支持 op_host、op_kernel 和 triton 板块。
    支持两种模式：
    1. view模式：查看板块下的代码文件
    2. backfill模式：将模型生成的代码回填到指定文件

    参数:
        - mode: 工作模式（必需，'view' 或 'backfill'）
        - board: 板块名称（必需，'op_host'、'op_kernel' 或 'triton'）
        - operator_name: 算子名称（必需，字符串）
        - code_items: 代码项列表（backfill模式必需，每项包含file_path、code）
        - context: 可选回显字段（兼容用，不参与逻辑）

    返回:
        view模式: 返回文件列表和代码内容
        backfill模式: 返回回填结果
    """
    try:
        required_fields = ['mode', 'board', 'operator_name']
        check_tool_para(data, required_fields)

        # 验证mode
        mode = data['mode']
        if mode not in ['view', 'backfill']:
            raise ValidationError("Parameter 'mode' must be 'view' or 'backfill'")

        # 验证board
        board = data['board']
        supported_boards = ['op_host', 'op_kernel', 'triton']
        if board not in supported_boards:
            raise ValidationError(f"Parameter 'board' must be one of {supported_boards}")

        # 验证operator_name
        operator_name = data['operator_name']
        if not isinstance(operator_name, str) or not operator_name.strip():
            raise ValidationError("Parameter 'operator_name' must be a non-empty string")

        # 回填模式的额外验证
        if mode == 'backfill':
            # context 字段保留向后兼容（若提供则原样回显），但不再校验 dtype/soc/functionality —
            # 它们在 _2d_service.process 中只是反射进响应里，未参与实际逻辑，
            # 强行校验会让 orchestrator / LLM 频繁踩 schema 而无业务收益。

            # 检查code_items
            if 'code_items' not in data or not isinstance(data['code_items'], list):
                raise ValidationError("Backfill mode requires 'code_items' parameter as a list")

            code_items = data['code_items']
            if len(code_items) == 0:
                raise ValidationError("Parameter 'code_items' must contain at least one item")

            for i, item in enumerate(code_items):
                if not isinstance(item, dict):
                    raise ValidationError(f"code_items[{i}] must be a dictionary")
                if not item.get('file_path') or not isinstance(item['file_path'], str):
                    raise ValidationError(f"code_items[{i}] missing required field: 'file_path'")
                if 'code' not in item or not isinstance(item.get('code'), str):
                    raise ValidationError(f"code_items[{i}] missing required field: 'code'")

        # 调用服务
        result = code_backfill_service.process(
            mode=mode,
            board=board,
            operator_name=operator_name,
            tiling_type='2d',
            context=data.get('context'),
            code_items=data.get('code_items')
        )

        return jsonify({
            'success': result.get('success', False),
            'tool_name': 'code_backfill',
            'tool_call_result': result
        })

    except ValidationError as e:
        raise e
    except Exception as e:
        logger.error(f"Error in code_backfill: {str(e)}")
        raise ServiceError(f"Failed to execute code backfill: {str(e)}")

def ascend_code_fix(data):
    """
    Ascend 算子代码精确替换工具，支持 op_host、op_kernel 和 triton 板块。
    支持两种模式：
    1. view模式：查看指定文件的当前代码
    2. fix模式：修复文件代码（支持函数粒度替换或文件级覆盖）

    参数:
        - mode: 工作模式（必需，'view' 或 'fix'）
        - operator_name: 算子名称（必需，字符串）
        - code_items: 代码项列表（必需，每项包含board、file_path）
            - board: 板块名称（必需，'op_host'、'op_kernel' 或 'triton'）
            - file_path: 文件相对路径（必需）
            - function_name: 函数名称（可选，传入则进行函数粒度替换）
            - code: 新的代码（仅fix模式必需）

    返回:
        view模式: 返回文件的当前代码
        fix模式: 返回修复结果
    """
    try:
        # 检查必需参数
        required_fields = ['mode', 'operator_name', 'code_items']
        check_tool_para(data, required_fields)

        # 验证mode
        mode = data['mode']
        if mode not in ['view', 'fix']:
            raise ValidationError("Parameter 'mode' must be 'view' or 'fix'")

        # 验证operator_name
        operator_name = data['operator_name']
        if not isinstance(operator_name, str) or not operator_name.strip():
            raise ValidationError("Parameter 'operator_name' must be a non-empty string")

        # 验证code_items
        if 'code_items' not in data or not isinstance(data['code_items'], list):
            raise ValidationError("Parameter 'code_items' is required and must be a list")

        code_items = data['code_items']
        if len(code_items) == 0:
            raise ValidationError("Parameter 'code_items' must contain at least one item")

        # 验证每个code_item
        supported_boards = ['op_host', 'op_kernel', 'triton']
        for i, item in enumerate(code_items):
            if not isinstance(item, dict):
                raise ValidationError(f"code_items[{i}] must be a dictionary")

            # board 必需
            if 'board' not in item or not item['board']:
                raise ValidationError(f"code_items[{i}] missing required field: 'board'")
            if item['board'] not in supported_boards:
                raise ValidationError(
                    f"code_items[{i}].board must be one of {supported_boards}, got '{item['board']}'"
                )

            # file_path 必需
            if 'file_path' not in item or not item['file_path']:
                raise ValidationError(f"code_items[{i}] missing required field: 'file_path'")
            if not isinstance(item['file_path'], str):
                raise ValidationError(f"code_items[{i}].file_path must be a string")

            # fix模式需要code字段
            if mode == 'fix':
                if 'code' not in item or not item['code']:
                    raise ValidationError(f"Fix mode requires 'code' field in code_items[{i}]")
                if not isinstance(item['code'], str):
                    raise ValidationError(f"code_items[{i}].code must be a string")

        result = ascendc_code_fix_rus_service.process(
            mode=mode,
            operator_name=operator_name,
            code_items=code_items,
            context=data.get('context')
        )

        return jsonify({
            'success': result.get('success', False),
            'tool_name': 'ascend_code_fix',
            'tool_call_result': result
        })

    except ValidationError as e:
        raise e
    except Exception as e:
        logger.error(f"Error in ascend_code_fix: {str(e)}")
        raise ServiceError(f"Failed to execute Ascend code fix: {str(e)}")


def list_board_files(data):
    """
    列出算子文件列表。指定 board 时返回该板块文件，board 为空或未传时返回算子根目录下一级的所有文件和文件夹。

    参数:
        - op_name: 算子名称（必需，驼峰命名格式，如'AddCustom'）
        - board: 板块名称（可选，'op_host'、'op_kernel'、'op_speed_test' 或 'triton'）

    返回:
        文件列表及统计信息
    """
    try:
        # 检查必需参数
        required_fields = ['op_name']
        check_tool_para(data, required_fields)

        # 验证op_name
        op_name = data['op_name']
        if not isinstance(op_name, str) or not op_name.strip():
            raise ValidationError("Parameter 'op_name' must be a non-empty string")

        # 验证board
        board = data.get('board', '')
        supported_boards = ['op_host', 'op_kernel', 'op_speed_test', 'triton']
        if board not in supported_boards:
            raise ValidationError(f"Parameter 'board' must be one of {supported_boards}")

        # 调用服务
        result = board_files_list_service.list_board_files(
            op_name=op_name,
            board=board,
            tiling='2d'
        )

        return jsonify({
            'success': result.get('success', False),
            'tool_name': 'list_board_files',
            'tool_call_result': result
        })

    except ValidationError as e:
        raise e
    except Exception as e:
        logger.error(f"Error in list_board_files: {str(e)}")
        raise ServiceError(f"Failed to list board files: {str(e)}")


def view_all_file_content(data):
    """
    查看文件完整内容工具

    参数:
        - op_name: 算子名称（必需，驼峰命名格式，如'AddCustom'）
        - board: 板块名称（可选，'op_host'、'op_kernel'、'op_speed_test' 或 'triton'）
        - file_name: 文件名（必需，如'test.cpp'）

    返回:
        文件的完整内容及元信息
    """
    try:
        # 检查必需参数
        required_fields = ['op_name', 'file_name']
        check_tool_para(data, required_fields)

        # 验证op_name
        op_name = data['op_name']
        if not isinstance(op_name, str) or not op_name.strip():
            raise ValidationError("Parameter 'op_name' must be a non-empty string")

        # 验证board
        board = data.get('board', '')
        supported_boards = ['op_host', 'op_kernel', 'op_speed_test', 'triton']
        if board not in supported_boards:
            raise ValidationError(f"Parameter 'board' must be one of {supported_boards}")

        # 验证file_name
        file_name = data['file_name']
        if not isinstance(file_name, str) or not file_name.strip():
            raise ValidationError("Parameter 'file_name' must be a non-empty string")

        # 调用服务
        result = view_all_file_content_service.view_file_content(
            op_name=op_name,
            board=board,
            file_name=file_name,
            tiling='2d'
        )

        return jsonify({
            'success': result.get('success', False),
            'tool_name': 'view_all_file_content',
            'tool_call_result': result
        })

    except ValidationError as e:
        raise e
    except Exception as e:
        logger.error(f"Error in view_all_file_content: {str(e)}")
        raise ServiceError(f"Failed to view file content: {str(e)}")


def load_cann_skills(data):
    """
    加载GTSOPS-CANN算子开发相关的Skills知识

    从GTSOPS-CANN-skills知识库中加载指定类型的skill内容，说明GTSOPS与CANN的差异，
    以及可以从CANN-skills复用的知识。

    参数:
        - skills: 要加载的skill名称列表（必需，1-4个）
            - "gtsops-op-scaffold": 算子开发要点（推荐首先加载）
            - "gtsops-kernel": Kernel实现
            - "gtsops-tiling": Tiling策略
            - "gtsops-debugging": 调试技巧
        - context: 当前开发上下文描述（可选）

    返回:
        加载的skills内容列表及摘要
    """
    try:
        # 检查必需参数
        required_fields = ['skills']
        check_tool_para(data, required_fields)

        # 验证skills参数
        skills = data['skills']
        if not isinstance(skills, list):
            raise ValidationError("Parameter 'skills' must be a list")

        if len(skills) < 1 or len(skills) > 4:
            raise ValidationError("Parameter 'skills' must contain 1-4 items")

        # 验证每个skill是否有效
        valid_skills = ["gtsops-op-scaffold", "gtsops-kernel", "gtsops-tiling", "gtsops-debugging"]
        for skill in skills:
            if skill not in valid_skills:
                raise ValidationError(f"Invalid skill '{skill}'. Valid options are: {valid_skills}")

        # 获取可选的context参数
        context = data.get('context', '')

        # 调用服务
        result = load_cann_skills_service.load_skills(
            skills=skills,
            context=context if context else None
        )

        return jsonify({
            'success': result.get('success', False),
            'tool_name': 'load_cann_skills',
            'tool_call_result': result
        })

    except ValidationError as e:
        raise e
    except Exception as e:
        logger.error(f"Error in load_cann_skills: {str(e)}")
        raise ServiceError(f"Failed to load CANN skills: {str(e)}")


def ascendc_code_profiling(data):
    """
    AscendC Profiling 工具

    先执行沙盒编译测试确保算子无错误，再采集 NPU profiling 数据并生成报告。

    参数:
        - op_name: 算子名称（必需，驼峰命名，如 'AddCustom'）

    返回:
        成功时: { status: 'success', data: <profiling_report>, profiling_report_path }
        编译失败时: { status: 'compile_failed', message: '代码无法编译通过...', errors, ... }
        运行异常时: { status: 'test_failed', message: '代码运行时发生异常...', errors, ... }
        精度不对齐时: { status: 'test_mismatch', message: '代码运行结果精度不对齐...', errors, ... }
        profiling失败时: { status: 'profiling_failed', errors, ... }
    """
    try:
        required_fields = ['op_name']
        check_tool_para(data, required_fields)

        op_name = data['op_name']
        if not isinstance(op_name, str) or not op_name.strip():
            raise ValidationError("Parameter 'op_name' must be a non-empty string")

        result = ascendc_code_profiling_service.execute(op_name=op_name)

        # 根据 debug 阶段的失败状态，补充面向模型的修复指引 message
        status = result.get('status') or result.get('debug_status', '')
        debug_status = result.get('debug_status', status)

        if status == 'debug_failed' or status not in ('success', 'profiling_failed'):
            # 统一把 debug_status 映射为顶层 status，方便调用方判断
            if debug_status == 'compile_failed':
                result['status'] = 'compile_failed'
                result['message'] = (
                    '【编译失败】代码无法通过编译。请根据下方 errors 中的编译错误信息修复代码后重新调用本工具。'
                    f'原始错误: {result.get("message", "")}'
                )
            elif debug_status == 'test_failed':
                result['status'] = 'test_failed'
                result['message'] = (
                    '【运行异常】代码编译通过，但运行时发生异常（进程崩溃或非零退出码）。'
                    '请根据下方 errors 中的运行时错误信息修复代码后重新调用本工具。'
                    f'原始错误: {result.get("message", "")}'
                )
            elif debug_status == 'test_mismatch':
                result['status'] = 'test_mismatch'
                result['message'] = (
                    '【精度不对齐】代码编译通过且运行正常，但算子输出与参考结果精度不对齐。'
                    '请根据下方 errors 中的精度错误信息修复算子计算逻辑后重新调用本工具。'
                    f'原始错误: {result.get("message", "")}'
                )
            else:
                # 其他 debug 失败类型（file_not_found、timeout 等）
                result['status'] = 'debug_failed'
                result['message'] = (
                    f'【调试前置检查失败】({debug_status}) {result.get("message", "")}'
                    '请检查算子文件是否存在并重新调用本工具。'
                )

        return jsonify({
            'success': result.get('status') == 'success',
            'tool_name': 'ascend_code_profiling',
            'tool_call_result': result,
        })

    except ValidationError as e:
        raise e
    except Exception as e:
        logger.error(f"Error in ascendc_code_profiling: {str(e)}")
        raise ServiceError(f"Failed to run ascendc_code_profiling: {str(e)}")


def finish_task(data):
    """Acknowledge an explicit model request to finish the current bare run."""
    try:
        check_tool_para(data, ['op_name', 'reason', 'summary'])
        op_name = data['op_name']
        reason = data['reason']
        summary = data['summary']
        if not isinstance(op_name, str) or not op_name.strip():
            raise ValidationError("Parameter 'op_name' must be a non-empty string")
        if not isinstance(reason, str):
            raise ValidationError("Parameter 'reason' must be a string")
        if not isinstance(summary, str) or not summary.strip():
            raise ValidationError("Parameter 'summary' must be a non-empty string")
        try:
            result = finish_task_service.execute(op_name, reason, summary)
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc
        return jsonify({
            'success': True,
            'tool_name': 'finish_task',
            'tool_call_result': result,
        })
    except ValidationError:
        raise
    except Exception as exc:
        logger.error("Error in finish_task: %s", exc)
        raise ServiceError(f"Failed to execute finish_task: {exc}")


def get_op_version(data):
    """
    获取算子指定版本的代码内容。

    参数:
        - op_name: 算子名称（必需，驼峰命名）
        - version: 版本标识（必需）
            - 'v{N}'：第 N 次写操作前的快照，如 'v3'
            - 'best'：历史 profiling 最优版本
            - 'current'：当前工作目录中的代码

    返回:
        { version_dir, files: [{file_path, content, line_count}], best_exec_time_us（best版本）}
    """
    try:
        required_fields = ['op_name', 'version']
        check_tool_para(data, required_fields)

        op_name = data['op_name']
        if not isinstance(op_name, str) or not op_name.strip():
            raise ValidationError("Parameter 'op_name' must be a non-empty string")

        version = data['version']
        if not isinstance(version, str) or not version.strip():
            raise ValidationError("Parameter 'version' must be a non-empty string")

        result = get_op_version_service.get_version(op_name=op_name, version=version)

        return jsonify({
            'success': result.get('success', False),
            'tool_name': 'get_op_version',
            'tool_call_result': result,
        })

    except ValidationError as e:
        raise e
    except Exception as e:
        logger.error(f"Error in get_op_version: {str(e)}")
        raise ServiceError(f"Failed to get op version: {str(e)}")



def restore_op_version(data):
    """
    Restore an operator snapshot into the mutable workspace without returning
    source contents to the caller.

    Parameters:
        - op_name: operator name
        - version: 'best' | 'latest' | 'v{N}' | '{op_name}_version{N}'
    """
    try:
        check_tool_para(data, ['op_name', 'version'])
        op_name = data.get('op_name')
        version = data.get('version')
        if not isinstance(op_name, str) or not op_name.strip():
            raise ValidationError("Parameter 'op_name' must be a non-empty string")
        if not isinstance(version, str) or not version.strip():
            raise ValidationError("Parameter 'version' must be a non-empty string")

        result = restore_op_version_service.restore(
            op_name=op_name.strip(),
            version=version.strip(),
        )

        return jsonify({
            'success': result.get('success', False),
            'tool_name': 'restore_op_version',
            'tool_call_result': result,
        })

    except ValidationError as e:
        raise e
    except Exception as e:
        logger.error(f"Error in restore_op_version: {str(e)}")
        raise ServiceError(f"Failed to restore op version: {str(e)}")

def skill_summary(data):
    """
    保存性能优化 Skill 文档到 op-path-version/skills/<op_name>/ 目录。

    参数:
        - op_name: 算子名称（必需）
        - version_tag: 本次优化版本标识（必需）
        - baseline_duration_ms / current_duration_ms: 耗时对比（必需）
        - profiling_summary: 性能数据摘要（必需）
        - optimization_strategies: 优化策略列表（必需）
    """
    try:
        required_fields = ['op_name', 'version_tag', 'baseline_duration_ms',
                           'current_duration_ms', 'profiling_summary', 'optimization_strategies']
        check_tool_para(data, required_fields)

        op_name = data['op_name']
        version_tag = data['version_tag']

        from datetime import datetime as _dt
        import config.settings as _settings
        rus_op_path_version = os.environ.get('RUS_OP_PATH_VERSION', _settings.RUS_OP_PATH_VERSION)

        skills_dir = Path(rus_op_path_version) / 'skills' / op_name
        skills_dir.mkdir(parents=True, exist_ok=True)

        timestamp = _dt.now().strftime('%Y%m%d_%H%M%S')
        doc_path = skills_dir / f'skill_{version_tag}_{timestamp}.json'
        doc_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8'
        )
        logger.info(f"skill_summary saved: {doc_path}")

        speedup = round(data['baseline_duration_ms'] / data['current_duration_ms'], 4) \
            if data.get('current_duration_ms') else None

        return jsonify({
            'success': True,
            'tool_name': 'skill_summary',
            'tool_call_result': {
                'message': f'Skill 文档已保存: {doc_path}',
                'doc_path': str(doc_path),
                'speedup': speedup,
            }
        })

    except ValidationError as e:
        raise e
    except Exception as e:
        logger.error(f"Error in skill_summary: {str(e)}")
        raise ServiceError(f"Failed to save skill summary: {str(e)}")


def solve_gemm_tiling_optimization(data):
    """
    GEMM Tiling 参数优化求解工具

    基于华为昇腾910B3硬件配置和GEMM矩阵乘维度[M, K, N]，
    通过MIP求解器自动计算最优的L0/L1级tiling参数。

    参数:
        - mkn: GEMM矩阵乘维度 [M, K, N]（必需）
        - data_b: 输入输出数据位宽(字节)（必需）
        - isnd_abc: ABC矩阵是否为ND格式（必需）
        - bias: 是否需要加bias（必需）

    返回:
        最优tiling方案及预估性能指标
    """
    try:
        required_fields = ['mkn', 'data_b', 'isnd_abc', 'bias']
        check_tool_para(data, required_fields)

        # 验证 mkn
        mkn = data['mkn']
        if not isinstance(mkn, list) or len(mkn) != 3:
            raise ValidationError("Parameter 'mkn' must be an array with 3 elements [M, K, N]")
        if not all(isinstance(x, (int, float)) and x > 0 for x in mkn):
            raise ValidationError("Parameter 'mkn' elements must be positive numbers")

        # 验证 data_b
        data_b = data['data_b']
        if not isinstance(data_b, list) or len(data_b) != 2:
            raise ValidationError("Parameter 'data_b' must be an array with 2 elements")
        if not all(isinstance(x, int) and x > 0 for x in data_b):
            raise ValidationError("Parameter 'data_b' elements must be positive integers")

        # 验证 isnd_abc
        isnd_abc = data['isnd_abc']
        if not isinstance(isnd_abc, list) or len(isnd_abc) != 3:
            raise ValidationError("Parameter 'isnd_abc' must be an array with 3 boolean elements")
        if not all(isinstance(x, bool) for x in isnd_abc):
            raise ValidationError("Parameter 'isnd_abc' elements must be booleans")

        # 验证 bias
        bias = data['bias']
        if not isinstance(bias, bool):
            raise ValidationError("Parameter 'bias' must be a boolean")

        result = solve_gemm_tiling_service.execute(
            mkn=mkn,
            data_b=data_b,
            isnd_abc=isnd_abc,
            bias=bias,
        )

        return jsonify({
            'success': result.get('success', False),
            'tool_name': 'solve_gemm_tiling_optimization',
            'tool_call_result': result,
        })

    except ValidationError as e:
        raise e
    except Exception as e:
        logger.error(f"Error in solve_gemm_tiling_optimization: {str(e)}")
        raise ServiceError(f"Failed to solve GEMM tiling optimization: {str(e)}")


@app.route('/api/', methods=['POST'])
def tool_entrance():
    """工具入口"""
    tool_name = None
    try:
        data = request.get_json()
        if _NATIVE_TOOL_MODE and isinstance(data, str):
            data = parse_native_tool_call(data)
        check_result = check_tool_format(data)

        # 判断返回类型
        if isinstance(check_result, tuple) and len(check_result) == 3:
            tool_name, tool_para, ground_truth = check_result
        else:
            # 返回的是错误响应
            return check_result

        if tool_name == "code_backfill":
            return code_backfill(tool_para)
        elif tool_name == "search_from_windrag" or tool_name == "search_from_windrag_error":
            return search_from_windrag(tool_para)
        elif tool_name == "ascend_code_fix":
            return ascend_code_fix(tool_para)
        elif tool_name == "list_board_files":
            return list_board_files(tool_para)
        elif tool_name == "view_all_file_content":
            return view_all_file_content(tool_para)
        elif tool_name == "load_cann_skills":
            return load_cann_skills(tool_para)
        elif tool_name == "ascend_code_profiling":
            return ascendc_code_profiling(tool_para)
        elif tool_name == "get_op_version":
            return get_op_version(tool_para)
        elif tool_name == "restore_op_version":
            return restore_op_version(tool_para)
        elif tool_name == "finish_task":
            return finish_task(tool_para)
        elif tool_name == "skill_summary":
            return skill_summary(tool_para)
        elif tool_name == "solve_gemm_tiling_optimization":
            return solve_gemm_tiling_optimization(tool_para)
        else:
            return jsonify({
                'success': False,
                'tool_name': tool_name,
                'tool_call_result': f"未实现的工具: {tool_name}"
            })
    except ValidationError as e:
        logger.warning(f"Validation error: {str(e)}")
        return jsonify({
            'success': False,
            'tool_name': tool_name,
            'error_type': 'ValidationError',
            'tool_call_result': str(e)
        })
    except Exception as e:
        logger.error(f"Error in overall api: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'tool_name': tool_name,
            'error_type': 'ServiceError',
            'tool_call_result': f"API调用失败: {str(e)}",
            'details': {
                'exception_type': type(e).__name__,
                'exception_message': str(e)
            }
        })


@app.route('/health', methods=['GET'])
def health_check():
    """健康检查端点"""
    return jsonify({
        'status': 'healthy',
        'message': 'AscendC Performance Tools Backend is running',
        'available_tools': tool_list,
    })


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='AscendC Tools Backend')
    parser.add_argument('--op-path', type=str, default=None,
                        help='算子代码根目录（待优化的原始算子代码），覆盖 settings.py 中的 RUS_OP_PATH，'
                             '默认: ${ASCEND_GYM_DATA_ROOT}/rus_ops')
    parser.add_argument('--op-path-version', type=str, default=None,
                        help='算子代码输出路径（记录每个版本的算子代码更新），覆盖 settings.py 中的 RUS_OP_PATH_VERSION，'
                             '默认: ${ASCEND_GYM_DATA_ROOT}/rus_ops_improved')
    parser.add_argument('--port', type=int, default=7077,
                        help='服务监听端口（默认 7077）')
    parser.add_argument('--native-tool', action='store_true', default=False,
                        help='启用原生工具调用格式解析（Qwen3.6 等模型的 <function=...> 格式），'
                             '默认 False，保持 OpenAI JSON 格式兼容')
    parser.add_argument('--min-improvement-epsilon', type=float, default=None,
                        help='接受为新 best snapshot 所需的相对提升幅度（如 0.02 表示 2%）。'
                             '覆盖 settings.PROFILING_CONFIG["min_improvement_epsilon"]。'
                             '必须与 Client orchestrator.min_improvement_epsilon 对齐。'
                             '默认从环境变量 PROFILING_MIN_IMPROVEMENT_EPSILON 读取，最终默认 0.02。')
    args = parser.parse_args()

    if args.native_tool:
        _NATIVE_TOOL_MODE = True
    if args.op_path:
        from pathlib import Path as _Path
        code_backfill_service._service_2d.work_dir = _Path(args.op_path)
        sandbox_compile_test_service.rus_op_path = args.op_path
        ascendc_code_profiling_service.rus_op_path = args.op_path
        ascendc_code_fix_rus_service.work_dir = _Path(args.op_path)
        board_files_list_service.rus_op_path = _Path(args.op_path)
        view_all_file_content_service.rus_op_path = _Path(args.op_path)
        get_op_version_service.op_path = args.op_path
        restore_op_version_service.version_service.op_path = args.op_path
        logger.info(f"RUS op_path overridden to: {args.op_path}")

    # 若命令行传入了 op_path_version，注入到相关服务
    if args.op_path_version:
        os.environ['RUS_OP_PATH_VERSION'] = args.op_path_version
        ascendc_code_profiling_service.rus_op_path_version = args.op_path_version
        code_backfill_service._service_2d.op_path_version = args.op_path_version
        ascendc_code_fix_rus_service.op_path_version = args.op_path_version
        board_files_list_service.rus_op_path_version = _Path(args.op_path_version)
        view_all_file_content_service.rus_op_path_version = _Path(args.op_path_version)
        get_op_version_service.op_path_version = args.op_path_version
        restore_op_version_service.version_service.op_path_version = args.op_path_version
        logger.info(f"RUS op_path_version overridden to: {args.op_path_version}")

    # 若命令行传入了 epsilon，注入到 profiling service（settings 提供默认 0.02）。
    # 直接覆盖实例的 profiling_config 副本即可，不动 module-level PROFILING_CONFIG。
    if args.min_improvement_epsilon is not None:
        ascendc_code_profiling_service.profiling_config['min_improvement_epsilon'] = (
            float(args.min_improvement_epsilon)
        )
        logger.info(
            "PROFILING_CONFIG['min_improvement_epsilon'] overridden to: %.4f",
            args.min_improvement_epsilon,
        )

    # 启动Flask应用（多线程模式，支持并发请求）
    app.run(debug=True, host='0.0.0.0', port=args.port, use_reloader=False, threaded=True)
