#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
华为昇腾(Ascend) NPU Profiling报告生成服务
通过监听窗口接收参数，生成profiling报告
"""

import json
import subprocess
import os
import re
from pathlib import Path
from typing import Dict, Any, Optional
from flask import Flask, request, jsonify


# Profiling执行配置（用于profiling命令）
PROFILING_CONFIG = {
    'ascend_c_path': '${ASCEND_GYM_DATA_ROOT}/models/user/SandboxFusion/sandbox/tests/runners/samples/ascend_c',
    'main_py_path': './main.py',
    'host': '127.0.0.1',
    'port': '8080',
    'compile_timeout': '180000',
    'run_timeout': '300000',
}

# Flask服务配置（用于app.run）
FLASK_CONFIG = {
    'host': '0.0.0.0',
    'port': 1679,
}

# 创建Flask应用
app = Flask(__name__)


@app.route('/health', methods=['GET'])
def health():
    """健康检查接口"""
    response = build_response(
        success=True,
        message='Service is healthy',
        data={
            'service': 'profiling_report_generator',
            'version': '1.0'
        },
        status_code=200
    )
    return jsonify(response), 200


@app.route('/generate', methods=['POST'])
def generate_profiling_report():
    """
    生成profiling报告接口

    请求体格式:
    {
        "ascend_c_path": "AscendC路径",
        "op_root_path": "算子根路径",
        "profiling_location": "profiling原始信息位置"
    }
    """
    try:
        # 获取请求参数
        data = request.get_json()

        if not data:
            return jsonify({
                'success': False,
                'error': '请求体不能为空'
            }), 400

        # 验证必需参数
        required_fields = ['ascend_c_path', 'op_root_path', 'profiling_location']
        missing_fields = [f for f in required_fields if f not in data]
        if missing_fields:
            return jsonify({
                'success': False,
                'error': f'缺少必需参数: {", ".join(missing_fields)}'
            }), 400

        ascend_c_path = data['ascend_c_path']
        op_root_path = data['op_root_path']
        profiling_location = data['profiling_location']

        # 从op_root_path推导其他参数
        # 1. 查找算子代码子目录（按"含 op_host"判据，不依赖目录命名）
        import os
        from pathlib import Path

        op_root = Path(op_root_path)
        code_folders = [d for d in op_root.iterdir()
                        if d.is_dir() and (d / 'op_host').is_dir()]

        if not code_folders:
            return jsonify({
                'success': False,
                'error': f'在 {op_root_path} 中未找到含 op_host 的算子代码子目录'
            }), 400

        # 使用第一个找到的代码子目录
        op_folder_path = str(code_folders[0])

        # 2. 提取op_name：子目录名（snake-case）转驼峰
        folder_name = code_folders[0].name
        # 例如: sparse_attn_sharedkv -> SparseAttnSharedkv
        op_name = to_camel_case(folder_name)

        # 3. 推导exec_code_path：op_root_path下的.py文件
        # 查找op_root_path下的.py文件
        py_files = list(op_root.glob('*.py'))

        if not py_files:
            return jsonify({
                'success': False,
                'error': f'在 {op_root_path} 中未找到.py文件'
            }), 400

        # 使用找到的第一个.py文件
        exec_code_path = str(py_files[0])

        # 4. profiling输出目录使用profiling_location
        profiling_output_dir = profiling_location

        print(f"收到profiling请求:")
        print(f"  AscendC路径: {ascend_c_path}")
        print(f"  算子根路径: {op_root_path}")
        print(f"  算子文件夹路径: {op_folder_path}")
        print(f"  算子名称: {op_name}")
        print(f"  执行代码路径: {exec_code_path}")
        print(f"  Profiling输出目录: {profiling_output_dir}")
        print("=" * 80)

        # 调用profiling处理函数
        result = run_profiling(
            ascend_c_path=ascend_c_path,
            op_folder_path=op_folder_path,
            op_name=op_name,
            exec_code_path=exec_code_path,
            profiling_output_dir=profiling_output_dir,
            config=PROFILING_CONFIG
        )

        return jsonify(result), result.get('status_code', 500)

    except Exception as e:
        print(f"[ERROR] 处理请求时发生错误: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'服务器内部错误: {str(e)}'
        }), 500


def build_response(success: bool, message: str, data=None, error_details=None, status_code: int = 200) -> Dict[str, Any]:
    """
    构建统一的API响应格式

    Args:
        success: 是否成功
        message: 响应消息
        data: 返回的数据（成功时）
        error_details: 错误详情（失败时）
        status_code: HTTP状态码

    Returns:
        Dict: 包含status_code的响应字典
    """
    response = {
        'success': success,
        'message': message,
        'status_code': status_code
    }

    if data is not None:
        response['data'] = data

    if error_details is not None:
        response['error_details'] = error_details

    return response


def check_profiling_success(stdout: str, stderr: str) -> Dict[str, Any]:
    """
    检查profiling命令是否成功执行

    根据stdout中的服务器响应判断成功与否
    检查 compile_result.return_code 和 run_result.return_code 是否都为 0

    Args:
        stdout: 标准输出
        stderr: 标准错误输出

    Returns:
        Dict: {'success': bool, 'message': str, 'error_details': dict}
    """
    if not stdout:
        return {
            'success': False,
            'message': 'No stdout output to check',
            'error_details': {}
        }

    try:
        # 查找 "Full server response:" 和 "---- COMPILATION DETAILS ---" 之间的所有内容
        json_match = re.search(r'Full server response:\s*(.+?)\s*--- COMPILATION DETAILS ---', stdout, re.DOTALL)
        if not json_match:
            return {
                'success': False,
                'message': 'Cannot find server response JSON in stdout',
                'error_details': {'stdout_sample': stdout[:500]}
            }

        # 解析JSON
        server_response = json.loads(json_match.group(1))

        # 检查compile_result的return_code
        compile_result = server_response.get('compile_result', {})
        compile_return_code = compile_result.get('return_code', -1)
        compile_status = compile_result.get('status', 'Unknown')

        # 检查run_result的return_code
        run_result = server_response.get('run_result', {})
        run_return_code = run_result.get('return_code', -1)
        run_status = run_result.get('status', 'Unknown')

        # 检查整体status
        overall_status = server_response.get('status', 'Unknown')

        # 判断成功：两个return_code都必须为0
        success = (
            overall_status == 'Success' and
            compile_return_code == 0 and
            run_return_code == 0
        )

        if success:
            return {
                'success': True,
                'message': 'Profiling executed successfully',
                'error_details': None
            }
        else:
            return {
                'success': False,
                'message': f'Profiling failed - overall_status: {overall_status}, compile_rc: {compile_return_code}, run_rc: {run_return_code}',
                'error_details': {
                    'overall_status': overall_status,
                    'compile_status': compile_status,
                    'compile_return_code': compile_return_code,
                    'run_status': run_status,
                    'run_return_code': run_return_code
                }
            }

    except json.JSONDecodeError as e:
        return {
            'success': False,
            'message': f'Failed to parse server response JSON: {e}',
            'error_details': {
                'parse_error': str(e),
                'json_sample': json_match.group(1)[:500] if json_match else stdout[:500]
            }
        }
    except Exception as e:
        return {
            'success': False,
            'message': f'Error checking profiling success: {e}',
            'error_details': {'exception': str(e)}
        }


def to_camel_case(text: str) -> str:
    """
    将文本转换为驼峰命名方式
    例如: new_gelu -> NewGelu
    """
    # 按下划线分割
    words = text.split('_')

    # 每个单词首字母大写，其余小写
    camel_words = [word.capitalize() for word in words]

    # 连接成驼峰
    return ''.join(camel_words)


def run_profiling(
    ascend_c_path: str,
    op_folder_path: str,
    op_name: str,
    exec_code_path: str,
    profiling_output_dir: str,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    运行profiling命令并生成报告

    Args:
        ascend_c_path: AscendC路径
        op_folder_path: 算子代码子目录路径（即含 op_host/op_kernel 的那一层）
        op_name: 算子名称（驼峰命名）
        exec_code_path: 执行代码路径
        profiling_output_dir: profiling输出目录
        config: 配置参数

    Returns:
        Dict: 包含status_code的响应字典
    """
    # 生成profiling命令（使用fstring，参考bash命令）
    profiling_cmd = f"""set -euo pipefail
cd {ascend_c_path}
python {config['main_py_path']} \\
--custom-op-dir {op_folder_path} \\
--exec-code-path {exec_code_path} \\
--host {config['host']} \\
--port {config['port']} \\
--compile-timeout {config['compile_timeout']} \\
--run-timeout {config['run_timeout']} \\
--profile \\
--profiling-output-dir {profiling_output_dir}
"""

    print(f"Step 1: 执行profiling命令")
    print(f"Command:\n{profiling_cmd}")
    print("=" * 80)

    try:
        # 执行profiling命令
        result = subprocess.run(
            profiling_cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=int(config['run_timeout']) / 1000 + 60  # 转换为秒并添加缓冲时间
        )

        print(f"Command execution completed")
        print(f"Return code: {result.returncode}")

        if result.stdout:
            print(f"\nStandard Output:\n{result.stdout}")

        if result.stderr:
            print(f"\nStandard Error:\n{result.stderr}")

        # 检查profiling是否成功 - 基于服务器响应
        profiling_check = check_profiling_success(result.stdout, result.stderr)

        if profiling_check['success']:
            print("\n[INFO] Profiling command executed successfully and verified.")

            # 2. 查找profiling输出目录中的CSV文件
            print(f"\nStep 2: 处理profiling CSV文件，从 {profiling_output_dir}")

            if os.path.exists(profiling_output_dir):
                # 查找CSV文件
                csv_files = list(Path(profiling_output_dir).glob("*.csv"))

                if not csv_files:
                    return build_response(
                        success=False,
                        message='No CSV files found in profiling output directory',
                        error_details={'directory': profiling_output_dir},
                        status_code=500
                    )

                print(f"Found {len(csv_files)} CSV file(s)")
                for csv_file in csv_files:
                    print(f"  - {csv_file.name}")

                # 3. 调用extract_msprof_fields.py中的函数生成最终JSON
                print(f"\nStep 3: 生成最终JSON报告，使用 extract_msprof_fields.py")

                # 导入MSProfFieldExtractor
                from extract_msprof_fields import MSProfFieldExtractor

                extractor = MSProfFieldExtractor()
                json_output_path = os.path.join(profiling_output_dir, "profiling_report.json")

                # 调用save_to_json方法
                extractor.save_to_json(profiling_output_dir, json_output_path)

                print(f"\n[SUCCESS] Final profiling report generated: {json_output_path}")

                # 读取并返回JSON内容
                with open(json_output_path, 'r', encoding='utf-8') as f:
                    json_content = f.read()
                    json_data = json.loads(json_content)

                return build_response(
                    success=True,
                    message='Profiling report generated successfully',
                    data=json_data,
                    status_code=200
                )
            else:
                return build_response(
                    success=False,
                    message='Profiling output directory not found',
                    error_details={'directory': profiling_output_dir},
                    status_code=500
                )

        else:
            # 基于profiling_check返回的error_details来构建错误信息
            error_details = profiling_check.get('error_details', {})

            print(f"\n[ERROR] Profiling command failed: {profiling_check['message']}")

            return build_response(
                success=False,
                message=f'Profiling command failed: {profiling_check["message"]}',
                error_details=error_details,
                status_code=500
            )

    except subprocess.TimeoutExpired:
        return build_response(
            success=False,
            message=f'Command execution timed out after {config["run_timeout"]}ms',
            error_details={'timeout_ms': config['run_timeout']},
            status_code=504
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return build_response(
            success=False,
            message=f'Unexpected error: {str(e)}',
            error_details={'exception_type': type(e).__name__},
            status_code=500
        )


def main():
    """主函数 - 启动Flask服务"""
    import sys

    print("=" * 80)
    print("华为昇腾(Ascend) NPU Profiling报告生成服务")
    print("=" * 80)
    print()
    print("配置信息:")
    print(f"  Flask服务: http://{FLASK_CONFIG['host']}:{FLASK_CONFIG['port']}")
    print(f"  Profiling服务: http://{PROFILING_CONFIG['host']}:{PROFILING_CONFIG['port']}")
    print()
    print("=" * 80)
    print(f"健康检查: http://{FLASK_CONFIG['host']}:{FLASK_CONFIG['port']}/health")
    print(f"生成报告: http://{FLASK_CONFIG['host']}:{FLASK_CONFIG['port']}/generate")
    print("=" * 80)
    print()
    print("API文档:")
    print()
    print("1. 健康检查接口")
    print("   URL: GET /health")
    print("   响应格式:")
    print("   {")
    print("     \"success\": true,")
    print("     \"message\": \"Service is healthy\",")
    print("     \"data\": {")
    print("       \"service\": \"profiling_report_generator\",")
    print("       \"version\": \"1.0\"")
    print("     },")
    print("     \"status_code\": 200")
    print("   }")
    print()
    print("2. 生成profiling报告接口")
    print("   URL: POST /generate")
    print("   Content-Type: application/json")
    print("   请求体:")
    print("   {")
    print("     \"ascend_c_path\": \"AscendC路径\",")
    print("     \"op_root_path\": \"算子根路径\",")
    print("     \"profiling_location\": \"profiling原始信息位置\"")
    print("   }")
    print("   说明:")
    print("     - exec_code_path: 自动从op_root_path查找.py文件")
    print("     - op_folder_path: 自动从op_root_path查找含 op_host 的代码子目录")
    print("     - op_name: 从op_folder_path文件名推导（snake-case 转驼峰命名）")
    print()
    print("   成功响应 (200):")
    print("   {")
    print("     \"success\": true,")
    print("     \"message\": \"Profiling report generated successfully\",")
    print("     \"data\": {")
    print("       \"op_name\": \"...\",")
    print("       \"device_id\": \"...\",")
    print("       \"...\": \"其他profiling字段\"")
    print("     },")
    print("     \"status_code\": 200")
    print("   }")
    print()
    print("   失败响应 (400/500/504):")
    print("   {")
    print("     \"success\": false,")
    print("     \"message\": \"错误原因描述\",")
    print("     \"error_details\": {")
    print("       \"reason\": \"详细原因\",")
    print("       \"stdout\": \"...\",")
    print("       \"stderr\": [\"错误1\", \"错误2\"],")
    print("       \"return_code\": 1")
    print("     },")
    print("     \"status_code\": 500")
    print("   }")
    print()
    print("使用curl示例:")
    print(f"   curl -X POST http://{FLASK_CONFIG['host']}:{FLASK_CONFIG['port']}/generate \\")
    print("     -H \"Content-Type: application/json\" \\")
    print("     -d '{")
    print("       \"ascend_c_path\": \"${ASCEND_GYM_DATA_ROOT}/models/user/SandboxFusion/sandbox/tests/runners/samples/ascend_c\",")
    print("       \"op_root_path\": \"${ASCEND_GYM_DATA_ROOT}/models/user/ascendc_agent_eval/0331/NewGELU\",")
    print("       \"profiling_location\": \"${ASCEND_GYM_DATA_ROOT}/profiling_results_NewGeluCustom\"")
    print("     }'")
    print()
    print("使用Python示例:")
    print("   import requests")
    print("   import json")
    print(f"   url = 'http://{FLASK_CONFIG['host']}:{FLASK_CONFIG['port']}/generate'")
    print("   data = {")
    print("       'ascend_c_path': '${ASCEND_GYM_DATA_ROOT}/models/user/SandboxFusion/sandbox/tests/runners/samples/ascend_c',")
    print("       'op_root_path': '${ASCEND_GYM_DATA_ROOT}/models/user/ascendc_agent_eval/0331/NewGELU',")
    print("       'profiling_location': '${ASCEND_GYM_DATA_ROOT}/profiling_results_NewGeluCustom'")
    print("   }")
    print("   response = requests.post(url, json=data)")
    print("   result = response.json()")
    print("=" * 80)
    print()

    # 启动Flask服务
    app.run(
        host=FLASK_CONFIG['host'],
        port=int(FLASK_CONFIG['port']),
        debug=False
    )


if __name__ == '__main__':
    main()
