import os


def extract_files_as_code_list(root_dir):
    """
    递归扫描 root_dir 目录，提取所有 .cpp、.h、.hpp、.cc 文件的内容，
    返回符合 code_list 格式的列表：
        [
            {"file_rel_path": "relative/path/to/file.cpp", "code_content": "..."},
            ...
        ]

    参数:
        root_dir (str): 要扫描的根目录路径（绝对或相对）

    返回:
        list[dict]: 符合 AscendC 编译服务要求的 code_list
    """
    code_list = []
    root_dir = os.path.abspath(root_dir)  # 确保是绝对路径，便于计算相对路径

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(('.cpp', '.h', '.hpp', '.cc')):
                file_path = os.path.join(dirpath, filename)
                # 计算相对于 root_dir 的相对路径
                rel_path = os.path.relpath(file_path, root_dir).replace(os.sep, '/')  # 统一使用正斜杠
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                except UnicodeDecodeError:
                    # 如果 UTF-8 失败，尝试 latin-1 或忽略错误（根据实际需求调整）
                    with open(file_path, 'r', encoding='latin-1') as f:
                        content = f.read()
                except Exception as e:
                    print(f"Warning: Skipping file {file_path} due to read error: {e}")
                    continue

                code_list.append({
                    "file_rel_path": rel_path,
                    "code_content": content
                })

    return code_list


# 示例用法
if __name__ == "__main__":
    import json

    op_root = "${ASCEND_GYM_DATA_ROOT}/models/user/mse_loss_v2"
    code_list = extract_files_as_code_list(op_root)

    # 打印结果（或用于后续调用 compile_code）
    print(json.dumps(code_list, indent=2, ensure_ascii=False))