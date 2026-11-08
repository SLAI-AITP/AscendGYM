"""
AscendC工具包配置文件
"""
import os

# RUS算子（2D Tiling）相关路径配置
def _expand_env_path(value):
    return os.path.expanduser(os.path.expandvars(value))


RUS_OP_PATH = _expand_env_path(os.getenv(
    'RUS_OP_PATH',
    '${ASCEND_GYM_DATA_ROOT}/rus_ops',
))
# 算子代码输出路径（记录每个版本的算子代码更新，优先级高于 RUS_OP_PATH）
RUS_OP_PATH_VERSION = _expand_env_path(os.getenv(
    'RUS_OP_PATH_VERSION',
    '${ASCEND_GYM_DATA_ROOT}/rus_ops_improved',
))
SANDBOX_PATH = _expand_env_path(os.getenv(
    'SANDBOX_PATH', '${ASCEND_GYM_WORKSPACE}/bridge/bare_bench/sandbox_runner'))

# RAG 知识库配置
WINDVECTOR_CONFIG = {
    "windvector_url": os.getenv("WINDVECTOR_URL", "http://127.0.0.1:20999/wind/vector/rest/v1/query"),
    "windvector_table_matching": os.getenv("WINDVECTOR_TABLE_MATCHING", '{"all": "CANN_RAG_DATABASE"}'),
    # "windvector_table_matching": os.getenv("WINDVECTOR_TABLE_MATCHING", '{"all": "CANN_RAG_DATABASE"}'),
    "windvectord_original_num": os.getenv("WINDVECTOR_ORIGINAL_NUM", 100),
    "rerank_url": os.getenv("RERANK_URL", "http://127.0.0.1:8181/rerank"),
    "rerank_model": os.getenv("RERANK_MODEL", "v-bge-reranker-v2-m3"),
    "rag_qa_url": os.getenv("RAG_QA_URL", "http://127.0.0.1:6324/rag/search")
}

# 代码回填配置
CODE_BACKFILL_CONFIG = {
    '2d_work_dir': RUS_OP_PATH,
    'supported_boards': ['op_host', 'op_kernel'],
    'supported_dtypes': ["fp16", "fp32", "i8", "i16", "i32", "i64", "bf16"],
    'supported_socs': ['ascend910b', 'ascend910c', 'ascend310p'],
}

# 沙盒编译测试配置（供 sandbox_compile_test_service 使用）
SANDBOX_CONFIG = {
    'sandbox_path': SANDBOX_PATH,
    'timeout': int(os.getenv('SANDBOX_TIMEOUT', '3600')),
    'conda_env': os.getenv('SANDBOX_CONDA_ENV', 'sandbox-npu-runtime'),
}

# Profiling 执行配置（供 ascendc_code_profiling_service 使用）
PROFILING_CONFIG = {
    'ascend_c_path': os.getenv('PROFILING_ASCEND_C_PATH', SANDBOX_PATH),
    'main_py_path': os.getenv('PROFILING_MAIN_PY_PATH', './main.py'),
    'host': os.getenv('PROFILING_HOST', '127.0.0.1'),
    'port': os.getenv('PROFILING_PORT', '9000'),
    'compile_timeout': os.getenv('PROFILING_COMPILE_TIMEOUT', '1800000'),
    'run_timeout': os.getenv('PROFILING_RUN_TIMEOUT', '3600000'),
    # Fixed benchmark sampling contract. profiling_runs is TOTAL rounds,
    # including warmup; the first warmup_runs rounds are correctness-checked
    # but excluded from the arithmetic mean.
    'profiling_runs': int(os.getenv('PROFILING_RUNS', '8')),
    'warmup_runs': int(os.getenv('PROFILING_WARMUP_RUNS', '3')),
    'profiling_statistic': 'mean',
    # 接受为新 best snapshot 所需的相对提升幅度。新 exec_time_us 必须 ≤
    # prev_best * (1 - eps)。eps=0 退回到严格 < prev_best 的原始语义。
    #
    # 默认 0.0:Client 侧用 baseline-adaptive epsilon (orchestrator.epsilon_tiers)
    # 自适应 noise floor;Backend 不再尝试镜像同样的 epsilon (因为 Backend 不
    # 知道 baseline,无法做 tier 判定)。Client 自己用 canonical_hash gate 和
    # cycle/global_best gate 做双层过滤,Backend 只需"严格更快就接受"。
    # 如果需要强制 Backend 也按某个固定 eps 过滤,通过环境变量或 --min-improvement-epsilon
    # CLI flag 显式覆盖。
    'min_improvement_epsilon': float(os.getenv('PROFILING_MIN_IMPROVEMENT_EPSILON', '0.0')),
}

