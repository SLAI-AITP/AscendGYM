# transformer__chunk_gated_delta_rule

状态：保留的 golden 任务，未加入正式 catalog。

源码基线：ops_src/ops_transformer/stable/kernels/chunk_gated_delta_rule

测试入口：chunk_gated_delta_rule_test.py

编译后端：transformer；固定张量文件数：4。

修改范围为 op_host/ 和 op_kernel/；golden 与固定测试保持只读。编译与 profiling 由官方镜像或兼容自建服务提供；任务源码、冻结测试与 golden 由 bridge 提交。环境与接口要求见[自建环境](../../../../docs/SELF_HOSTED_ENVIRONMENT.md)和[服务协议](../../../../docs/EXECUTION_BACKEND.md)。
