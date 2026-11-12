# hc_pre_sinkhorn

状态：正式 batch1 任务。

源码基线：ops_src/vllm_ascend/kernels/hc_pre_sinkhorn

测试入口：hc_pre_sinkhorn_test.py

编译后端：vllm_ascend；固定张量文件数：0。

修改范围为 op_host/ 和 op_kernel/；golden 与固定测试保持只读。编译与 profiling 由官方镜像或兼容自建服务提供；任务源码、冻结测试与 golden 由 bridge 提交。环境与接口要求见[自建环境](../../../../docs/SELF_HOSTED_ENVIRONMENT.md)和[服务协议](../../../../docs/EXECUTION_BACKEND.md)。
