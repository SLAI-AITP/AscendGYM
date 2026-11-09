# fused_gdn_gating

状态：正式 batch1 任务。

源码基线：ops_src/vllm_ascend/kernels/fused_gdn_gating

测试入口：fused_gdn_gating_test.py

编译后端：vllm_ascend；固定张量文件数：0。

修改范围为 op_host/ 和 op_kernel/；golden 与固定测试保持只读。上游编译仓库按固定版本单独获取，存放于 AscendForge/runtime/ascend_c/，本目录不写入上游工作树。
