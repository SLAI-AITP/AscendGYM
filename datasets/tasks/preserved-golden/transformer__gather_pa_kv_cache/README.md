# transformer__gather_pa_kv_cache

状态：保留的 golden 任务，未加入正式 catalog。

源码基线：ops_src/ops_transformer/stable/kernels/gather_pa_kv_cache

测试入口：gather_pa_kv_cache_test.py

编译后端：transformer；固定张量文件数：80。

修改范围为 op_host/ 和 op_kernel/；golden 与固定测试保持只读。上游编译仓库按固定版本单独获取，存放于 AscendForge/runtime/ascend_c/，本目录不写入上游工作树。
