# lightning_indexer

状态：正式 batch1 任务。

源码基线：ops_src/ops_transformer/test_kernels/kernels/lightning_indexer

测试入口：lightning_indexer_test.py

编译后端：transformer；固定张量文件数：20。

修改范围为 op_host/ 和 op_kernel/；golden 与固定测试保持只读。上游编译仓库按固定版本单独获取，存放于 AscendForge/runtime/ascend_c/，本目录不写入上游工作树。
