from six import moves

# GEMM TILING OPTIMIZATION

## requires和运行
pip install pyscipopt \
conda install -c conda-forge pyomo scip \
或 pip install pyomo 并且去官网下载 SCIP Optimization Suite，添加exe到路径中

python -m unittest test_gemm.TestGemm.test_mip \
python example_mkn.py \
python example_white_list.py 

## 核心算法模块

- gemm_model_base.py: tiling参数、硬件参数、性能参数
- gemm_model_problem.py: 问题定义
- gemm_model_objs.py: 性能估计
- gemm_mip.py: 优化求解
- gemm_solver: 求解接口
```python
from gemm_model_base import HardwareParam
from gemm_model_objs import evaluate_tiling_key
from gemm_solver import solve_problem, tiling_key2white_list
from gemm_problem import Problem
from gemm_mip import GemmMipBottleneck
```

## 输入

- hardware.yml: 硬件参数

```python
with open('hardware.yml', 'r', encoding='utf-8') as f:
  param_dict = yaml.safe_load(f)
hardware_param = HardwareParam(param_dict["910B3_modify"])
```

- gemm_problem.Problem

```python
problem = Problem(hardware_param=hardware_param, MKN_orig=MKN)  # 硬件参数和矩阵乘维度
problem.dataB = [2, 4]  # 输入（AB矩阵）和输出（C矩阵）数据的位宽，单位Byte
problem.isnd_abc = [True, False, True]  # ABC矩阵是否为ND格式
problem.bias = False  # 是否需要加bias
problem.reuse_l0 = True  # l0是否数据复用
problem.reuse_l1 = True  # l1是否数据复用
# 固定部分参数
problem.fixed_solution.m1k1n1_tilde = [1, None, 1]  # L1只缓存K维
problem.fixed_solution.core_loop = 1  # 先切分核数据再核内tiling
# problem.fixed_solution.orders_l0l1l2 = ['KNM', 'KNM', 'MN']  # 注释掉可求解计算顺序
# problem.buffer_num_l1ab = [2, 2]  # L1AB >=2，注释掉或None作为决策变量
# problem.buffer_num_l0abc = [2, 2, None]  # L0ABC >=2，L0C=1或2，注释掉或None作为决策变量
```

## 求解和输出
### 单算子（固定MKN）
```python
tiling_key, model_performance, solver_status = solve_problem(problem=problem, solver=GemmMipBottleneck, time_limit=120)
```
- gemm_model_base.TilingKey
  - orders_l0l1l2: L0, L1, L2的遍历顺序, 例："KNM" -> 先K再N再M
  - buffer_num_l0abc: L0A, L0B, L0C的BufferNum
  - reuse_l0: L0是否数据复用
  - reuse_num_l0ab: L0A, L0B的复用缓存数
  - buffer_num_l1ab: L1 A、B矩阵的BufferNum
  - reuse_l1: L1是否数据复用
  - reuse_num_l1ab: L1 AB的复用缓存数
  - 
  - mckcnc_num: 每个维度上切分的核数
  - mckcnc: 单核单次迭代的数据shape
  - m0k0n0: L0 tiling shape
  - m1k1n1_tilde: 每次从GM搬运到L1的L0 tiling个数
  - m1k1n1_num: 单核单次迭代每个维度上从GM搬运到L1的次数
  - m2n2_tilde: L2 swizzle count，用于提高L2命中率，具体见伪代码
  - 
  - core_num: 核数
  - core_loop: 单核的迭代次数，可默认设定为1
  

- gemm_model_base.TilingPerformance
  - unit_time: 每个搬运单元的预估时间
  - core_loop_num = 每个核迭代几次 
  - util_cache_l0abc: L0利用率
  - util_cache_l1: L1利用率
  - util_core: 核利用率

### 单核单次迭代伪代码
计算数据量 [mckcnc[0], mckcnc[1]] * [mckcnc[1], mckcnc[2]] = [mckcnc[0], mckcnc[2]]
```python
# example orders_l0l1l2[2] = 'MN', first loop M, then loop N, change the MKN order when this value changed
for n in range(1, m2n2_num[1]):
    for m in range(1, m2n2_num[0]):
        process_l1()

def process_l1():
    # example orders_l0l1l2[1] = 'KMN', first loop K, then loop M then loop N, change the MKN order when this value changed
    for n in range(1, m2n2_tilde[1]):
        for m in range(1, m2n2_tilde[0]):
            for k in range(1, m1k1n1_num[1]):
                MTE2 move A1:shape[m1k1n1[0], m1k1n1[1]], B1:shape[m1k1n1[2], m1k1n1[1]]
                process_l0()
            if m1k1n1_num[1] > 1:
                FixPipe move C1:shape[m1k1n1[0], m1k1n1[2]]  # only when K loop is finished

def process_l0():
    # example orders_l0l1l2[0] = 'KNM', first loop K, then loop N then loop M, change the MKN order when this value changed
    for m in range(1, m1k1n1_tilde[0]):
        for n in range(1, m1k1n1_tilde[2]):
            for k in range(1, m1k1n1_tilde[1]):
                MTE1 move A0:shape[m0k0n0[0], m0k0n0[1]], B0:shape[m0k0n0[2], m0k0n0[1]]
                MAC cal C0 = A0 * B0
            if m1k1n1_num[1] == 1:
                FixPipe move C0:shape[m0k0n0[0], m0k0n0[2]]  # only when K loop is finished
```
- reuse_l0 = True
  - process_l0()采用头尾往返”的遍历方式
  - 采用数据复用，reuse_num_l0ab为L0A, L0B的复用缓存数
- reuse_l1 = True
  - process_l1()采用头尾往返”的遍历方式
  - 采用数据复用，reuse_num_l1ab为L1中A、B矩阵的复用缓存数
- buffer_num_l0abc和buffer_num_l1ab >= 2 时读写并行进行，== 1时读写串行

### GTS已有算子
已有算子实现方式已固定如下，可将solver修改为GemmGts求解：
- L1AB、L0A、L0B开double buffer，L0C不开double buffer
- L0层无数据复用，L1层当M维不切分时（m0=M），A矩阵进行数据复用，共占用reuse_num+2块缓存，其中reuse_num块缓存用于数据复用(CacheCnt)，2块用于pingpong（buffer_num_l1ab=reuse_num_l1ab时不申请这两块）
- m0k0n0，MN维切分核数，L1缓存块个数（k1_tilde），这五个参数可通过白名单进行修订 
- L1层，当M维切分数高于N维时，遍历顺序为K->N->M，反之为K->M->N
- L1缓存块只存在K维（m1_tilde=n1_tilde=1），所以L0层遍历只有K维
- 采用先tiling，再分配给各个核计算的实现方式
- 无针对L2命中率特定设计的swizzle
```python
from gemm_existing_operator import GemmGts
tiling_key, model_performance, solver_status = solve_problem(problem=problem, solver=GemmGts, time_limit=120)
```
GTS wind_matmul_v2白名单参数对应表：
  - baseM: tiling_key.m0k0n0[0]
  - baseK: tiling_key.m0k0n0[1]
  - baseN: tiling_key.m0k0n0[2]
  - mDim: tiling_key.mckcnc_num[0]
  - nDim: tiling_key.mckcnc_num[2]
  - l1CopyBufferNum: tiling_key.m1k1n1_tilde[1]

可直接使用以下代码获得
```python
tiling_str = tiling_key2white_list(tiling_key)
```
  
### 白名单（固定KN维，随机M维）
```python
white_list = solve_white_list(problem=problem, solver=GemmGts, m_sep=16, time_limit=600, work_num=8)
white_list.write(r'white_list.txt')
```
- gemm_model_base.WhiteList
  - hardware_name: 硬件名称
  - K: K维
  - N: N维
  - wl: 白名单 [(M_min, M_max, m0, k0, n0, mc_num, nc_num, L1缓存块), ]