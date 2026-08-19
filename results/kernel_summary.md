# Kernel 与 NCCL 汇总

本文件由两份 `.nsys-rep` 导出 SQLite 查询。

## Trace 元数据

| 框架 | GPU | GPU 数 | `RUN_DURATION_MS` |
|---|---|---:|---:|
| vLLM | NVIDIA A800-SXM4-80GB | 2 | 323,776 |
| SGLang | NVIDIA A800-SXM4-80GB | 2 | 256,360 |

## 热点 Kernel

| 框架 | Kernel | 次数 | 累计耗时 |
|---|---|---:|---:|
| vLLM | `flash_fwd_kernel` | 54 | 3,561.92 ms |
| vLLM | `ampere_bf16...64x64...gemm` | 8,040 | 1,147.18 ms |
| vLLM | `fused_moe_kernel` | 17,280 | 1,007.93 ms |
| SGLang | `BatchPrefillWithPagedKVCacheKernel` | 666 | 2,337.92 ms |
| SGLang | `fused_moe_kernel` | 3,580 | 1,808.01 ms |
| SGLang | `ampere_bf16...128x256...gemm` | 5,820 | 1,477.20 ms |


## `fused_moe_kernel`

| 框架 | 次数 | 总耗时 | 平均 | 最小 | 最大 |
|---|---:|---:|---:|---:|---:|
| vLLM | 17,280 | 1,007.93 ms | 58.33 μs | 13.438 μs | 741.484 μs |
| SGLang | 3,580 | 1,808.01 ms | 505.03 μs | 243.124 μs | 1,051.322 μs |

派生比值：

- 平均单次：505.03 / 58.33 = **8.66×**；
- trace 内累计：1,808.01 / 1,007.93 = **1.79×**；
- 调用次数：17,280 / 3,580 = **4.83×**，vLLM 更多。

### blockX 聚合

| 框架 | blockX | 次数 | 总耗时 | 平均 |
|---|---:|---:|---:|---:|
| vLLM | 256 | 11,200 | 712.97 ms | 63.66 μs |
| vLLM | 128 | 6,080 | 294.96 ms | 48.51 μs |
| SGLang | 128 | 3,580 | 1,808.01 ms | 505.03 μs |

## NCCL AllReduce

| 框架 | 次数 | 总耗时 | 平均 |
|---|---:|---:|---:|
| vLLM | 274 | 504.71 ms | 1,841.99 μs |
| SGLang | 3,581 | 858.25 ms | 239.67 μs |

派生比值：

- 调用次数：3,581 / 274 = **13.07×**，SGLang 更多；
- trace 内累计：858.25 / 504.71 = **1.70×**；
- SGLang 单次平均更短，说明调用粒度不同。

## Timeline 核对

| 框架 | duration | grid | block | registers/thread | occupancy |
|---|---:|---:|---:|---:|---:|
| vLLM | 64.662 μs | 1240×1×1 | 256×1×1 | 95 | 25% |
| SGLang | 643.447 μs | 8168×1×1 | 128×1×1 | 96 | 31.25% |

SQLite 中存在与截图 duration、grid、block 和 registers 完全匹配的记录。

## GDN 观察

两份 trace 都出现 GDN/causal-conv 相关 Kernel。它们用于建立“模型组件 → GPU 执行轨迹”的认识，不作为本项目核心归因，因为 trace 窗口与 workload 尚未对齐。
