# Nsight Systems 与 SQLite 分析

## 数据来源

本项目使用两份 Nsight Systems 原始报告进行复核：

- `vllm_qwen35.nsys-rep`
- `sglang_qwen35.nsys-rep`

原始 `.nsys-rep` 文件体积较大且可能包含本机路径、进程信息和环境数据，因此不默认提交到公开仓库。仓库保留查询 SQL、汇总结果和两张经过人工核对的 Timeline 截图。

## 导出 SQLite

```bash
nsys export --type sqlite --force-overwrite=true \
  --output vllm_qwen35.sqlite vllm_qwen35.nsys-rep

nsys export --type sqlite --force-overwrite=true \
  --output sglang_qwen35.sqlite sglang_qwen35.nsys-rep
```

运行通用查询：

```bash
sqlite3 -header -column vllm_qwen35.sqlite \
  < scripts/kernel_analysis.sql
```

## Trace 概览

Nsight 元数据记录的 `RUN_DURATION_MS`：

| Trace | 采集窗口 |
|---|---:|
| vLLM | 323,776 ms |
| SGLang | 256,360 ms |

## `fused_moe_kernel` 汇总

| 框架 | 次数 | 总耗时 | 平均 | 最小 | 最大 |
|---|---:|---:|---:|---:|---:|
| vLLM | 17,280 | 1,007.93 ms | 58.33 μs | 13.438 μs | 741.484 μs |
| SGLang | 3,580 | 1,808.01 ms | 505.03 μs | 243.124 μs | 1,051.322 μs |

### vLLM 的 Launch 分布

按 `blockX` 聚合：

| blockX | 次数 | 总耗时 | 平均 |
|---:|---:|---:|---:|
| 256 | 11,200 | 712.97 ms | 63.66 μs |
| 128 | 6,080 | 294.96 ms | 48.51 μs |

`blockX=256` 不是单一配置：

| blockX | registers/thread | 次数 | 总耗时 | 平均 | gridX 范围 |
|---:|---:|---:|---:|---:|---:|
| 256 | 95 | 5,440 | 318.37 ms | 58.52 μs | 1,076–1,264 |
| 256 | 126 | 5,440 | 254.94 ms | 46.86 μs | 4,304–5,056 |
| 256 | 163 | 160 | 70.63 ms | 441.42 μs | 1,272–3,064 |
| 256 | 164 | 160 | 69.04 ms | 431.49 μs | 5,088–12,256 |

`blockX=128` 也包含多组寄存器配置（66、72、80、121、122、168、186）。

### SGLang 的 Launch 分布

| blockX | registers/thread | 次数 | 总耗时 | 平均 | gridX 范围 |
|---:|---:|---:|---:|---:|---:|
| 128 | 96 | 3,580 | 1,808.01 ms | 505.03 μs | 5,352–40,864 |

### Timeline 实例

SQLite 精确找到了截图中的两条记录：

| 框架 | duration | gridX | blockX | registers/thread | dynamic shared memory |
|---|---:|---:|---:|---:|---:|
| vLLM | 64.662 μs | 1,240 | 256 | 95 | 49,152 B |
| SGLang | 643.447 μs | 8,168 | 128 | 96 | 16,384 B |

这两个实例适合证明“存在不同 Launch 形态”，不适合当成同 workload Kernel 的微基准。

## NCCL AllReduce

按 Kernel 名称拆分：

| 框架 | Kernel | 次数 | 总耗时 | 平均 |
|---|---|---:|---:|---:|
| vLLM | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` | 270 | 504.43 ms | 1,868.25 μs |
| vLLM | `ncclDevKernel_AllReduce_Sum_f32_RING_LL` | 4 | 0.28 ms | 69.65 μs |
| SGLang | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` | 3,579 | 853.78 ms | 238.55 μs |
| SGLang | `ncclDevKernel_AllReduce_Sum_f32_RING_LL` | 2 | 4.47 ms | 2,234.98 μs |

合并后：

| 框架 | 次数 | 总耗时 | 平均 |
|---|---:|---:|---:|
| vLLM | 274 | 504.71 ms | 1,841.99 μs |
| SGLang | 3,581 | 858.25 ms | 239.67 μs |

vLLM trace 还包含 3,844 次 NCCL AllGather、累计 107.49 ms。它不属于本项目的 AllReduce 对比行，SQL 中必须用明确的 `AllReduce` 过滤条件，避免把所有 NCCL Kernel 混在一起。

## GDN 相关观察

Trace 中能检索到以下名称：

- `fused_recurrent_gated_delta_rule_packed_decode_kernel`
- `chunk_gated_delta_rule_fwd_kernel_h_blockdim64`
- `chunk_gated_delta_rule_fwd_kkt_solve_kernel`
- `fused_qkv_split_gdn_prefill_kernel`
- `fused_gdn_gating_kernel`
- `_causal_conv1d_fwd_kernel` / `_causal_conv1d_update_kernel`

这项发现的价值在于：此前从模型架构学习时未重点关注的 Gated Delta Network 组件，可以从实际 GPU Kernel 路径中被识别出来。

