# Figures

## `vllm_fused_moe_timeline.png`

Nsight Systems Timeline 中选中的 vLLM `fused_moe_kernel`：

- Duration: 64.662 μs
- Grid: 1240 × 1 × 1
- Block: 256 × 1 × 1
- Registers per thread: 95
- Dynamic shared memory: 49,152 bytes
- Theoretical occupancy: 25%

## `sglang_fused_moe_timeline.png`

Nsight Systems Timeline 中选中的 SGLang `fused_moe_kernel`：

- Duration: 643.447 μs
- Grid: 8168 × 1 × 1
- Block: 128 × 1 × 1
- Registers per thread: 96
- Dynamic shared memory: 16,384 bytes
- Theoretical occupancy: 31.25%

## 使用边界

两张图证明存在不同 Launch 形态。由于 `gridX` 和潜在 workload 不同，不能仅用这两个实例量化框架整体 MoE 性能差距。

截图显示的 GUI 版本为 NVIDIA Nsight Systems 2026.1.2。
