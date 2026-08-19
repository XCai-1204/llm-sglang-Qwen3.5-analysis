# 结论、推断与待验证问题

## 已由现有证据支持

1. 在记录的 Benchmark 中，vLLM 单并发吞吐为 171.17 tok/s，SGLang 为 143.74 tok/s，vLLM 领先约 19.1%。
2. 并发 8 时，vLLM 为 704.27 tok/s，SGLang 为 665.66 tok/s，vLLM 领先约 5.8%。
3. SGLang 的 `fused_moe_kernel` 平均单次耗时约为 vLLM 的 8.66×；但调用次数更少。
4. 本次 trace 中 SGLang 的 `fused_moe_kernel` 累计时间约为 vLLM 的 1.79×。
5. SGLang 的 AllReduce 次数显著更多，单次更短，trace 内累计耗时约为 vLLM 的 1.70×。
6. Timeline 与 SQLite 都验证了两边存在不同的 `gridX`、`blockX`、寄存器和动态共享内存配置。
7. 两份 trace 都能识别到 GDN/causal-conv 相关 Kernel。

## 可以提出、但尚未证明的解释

- 两个框架对 MoE 工作量采用了不同粒度的切分；
- 两边可能选择了不同 Kernel 变体或 MoE backend；
- SGLang 的 TP 通信被组织成更细粒度、更频繁的 AllReduce；
- batching、CUDA Graph、调度或专家路由分布可能共同影响 Kernel 形态。

## 当前项目结论

> 本实验通过 Benchmark 确认 vLLM 在给定环境中的吞吐优势，并通过 Nsight Systems 与 SQLite 将后续调查聚焦到 MoE Kernel 的执行粒度和 TP 通信组织。现有数据证明两框架采用了明显不同的 GPU 执行形态，但尚不足以归因于某一个 block size、单一 Kernel 实现或单一通信策略。
