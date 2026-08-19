# Benchmark 方法与结果

## 目标

Benchmark 的职责是确认端到端性能现象，而不是直接解释原因。本实验比较 vLLM 与 SGLang 在相同模型、GPU、TP、输入长度和输出长度下的吞吐。

## 已记录配置

| 参数 | 值 |
|---|---|
| 模型 | Qwen3.5-35B-A3B MoE |
| GPU | NVIDIA A800-SXM4-80GB ×2 |
| Tensor Parallel | 2 |
| Prompt 长度 | 512 tokens |
| Output 长度 | 128 tokens |
| 并发 | 1、8 |


## 结果

| 并发 | vLLM | SGLang | 绝对差值 | vLLM 相对提升 |
|---:|---:|---:|---:|---:|
| 1 | 171.17 tok/s | 143.74 tok/s | 27.43 tok/s | 19.08% |
| 8 | 704.27 tok/s | 665.66 tok/s | 38.61 tok/s | 5.80% |

相对提升按下式计算：

```text
(vLLM - SGLang) / SGLang × 100%
```

## 如何使用脚本

先分别启动对应框架服务，再执行 Benchmark。脚本默认使用本实验中已记录的 512/128 token 长度，但要求显式设置 `MODEL_PATH` 和 `NUM_PROMPTS`。

### vLLM

```bash
export MODEL_PATH=/path/to/Qwen3.5-35B-A3B
export NUM_PROMPTS=128

MAX_CONCURRENCY=1 bash scripts/benchmark_vllm.sh
MAX_CONCURRENCY=8 bash scripts/benchmark_vllm.sh
```

### SGLang

```bash
export MODEL_PATH=/path/to/Qwen3.5-35B-A3B
export NUM_PROMPTS=128

MAX_CONCURRENCY=1 bash scripts/benchmark_sglang.sh
MAX_CONCURRENCY=8 bash scripts/benchmark_sglang.sh
```

`NUM_PROMPTS=128` 只是命令示例，不是对原实验请求总数的声称。

## 公平比较检查表

- 固定相同模型权重、dtype、量化状态和 tokenizer；
- 固定 TP、可见 GPU、显存策略和最大上下文长度；
- 固定输入/输出 token 长度、请求数、随机种子和并发；
- 两边执行相同数量的 warm-up；
- 记录框架 commit/version、CUDA、PyTorch、驱动和 NCCL 版本；
- 至少重复三次并报告均值与波动；
- 确认服务端日志没有 OOM、fallback 或请求失败；
- 将 Benchmark 与 Profiling 的 workload 分开：Profiler 会引入额外开销。

## 如何解读

单并发领先更明显，可能涉及单请求 Kernel 路径、调度开销或 decode 效率；并发 8 差距缩小，可能说明 batching 后两边 GPU 利用率更接近。但这些都只是研究方向，不能由两行吞吐数字直接证明。

原始记录整理见 [../results/benchmark_results.md](../results/benchmark_results.md)。
