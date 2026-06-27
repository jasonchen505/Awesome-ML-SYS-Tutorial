# Awesome-ML-SYS-Tutorial 全链路复现计划（8×4090）

> 基于 Awesome-ML-SYS-Tutorial + slime + SGLang 三个项目的系统性学习与复现方案
> 硬件：8×RTX 4090（24GB 显存，PCIe 4.0，无 NVLink）
> 目标：以算法岗面试为导向，完整掌握 LLM 推理 & Agent & RL 后训练的核心技术

---

## 一、硬件能力评估与限制分析

### 1.1 RTX 4090 关键参数

| 参数 | 数值 | 对复现的影响 |
|------|------|-------------|
| 显存 | 24GB GDDR6X | 单卡可跑 8B FP16（~16GB），4B 舒适 |
| 内存带宽 | 1008 GB/s | decode 阶段 memory-bound，影响推理速度 |
| FP16 算力 | 82.6 TFLOPS | 足够训练小模型 |
| NVLink | **不支持** | TP 通信走 PCIe（~32GB/s），TP≤2 |
| FP8 训练 | **不支持**（仅推理） | 只能用 BF16 训练 |
| 总显存 | 192GB（8卡） | 可支持 4B 模型 colocate 训练 |

### 1.2 模型规模可行性

| 模型 | 参数量 | 4090 可行性 | 推荐配置 | 阶段 |
|------|--------|-------------|----------|------|
| Qwen2.5-0.5B | 500M | ✅ 完全可行 | TP=1, colocate | 入门验证 |
| Qwen3-1.7B | 1.7B | ✅ 可行 | TP=1, colocate | 基础训练 |
| Qwen2.5-3B | 3B | ✅ 可行 | TP=2, colocate | 标准训练 |
| Qwen3-4B | 4B | ✅ 可行 | TP=2, colocate | 标准训练 |
| Qwen2.5-7B | 7B | ⚠️ 需优化 | TP=2, separated | 挑战目标 |
| Qwen3-14B | 14B | ❌ 困难 | 需要更多显存 | 超出能力 |
| MoE 模型 | 30B+ | ❌ 不可行 | 需要 H100 | 超出能力 |

### 1.3 核心限制与应对策略

| 限制 | 影响 | 应对策略 |
|------|------|---------|
| 显存不足 | 大模型/OOM | 降低 `mem-fraction-static`、减小 batch size、启用重计算 |
| PCIe 带宽 | TP 效率低 | TP≤2、优先 DP、colocate 模式 |
| 无 FP8 训练 | 训练速度慢 | 使用 BF16、FP8 仅用于推理 |
| 无 NVLink | 权重同步慢 | 使用 delta weight sync |

---

## 二、总体复现规划（6 周）

```
┌─────────────────────────────────────────────────────────────────────┐
│                    6 周复现路线图                                    │
├─────────────────────────────────────────────────────────────────────┤
│ Week 1: 环境搭建 + SGLang 基础推理                                  │
│   └─ 产出：SGLang 推理服务正常运行，理解核心架构                      │
│ Week 2: SGLang 核心特性深入 + 性能基准                               │
│   └─ 产出：Prefix Caching / 推测解码 / 连续批处理的实验报告           │
│ Week 3: slime 小模型 GRPO 训练                                      │
│   └─ 产出：0.5B 模型训练曲线，理解 RL 后训练全流程                    │
│ Week 4: slime 中等模型 + RL 算法对比                                 │
│   └─ 产出：3B-4B 模型训练，GRPO vs PPO 对比实验                      │
│ Week 5: Agent 训练 + 高级特性                                       │
│   └─ 产出：多轮交互训练、Loss Masking、Session Routing               │
│ Week 6: 端到端项目 + 总结复盘                                       │
│   └─ 产出：完整项目报告，面试准备材料                                │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 三、Week 1：环境搭建 + SGLang 基础推理

### 3.1 目标
- 搭建完整的开发环境
- 验证 SGLang 基础推理功能
- 理解 SGLang 的核心架构

### 3.2 Day 1-2：环境搭建

**步骤 1：创建 Conda 环境**
```bash
conda create -n sglang python=3.12 -y
conda activate sglang

# 安装 SGLang（从源码）
cd /data/home/yizhou/sglang
pip install -e "python[all]"

# 验证安装
python -c "import sglang; print(sglang.__version__)"
```

**步骤 2：下载模型**
```bash
# 小模型（快速验证）
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --local-dir /root/models/Qwen2.5-0.5B

# 中等模型（标准测试）
huggingface-cli download Qwen/Qwen2.5-3B-Instruct --local-dir /root/models/Qwen2.5-3B

# 推理蒸馏模型（推理基准）
huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --local-dir /root/models/DeepSeek-R1-1.5B
```

**步骤 3：下载训练数据**
```bash
# slime 使用的数据集
huggingface-cli download --repo-type dataset zhuzilin/gsm8k --local-dir /root/data/gsm8k
huggingface-cli download --repo-type dataset aaabiao/dapo-math-17k --local-dir /root/data/dapo-math-17k
```

### 3.3 Day 3-4：SGLang 基础验证

**实验 1：最简验证（无需下载模型）**
```bash
python -m sglang.benchmark.one_batch \
  --model-path TinyLlama/TinyLlama-1.1B-Chat-v0.4 \
  --correct
```
学习点：理解 SGLang 的启动流程、ForwardMode、dummy weights

**实验 2：单卡推理服务器**
```bash
python3 -m sglang.launch_server \
  --model-path /root/models/Qwen2.5-0.5B \
  --host 0.0.0.0 --port 30000

# 测试请求
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"default","messages":[{"role":"user","content":"Hello!"}]}'
```

**实验 3：Engine API 离线批处理**
```python
import sglang as sgl

llm = sgl.Engine(model_path="/root/models/Qwen2.5-0.5B")
outputs = llm.generate(
    ["Hello, my name is", "The capital of France is"],
    {"temperature": 0.8, "top_p": 0.95}
)
print(outputs)
llm.shutdown()
```

### 3.4 Day 5-7：理解核心架构

**阅读顺序**（配合 Awesome-ML-SYS-Tutorial 的文档）：
1. `sglang/code-walk-through/readme.md` — 请求处理全流程
2. `sglang/scheduler/readme-en.md` — 调度器设计
3. `python/sglang/srt/mem_cache/radix_cache.py` — RadixCache 实现

**画架构图**：
```
用户请求 → TokenizerManager → Scheduler → ModelRunner → DetokenizerManager
                ↓                   ↓            ↓              ↓
            ZMQ IPC            RadixCache    CUDA Graph    ZMQ IPC
                              SchedulePolicy  ForwardBatch
```

### 3.5 Week 1 验收标准
- [ ] SGLang 环境安装成功
- [ ] 0.5B 模型推理正常
- [ ] 理解 SGLang 的多进程架构
- [ ] 理解 RadixCache 的基本原理
- [ ] 能解释连续批处理的工作流程

---

## 四、Week 2：SGLang 核心特性深入 + 性能基准

### 4.1 目标
- 深入理解 Prefix Caching
- 验证推测解码效果
- 运行性能基准测试

### 4.2 Day 1-2：Prefix Caching 验证

**对比实验设计**：
```bash
# 实验组：开启 Prefix Caching（默认）
python3 -m sglang.launch_server \
  --model-path /root/models/Qwen2.5-3B \
  --host 0.0.0.0 --port 30000 \
  --mem-fraction-static 0.85 \
  --schedule-policy lpm

# 对照组：关闭 Prefix Caching
python3 -m sglang.launch_server \
  --model-path /root/models/Qwen2.5-3B \
  --host 0.0.0.0 --port 30001 \
  --mem-fraction-static 0.85 \
  --disable-radix-cache

# 基准测试
python3 -m sglang.bench_serving \
  --backend sglang --port 30000 \
  --dataset-name random --num-prompts 200 \
  --random-input-len 1024 --random-output-len 256 \
  --output-file results_with_cache.jsonl

python3 -m sglang.bench_serving \
  --backend sglang --port 30001 \
  --dataset-name random --num-prompts 200 \
  --random-input-len 1024 --random-output-len 256 \
  --output-file results_without_cache.jsonl
```

**学习点**：Cache Hit Rate、TTFT 变化、吞吐量变化

### 4.3 Day 3-4：推测解码验证

**EAGLE 推测解码**：
```bash
python3 -m sglang.launch_server \
  --model-path /root/models/Qwen2.5-3B \
  --speculative-algorithm eagle \
  --speculative-num-steps 5 \
  --speculative-eagle-topk 8 \
  --host 0.0.0.0 --port 30000
```

**NGRAM 推测解码**（无需 draft 模型）：
```bash
python3 -m sglang.launch_server \
  --model-path /root/models/Qwen2.5-3B \
  --speculative-algorithm ngram \
  --speculative-num-steps 3
```

### 4.4 Day 5-7：性能基准测试

**不同 batch size 延迟测试**：
```bash
python3 -m sglang.benchmark.one_batch \
  --model-path /root/models/Qwen2.5-3B \
  --batch-size 1 4 8 16 24 \
  --input-len 256 512 1024 \
  --output-len 128 256
```

**不同并发数吞吐测试**：
```bash
for concurrency in 1 4 8 16 32 64; do
  python3 -m sglang.bench_serving \
    --backend sglang \
    --dataset-name random --num-prompts 100 \
    --random-input-len 512 --random-output-len 128 \
    --max-concurrency $concurrency \
    --output-file results_concurrency_${concurrency}.jsonl
done
```

### 4.5 Week 2 验收标准
- [ ] 能解释 Prefix Caching 的工作原理和效果
- [ ] 能解释推测解码的加速原理
- [ ] 理解不同配置对性能的影响
- [ ] 能根据监控指标进行调优
- [ ] 完成至少 3 组对比实验

---

## 五、Week 3：slime 小模型 GRPO 训练

### 5.1 目标
- 搭建 slime 训练环境
- 完成 Qwen2.5-0.5B 的 GRPO 训练
- 理解 RL 后训练的完整流程

### 5.2 Day 1-2：slime 环境搭建

**步骤 1：Docker 环境**
```bash
docker pull slimerl/slime:latest
docker run --rm --gpus all --ipc=host --shm-size=16g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /data/home/yizhou:/workspace \
  -it slimerl/slime:latest /bin/bash
```

**步骤 2：模型权重转换**
```bash
cd /root/slime
source scripts/models/qwen2.5-0.5B.sh

PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/models/Qwen2.5-0.5B \
    --save /root/models/Qwen2.5-0.5B_torch_dist
```

### 5.3 Day 3-5：运行 GRPO 训练

**4090 优化配置脚本**：
```bash
#!/bin/bash
# run-4090-0.5B.sh

pkill -9 sglang; sleep 3; ray stop --force; pkill -9 ray; pkill -9 python; sleep 3
set -ex

source scripts/models/qwen2.5-0.5B.sh

CKPT_ARGS=(
   --hf-checkpoint /root/models/Qwen2.5-0.5B/
   --ref-load /root/models/Qwen2.5-0.5B_torch_dist/
   --save /root/slime_output/0.5B/
   --save-interval 20
)

ROLLOUT_ARGS=(
   --prompt-data /root/data/gsm8k/train.parquet
   --input-key messages --label-key label
   --apply-chat-template --rollout-shuffle
   --rm-type math
   --num-rollout 100
   --rollout-batch-size 16
   --n-samples-per-prompt 4
   --rollout-max-response-len 512
   --rollout-temperature 1
   --global-batch-size 64
)

PERF_ARGS=(
   --tensor-model-parallel-size 1
   --sequence-parallel
   --use-dynamic-batch-size
   --max-tokens-per-gpu 4096
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss --kl-loss-coef 0.001 --kl-loss-type low_var_kl
   --eps-clip 0.2 --eps-clip-high 0.28
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.6
)

ray start --head --node-ip-address 127.0.0.1 --num-gpus 8 --disable-usage-stats

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"env_vars":{"PYTHONPATH":"/root/Megatron-LM","CUDA_DEVICE_MAX_CONNECTIONS":"1"}}' \
   -- python3 train.py \
   --actor-num-nodes 1 --actor-num-gpus-per-node 8 --colocate \
   --calculate-per-token-loss \
   ${MODEL_ARGS[@]} ${CKPT_ARGS[@]} ${ROLLOUT_ARGS[@]} \
   ${GRPO_ARGS[@]} ${PERF_ARGS[@]} ${SGLANG_ARGS[@]}
```

### 5.4 Day 6-7：分析训练结果

**关键指标监控**：
- Reward 曲线：是否收敛
- KL 散度：是否稳定
- Response Length：是否合理
- Grad Norm：是否正常

### 5.5 Week 3 验收标准
- [ ] slime 环境搭建成功
- [ ] 0.5B 模型 GRPO 训练正常运行
- [ ] 理解 RL 后训练的完整流程（Rollout → Reward → Update）
- [ ] 能解释 GRPO 的 advantage 计算方式
- [ ] 能分析训练曲线，判断训练是否正常

---

## 六、Week 4：中等模型 + RL 算法对比

### 6.1 目标
- 训练 3B-4B 模型
- 对比 GRPO vs PPO 的效果
- 理解 KL 散度控制的作用

### 6.2 Day 1-3：3B 模型训练

**4090 配置（3B，TP=2）**：
```bash
# 使用 4 GPU 训练 + 4 GPU 推理（分离模式）
--tensor-model-parallel-size 2
--sglang-mem-fraction-static 0.6
--rollout-batch-size 8
--n-samples-per-prompt 4
--max-tokens-per-gpu 4096
--recompute-granularity full
--recompute-method uniform
--recompute-num-layers 1
```

### 6.3 Day 4-5：GRPO vs PPO 对比

**GRPO 配置**：
```bash
--advantage-estimator grpo
--use-kl-loss --kl-loss-coef 0.001
```

**PPO 配置**：
```bash
--advantage-estimator ppo
--use-kl-loss --kl-loss-coef 0.001
```

**对比指标**：
- 训练速度
- Reward 收敛曲线
- KL 散度变化
- 生成质量

### 6.4 Day 6-7：KL 散度控制实验

**不同 KL 系数对比**：
```bash
# 无 KL 惩罚
--kl-loss-coef 0.0

# 轻度 KL 惩罚
--kl-loss-coef 0.001

# 重度 KL 惩罚
--kl-loss-coef 0.01
```

### 6.5 Week 4 验收标准
- [ ] 3B 模型训练正常运行
- [ ] 完成 GRPO vs PPO 对比实验
- [ ] 理解 KL 散度控制的作用
- [ ] 能解释不同 KL 系数对训练的影响

---

## 七、Week 5：Agent 训练 + 高级特性

### 7.1 目标
- 实现多轮交互训练
- 理解 Loss Masking 机制
- 掌握 Session Routing

### 7.2 Day 1-3：Search-R1 复现

**配置**：
```bash
# 使用 slime 的 search-r1 示例
--custom-generate-function-path examples/search-r1/generate_with_search.py
--custom-rm-path examples/search-r1/reward_func.py
--rollout-max-response-len 8192
--router-policy consistent_hashing
```

**学习点**：
- 多轮交互的 tokenizer 处理
- Loss Masking 的实现
- Session Routing 的作用

### 7.3 Day 4-5：VLM Multi-Turn 训练

**配置**（如果有 VLM 模型）：
```bash
# 使用 Qwen2.5-VL-2B
--rollout-function-path examples/geo3k_vlm_multi_turn/run_geo3k_vlm_multi_turn.py
```

### 7.4 Day 6-7：训推不一致验证

**实验设计**：
```bash
# 启用 TIS/MIS
--use-tis
--custom-config-path examples/train_infer_mismatch_helper/mis.yaml
--custom-tis-function-path examples.train_infer_mismatch_helper.mis.compute_mis_weights_with_cp
```

### 7.5 Week 5 验收标准
- [ ] 理解 Agent 训练的完整流程
- [ ] 理解 Loss Masking 的必要性
- [ ] 理解 Session Routing 的作用
- [ ] 能解释训推不一致的原因和解决方案

---

## 八、Week 6：端到端项目 + 总结复盘

### 8.1 目标
- 完成一个完整的端到端项目
- 总结学习成果
- 准备面试材料

### 8.2 Day 1-3：端到端项目

**项目选择**（根据兴趣）：
1. **数学推理 RL**：用 GRPO 训练 Qwen3-4B 解数学题
2. **代码生成 RL**：用 GRPO 训练 Qwen3-4B 写代码
3. **Agent 训练**：用多轮交互训练搜索 Agent

### 8.3 Day 4-5：总结复盘

**输出文档**：
1. 项目报告：技术方案、实验结果、问题分析
2. 面试准备：关键知识点、项目介绍、深挖问题
3. 学习笔记：增量学习点、心得体会

### 8.4 Day 6-7：面试准备

**准备内容**：
1. 项目介绍模板（3-4 分钟）
2. 技术深挖问题（16 道题）
3. 代码实现细节

### 8.5 Week 6 验收标准
- [ ] 完成端到端项目
- [ ] 输出项目报告
- [ ] 准备面试材料
- [ ] 能流利介绍项目经验

---

## 九、OOM 应急处理流程

当遇到 OOM 时，按以下顺序逐步降低配置：

```
Step 1: --sglang-mem-fraction-static 0.5
Step 2: --max-tokens-per-gpu 2048
Step 3: --rollout-batch-size 4
Step 4: --n-samples-per-prompt 2
Step 5: --recompute-granularity full
Step 6: 分离模式：--actor-num-gpus-per-node 4 --rollout-num-gpus 4
```

---

## 十、关键配置速查表

### 10.1 SGLang 推理配置

| 参数 | 默认值 | 4090 推荐值 | 说明 |
|------|--------|------------|------|
| `--mem-fraction-static` | 0.9 | 0.6-0.85 | KV 缓存池比例 |
| `--cuda-graph-max-bs` | auto(24) | 16-24 | CUDA Graph 最大 BS |
| `--chunked-prefill-size` | auto(2048) | 2048 | Prefill 分块大小 |
| `--schedule-policy` | lpm | lpm | 调度策略 |
| `--kv-cache-dtype` | auto | fp8_e4m3 | KV 缓存精度 |

### 10.2 slime 训练配置

| 参数 | 默认值 | 4090 推荐值 | 说明 |
|------|--------|------------|------|
| `--tensor-model-parallel-size` | 1 | 1-2 | 张量并行 |
| `--max-tokens-per-gpu` | 9216 | 4096 | 每 GPU 最大 token 数 |
| `--rollout-batch-size` | 32 | 8-16 | Rollout batch size |
| `--n-samples-per-prompt` | 8 | 4 | 每 prompt 采样数 |
| `--rollout-max-response-len` | 4096 | 512-2048 | 最大生成长度 |
| `--sglang-mem-fraction-static` | 0.8 | 0.6 | SGLang 显存比例 |

### 10.3 RL 算法配置

| 参数 | GRPO 推荐值 | PPO 推荐值 | 说明 |
|------|------------|-----------|------|
| `--advantage-estimator` | grpo | ppo | 优势估计器 |
| `--eps-clip` | 0.2 | 0.2 | PPO 裁剪下界 |
| `--eps-clip-high` | 0.28 | 0.28 | PPO 裁剪上界 |
| `--kl-loss-coef` | 0.001 | 0.001 | KL 损失系数 |
| `--kl-loss-type` | low_var_kl | low_var_kl | KL 估计器类型 |

---

## 十一、学习资源索引

### 11.1 核心文档

| 文档 | 路径 | 内容 |
|------|------|------|
| SGLang 代码走读 | `sglang/code-walk-through/readme.md` | 请求处理全流程 |
| SGLang 调度器 | `sglang/scheduler/readme-en.md` | 调度器设计 |
| slime 代码走读 | `rlhf/slime/code-walk-through/readme_en.md` | RL 框架架构 |
| 权重更新机制 | `rlhf/sys-design/readme-1-EN.md` | 三种更新方式 |
| 训推不一致 | `rlhf/slime/mismatch/blog-en.md` | 根因分析与解决方案 |
| 推测解码 in RL | `rlhf/slime/spec/readme-en.md` | Online SFT draft model |

### 11.2 示例代码

| 示例 | 路径 | 说明 |
|------|------|------|
| Search-R1 | `slime/examples/search-r1/` | 多轮搜索 Agent |
| Geo3K VLM | `slime/examples/geo3k_vlm/` | VLM 单轮 RL |
| Geo3K VLM Multi-Turn | `slime/examples/geo3k_vlm_multi_turn/` | VLM 多轮 RL |
| Fully Async | `slime/examples/fully_async/` | 异步训练 |

### 11.3 面试准备文档

| 文档 | 路径 | 内容 |
|------|------|------|
| LLM 面试准备 | `LLM_Interview_Preparation.md` | infra 视角 |
| 技术面试深挖 | `Technical_Interview_Deep_Dive.md` | 五类问题应对 |
| 算法岗面试 | `LLM_Algorithm_Intern_Interview_Guide.md` | 算法视角 |
| 五类能力应对 | `Technical_Interview_Five_Categories_Algorithm.md` | 算法岗版 |

---

**最后更新**：2026-06-27

**适用场景**：8×RTX 4090 硬件条件下的 LLM 推理 & Agent & RL 后训练全链路复现
