# 复现过程增量学习笔记

> 基于 Awesome-ML-SYS-Tutorial + slime + SGLang 三个项目的实际复现过程
> 记录每一轮学习中**对比之前新增的知识点和理解**
> 用途：面试准备时快速回顾"我学到了什么"

---

## 使用说明

每次完成一个阶段的学习/实验后，在对应章节记录：
1. **新增知识点**：之前不知道，现在理解了的
2. **纠正的误解**：之前理解有误，现在纠正了的
3. **深度理解**：之前只知道概念，现在能讲清楚原理的
4. **实际体感**：之前只看文档，现在动手验证了的

---

## 第一轮：SGLang 推理框架基础

### 新增知识点

#### 1. SGLang 的多进程架构
- **之前**：以为推理引擎就是"加载模型 → 接收请求 → 返回结果"
- **现在**：理解了 SGLang 使用三个独立进程（TokenizerManager、Scheduler、DetokenizerManager），通过 ZMQ 通信。这样做的原因是**避免 Python GIL 限制**，实现真正的并行。

#### 2. RadixCache 的三层设计
- **之前**：只知道"Prefix Caching 可以复用 KV Cache"
- **现在**：理解了三层设计：
  - L1 RadixCache（策略层）：Radix Tree 管理逻辑 token 序列到物理 slot 的映射
  - L2 ReqToTokenPool（页表）：请求到 token 位置的映射
  - L3 TokenToKVPool（物理内存）：实际的 KV tensor 存储
- **类比**：和操作系统的虚拟内存管理类似（页表 + 物理内存 + 缓存策略）

#### 3. 连续批处理的 Prefill-Decode 分离
- **之前**：以为连续批处理就是"动态添加/删除请求"
- **现在**：理解了 Prefill 和 Decode 的调度优先级——**Prefill 优先**。新请求先做 Prefill，然后合并到 running batch 做 Decode。Chunked Prefill 将长 prompt 分块，避免阻塞 Decode 请求。

#### 4. Overlap Scheduling 的实现
- **之前**：以为 CPU 调度和 GPU 计算是串行的
- **现在**：理解了 Overlap Scheduling 通过 **FutureMap** 机制实现 CPU-GPU 重叠——CPU 准备下一批时，GPU 还在计算当前批。具体做法是：CPU 侧用符号引用（symbolic reference）预留 GPU 位置，GPU 侧通过一个小 kernel 解析实际 token。

### 纠正的误解

#### 5. RadixCache 更新时机
- **之前**：以为 RadixCache 在 Pre Schedule 时更新
- **现在**：理解了 RadixCache 在 **Post Schedule** 时更新。原因是防止并发请求命中尚未完全写入 KV 数据的前缀。

#### 6. 推测解码不影响输出分布
- **之前**：以为推测解码是一种"近似"方法，会改变输出
- **现在**：理解了推测解码通过 **rejection sampling** 保证输出分布与原模型完全一致。这是推测解码的核心优势——加速但不改变分布。

### 深度理解

#### 7. Prefill vs Decode 的计算特性差异
- **之前**：知道 Prefill 是 compute-bound，Decode 是 memory-bound
- **现在**：能解释具体原因——Prefill 处理整个输入序列，矩阵大，计算密集；Decode 每步只生成一个 token，矩阵小，但需要读取整个 KV Cache，带宽密集。这就是为什么 PD 分离可以优化资源利用。

#### 8. CUDA Graph 的作用
- **之前**：知道 CUDA Graph 可以加速
- **现在**：理解了 CUDA Graph 的原理——将一系列 CUDA 操作录制为一个图，然后重放，避免 CPU-GPU 之间的 kernel launch 开销。在 Decode 阶段，每个 iteration 的计算图相同（都是处理一个 token），所以 CUDA Graph 特别有效。

---

## 第二轮：slime RL 训练框架

### 新增知识点

#### 9. RL 后训练的闭环流程
- **之前**：只知道"采样 → 训练"的简单流程
- **现在**：理解了完整闭环：`Data Sampling → Weight Update → Data Sampling → ...`，以及关键参数约束：`(rollout-batch-size × n-samples-per-prompt) = (global-batch-size × num-steps-per-rollout)`

#### 10. GRPO 的 advantage 是 sample 级别的
- **之前**：知道 GRPO 不需要 Critic，但没想过 advantage 的粒度
- **现在**：理解了 GRPO 的 advantage 是 **sample 级别的标量**，不是 token 级别的向量。这意味着一个 sample 内所有 token 共享同一个 advantage，信用分配是粗粒度的。

#### 11. 三种 KL 估计器的区别
- **之前**：只知道"KL 散度可以控制策略偏离"
- **现在**：理解了三种估计器：
  - k1：`log(π/π_ref)` — 一阶近似，可能为负
  - k2：`(log(π/π_ref))²/2` — 二阶近似，始终非负
  - k3 (low_var_kl)：`π_ref/π - 1 - log(π_ref/π)` — **非负、无偏、低方差**

#### 12. 权重同步的四种方式
- **之前**：以为权重同步就是"广播参数"
- **现在**：理解了四种方式：
  - full + nccl：标准全量广播
  - full + disk：写 checkpoint 到磁盘
  - delta + nccl：只传输变化的权重（~3% 密度）
  - delta + disk：稀疏增量写入共享文件系统

### 纠正的误解

#### 13. 训练-推理解耦 vs 共置
- **之前**：以为"分离模式一定比共置模式好"
- **现在**：理解了各有优劣——分离模式资源利用率低但显存管理简单；共置模式资源利用率高但需要精细的显存管理（offload/reload）。

#### 14. 动态采样的目的
- **之前**：以为动态采样是为了"提高效率"
- **现在**：理解了动态采样的核心目的是**避免同质化数据**。如果一个 prompt 的所有 sample 的 reward 都相同（`std = 0`），advantage 全为 0，无法学习。动态采样过滤掉这些无效样本。

### 深度理解

#### 15. Loss Masking 的必要性
- **之前**：知道 Agent 训练需要 Loss Masking，但不理解深层原因
- **现在**：理解了如果工具返回参与 loss，模型会学习"复制"工具输出（因为这些 token 的梯度会降低 loss），而不是学习如何正确使用工具。这会导致模型过度依赖工具输出，而不是学习推理能力。

#### 16. PPO Clipping 的悲观估计
- **之前**：知道 PPO 有 clipping，但不理解为什么取 max
- **现在**：理解了取 max 是**悲观估计**——当 advantage 为正时，我们想增大 ratio，但 clipping 限制了增大幅度；取 max 确保不会过度乐观。当 advantage 为负时，我们想减小 ratio，取 max 确保不会过度惩罚。

---

## 第三轮：训推不一致与算法-系统交叉

### 新增知识点

#### 17. 训推不一致的根因
- **之前**：以为"权重相同，输出就相同"
- **现在**：理解了即使权重完全相同，推理引擎和训练引擎也会产生不同的 log-probabilities，原因是：
  1. 浮点加法非结合性
  2. 不同 CUDA kernel
  3. 不同 batch size 触发不同 reduction order
  4. 不同矩阵形状
  5. MoE 专家选择的级联放大

#### 18. MoE 模型为什么受影响最大
- **之前**：知道 MoE 模型的训推不一致更严重，但不理解原因
- **现在**：理解了 MoE 使用 top-k 路由，logit 的微小差异可能导致选择不同的专家。一旦专家选择不同，后续所有计算都会 diverge，产生**级联放大效应**。

#### 19. TIS/MIS 的实验结论
- **之前**：知道 TIS/MIS 可以缓解训推不一致，但不知道具体效果
- **现在**：理解了实验结论：
  - config 1: token TIS [0.5, 2.0] + geometric MIS [0.99, 1.001] + batch norm → 仍然崩溃
  - config 2: token TIS [0.5, 1.5] + geometric MIS [0.99, 1.001] + batch norm → **没有崩溃**
  - 结论：**MIS + batch normalization 是推荐的默认配置**

### 纠正的误解

#### 20. KL 散度在训推不一致中的角色
- **之前**：以为 KL 散度只是"正则化"
- **现在**：理解了 KL 散度是衡量训推不一致程度的核心指标。K3 KL 持续增大时，说明训推不一致在加剧，可能导致训练崩溃。

### 深度理解

#### 21. Truly On-Policy vs Algorithmic Mitigation
- **之前**：以为"消除训推不一致"是唯一目标
- **现在**：理解了两种策略的权衡：
  - Truly On-Policy：K3 KL = 0，但需要侵入性修改，有性能损失，仅适用于 dense 模型
  - Algorithmic Mitigation：通过 TIS/MIS 修正，轻量有效，适用于所有模型

---

## 第四轮：Agent 训练与多轮交互

### 新增知识点

#### 22. Dummy Messages + Delta Tokens 技巧
- **之前**：以为每轮交互都要重新编码完整上下文
- **现在**：理解了使用 `DUMMY_MESSAGES` 作为模板，只编码增量 token（delta tokens），避免重复的 system prompt / tool instruction 占用 token budget。

#### 23. Session Routing 的实现
- **之前**：知道 Agent 需要"路由到同一 worker"
- **现在**：理解了使用 consistent hashing 路由，确保同一 session 的请求路由到同一 worker，复用 prefix cache。

#### 24. Fan-out 训练
- **之前**：以为一次 rollout 只能返回一个 sample
- **现在**：理解了一次 rollout 可以拆分为多个可训练段（如 subagent 轨迹、主 agent 继续），所有段共享同一个 rollout_id。

### 深度理解

#### 25. Agent 训练的四大挑战
- **之前**：只知道"Agent 训练很难"
- **现在**：能具体列出四大挑战：
  1. **信用分配**：多轮交互中，如何分配奖励到每一步
  2. **Loss Masking**：工具返回不应参与 loss 计算
  3. **Off-Policy 问题**：使用旧策略生成的数据训练新策略
  4. **长序列处理**：Agent 轨迹可能非常长

---

## 第五轮：推理优化与工程落地

### 新增知识点

#### 26. RadixAttention 对 RL 训练的价值
- **之前**：知道 RadixAttention 可以复用 KV Cache
- **现在**：理解了对 RL 训练的特殊价值：
  1. 多轮 Agent：system prompt + 历史对话每轮复用
  2. GRPO 组内采样：同一 prompt 的多个 sample 共享前缀
  3. 权重更新后：旧 KV Cache 失效，需要 flush

#### 27. 推测解码在 RL 中的在线更新
- **之前**：以为推测解码的 draft model 是固定的
- **现在**：理解了在 RL 训练中，draft model（MTP layer）使用 target model 的 hidden states 进行在线 SFT，保持两者对齐。实验表明，在线更新比冻结 draft model 提升 25% 的接受率。

#### 28. Delta Weight Sync 的原理
- **之前**：以为权重同步就是"广播所有参数"
- **现在**：理解了 Delta Weight Sync 只传输变化的权重（~3% 密度），大幅降低通信成本。实现步骤：Diff → Encode → 传输 → 应用。

### 纠正的误解

#### 29. 4090 的 FP8 支持
- **之前**：以为 4090 可以用 FP8 训练
- **现在**：理解了 4090（Ada Lovelace）只支持 FP8 推理，不支持 FP8 训练。训练只能用 BF16。

#### 30. TP 在 4090 上的限制
- **之前**：以为 TP 越大越好
- **现在**：理解了 4090 没有 NVLink，TP 通信走 PCIe（~32GB/s），TP 应该 ≤ 2。对于 8 卡 4090，推荐 `dp=4, tp=2` 而不是 `tp=8`。

---

## 第六轮：业务理解与场景适配

### 新增知识点

#### 31. Prefix Caching 的适用场景
- **之前**：以为"所有场景都应该开启 Prefix Caching"
- **现在**：理解了不适合的场景：
  - 纯生成任务：无共享前缀
  - 高度个性化：每个请求前缀不同
  - 实时性要求极高：缓存匹配有开销

#### 32. 推测解码的适用场景
- **之前**：以为"推测解码总是能加速"
- **现在**：理解了 Accept Rate > 60% 时加速效果明显；< 40% 时可能反而更慢。适合代码补全、多轮对话等高重复性场景。

#### 33. RL 后训练 vs SFT 的选择
- **之前**：以为"RL 一定比 SFT 好"
- **现在**：理解了选择依据：
  - 先做 SFT：数据充足、任务明确、需要快速迭代
  - 先做 RL：需要探索新策略、有明确 reward 信号、需要提升推理能力

### 深度理解

#### 34. 资源有限时的优化优先级
- **之前**：不知道应该优先优化什么
- **现在**：理解了按 ROI 排序的优先级：
  1. KV 缓存大小（`--mem-fraction-static`）：直接影响并发能力
  2. CUDA Graph（`--cuda-graph-max-bs`）：直接影响 decode 吞吐
  3. 调度策略（`--schedule-policy lpm`）：间接影响缓存命中率
  4. Chunked Prefill：影响长 prompt 处理能力
  5. DP 实例：水平扩展应对流量增长

---

## 附录：关键代码位置速查

| 模块 | 文件路径 | 关键内容 |
|------|----------|---------|
| SGLang 调度器 | `python/sglang/srt/managers/scheduler.py` | 事件循环、批处理策略 |
| RadixCache | `python/sglang/srt/mem_cache/radix_cache.py` | 前缀缓存、LRU 淘汰 |
| 推测解码 | `python/sglang/srt/speculative/` | EAGLE、MTP、NGRAM |
| RL 算法 | `slime/utils/ppo_utils.py` | GRPO/PPO/CISPO 实现 |
| 损失函数 | `slime/backends/megatron_utils/loss.py` | policy_loss_function |
| 权重同步 | `slime/backends/megatron_utils/update_weight/` | Delta Weight Sync |
| Agent 适配 | `slime/agent/adapters/` | Anthropic/OpenAI 协议 |

---

**最后更新**：2026-06-27

**使用建议**：
1. 每完成一个阶段的学习/实验后，更新对应章节
2. 重点记录"纠正的误解"——这些是最有价值的面试素材
3. 面试前快速回顾本文档，唤醒记忆
