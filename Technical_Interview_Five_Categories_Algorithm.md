# 技术面试五类能力深度应对：算法岗版

> 基于 Awesome-ML-SYS-Tutorial + slime + SGLang 三个项目的深度分析
> 面向：LLM & Agent 应用 / 后训练方向的 MS 在读算法岗候选人
> 核心定位：**不是背答案，而是展示"真正做过项目的人"的思维方式**

---

## 使用说明

本文档与 `Technical_Interview_Deep_Dive.md`（偏 infra 视角）互补，**本文档偏算法视角**。每道题都按以下结构组织：

```
【面试官问法】→ 面试官实际会怎么问
【思维框架】→ 回答的思考路径（不是背答案）
【深度回答】→ 展示真正理解的回答
【面试官会追问什么】→ 预判追问，展示更深理解
【如果你不会】→ 诚实承认 + 展示相关知识的策略
```

---

## 第一类：底层原理深入理解

> **面试官考察目标**：不是看你能不能背概念，而是看你能不能讲清楚**这个方法解决什么问题、存在哪些局限性、有哪些改进方法**。能讲清楚这些的人，才是真正理解了的人。

### 1.1 GRPO 算法

**【面试官问法】**
- "解释一下 GRPO 算法，它和 PPO 有什么区别？"
- "为什么 DeepSeek 选择 GRPO 而不是 PPO？"
- "GRPO 的 advantage 是 sample 级别的，这有什么影响？"

**【思维框架】**
回答这类问题的核心是：**先讲清楚"解决什么问题"，再讲"怎么解决的"，最后讲"有什么代价"**。

**【深度回答】**

GRPO 解决的核心问题是 **PPO 需要 Critic 模型带来的高计算成本和训练不稳定**。

PPO 需要一个 Critic 模型 V(s) 来估计每个 token 的价值，然后通过 GAE 计算 advantage：
```
δ_t = r_t + γ·V(s_{t+1}) - V(s_t)
A_t = δ_t + (γλ)·A_{t+1} + (γλ)²·A_{t+2} + ...
```

这带来三个问题：
1. **计算成本翻倍**：需要额外的 Critic 模型前向传播
2. **Critic 训练不稳定**：价值函数估计可能不准确，导致 advantage 估计有偏
3. **超参数多**：GAE 的 γ 和 λ 需要调参

GRPO 的解决方案很简洁：**用组内相对优势代替绝对优势**，直接省掉 Critic：
```
A_i = (r_i - mean(r_group)) / std(r_group)
```

对每个 prompt 生成一组响应（如 8 个），用组内 reward 的均值和标准差做标准化，得到相对优势。

**但这个设计是有代价的**：
- **信用分配是粗粒度的**：一个 sample 内所有 token 共享同一个 advantage。在稀疏奖励场景下合理，但在密集奖励场景下不够精细
- **稀疏奖励问题**：如果一个 prompt 的所有 sample 的 reward 都相同（`std = 0`），advantage 全为 0，无法学习。这就是**动态采样**的动机
- **组内偏差**：advantage 只反映组内相对排名，可能与全局最优不一致

**【面试官会追问什么】**

追问 1："GRPO 的 advantage 是 sample 级别的标量，PPO 的是 token 级别的向量，这在训练效果上有什么具体影响？"

→ GRPO 的所有 token 共享同一个 advantage，意味着"如果这个 sample 最终答对了，所有 token 都被认为是好的"。这在**稀疏奖励**场景下是合理的（只有最终结果有奖励），但在**密集奖励**场景下可能有问题——比如 Agent 的前几步很好但最后一步出错，GRPO 会把前面的好步骤也标记为"坏的"。

追问 2："你提到动态采样，具体是怎么实现的？"

→ 采样比实际需要更多的 prompts（`over_sampling_batch_size > rollout_batch_size`），每个 prompt 生成多个响应，然后过滤掉 reward 标准差为 0 的样本。如果过滤太严格，自动触发新一轮过采样。在 slime 中通过 `--dynamic-sampling-filter-path` 参数配置。

### 1.2 PPO Clipping 机制

**【面试官问法】**
- "PPO 的 clipping 机制是怎么工作的？为什么取 max 而不是 min？"
- "有没有什么改进的 clipping 方法？"

**【思维框架】**
先讲"解决什么问题"，再讲"怎么解决的"，最后讲"有什么局限性"。

**【深度回答】**

PPO Clipping 解决的核心问题是 **RL 训练中策略更新幅度过大导致的训练不稳定**。

标准策略梯度的问题是：如果一次更新太大，策略可能跳到一个很差的区域，而且很难恢复。PPO 通过 clipping 限制 ratio 的范围来控制更新幅度：

```python
ratio = exp(new_logp - old_logp)  # π_θ / π_old
pg_losses1 = -ratio * advantages
pg_losses2 = -ratio.clamp(1 - ε, 1 + ε_high) * advantages
loss = max(pg_losses1, pg_losses2)  # 悲观估计
```

**为什么取 max（悲观估计）？**
- 当 advantage > 0（好的动作）时，我们想增大 ratio，但 clipping 限制了增大幅度；取 max 确保我们不会过度乐观
- 当 advantage < 0（坏的动作）时，我们想减小 ratio，但 clipping 限制了减小幅度；取 max 确保我们不会过度惩罚

**局限性**：
1. **保守性**：悲观估计可能限制策略改进
2. **负 advantage 问题**：当 advantage 为负时，ratio 越大 loss 越小，可能导致过度优化
3. **不对称裁剪**：`eps_clip` 和 `eps_clip_high` 需要分别调参

**改进方法**：
- **Dual-Clip PPO**：对负 advantage 额外施加上界裁剪
- **CISPO**：ratio 在 stop-gradient 下裁剪，被裁剪的 token 仍然贡献梯度

**【面试官会追问什么】**

追问："CISPO 和 PPO 的关键区别是什么？为什么 CISPO 的样本效率更高？"

→ PPO 中，梯度通过 ratio 流动，被裁剪的 token **不贡献梯度**。CISPO 中，ratio 被 detach，梯度通过 `log_probs` 流动，被裁剪的 token **仍然贡献梯度**。这意味着 CISPO 可以从更多样本中学习。

### 1.3 KL 散度控制

**【面试官问法】**
- "KL 散度在 RL 训练中有什么作用？"
- "你了解几种 KL 估计器？哪个最好？为什么？"

**【深度回答】**

KL 散度在 RL 中的作用**不仅仅是正则化**，它有多种使用方式：

1. **作为 reward 惩罚**：`rewards[t] -= kl_coef * KL_t`（token 级别）
2. **作为额外 loss 项**：`loss = loss + kl_loss_coef * kl`
3. **作为监控指标**：`kl_coef = 0` 时只记录不参与梯度
4. **作为 TIS 的判断依据**：序列级 KL 超阈值时 mask 掉

三种 KL 估计器（来自 Schulman 的博客）：
- **k1**：`KL = log(π/π_ref)` — 一阶近似，可能为负
- **k2**：`KL = (log(π/π_ref))²/2` — 二阶近似，始终非负
- **k3 (low_var_kl)**：`KL = π_ref/π - 1 - log(π_ref/π)` — **非负、无偏、低方差**

**为什么 k3 最好？** 它是**非负的**（不会出现负 KL）、**无偏的**（期望等于真实 KL）、**低方差的**（比 k1 和 k2 更稳定）。在 RL 训练中，KL 估计的稳定性直接影响训练稳定性。

**【面试官会追问什么】**

追问："KL 散度在训推不一致问题中扮演什么角色？"

→ KL 散度是衡量训推不一致程度的核心指标。我们用 K3 KL 来量化推理引擎和训练引擎产生的 log-probabilities 的差异。Dense 模型的 K3 KL 约 10⁻⁵ 到 10⁻³，MoE 模型约 10⁻³ 到 10⁻¹。当 K3 KL 持续增大时，说明训推不一致在加剧，可能导致训练崩溃。

### 1.4 训推不一致（Training-Inference Mismatch）

**【面试官问法】**
- "什么是训推不一致？为什么会出现这个问题？"
- "为什么 MoE 模型的训推不一致比 dense 模型严重得多？"
- "怎么解决这个问题？"

**【思维框架】**
这是**算法与系统交叉**的问题，最能体现"系统思维"。回答时要从**根因 → 影响 → 解决方案**三个层次展开。

**【深度回答】**

**定义**：即使权重完全相同，推理引擎（SGLang）和训练引擎（Megatron/FSDP）对同一 token 产生的 log-probabilities 也不同。

**根因**（按严重程度排序）：
1. **浮点加法非结合性**：`(a + b) + c ≠ a + (b + c)`，不同 reduction order 产生不同结果
2. **不同 CUDA kernel**：训练和推理使用不同实现
3. **不同 batch size**：触发不同 reduction order
4. **不同矩阵形状**：推理是逐 token 小矩阵，训练是全序列大 batch
5. **MoE 专家选择**：logit 微小差异导致不同专家路由（**级联放大**）

**为什么 MoE 模型受影响最大？** MoE 模型使用 top-k 路由，logit 的微小差异可能导致选择不同的专家。一旦专家选择不同，后续所有计算都会 diverge，产生**级联放大效应**。Dense 模型没有这种离散选择，差异是连续的、可控的。

**对 PPO 的影响**：标准 PPO 公式假设 sampling distribution 和 training distribution 一致。但实际中，分子 π_θ 来自训练引擎，分母 π_old 来自推理引擎，两者不一致导致 importance sampling ratio 有偏。

**两种解决方案**：

1. **Truly On-Policy（位对齐）**：让训练和推理引擎产生完全相同的 log-probabilities。使用 batch-invariant kernels、FlashAttention-3、DeepGEMM。结果：K3 KL = 0。局限：仅适用于 vanilla dense 模型，有性能损失。

2. **算法缓解（Importance Sampling 修正）**：显式地用 importance sampling 修正分布差异。包括 TIS（裁剪 importance ratio）、MIS（序列级 KL 超阈值时 mask 掉）、Self-Normalization（归一化 batch 权重）。

**【面试官会追问什么】**

追问 1："TIS 的裁剪上下界如何选择？"

→ 上界防止过度信任推理引擎的数据（`w > upper` 说明推理引擎的概率远高于训练引擎），下界防止丢弃有用数据（`w < lower` 说明推理引擎的概率远低于训练引擎）。典型值：`[0.5, 2.0]`。在 MoE 模型上，可能需要更紧的界如 `[0.5, 1.5]`。

追问 2："MIS 的实验结果如何？什么配置最有效？"

→ 在 Qwen30B-A3B 的实验中：
- config 1: token TIS [0.5, 2.0] + geometric MIS [0.99, 1.001] + batch norm → 仍然崩溃
- config 2: token TIS [0.5, 1.5] + geometric MIS [0.99, 1.001] + batch norm → **没有崩溃**
- config 3: token TIS [0.5, 1.5] + geometric MIS [0.99, 1.001] → **没有崩溃**
- config 4: token TIS [0.5, 1.5] → 崩溃

结论：**MIS + batch normalization 是推荐的默认配置**。

---

## 第二类：实验和方案验证能力

> **面试官考察目标**：不仅关注你做了什么，更关注**怎么证明它是有效的**。追问实验细节的过程中，能看出你对项目是否有真正深入理解。

### 2.1 如何验证 RL 训练的正确性？

**【面试官问法】**
- "你怎么知道你的 RL 训练是正确的？"
- "如果 KL 在第一步不为 0，你会怎么排查？"
- "如何验证 Advantage 计算的正确性？"

**【思维框架】**
展示**分层验证**的思维方式：从最基础的数值检查到端到端的训练验证。

**【深度回答】**

RL 训练的正确性验证需要**分层进行**，因为 RL 系统太复杂了，一步到位很难定位问题。

**第一层：精度对齐验证**
```python
# 检查 log_probs 和 ref_log_probs 是否相等（第一步 KL 应为 0）
assert torch.allclose(log_probs, ref_log_probs, atol=1e-6)
# 检查 grad_norm 是否较小
assert grad_norm < 1.0
```
这是最基础的检查。如果第一步 KL 不为 0，说明推理引擎和训练引擎的数值不一致（训推不一致问题）。

**第二层：权重同步验证**
```bash
--check-weight-update-equal  # 验证 Megatron -> SGLang 权重同步
```
验证权重从训练引擎同步到推理引擎后是否一致。

**第三层：数值一致性验证**
```python
# 验证不同 CP 大小下梯度一致
for cp_size in [1, 2, 4]:
    grad = compute_loss(cp_size=cp_size)
    assert torch.allclose(grad, expected_grad, atol=1e-5)
```
验证不同并行策略下，梯度计算是否一致。

**第四层：确定性复现**
```bash
--sglang-enable-deterministic-inference
--deterministic-mode
NCCL_ALGO=Ring
NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
CUBLAS_WORKSPACE_CONFIG=:4096:8
```
使用确定性模式，确保结果可复现。

**第五层：分离调试**
```bash
--debug-rollout-only  # 仅调试推理
--debug-train-only    # 仅调试训练
--save-debug-rollout-data /tmp/rollout.pt
--load-debug-rollout-data /tmp/rollout.pt
```
将推理和训练分离，逐步排查问题。

**【面试官会追问什么】**

追问："如果第一步 KL 不为 0，你会怎么排查？"

→ 可能原因：
1. **Transformer Engine 中的非确定性 kernel** → 使用 `--attention-backend flash` 强制 Flash Attention
2. **参数加载错误** → 检查 Megatron checkpoint 加载日志
3. **参数更新错误** → 检查并行策略下的参数名映射
4. **精度不匹配** → 检查训练和推理是否使用相同精度（bf16/fp16）

### 2.2 如何验证推测解码的加速效果？

**【面试官问法】**
- "推测解码的加速效果怎么验证？"
- "Accept Rate 是怎么计算的？"
- "为什么在线更新 draft model 比冻结的好？"

**【深度回答】**

**实验设计**：
```bash
# 基线：无推测解码
python -m sglang.launch_server --disable-speculative-decoding

# 实验：EAGLE 推测解码
python -m sglang.launch_server \
  --speculative-algorithm eagle \
  --speculative-num-steps 5 \
  --speculative-eagle-topk 8
```

**关键指标**：
- **Decode Throughput**：token/s
- **Accept Rate**：被接受的 draft token 数 / 总 draft token 数
- **额外开销**：draft 模型的 GPU 内存

**实验结果**（Mimo-7B-RL，H200 集群）：

| 配置 | Rollout Throughput | Accept Rate | 额外开销 |
|------|-------------------|-------------|---------|
| 无推测解码 | 1219 token/s | - | - |
| 冻结 draft model | 1464 token/s | ~60% | 5% GPU 内存 |
| 在线更新 draft model | 1667 token/s | ~70% | 5% GPU 内存 |

**为什么在线更新比冻结好？** RL 训练过程中，target model 不断变化。如果 draft model 冻结，它会逐渐与 target model 偏离，导致接受率下降。在线 SFT 保持两者对齐，在后期训练阶段，在线更新比冻结提升 25% 的接受率。

**【面试官会追问什么】**

追问："推测解码会影响 RL 训练的质量吗？"

→ 不会。理论上，推测解码通过 rejection sampling 保证输出分布与原模型一致。实验中，reward 曲线在有无推测解码的情况下保持一致。这是推测解码的核心优势——**加速但不改变分布**。

### 2.3 如何验证 Agent 训练的 Loss Masking？

**【面试官问法】**
- "为什么 Agent 训练需要 Loss Masking？"
- "怎么验证 Loss Masking 是有效的？"
- "Loss Masking 是怎么实现的？"

**【深度回答】**

**实验设计**：
```python
# 1. 不使用 Loss Masking
loss = compute_loss(logits, labels, loss_mask=None)

# 2. 使用 Loss Masking
loss_mask = [1] * model_tokens + [0] * tool_tokens
loss = compute_loss(logits, labels, loss_mask=loss_mask)
```

**对比指标**：
| 配置 | 最终 Reward | 训练稳定性 | 工具使用准确率 |
|------|------------|-----------|--------------|
| 无 Loss Masking | 0.6 | 差（loss 波动大） | 0.5 |
| 有 Loss Masking | 0.85 | 好（loss 平稳） | 0.8 |

**为什么无 Loss Masking 会导致训练不稳定？** 工具返回的 token 不是模型生成的，如果参与 loss 计算，会让模型学习"复制"工具输出，而不是学习如何正确使用工具。这会导致梯度方向不准确，训练不稳定。

**实现细节**：在 slime 的 `Sample` 数据类中，`loss_mask` 字段标记每个 token 是否参与 loss 计算。在 `loss.py` 中：
```python
loss = (loss * loss_mask).sum() / loss_mask.sum()
```

---

## 第三类：问题定位能力

> **面试官考察目标**：很多同学只讲实验结果、堆一堆复杂流程，但没有讲**优化思路与解决方案**。面试官想看的是你遇到问题时的**排查思路**。

### 3.1 模型上线后能力突然下降

**【面试官问法】**
- "模型上线后能力突然下降，你会怎么排查？"
- "如果权重同步出错了，会有什么表现？"

**【思维框架】**
展示**系统性的排查思路**，而不是东一榔头西一棒子。

**【深度回答】**

**排查步骤**：

**Step 1：确认问题范围**
- 是所有请求都下降，还是特定类型请求？
- 是突然下降，还是逐渐下降？
- 有没有发布新版本或修改配置？

**Step 2：检查权重同步**
```bash
# 验证权重同步是否正确
--check-weight-update-equal
```
如果权重同步出错，模型的输出会完全偏离预期，因为模型参数本身就是错的。

**Step 3：检查数值精度**
```bash
# 确保 dtype 与训练一致
--dtype bf16

# 检查 KV Cache 精度
--kv-cache-dtype auto  # 而非 fp8
```
如果训练用 bf16 但推理用 fp16，数值差异可能导致输出不同。

**Step 4：检查前缀缓存**
```bash
# 禁用前缀缓存，排除缓存污染
--disable-radix-cache
```
如果旧的 KV Cache 被错误复用（比如权重更新后没有 flush），会导致输出错误。

**Step 5：检查训推不一致**
```bash
# 检查 K3 KL 指标
curl http://localhost:30000/metrics | grep k3_kl
```
如果 K3 KL 持续增大，说明训推不一致在加剧，可能导致训练崩溃。

**【面试官会追问什么】**

追问："你能举一个你实际遇到的例子吗？"

→ 在 MoE 模型的 RL 训练中，我们观察到：在 step 320 左右，grad norm 突然从 0.07 降到 0.02，然后 reward 急剧下降，K3 KL 急剧上升。排查后发现是 MoE 的专家选择导致的训推不一致——logit 的微小差异导致选择不同的专家，产生级联放大效应。解决方案是启用 MIS + batch normalization。

### 3.2 训练结果和预期不一致

**【面试官问法】**
- "实验结果和预期不一致，你会怎么排查？"
- "如果 reward 不收敛，你会检查哪些地方？"

**【深度回答】**

**排查思路**：

**Step 1：检查数据**
```python
# 检查训练数据是否正确
print(sample.prompt)
print(sample.response)
print(sample.reward)

# 检查 loss_mask 是否正确
print(sample.loss_mask)
```
数据问题是最高频的 bug 来源。

**Step 2：检查超参数**
```bash
# 打印所有超参数
--print-args

# 检查关键超参数
--advantage-estimator grpo
--eps-clip 0.2
--kl-loss-coef 0.001
```

**Step 3：检查数值**
```python
# 检查 log_probs 是否合理
print(f"log_probs range: {log_probs.min()}, {log_probs.max()}")
print(f"KL: {kl}")
print(f"advantage range: {advantages.min()}, {advantages.max()}")
```

**Step 4：检查梯度**
```python
# 检查梯度是否合理
print(f"grad_norm: {grad_norm}")
assert not torch.isnan(grad).any()
assert not torch.isinf(grad).any()
```

**Step 5：逐步验证**
```python
# 1. 验证 rollout 是否正确
--debug-rollout-only --save-debug-rollout-data /tmp/rollout.pt

# 2. 验证训练是否正确
--load-debug-rollout-data /tmp/rollout.pt --debug-train-only
```

**【面试官会追问什么】**

追问："如果 reward 不收敛，但 loss 在下降，可能是什么原因？"

→ 可能原因：
1. **Reward Hacking**：模型找到了一种"骗"reward 的方式，但实际上并没有真正解决问题
2. **KL 惩罚太弱**：策略偏离太远，虽然 loss 下降但实际效果变差
3. **数据分布问题**：训练数据不能代表实际场景
4. **评估指标问题**：reward 指标和实际效果不一致

### 3.3 系统上线后突然十分缓慢

**【面试官问法】**
- "系统上线后突然变慢，你会怎么排查？"
- "KV 缓存满了会有什么表现？"

**【深度回答】**

**排查步骤**：

**Step 1：检查 GPU 利用率**
```bash
nvidia-smi  # 查看 GPU 利用率和显存使用
```

**Step 2：检查队列状态**
```bash
curl http://localhost:30000/metrics | grep -E "num_running_reqs|num_queue_reqs"
```

**Step 3：检查关键指标**
```bash
# 检查 token usage（KV 缓存利用率）
curl http://localhost:30000/metrics | grep token_usage

# 检查是否有频繁 retract
curl http://localhost:30000/metrics | grep num_retracted_reqs_total
```

**Step 4：检查日志**
```bash
# 检查 Decode batch 日志
grep "Decode batch" server.log | tail -20
```

**KV 缓存满了会有什么表现？**
- `token_usage` 接近 1.0
- 频繁 retract（请求被回退到等待队列）
- 吞吐量急剧下降
- 延迟波动大

**解决方案**：
| 原因 | 解决方案 |
|------|---------|
| KV 缓存满 | 降低 `--mem-fraction-static`，或增大 `--schedule-conservativeness` |
| CUDA Graph 未启用 | 检查 `--cuda-graph-max-bs`，适当增大 |
| 队列过长 | 增加 DP 实例，或使用 PD 分离 |
| 频繁 retract | 增大 `--schedule-conservativeness` 到 1.3 |

---

## 第四类：工程落地能力

> **面试官考察目标**：不仅看理论，更看实际动手与工程落地能力。理论可行的方案，实际工程落地中可能不可行。

### 4.1 如何部署一个生产级的 RL 训练系统？

**【面试官问法】**
- "你会怎么部署一个 RL 训练系统？"
- "训练和推理应该放在同一台机器上还是分开？"
- "权重同步有哪些方式？各有什么优缺点？"

**【思维框架】**
展示**权衡思考**的能力，不是只给一个答案，而是讲清楚不同方案的优缺点。

**【深度回答】**

**部署模式的权衡**：

| 方面 | 分离模式 | 共置模式 |
|------|----------|----------|
| 资源利用率 | 较低（需要两套 GPU） | 较高（共享 GPU） |
| 通信成本 | 高（需要权重同步） | 低（本地访问） |
| 显存管理 | 简单 | 复杂（需要卸载） |
| 适用场景 | 大规模训练 | 资源受限 |

**权重同步方式**：

| 方式 | 原理 | 优点 | 缺点 |
|------|------|------|------|
| 全量 NCCL 广播 | 广播所有参数 | 简单可靠 | 成本与模型大小线性相关 |
| Delta Weight Sync | 只传输变化的权重（~3% 密度） | 通信成本低 | 实现复杂 |
| CUDA IPC Handle | 传输 GPU 内存指针 | 零拷贝 | 仅限同节点 |
| 磁盘传输 | 写 checkpoint 到磁盘 | 简单 | I/O 瓶颈 |

**共置模式下的显存管理**：
```bash
--colocate           # 共置模式
--offload-train      # 训练时将推理模型卸载到 CPU
--offload-rollout    # 推理时将训练模型卸载到 CPU
--sglang-mem-fraction-static 0.8  # 调整 SGLang 显存占用
```

**【面试官会追问什么】**

追问："Delta Weight Sync 的原理是什么？为什么只传输 3% 的权重？"

→ RL 步之间，只有少量权重变化。Delta Weight Sync 通过逐字节比较当前权重与快照，只传输差异部分。实现步骤：
1. **Diff**：逐字节比较当前权重与快照
2. **Encode**：编码变化的 (position, value) 对
3. **传输**：通过 NCCL 或磁盘
4. **应用**：接收端 NaN-masked overwrite

### 4.2 如何保证 RL 训练的稳定性？

**【面试官问法】**
- "RL 训练容易不稳定，你会怎么保证稳定性？"
- "如果训练突然崩溃，你会怎么恢复？"

**【深度回答】**

**稳定性保证的四个层次**：

**第一层：算法层面**
- PPO Clipping：限制策略更新幅度
- KL 惩罚：防止策略偏离太远
- Dual-Clip PPO：对负 advantage 额外裁剪
- OPSM：序列级 KL 超阈值时 mask 掉

**第二层：系统层面**
- 权重同步验证：`--check-weight-update-equal`
- 训推不一致修正：TIS/MIS
- KV Cache Flush：权重更新前清除旧缓存

**第三层：工程层面**
- 检查点：定期保存训练状态
- 容错机制：Watchdog、Crash Dump
- 健康检查：心跳检测、超时重试

**第四层：监控层面**
- 关键指标监控：reward、KL、grad_norm、response_length
- 告警规则：grad_norm 突降、reward 突降、KL 突增
- Trace Viewer：可视化训练过程

**【面试官会追问什么】**

追问："如果训练突然崩溃，你会怎么恢复？"

→ 
1. **检查崩溃前的指标**：grad_norm 是否突降？KL 是否突增？reward 是否突降？
2. **回滚到最近的检查点**：使用崩溃前的 checkpoint 继续训练
3. **启用更保守的配置**：降低学习率、增大 KL 惩罚、启用 MIS
4. **分析崩溃原因**：是算法问题（训推不一致）还是系统问题（OOM、NCCL hang）？

### 4.3 如何进行性能调优？

**【面试官问法】**
- "如果推理速度太慢，你会怎么优化？"
- "资源有限时，应该优先优化什么？"

**【深度回答】**

**调优优先级**（按 ROI 排序）：

1. **KV 缓存大小**（`--mem-fraction-static`）：直接影响并发能力
   ```bash
   --mem-fraction-static 0.85  # 起始值
   --mem-fraction-static 0.87  # 逐步增加
   --mem-fraction-static 0.89  # 直到 OOM 后回退
   ```

2. **CUDA Graph**（`--cuda-graph-max-bs`）：直接影响 decode 吞吐
   ```bash
   --cuda-graph-max-bs 256  # A100
   --cuda-graph-max-bs 512  # H100
   ```

3. **调度策略**（`--schedule-policy lpm`）：间接影响缓存命中率

4. **Chunked Prefill**（`--chunked-prefill-size`）：影响长 prompt 处理能力

5. **DP 实例**：水平扩展应对流量增长

**【面试官会追问什么】**

追问："推测解码在什么场景下应该启用？"

→ 
- **适合**：代码补全（高重复性）、多轮对话（历史上下文可预测）、文本摘要（结构化输出）
- **不适合**：创造性写作（低重复性）、翻译任务（语言差异大）
- **判断依据**：Accept Rate > 60% 时，加速效果明显；< 40% 时，可能反而更慢

---

## 第五类：业务与实际场景理解

> **面试官考察目标**：关注**实际场景价值和业务价值**。实验环境里能跑通证明有效，但在实际生产中不一定有用。

### 5.1 RL 后训练适合什么场景？

**【面试官问法】**
- "RL 后训练适合什么场景？什么场景不适合？"
- "如果资源有限，应该优先做 RL 还是 SFT？"
- "RL 后训练的上线成本有多高？"

**【深度回答】**

**适合的场景**：

| 场景 | 原因 | 预期收益 |
|------|------|---------|
| 对齐人类偏好 | RLHF/DPO | 输出更符合人类期望 |
| 提高推理能力 | GRPO/PPO | 数学/代码推理提升 |
| Agent 训练 | 工具调用 RL | 工具使用准确率提升 |
| 特定任务优化 | 定制 reward | 任务特定指标提升 |

**不适合的场景**：

| 场景 | 原因 | 替代方案 |
|------|------|---------|
| 数据量不足 | RL 需要大量采样 | 先做 SFT |
| Reward 难以定义 | 需要明确的 reward 信号 | 使用 DPO/RLHF |
| 计算资源有限 | RL 训练成本高 | 使用 DPO |
| 快速迭代 | RL 训练周期长 | 先做 Prompt Engineering |

**上线成本估算**：
- **GPU 成本**：70B 模型约 $10-20/hour × 24h × 7天
- **数据成本**：人工标注 + 数据清洗
- **人力成本**：2-4 周开发 + 调优
- **运维成本**：监控 + 故障处理

**【面试官会追问什么】**

追问："如果资源有限，应该优先做 RL 还是 SFT？"

→ 
- **先做 SFT**：如果数据充足、任务明确、需要快速迭代
- **先做 RL**：如果需要探索新的策略空间、有明确的 reward 信号、需要提升推理能力
- **判断依据**：SFT 是模仿学习，只能学习训练数据中的分布；RL 是策略优化，可以探索新的、更好的策略。如果训练数据已经覆盖了最优策略，SFT 就够了；如果最优策略不在训练数据中，需要 RL。

### 5.2 Agent 训练适合什么场景？

**【面试官问法】**
- "Agent 训练适合什么场景？"
- "Agent 训练的上线成本有多高？"
- "如果资源有限，应该优先优化 Agent 的哪个部分？"

**【深度回答】**

**适合的场景**：

| 场景 | 原因 | 预期收益 |
|------|------|---------|
| 代码生成 | 可验证的 reward（测试通过） | 代码质量提升 |
| 数学推理 | 可验证的 reward（答案正确） | 推理能力提升 |
| 搜索增强 | 环境反馈 | 信息检索能力提升 |
| 工具调用 | 工具反馈 | 工具使用准确率提升 |

**不适合的场景**：

| 场景 | 原因 | 替代方案 |
|------|------|---------|
| 开放式对话 | 难以定义 reward | 使用 RLHF |
| 创造性任务 | 难以评估质量 | 使用人工评估 |
| 实时性要求高 | Agent 交互延迟大 | 简化工具调用 |

**资源有限时的优先级**：
1. **工具设计**：简洁、明确的工具接口
2. **Reward 设计**：可验证的 reward 信号
3. **Loss Masking**：正确处理工具返回
4. **Session Routing**：复用 prefix cache
5. **异步训练**：提高采样效率

### 5.3 如何评估方案的业务价值？

**【面试官问法】**
- "这个方案的业务价值是什么？"
- "用户更关心的是什么？"
- "如果资源有限，应该首先优化哪些部分？"

**【深度回答】**

**评估框架**：

**技术指标**：
- 延迟：TTFT、TPOT、E2E Latency
- 吞吐：token/s、requests/s
- 资源利用率：GPU 利用率、显存利用率

**业务指标**：
- 用户满意度：NPS、CSAT
- 任务完成率：Agent 任务成功率
- 成本效率：$/request、$/token
- ROI：投入产出比

**用户关心什么？**
1. **延迟**：TTFT 和 TPOT 是否满足 SLA
2. **成本**：GPU 成本是否降低
3. **稳定性**：延迟是否稳定，是否有毛刺
4. **可扩展性**：能否水平扩展应对流量增长

**【面试官会追问什么】**

追问："如果资源有限，你会优先优化哪个指标？"

→ 
- **优先优化延迟**：如果用户对延迟敏感（如实时对话）
- **优先优化吞吐**：如果需要处理大量请求（如批量推理）
- **优先优化成本**：如果 GPU 预算有限
- **判断依据**：看业务场景的核心需求。对于实时对话，延迟 > 吞吐 > 成本；对于批量处理，吞吐 > 成本 > 延迟。

---

## 附录：面试回答模板总结

### 模板 1：解决什么问题

"这个技术解决的核心问题是 [问题描述]。具体痛点包括：1）[痛点1]；2）[痛点2]；3）[痛点3]。解决方案是 [方案描述]。但这个设计也有代价：[代价描述]。"

### 模板 2：局限性与改进

"它的局限性在于：1）[局限性1]；2）[局限性2]；3）[局限性3]。改进方向包括：1）[改进1]，原理是 [原理]；2）[改进2]，原理是 [原理]。"

### 模板 3：实验验证

"为了验证有效性，我设计了以下实验：1）[实验1]，指标是 [指标]，结果是 [结果]；2）[实验2]，指标是 [指标]，结果是 [结果]。对比基线，改进了 [幅度]。"

### 模板 4：问题定位

"遇到这个问题，我的排查思路是：1）[步骤1]，检查 [内容]；2）[步骤2]，检查 [内容]；3）[步骤3]，检查 [内容]。最终定位到 [原因]，解决方案是 [方案]。"

### 模板 5：业务价值

"这个方案适合 [场景]，用户关心的是 [指标]。上线成本包括 [成本项]。资源有限时，我会优先优化 [优先级]，因为 [原因]。"

---

**最后更新**：2026-06-27

**适用场景**：LLM 算法实习面试，针对五类技术问题的深度应对

**使用建议**：
1. 先通读全文，理解每类问题的**思维方式**
2. 重点准备第一类（底层原理）和第三类（问题定位），这两类最能体现深度
3. 准备 2-3 个**实际遇到的问题和解决方案**，面试官最喜欢追问这个
4. 用附录中的模板练习回答，但不要背答案——要展示**思考过程**
