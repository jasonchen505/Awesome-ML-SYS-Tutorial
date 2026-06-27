# LLM 算法实习面试深度准备指南

> 基于 Awesome-ML-SYS-Tutorial + slime + SGLang 三个项目的系统性学习与面试准备
> 面向：LLM & Agent 应用 / 后训练方向的 MS 在读学生
> 定位：**算法视角**，侧重面试官可深挖考察实习生水位的核心知识点

---

## 目录

- [第一篇：全局认知——三个项目的关系与学习路径](#第一篇全局认知三个项目的关系与学习路径)
- [第二篇：RL 后训练算法深度理解](#第二篇rl-后训练算法深度理解)
- [第三篇：训推不一致——算法与系统的交叉地带](#第三篇训推不一致算法与系统的交叉地带)
- [第四篇：Agent RL 训练的算法核心挑战](#第四篇agent-rl-训练的算法核心挑战)
- [第五篇：推理系统的算法级理解](#第五篇推理系统的算法级理解)
- [第六篇：面试深挖题库——分层考察实习生水位](#第六篇面试深挖题库分层考察实习生水位)
- [第七篇：项目经验介绍与引导策略](#第七篇项目经验介绍与引导策略)
- [第八篇：算法岗 vs Infra 岗的差异化准备](#第八篇算法岗-vs-infra-岗的差异化准备)
- [附录：核心代码索引与学习资源](#附录核心代码索引与学习资源)

---

## 第一篇：全局认知——三个项目的关系与学习路径

### 1.1 三个项目的定位与关系

```
┌─────────────────────────────────────────────────────────────────────┐
│                    LLM 后训练全链路                                   │
│                                                                     │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────┐  │
│  │  Awesome-ML-SYS  │    │     slime        │    │    SGLang    │  │
│  │    Tutorial      │    │  (RL 训练框架)    │    │ (推理引擎)   │  │
│  │  (学习笔记/博客) │    │                  │    │              │  │
│  └────────┬─────────┘    └────────┬─────────┘    └──────┬───────┘  │
│           │                       │                      │          │
│           │  理论基础              │  调用关系             │          │
│           └───────────────────────┼──────────────────────┘          │
│                                   │                                 │
│                    slime 唯一使用 SGLang 作为推理后端                │
└─────────────────────────────────────────────────────────────────────┘
```

**对算法岗面试者的启示**：
- **Awesome-ML-SYS-Tutorial**：理解"为什么这样设计"的理论基础
- **slime**：理解 RL 后训练的完整算法流程与工程实现
- **SGLang**：理解推理优化如何影响 RL 训练的正确性和效率

### 1.2 算法岗面试者的差异化学习路径

| 层次 | Infra 岗关注点 | 算法岗关注点 |
|------|---------------|-------------|
| **L1 算法原理** | 实现细节 | 公式推导、变体对比、适用场景 |
| **L2 系统影响** | 性能优化 | 系统行为如何影响算法正确性 |
| **L3 调试验证** | 工具使用 | 如何验证算法效果、定位算法bug |
| **L4 业务价值** | 部署运维 | 算法选择的 ROI、场景适配 |

---

## 第二篇：RL 后训练算法深度理解

### 2.1 从第一性原理理解 RL 后训练

**核心问题**：为什么预训练之后还需要 RL 后训练？

```
预训练：学习 P(next_token | context) → 语言建模能力
SFT：学习 P(response | prompt) → 指令遵循能力
RL：学习 E[reward(policy)] → 偏好对齐 / 推理能力 / 工具使用
```

**算法岗面试深挖点**：
- Q: RL 后训练和 SFT 的本质区别是什么？
- A: SFT 是**模仿学习**（behavior cloning），RL 是**策略优化**（policy optimization）。SFT 只能学习训练数据中的分布，RL 可以探索新的、更好的策略。关键区别：RL 有 **exploration**（探索）和 **credit assignment**（信用分配）。

### 2.2 PPO vs GRPO vs REINFORCE++：算法选择的深层逻辑

#### PPO（Proximal Policy Optimization）

**核心思想**：通过 clipping 机制限制策略更新幅度，保证训练稳定性。

**公式推导**：
```
目标函数：L(θ) = E_t[min(r_t(θ)·A_t, clip(r_t(θ), 1-ε, 1+ε)·A_t)]
其中：r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)  （重要性采样比率）
      A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...  （GAE 优势估计）
      δ_t = r_t + γV(s_{t+1}) - V(s_t)  （TD 误差）
```

**代码实现**（`slime/utils/ppo_utils.py`）：
```python
def compute_policy_loss(ppo_kl, advantages, eps_clip, eps_clip_high):
    ratio = (-ppo_kl).exp()  # ratio = exp(new_logp - old_logp)
    pg_losses1 = -ratio * advantages
    pg_losses2 = -ratio.clamp(1 - eps_clip, 1 + eps_clip_high) * advantages
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)  # 悲观估计
    return clip_pg_losses1
```

**面试深挖**：
- Q: 为什么 PPO 取 max（悲观估计）而不是 min？
- A: 取 max 是**悲观估计**（conservative policy improvement）。当 advantage 为正时，我们想增大 ratio，但 clipping 限制了增大幅度；取 max 确保我们不会过度乐观地更新。这保证了策略更新是保守的、稳定的。

- Q: `eps_clip` 和 `eps_clip_high` 为什么要分开设置？
- A: 对于正 advantage（好的动作），我们允许更大幅度的更新（`eps_clip_high` 更大）；对于负 advantage（坏的动作），我们更保守地裁剪（`eps_clip` 更小）。这是**非对称裁剪**，符合"鼓励好行为比惩罚坏行为更激进"的直觉。

#### GRPO（Group Relative Policy Optimization）

**核心思想**：用**组内相对优势**代替**绝对优势**，不需要 Critic 模型。

**公式**：
```
A_i = (r_i - mean(r_group)) / std(r_group)
```

**代码实现**（`slime/utils/ppo_utils.py`）：
```python
def get_grpo_returns(rewards, kl):
    # rewards: [batch_size] 每个 sample 的标量 reward
    # 对每个 token，advantage 都等于该 sample 的标量 reward
    returns = []
    for i in range(len(rewards)):
        returns.append(torch.ones_like(kl[i]) * rewards[i])
    return returns
```

**面试深挖**：
- Q: GRPO 的 advantage 是**sample 级别**的标量，而不是 **token 级别**的向量，这有什么影响？
- A: 这意味着 GRPO 的**信用分配是粗粒度的**——一个 sample 内所有 token 共享同一个 advantage。这在稀疏奖励场景下是合理的（只有最终结果有奖励），但在密集奖励场景下可能不够精细。PPO 通过 Critic 模型提供 token 级别的 value 估计，信用分配更精细。

- Q: GRPO 什么时候会失效？
- A: 当一个 prompt 的所有 sample 的 reward 都相同时（`std = 0`），advantage 全为 0，无法学习。这就是**动态采样**（Dynamic Sampling）的动机——过滤掉 reward 标准差为 0 的样本。

#### REINFORCE++

**核心思想**：带折扣回报的 REINFORCE，加入 KL 惩罚作为 token 级别奖励。

**公式**：
```
G_t = Σ_{k=t}^T γ^{k-t} * r_k
其中 r_t = -kl_coef * KL_t  (token 级别 KL 惩罚)
     r_T = R  (终端奖励)
```

**面试深挖**：
- Q: REINFORCE++ 和 GRPO 的关键区别？
- A: GRPO 是**组内相对**的（需要多个 sample），REINFORCE++ 是**绝对的**（单个 sample 就能计算）。REINFORCE++ 通过 token 级别的 KL 惩罚提供更密集的信号，但方差可能更大。

### 2.3 KL 散度控制：不仅仅是正则化

**三种 KL 估计器**（`slime/utils/ppo_utils.py`）：

| 估计器 | 公式 | 特点 | 适用场景 |
|--------|------|------|---------|
| k1 | KL = log(π/π_ref) | 一阶近似，可能为负 | 快速监控 |
| k2 | KL = (log(π/π_ref))²/2 | 二阶近似，始终非负 | 一般用途 |
| k3 (low_var_kl) | KL = π_ref/π - 1 - log(π_ref/π) | 非负、无偏、低方差 | **推荐默认** |

**面试深挖**：
- Q: 为什么 k3 是最好的 KL 估计器？
- A: k3 来自 [Schulman 的博客](http://joschu.net/blog/kl-approx.html)，它是**非负的**（不会出现负 KL）、**无偏的**（期望等于真实 KL）、**低方差的**（比 k1 和 k2 更稳定）。在 RL 训练中，KL 估计的稳定性直接影响训练稳定性。

- Q: KL 散度在 RL 中有哪几种使用方式？
- A: 
  1. **作为 reward 惩罚**：`rewards[t] -= kl_coef * KL_t`（token 级别）
  2. **作为额外 loss 项**：`loss = loss + kl_loss_coef * kl`
  3. **作为监控指标**：`kl_coef = 0` 时只记录不参与梯度
  4. **作为 TIS 的判断依据**：序列级 KL 超阈值时 mask 掉

### 2.4 PPO Clipping 的变体：Dual-Clip 与 CISPO

#### Dual-Clip PPO

**问题**：标准 PPO 在 advantage 为负时，ratio 越大 loss 越小，可能导致过度优化。

**解决方案**：对负 advantage 额外施加上界裁剪：
```python
# 当 advantage < 0 时，额外裁剪
pg_losses3 = -eps_clip_c * advantages  # eps_clip_c > 1.0
clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
pg_losses = where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
```

#### CISPO（Clipped IS-Weighted Stop-Gradient）

**来源**：MiniMax-M1 论文

**核心创新**：IS ratio 在 **stop-gradient** 下裁剪，梯度通过 `log_probs` 流动。

```python
# CISPO: ratio 被 detach，梯度通过 log_probs 流动
ratio_truncated = torch.clamp(ratio, min=1.0 - eps_clip, max=1.0 + eps_clip_high)
pg_losses = -ratio_truncated.detach() * advantages * log_probs
```

**面试深挖**：
- Q: CISPO 和 PPO 的关键区别？
- A: PPO 中，梯度通过 ratio 流动（`ratio = exp(new_logp - old_logp)`），被裁剪的 token **不贡献梯度**。CISPO 中，ratio 被 detach，梯度通过 `log_probs` 流动，被裁剪的 token **仍然贡献梯度**。这意味着 CISPO 可以从更多样本中学习，提高样本效率。

### 2.5 GAE 的实现与优化

**标准 GAE**（后向递推）：
```python
for t in reversed(range(response_len)):
    nextvalues = full_values[t + 1] if t < response_len - 1 else 0.0
    delta = full_rewards[t] + gamma * nextvalues - full_values[t]
    lastgaelam = delta + gamma * lambd * lastgaelam
    advantages_reversed.append(lastgaelam)
```

**Chunked GAE**（并行前缀扫描）：
- 将标准后向递推改写为基于 chunk 的并行前缀扫描
- 复杂度从 O(T) 降低到 O(T/chunk_size) 的串行依赖
- 在 slime 中实现约 100×–300× 加速

**面试深挖**：
- Q: GAE 的 γ 和 λ 分别控制什么？
- A: **γ（折扣因子）**控制对未来奖励的重视程度，**λ（GAE 参数）**控制优势估计的偏差-方差权衡。λ=0 时退化为 TD(0)（低方差、高偏差），λ=1 时退化为 Monte Carlo（高方差、零偏差）。

---

## 第三篇：训推不一致——算法与系统的交叉地带

> 这是算法岗面试中最能体现"系统思维"的考察点

### 3.1 什么是训推不一致？

**定义**：即使权重完全相同，推理引擎（SGLang）和训练引擎（Megatron/FSDP）对同一 token 产生的 log-probabilities 也不同。

**量化指标**：K3 KL 散度
- Dense 模型：K3 KL ~ 10⁻⁵ 到 10⁻³
- MoE 模型：K3 KL ~ 10⁻³ 到 10⁻¹（**差 100 倍**）

### 3.2 根因分析

| 根因 | 影响 | 严重程度 |
|------|------|---------|
| 浮点加法非结合性 | 不同 reduction order 产生不同结果 | 低 |
| 不同 CUDA kernel | 训练和推理使用不同实现 | 中 |
| 不同 batch size | 触发不同 reduction order | 中 |
| 不同矩阵形状 | 推理是逐 token 小矩阵，训练是全序列大 batch | 高 |
| MoE 专家选择 | logit 微小差异导致不同专家路由 | **极高** |

### 3.3 对 PPO 的影响

标准 PPO 公式假设 sampling distribution 和 training distribution 一致：
```
L(θ) = E_{a~π_old}[min(r(θ)·A, clip(r(θ))·A)]
其中 r(θ) = π_θ(a|s) / π_old(a|s)
```

但实际中：
- **分子** π_θ(a|s)：来自训练引擎（Megatron）
- **分母** π_old(a|s)：来自推理引擎（SGLang）

两者不一致导致 **importance sampling ratio 有偏**。

### 3.4 两种解决方案

#### 方案一：Truly On-Policy（位对齐）

**思路**：让训练和推理引擎产生完全相同的 log-probabilities。

**实现**：
- 使用 batch-invariant kernels（Thinking Machines Lab）
- FlashAttention-3 同时用于训练和推理
- DeepGEMM 实现确定性矩阵乘法
- 结果：绝对差 = 0，K3 KL = 0

**局限**：仅适用于 vanilla dense 模型，需要侵入性修改，有一定性能损失。

#### 方案二：算法缓解（Importance Sampling 修正）

**核心思想**：显式地用 importance sampling 修正分布差异。

**TIS（Truncated Importance Sampling）**：
```python
# 裁剪 importance ratio
w = π_old / π_SGLang
w_clipped = clamp(w, lower_bound, upper_bound)
loss = w_clipped * loss
```

**MIS（Masked Importance Sampling）**：
```python
# 序列级 KL 超阈值时 mask 掉整个序列
if sequence_kl > threshold:
    mask = 0  # 不参与训练
```

**面试深挖**：
- Q: 为什么 MoE 模型的训推不一致比 dense 模型严重得多？
- A: MoE 模型使用 top-k 路由，logit 的微小差异可能导致选择不同的专家。一旦专家选择不同，后续所有计算都会 diverge，产生**级联放大效应**。Dense 模型没有这种离散选择，差异是连续的、可控的。

- Q: TIS 的裁剪上下界如何选择？
- A: 上界防止过度信任推理引擎的数据（`w > upper` 说明推理引擎的概率远高于训练引擎），下界防止丢弃有用数据（`w < lower` 说明推理引擎的概率远低于训练引擎）。典型值：`[0.5, 2.0]`。

---

## 第四篇：Agent RL 训练的算法核心挑战

### 4.1 Agent 训练的四大核心挑战

#### 挑战一：信用分配（Credit Assignment）

**问题**：多轮交互中，最终奖励应该归因到哪一步？

**示例**：
```
Turn 1: 模型生成搜索查询 → 搜索成功
Turn 2: 模型生成答案 → 答案错误
最终奖励：0（错误）
```
问题：Turn 1 的搜索查询是好的（搜索成功了），但因为 Turn 2 的错误，整体 reward 为 0。

**解决方案**：
1. **稀疏奖励**：只在最终结果给予奖励，依赖 RL 算法自动分配（GRPO/PPO）
2. **密集奖励**：每一步都给予奖励（如工具调用成功/失败）
3. **GAE**：使用 Critic 模型估计每一步的价值
4. **TIS**：通过重要性采样修正 off-policy 数据

#### 挑战二：Loss Masking

**问题**：Agent 轨迹包含模型生成和工具返回两部分，工具返回不应参与 loss 计算。

**代码示例**：
```python
async def generate(args, sample, sampling_params):
    for turn in range(max_turns):
        model_output = await call_sglang(prompt + full_response)
        action, content = parse_action(model_output)
        
        if action == "search":
            tool_output = await search(content)
            # 工具输出的 token 不参与 loss
            loss_masks += [0] * len(tokenize(tool_output))
        
        full_response += model_output + tool_output
    
    sample.loss_mask = loss_masks
    return sample
```

**面试深挖**：
- Q: 为什么工具返回的 token 不能参与 loss？
- A: 如果工具返回参与 loss，模型会学习"复制"工具输出（因为这些 token 的梯度会降低 loss），而不是学习**如何正确使用工具**。这会导致模型过度依赖工具输出，而不是学习推理能力。

#### 挑战三：Off-Policy 问题

**问题**：Agent 生成时间长且不均匀，使用旧策略生成的数据训练新策略。

**解决方案**：
1. **TIS（Truncated Importance Sampling）**：裁剪 importance ratio
2. **OPSM（Off-Policy Sequence Masking）**：序列级 KL 超阈值时 mask 掉
3. **KL 惩罚**：限制策略偏离参考模型

#### 挑战四：长序列处理

**问题**：Agent 轨迹可能非常长（数十轮交互），超出模型上下文窗口。

**解决方案**：
1. **上下文压缩**：定期压缩历史上下文
2. **分段训练**：将长轨迹分成多个段分别训练
3. **滑动窗口**：只保留最近的 N 轮交互
4. **高效注意力**：使用 Flash Attention 等优化

### 4.2 Multi-Turn RL 的 Tokenizer 视角

**核心问题**：多轮交互中，如何正确处理 chat template？

**Dummy Messages + Delta Tokens 技巧**：
```python
# 预编码 system prompt 和工具定义
dummy_ids = encode(DUMMY_MESSAGES)

# 每轮只编码新增的 observation
obs_ids = encode(DUMMY_MESSAGES + [obs_msg])[len(dummy_ids):]
```

**面试深挖**：
- Q: 为什么要用 delta tokens 而不是每轮重新编码完整上下文？
- A: 
  1. 避免重复的 system prompt / tool instruction 占用 token budget
  2. 保持 token 分布的一致性（相同前缀的 token 应该有相同的 token id）
  3. 支持 prefix caching 复用

### 4.3 Session Routing：Agent 训练的系统级优化

**问题**：Agent 的多轮交互需要将请求路由到同一 worker，复用 prefix cache。

**解决方案**：consistent hashing 路由
```bash
--router-policy consistent_hashing
```

**面试深挖**：
- Q: Session Routing 对 Agent 训练的效率影响有多大？
- A: 在多轮交互中，system prompt + 历史对话是共享的。Session Routing 确保这些共享前缀的 KV Cache 被复用，而不是每轮重新计算。在典型 Agent 场景下，可以减少 50-70% 的 TTFT（首 token 延迟）。

### 4.4 Fan-out 训练：复杂 Agent 工作流

**概念**：一次 rollout 拆分为多个可训练段（如 subagent 轨迹、主 agent 继续），所有段共享同一个 rollout_id。

**代码示例**：
```python
async def generate(args, sample, sampling_params):
    # 主 agent 生成
    main_output = await call_sglang(main_prompt)
    
    # 子 agent 生成（可能并行）
    sub_output = await call_sglang(sub_prompt)
    
    # 返回多个可训练段
    return [main_sample, sub_sample]
```

---

## 第五篇：推理系统的算法级理解

### 5.1 RadixAttention：为什么对 RL 训练至关重要？

**核心思想**：使用 Radix Tree 管理 KV Cache，实现前缀级别的缓存复用。

**对 RL 训练的价值**：
1. **多轮 Agent**：system prompt + 历史对话每轮复用
2. **GRPO 组内采样**：同一 prompt 的多个 sample 共享前缀
3. **权重更新后**：旧 KV Cache 失效，需要刷新

**面试深挖**：
- Q: 权重更新后，Radix Cache 中的旧 KV Cache 怎么办？
- A: 权重更新前，所有 rollout 引擎必须 flush radix tree（清除所有缓存）。因为旧权重计算的 KV Cache 与新权重不兼容，使用旧 KV Cache 会产生错误结果。这在 slime 中通过 `KV Cache Flush` 机制实现。

### 5.2 推测解码在 RL 中的特殊应用

**核心创新**：将推测解码引入 RL 采样，并且 **draft model 在训练过程中更新**。

**Online SFT for Draft Model**：
- Draft model（MTP layer）使用 target model 的 hidden states 和生成的 token 进行训练
- 训练目标：给定 position t 的 hidden state 和 t+1 的 token embedding，预测 t+2 的 token
- GRPO loss 和 MTP CE loss 同时反向传播
- Hidden states 和 shared LM head/embedding 被 **detach**，防止 MTP loss 污染主模型

**面试深挖**：
- Q: 为什么要在 RL 训练中在线更新 draft model？
- A: RL 训练过程中，target model 不断变化。如果 draft model 冻结，它会逐渐与 target model 偏离，导致接受率下降，在线 SFT 保持两者对齐。实验表明，在线 SFT 比冻结 draft model 提升 25% 的接受率。

### 5.3 Prefix Caching 的算法价值

**场景分析**：

| 场景 | 缓存命中 | TTFT 降低 | 吞吐提升 |
|------|---------|----------|---------|
| 多轮对话 | system prompt + 历史对话 | 50-70% | 2-3x |
| GRPO 组内采样 | 同一 prompt 的多个 sample | 30-50% | 1.5-2x |
| Agent 工作流 | system prompt + 工具定义 | 50-70% | 2-3x |
| 批量推理 | 相同 system prompt | 40-60% | 3-5x |

---

## 第六篇：面试深挖题库——分层考察实习生水位

### Level 1：基础概念（考察是否入门）

#### Q1: 什么是 RL 后训练？和 SFT 有什么区别？

**参考答案**：
RL 后训练是在预训练/SFT 之后，使用强化学习进一步优化模型的过程。与 SFT 的本质区别：
- SFT 是**模仿学习**，只能学习训练数据中的分布
- RL 是**策略优化**，可以探索新的、更好的策略
- RL 有 **exploration**（探索）和 **credit assignment**（信用分配）

**考察点**：是否理解 RL 的核心价值，而不是仅仅背诵定义。

#### Q2: 解释 GRPO 算法的原理

**参考答案**：
GRPO 用组内相对优势代替绝对优势，不需要 Critic 模型：
```
A_i = (r_i - mean(r_group)) / std(r_group)
```
- 对每个 prompt 生成一组响应（如 8 个）
- 计算组内 reward 的均值和标准差
- 用标准化后的 reward 作为 advantage
- 使用 PPO 风格的 clipping 限制策略更新

**考察点**：是否能推导公式，是否理解"相对"优势的含义。

### Level 2：深度理解（考察是否有独立思考能力）

#### Q3: GRPO 的 advantage 是 sample 级别的标量，这有什么影响？

**参考答案**：
这意味着 GRPO 的**信用分配是粗粒度的**——一个 sample 内所有 token 共享同一个 advantage。在稀疏奖励场景下是合理的（只有最终结果有奖励），但在密集奖励场景下可能不够精细。PPO 通过 Critic 模型提供 token 级别的 value 估计，信用分配更精细。

**考察点**：是否理解不同算法的 trade-off，而不是盲目使用。

#### Q4: 训推不一致为什么对 MoE 模型影响更大？

**参考答案**：
MoE 模型使用 top-k 路由，logit 的微小差异可能导致选择不同的专家。一旦专家选择不同，后续所有计算都会 diverge，产生**级联放大效应**。Dense 模型没有这种离散选择，差异是连续的、可控的。

**考察点**：是否理解 MoE 的离散路由特性，是否能分析级联效应。

#### Q5: 为什么 Agent 训练需要 Loss Masking？

**参考答案**：
工具返回的内容不是模型生成的，如果参与 loss 计算，模型会学习"复制"工具输出（因为这些 token 的梯度会降低 loss），而不是学习**如何正确使用工具**。这会导致模型过度依赖工具输出，而不是学习推理能力。

**考察点**：是否理解 loss 的本质是"让模型学习生成这些 token"。

### Level 3：系统思维（考察是否有工程落地能力）

#### Q6: 如何设计一个可扩展的 RL 训练框架？

**参考答案**（参考 slime 的设计）：
1. **模块化**：训练（Megatron）、推理（SGLang）、数据管理（Data Buffer）分离
2. **可扩展性**：通过函数路径参数实现自定义（17 个钩子点）
3. **正确性优先**：显式数据流，分离调试路径（debug-rollout-only / debug-train-only）
4. **性能优化**：异步执行、Delta Weight Sync、Partial Rollout
5. **生产级**：容错机制、检查点、Trace Viewer

**考察点**：是否有系统设计的全局视野，而不是只关注单个技术点。

#### Q7: 如何验证 RL 训练的正确性？

**参考答案**（参考 slime 的实践）：
1. **精度对齐验证**：检查 log_probs 和 ref_log_probs 是否相等（第一步 KL 应为 0）
2. **权重同步验证**：使用 `--check-weight-update-equal` 验证 Megatron -> SGLang 权重同步
3. **数值一致性验证**：验证不同 CP 大小下梯度一致
4. **确定性复现**：使用确定性模式（`--sglang-enable-deterministic-inference`）

**考察点**：是否有"验证"的意识，而不是"跑通就行"。

### Level 4：算法创新（考察是否有研究潜力）

#### Q8: 如何改进 GRPO 的信用分配？

**参考答案**（开放性问题）：
1. **混合方法**：GRPO + token-level KL penalty（类似 REINFORCE++）
2. **分层优势**：sample-level advantage + token-level baseline
3. **过程奖励**：引入 Process Reward Model（PRM）提供中间步骤的奖励
4. **课程学习**：从简单任务开始，逐步增加任务难度

**考察点**：是否有独立思考和创新能力。

#### Q9: 如何设计 Agent 的奖励函数？

**参考答案**（开放性问题）：
1. **结果奖励**：任务是否完成（如答案是否正确）
2. **过程奖励**：每一步是否合理（如工具调用是否成功）
3. **效率奖励**：完成任务的效率（如步数、token 数）
4. **安全奖励**：是否遵循安全约束

**考察点**：是否有奖励设计的系统性思考。

---

## 第七篇：项目经验介绍与引导策略

### 7.1 项目介绍模板（3-4 分钟）

**背景（30 秒）**：
"我深入研究了三个开源项目：Awesome-ML-SYS-Tutorial（学习笔记）、slime（RL 后训练框架）和 SGLang（推理引擎）。这三个项目构成了 LLM 后训练的完整技术栈。"

**技术亮点（2 分钟）**：
"通过学习这些项目，我深入理解了 RL 后训练的核心算法和系统设计：

1. **算法层面**：GRPO/PPO 的原理、KL 散度控制、信用分配、Loss Masking
2. **系统层面**：训推不一致的根因和解决方案、权重同步机制、Prefix Caching
3. **Agent 层面**：多轮交互的 tokenizer 处理、Session Routing、Fan-out 训练
4. **工程层面**：正确性验证、调试工具、容错机制"

**个人收获（1 分钟）**：
"最大的收获是理解了**算法和系统的交叉地带**——很多看似是系统问题的，根源在算法；看似是算法问题的，需要系统层面解决。例如训推不一致，本质上是浮点精度的系统问题，但解决方案需要算法层面的 importance sampling 修正。"

### 7.2 引导面试官深挖的策略

**主动提及可深挖的点**：
- "我对 GRPO 和 PPO 的区别有深入理解，包括它们在信用分配上的差异"
- "我研究过训推不一致的问题，特别是 MoE 模型为什么受影响更大"
- "我理解 Agent 训练中 Loss Masking 的原理和实现"

**准备 2-3 个技术细节**：
- Radix Tree 的数据结构和 LRU 淘汰策略
- PPO Clipping 的悲观估计原理
- Delta Weight Sync 的增量同步机制

### 7.3 应对"你不会"的问题

**策略**：
1. **诚实承认**："这个我还没有深入研究"
2. **展示相关知识**："但我理解相关的 XXX"
3. **表达学习意愿**："如果有机会，我会从 YYY 开始学习"

**示例**：
- 面试官："解释一下 Flash Attention 的原理"
- 你："Flash Attention 的具体实现我还没有深入研究，但我理解它的核心思想是通过 tiling 减少 HBM 访问，以及它在 SGLang 中作为注意力后端的使用。如果有机会，我会从 Tri Dao 的原始论文开始学习。"

---

## 第八篇：算法岗 vs Infra 岗的差异化准备

### 8.1 算法岗重点准备

| 主题 | 深度要求 | 准备策略 |
|------|---------|---------|
| RL 算法原理 | 公式推导、变体对比 | 能推导 PPO/GRPO 公式 |
| 信用分配 | 理解问题和解决方案 | 能分析 Loss Masking 的必要性 |
| 奖励设计 | 理解不同奖励的优劣 | 能设计简单任务的奖励函数 |
| 训练稳定性 | 理解 KL/Clipping 的作用 | 能解释为什么需要这些机制 |
| Agent 训练 | 理解多轮交互的挑战 | 能设计 Agent 训练的流程 |

### 8.2 Infra 岗重点准备

| 主题 | 深度要求 | 准备策略 |
|------|---------|---------|
| 推理优化 | 理解技术原理和实现 | 能解释 RadixAttention 的数据结构 |
| 分布式训练 | 理解并行策略和通信 | 能解释 TP/PP/EP 的区别 |
| 内存管理 | 理解显存优化技术 | 能解释 Paged Attention 的原理 |
| 性能调优 | 理解调优策略 | 能给出具体的调优建议 |
| 系统设计 | 理解架构权衡 | 能设计简单的推理系统 |

---

## 附录：核心代码索引与学习资源

### A.1 核心代码位置

#### slime 核心代码

| 模块 | 文件路径 | 关键内容 |
|------|----------|---------|
| RL 算法 | `slime/utils/ppo_utils.py` | GRPO/PPO/CISPO 实现、KL 估计器、GAE |
| 损失函数 | `slime/backends/megatron_utils/loss.py` | policy_loss_function、log_probs 计算 |
| Rollout | `slime/rollout/sglang_rollout.py` | 采样流程、DP 负载均衡 |
| 权重同步 | `slime/backends/megatron_utils/update_weight/` | Delta Weight Sync |
| Agent 适配 | `slime/agent/adapters/` | Anthropic/OpenAI 协议适配 |
| 参数定义 | `slime/utils/arguments.py` | 所有超参数定义 |

#### SGLang 核心代码

| 模块 | 文件路径 | 关键内容 |
|------|----------|---------|
| 调度器 | `python/sglang/srt/managers/scheduler.py` | 事件循环、批处理策略 |
| Radix Cache | `python/sglang/srt/mem_cache/radix_cache.py` | 前缀缓存、LRU 淘汰 |
| 推测解码 | `python/sglang/srt/speculative/` | EAGLE、MTP、NGRAM |
| 内存池 | `python/sglang/srt/mem_cache/memory_pool.py` | Paged Attention |

#### Awesome-ML-SYS-Tutorial 关键文档

| 主题 | 文件路径 | 关键内容 |
|------|----------|---------|
| slime 源码赏析 | `rlhf/slime/code-walk-through/readme_en.md` | 架构设计、数据流 |
| 权重更新机制 | `rlhf/sys-design/readme-1-EN.md` | 三种更新方式、CUDA IPC |
| 训推不一致 | `rlhf/slime/mismatch/blog-en.md` | 根因分析、TIS/MIS |
| 推测解码 in RL | `rlhf/slime/spec/readme-en.md` | Online SFT draft model |
| SGLang 调度 | `sglang/scheduler/readme-en.md` | KV Cache 三层设计、Overlap Scheduling |
| VLM Multi-Turn | `rlhf/slime/vlm-multi-turn/readme-en.md` | 统一 LLM/VLM 范式 |

### A.2 推荐学习路径

**第一阶段：建立全局认知（1-2 天）**
1. 阅读 slime README.md，理解架构设计
2. 阅读 SGLang README.md，理解推理引擎能力
3. 阅读 Awesome-ML-SYS-Tutorial README.md，了解学习路径

**第二阶段：深入算法原理（3-5 天）**
1. 阅读 `slime/utils/ppo_utils.py`，理解 GRPO/PPO 实现
2. 阅读 `slime/backends/megatron_utils/loss.py`，理解 loss 计算
3. 阅读 Awesome-ML-SYS-Tutorial 中的算法与理论部分

**第三阶段：理解系统影响（3-5 天）**
1. 阅读 `rlhf/sys-design/readme-1-EN.md`，理解权重更新机制
2. 阅读 `rlhf/slime/mismatch/blog-en.md`，理解训推不一致
3. 阅读 `sglang/scheduler/readme-en.md`，理解调度设计

**第四阶段：准备面试（2-3 天）**
1. 整理本文档中的面试题
2. 准备项目介绍模板
3. 模拟面试练习

### A.3 关键配置速查

```bash
# 基础 GRPO 训练
--advantage-estimator grpo
--use-kl-loss
--kl-loss-coef 0.001
--kl-loss-type low_var_kl
--eps-clip 0.2
--eps-clip-high 0.28

# Agent 训练
--custom-generate-function-path my_agent.generate
--custom-rm-path my_agent.reward_func
--rollout-max-response-len 8192
--router-policy consistent_hashing

# 大规模训练
--tensor-model-parallel-size 4
--pipeline-model-parallel-size 2
--context-parallel-size 2
--expert-model-parallel-size 4

# 异步训练
--update-weights-interval 5
--colocate
--offload-train

# 调试验证
--debug-rollout-only
--debug-train-only
--check-weight-update-equal
--sglang-enable-deterministic-inference
```

### A.4 进阶学习资源

**论文**：
- PPO: "Proximal Policy Optimization Algorithms" (Schulman et al., 2017)
- GRPO: "DeepSeekMath: Pushing the Limits of Mathematical Reasoning" (2024)
- CISPO: "MiniMax-M1: Scaling Test-Time Compute Efficiently" (2025)
- DAPO: "DAPO: An Open-Source LLM Reinforcement Learning System" (2025)
- EAGLE: "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty" (2024)

**博客**：
- [slime: An SGLang-Native Post-Training Framework for RL Scaling](https://lmsys.org/blog/2025-07-09-slime/)
- [Agent-Oriented Design: An Asynchronous and Decoupled Framework for Agentic RL](https://www.notion.so/Agent-Oriented-Design-An-Asynchronous-and-Decoupled-Framework-for-Agentic-RL-2278e692d081802cbdd5d37cef76a547)
- [SGLang Official Blog](https://lmsys.org/blog/)

---

**最后更新**：2026-06-27

**适用场景**：LLM 算法实习面试准备，重点考察算法理解深度、系统思维、工程落地能力

**使用建议**：
1. 先通读全文，建立全局认知
2. 重点学习第二篇（RL 算法）和第四篇（Agent 训练）
3. 准备第七篇中的项目介绍模板
4. 用第六篇的面试题进行自测
