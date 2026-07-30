# 导师汇报讲稿 / Supervisor Meeting Script
**Teams 会议 | June 2026**

---

## 中文版

### 开场

老师好，今天主要是想跟您汇报一下我论文目前的进展，以及接下来的计划。

---

### 一、文献阅读进展

文献阅读这块进展比较顺利。我主要梳理了三条技术线索：

**第一条：延迟反馈建模**
我精读了四篇核心论文——DFM、ES-DFM、FSIW 和 DEFER。
这四篇基本覆盖了 CVR 预测中延迟反馈问题的主流解法：
- DFM 用 EM 算法 + 指数分布对延迟时间建模；
- ES-DFM 提出了基于时间采样的重要性加权方案；
- FSIW 把问题转化为分布偏移，用重要性权重做纠偏；
- DEFER 的思路最直觉——直接把真实负样本也复制一份，从根本上修复训练集的分布。

**第二条：图神经网络**
我读了 LightGCN、TGN 和 DGSR。核心逻辑是：
- LightGCN 简化了静态协同过滤图的聚合方式；
- TGN 引入记忆模块，让图可以随时间动态更新；
- DGSR 专门针对序列推荐，把跨用户的交互历史合并成一张动态图，同时建模长期和短期偏好。

**第三条：LLM 与推荐系统的结合**（正在推进中）
下一步我计划精读 BERT4Rec、UniSRec、RLMRec 和 TALLRec，重点看 LLM 的文本表示如何替代或增强传统的 ID embedding。

---

### 二、研究问题（初步形式化）

目前我的三个研究问题是：

- **RQ1**：用 LLM 文本表示替换传统 ID embedding，能否在 CVR 预测任务中带来性能提升？
- **RQ2**：LLM 提供的语义特征，是否能够缓解延迟反馈造成的假负例问题？
- **RQ3**：引入 LLM 表示后的计算开销，和性能增益之间的权衡如何？

---

### 三、方法论设计思路（草案）

整体架构大致分三层：

1. **特征层**：用预训练 LLM（比如 BERT 或 LLaMA）对商品/用户的文本描述编码，生成语义 embedding，替换 DGSR 中原有的 ID embedding；
2. **图建模层**：在 DGSR 框架下，基于语义 embedding 构建动态图，进行邻居聚合；
3. **延迟反馈层**：在训练目标上引入延迟反馈纠偏机制（参考 DEFER 或 FSIW），减轻假负例对模型的干扰。

---

### 四、数据集方向

计划使用公开的 CVR 预测数据集，目前在考虑：
- **Ali-CCP**（阿里巴巴，点击 + 转化双标签，非常贴合任务设定）
- **Criteo Conversion**（工业界标准，有延迟标签）

具体选哪个还在评估中，主要看数据可及性和预处理难度。

---

### 五、接下来的计划

| 时间 | 任务 |
|------|------|
| 本周 | 完成 LLM 推荐论文精读（4篇）|
| 下周 | 确定数据集，开始预处理 |
| 两周后 | 完成方法论章节草稿 + 基线模型实现 |
| 一个月内 | 跑出初步实验结果 |

---

### 结尾

以上是我目前的进展和计划，请您给一些方向性的建议，比如数据集的选择、或者方法论设计上有没有需要调整的地方。谢谢老师！

---
---

## English Version

### Opening

Hi [Supervisor's name], thanks for making time for this meeting. I'd like to give you a quick update on where I am with my dissertation and walk you through the plan going forward.

---

### 1. Literature Review Progress

The literature review has been going well. I've been working through three main areas:

**Area 1: Delayed Feedback in CVR Prediction**
I've done a thorough read of four core papers — DFM, ES-DFM, FSIW, and DEFER.
These cover the main approaches to handling delayed feedback in conversion rate prediction:
- DFM models conversion delay using an EM algorithm with an exponential distribution;
- ES-DFM addresses the freshness–accuracy trade-off through elapsed-time sampling with importance weighting;
- FSIW reframes the problem as distribution shift and applies importance weights to correct the training distribution;
- DEFER takes the most intuitive approach — it duplicates real negative samples (not just positives) to fix the distributional mismatch from the ground up.

**Area 2: Graph Neural Networks for Recommendation**
I've reviewed LightGCN, TGN, and DGSR. The progression is:
- LightGCN simplifies static collaborative filtering by removing unnecessary transformations;
- TGN introduces a memory module for continuous-time dynamic graphs;
- DGSR is tailored for sequential recommendation — it merges all user interaction sequences into a single dynamic graph with both timestamp and order information, capturing cross-user collaborative signals while modelling both long-term and short-term preferences.

**Area 3: LLM-Enhanced Recommendation** *(in progress)*
My next step is to read BERT4Rec, UniSRec, RLMRec, and TALLRec, focusing on how LLM-derived text representations can replace or augment traditional ID-based embeddings.

---

### 2. Research Questions (Preliminary Formalisation)

I've refined my three research questions to:

- **RQ1**: Can LLM-derived text representations replace ID-based embeddings in dynamic graph networks for CVR prediction, and what are the performance gains?
- **RQ2**: Do LLM-based semantic features help mitigate the false-negative problem caused by delayed feedback?
- **RQ3**: What is the computational trade-off between the performance gains and the overhead of incorporating LLM representations?

---

### 3. Proposed Methodology (Draft)

The overall architecture has three components:

1. **Feature Layer**: Encode item/user textual descriptions using a pre-trained LLM (e.g., BERT or LLaMA) to produce semantic embeddings, replacing the ID embeddings in DGSR;
2. **Graph Layer**: Build the dynamic graph on top of these semantic embeddings within the DGSR framework, performing neighbour aggregation;
3. **Delayed Feedback Layer**: Incorporate a delayed feedback correction mechanism into the training objective (drawing from DEFER or FSIW) to reduce the impact of false negatives.

---

### 4. Dataset

I'm evaluating two publicly available CVR datasets:
- **Ali-CCP** (Alibaba, click + conversion dual labels — closely aligned with the task setting)
- **Criteo Conversion** (industry standard benchmark with delayed conversion labels)

The final choice will depend on data accessibility and preprocessing complexity — I expect to confirm this within the next week.

---

### 5. Timeline

| Timeframe | Task |
|-----------|------|
| This week | Complete LLM recommendation paper review (4 papers) |
| Next week | Finalise dataset selection, begin preprocessing |
| In 2 weeks | Draft methodology chapter + implement baselines |
| In 1 month | Produce initial experimental results |

---

### Closing

That's where I currently stand. I'd really welcome your thoughts — particularly on the dataset choice and whether the methodology design looks reasonable at this stage. Thank you!

---

*End of Script*
