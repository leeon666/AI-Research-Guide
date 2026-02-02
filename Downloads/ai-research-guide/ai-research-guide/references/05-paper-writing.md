# 论文写作指南

## 目录
- [写作前的准备](#写作前的准备)
- [确定核心贡献](#确定核心贡献)
- [论文结构设计](#论文结构设计)
- [各部分写作要点](#各部分写作要点)
- [提升可读性](#提升可读性)
- [写作技巧](#写作技巧)
- [常见错误与避免](#常见错误与避免)
- [AI辅助写作](#ai辅助写作)
- [实战案例](#实战案例)

## 写作前的准备

### 1. 明确核心贡献
**三类核心贡献**（Sebastian Nowozin分类）：

1. **Insight（洞察）**：解释了为什么某事物是这样的
   - 示例：BatchNorm解释了内部协变量偏移问题
   
2. **Performance（性能）**：做得更好
   - 示例：ResNet显著提升深层网络训练效果
   
3. **Capability（能力）**：做以前做不到的事
   - 示例：RAFT使密集光流估计成为可能

**选择原则**：
- 找到1-2个核心贡献，不要贪多
- 在论文开头就明确指出
- 全篇围绕核心贡献展开

### 2. 了解目标读者
**不同读者关注点不同**：
- **审稿人**：创新性、正确性、重要性
- **领域研究者**：方法细节、实验设置
- **应用工程师**：实用性、易用性、效率
- **跨领域读者**：背景介绍、直观理解

**写作策略**：
- 平衡不同读者的需求
- 先写技术细节，再写背景介绍
- 提供不同深度的信息

### 3. 收集素材
**需要准备的材料**：
- 实验结果（表格、图表）
- 相关论文列表
- 方法框架图
- 代码片段（如有）
- 消融实验结果

## 确定核心贡献

### Take-away 1: 清楚理解增量
- 明确相对previous methods的增量
- 找到1-2个核心idea
- 精炼出让读者容易记住的亮点

### Take-away 2: 发现新现象更重要
- 小的性能提升不代表新知识
- 读者学到新东西比展示更好结果更重要
- 关注"为什么"而非仅仅"更好"

### 避免的误区
❌ 重新包装别人的方法
```
错误示例：
"我们设计了使用大量3x3卷积（受VGGNet启发）和并行捷径（简化自GoogleNet）的模型，基于前两者"
```

✅ 讲述原创的故事
```
正确示例：
ResNet提出问题 → 抽象底层原理 → 提出解决方案 → 验证实验
```

## 论文结构设计

### 标准结构
```
1. Abstract（摘要）
2. Introduction（引言）
3. Related Work（相关工作）
4. Method（方法）
5. Experiments（实验）
6. Discussion（讨论）
7. Conclusion（结论）
8. References（参考文献）
```

### 渐进式展开原则
**三级结构**：
```
Level 1: Abstract（200-250词）
  ├─ 问题、方法、结果、贡献的简要概述

Level 2: Introduction（1-2页）
  ├─ 扩展Abstract，更详细但不涉及细节
  ├─ 研究动机
  ├─ 主要贡献
  └─ 论文结构

Level 3: Main Body（6-10页）
  ├─ 完整的细节
  ├─ 方法的数学描述
  ├─ 详细的实验
  └─ 深入的分析
```

**关键原则**：每层都是自完整的，下一层是上一层的扩展

## 各部分写作要点

### 1. Abstract（摘要）

**结构**：
1. **背景**（1-2句）：问题的重要性
2. **挑战**（1-2句）：现有方法的不足
3. **方法**（1-2句）：提出的方法核心思想
4. **结果**（1-2句）：主要实验结果
5. **贡献**（可选）：强调创新性

**示例**（检测论文）：
```
Recent object detectors have achieved impressive performance on standard benchmarks,
but still struggle with small-scale objects due to limited representation power.
To address this challenge, we propose a novel Multi-Scale Feature Fusion Network (MSFFNet)
that dynamically aggregates features from different levels through a learned attention mechanism.
Extensive experiments on COCO and PASCAL VOC demonstrate that MSFFNet consistently
outperforms state-of-the-art detectors, especially on small objects with a 5.3% AP improvement.
Our code is available at https://github.com/xxx/xxx.
```

**注意事项**：
- 长度控制在200-250词
- 使用主动语态（"We propose"而非"It is proposed"）
- 突出核心贡献
- 避免过多技术细节

### 2. Introduction（引言）

**标准结构（CARS模型）**：

**C**reate a Research Territory（建立研究领地）
```
a. 说明研究领域的重要性、中心地位、趣味性或问题性
b. 引入一般概念、定义和关键词
c. 提供该领域必要背景信息
```

**A**nnounce the Gap（指出空白）
```
a. 指出先前研究的不足
b. 或以某种方式扩展先前知识
c. 说明为什么现有方法不够好
```

**R**eflect on the Gap（反思空白）
```
a. 评价现有研究的局限性
b. 指出关键问题
c. 说明研究的重要性
```

**S**ettle the Gap（填补空白）
```
a. 概述研究目的或研究性质
b. 列出研究问题或假设
c. 宣布主要发现
d. 说明本研究价值
e. 指示论文结构
```

**写作技巧**：

**Knuth原则**：
> "Keep the reader upper-most in your mind"
> 始终以读者为中心

**直奔主题**：
- 不要写太多与论文主题无关的内容
- 新颖有趣的部分尽早出现
- 花更多空间描述原创新颖的想法

**尊重前人**：
- 在指出缺点之前，肯定历史贡献
- 客观评价，避免过度批评

**Page One Figure**：
- 在第一页放置最重要的图
- 突出论文核心
- 吸引读者注意力

**示例结构**：
```markdown
# Introduction

[Paragraph 1] General introduction and importance of the field

[Paragraph 2] Specific problem we aim to solve

[Paragraph 3] Limitations of existing methods (the gap)

[Paragraph 4] Our approach and main contributions:
  - Contribution 1
  - Contribution 2
  - Contribution 3

[Paragraph 5] Overview of the paper structure
```

### 3. Related Work（相关工作）

**平庸的做法**：
- 描述正确的历史
- 按时间顺序罗列方法

**更好的做法**：
- 关注不同方法与你的工作的关系
- 按主题分类，而非时间顺序
- 解释你是如何改进的

**组织策略**：

1. **选择3-4个最相关的主题**
   - 不要试图涵盖所有相关工作
   - 选择与你的工作最相关的

2. **每个主题下按历史演进组织**
   - 先写独立于你的文献综述
   - 再从你的角度重写

3. **避免负面评价**
   - 不要列出其他方法的负面方面
   - 解释你如何改进，而不是批评

4. **强调差异**
   - 突出你的方法与相关工作的区别
   - 明确你的独特贡献

**示例结构**：
```markdown
# Related Work

## 2.1 Feature Pyramid Networks for Object Detection
[早期工作]
[FPN及改进]
[你的方法在其中的定位]

## 2.2 Attention Mechanisms in Computer Vision
[Channel Attention]
[Spatial Attention]
[你的注意力设计的创新点]

## 2.3 Small Object Detection
[挑战概述]
[现有方法]
[你的方法的独特优势]
```

### 4. Method（方法）

**核心原则**：
- 清晰、完整、可复现
- 使用符号表统一记号
- 提供算法伪代码（如果复杂）

**结构**：
```markdown
# Method

## 3.1 Overview
[整体框架图]
[模块介绍]

## 3.2 Module A
[动机]
[数学公式]
[实现细节]

## 3.3 Module B
[动机]
[数学公式]
[实现细节]

## 3.4 Training Strategy
[损失函数]
[优化器]
[训练技巧]
```

**写作技巧**：

**算法描述**：
```
Algorithm 1: Our Proposed Method
Input: Input image I
Output: Detection results R

1: Extract multi-scale features F1, F2, F3, F4 from backbone
2: F_fused = FeatureFusion(F1, F2, F3, F4)
3: F_att = AttentionModule(F_fused)
4: R = DetectorHead(F_att)
5: return R
```

**数学公式**：
- 使用LaTeX格式
- 变量首次使用时定义
- 公式后面解释物理意义

**示例**：
```latex
We propose a novel attention mechanism that computes attention weights
as follows:

$$
\alpha_{ij} = \frac{\exp(f_{ij})}{\sum_{k=1}^{K} \exp(f_{ik})}
$$

where $f_{ij}$ represents the affinity between feature $i$ and feature $j$,
and $K$ is the number of features. This formulation allows the model to
dynamically focus on the most relevant features for each location.
```

### 5. Experiments（实验）

**核心原则**：
- 围绕贡献陈述做扎实的分析
- 不要夸大声明
- 诚实客观

**结构**：
```markdown
# Experiments

## 5.1 Experimental Setup
数据集
评估指标
实现细节
训练策略

## 5.2 Comparison with State-of-the-Arts
表格对比
结果分析

## 5.3 Ablation Studies
模块消融
超参数分析

## 5.4 Analysis
可视化
案例研究
失败案例分析

## 5.5 Generalization Tests
不同数据集
不同场景
```

**注意事项**：

**制作更多表格和图表**：
- 选择最重要的方面展示
- 表格要清晰，易于理解

**避免夸大声明**：
- 如果担心overclaim，与同行讨论
- 使用"may", "potentially"等谨慎表达

**示例表达**：
```
The performance improvement may be attributed to the fact that
our attention module can better capture long-range dependencies.

而不是：
The performance improvement is due to our attention module.
```

### 6. Discussion / Conclusion

**Conclusion内容**：
- 总结主要贡献
- 指出局限性
- 提出未来方向
- 不要重复引言和摘要

**示例**：
```markdown
# Conclusion

In this paper, we proposed XXX for YYY problem. Our key contributions include:
1. ...
2. ...
3. ...

Our method demonstrates significant improvements on XXX datasets,
especially in ZZZ scenario.

**Limitations**: Our method still has limitations in AAA, which we leave for future work.

**Future Directions**: Exploring BBB is a promising direction.
```

## 提升可读性

### 1. 增强逻辑强度
**Do not misuse connectives**
- 连接词是增强，不是逻辑本身
- 确保连接词与实际逻辑一致

**错误示例**：
```
We argue that problem A is critical. To this end, we propose method B.
```
"To this end"指哪个end？前面只有观点，没有行动或目标。

**正确示例**：
```
Problem A is critical. To address this problem, we propose method B.
```

**避免强加顺序**：
```
错误：
The system comprises three modules. First of all, Module A is...
Second, Module B is... Last but not least, Module C is...

正确：
The system comprises three modules. Module A is... Module B is...
Module C is...
```

### 2. 考虑可辩护性
**Make statements based on references and facts**

**添加引用**：
```
It is reported that problem A results in ... [1,2,3] and ... [4,5],
which are critical to ... because ... [6, 7, 8].
```

**客观描述结果**：
```
The improvement may be explained by the fact that XXX...
而非：
The improvement is due to XXX...
```

### 3. 缩短困惑时间
**Explain a concept as close as possible when it is proposed**

```
示例：
We propose XXX, which is implemented with a two-layer multilayer
perceptron (MLP).
```

**Resolve relative pronoun ambiguity**：
- 避免复杂的长句
- 尽量分解为短句

**Frequently use topic sentences**：
- 段落开头使用主题句
- 帮助读者快速获取主要信息

### 4. 增加信息密度
**Get to the point as soon as possible**
- 不要写太多历史
- 避免读者熟悉的内容

**Balanced text and figures**：
- 适当地平衡文本和图表
- 避免大图表只突出几个关键点
- 避免冗长的超参数描述（应放入附录）

**Important explanations close to figures**：
- 每个图表应能独立理解
- Caption中清晰说明主题和结论
- 分析结果与图表在同页

**示例**：
```
Table 5 shows that our method outperforms the baseline on all metrics.
This improvement is particularly significant for small objects (AP_s ↑5.3%),
which validates our claim that the attention module better captures
fine-grained details.

[Table 5]
```

## 写作技巧

### 1. 使用主动语态
```
Passive: The method is proposed to address this problem.
Active: We propose the method to address this problem.
```

### 2. 使用简单词汇
```
Complex: The methodology was implemented utilizing a sophisticated
          algorithmic framework.
Simple: We used an advanced algorithm.
```

### 3. 避免冗余
```
Redundant: In order to improve performance, we need to optimize the model.
Concise: To improve performance, we optimize the model.
```

### 4. 使用平行结构
```
Our method achieves three goals:
1. improves accuracy,
2. reduces complexity, and
3. enhances interpretability.
```

## 常见错误与避免

### 1. 过度使用被动语态
```
错误：It was observed that the results are improved.
正确：The results improved.
```

### 2. 复杂的句子结构
```
错误：The method, which was proposed by the authors who are from the
       university that is located in the city, performs well on the dataset.
正确：The method from X University performs well on the dataset.
```

### 3. 模糊的表达
```
错误：The results show that the method is good.
正确：The method achieves 78.5% mAP, outperforming the baseline by 2.3%.
```

### 4. 缺乏逻辑连接
```
错误：We added attention. The performance improved.
正确：We added attention to better capture long-range dependencies.
       As a result, the performance improved by 2.3%.
```

## AI辅助写作

### 推荐工具
- **ChatGPT / Claude**：通用写作辅助
- **跃问 / 豆包**：中文科研写作
- **Grammarly**：语法检查
- **DeepL**：中英翻译

### 使用技巧

**生成多个版本**：
```
Prompt: "Rewrite this paragraph in 3 different styles:
1. Academic formal
2. More concise
3. More natural"

选择最合适的版本。
```

**语言润色**：
```
Prompt: "Improve the readability of this paragraph.
Focus on:
- Logical flow
- Clarity
- Conciseness"
```

**避免过度依赖**：
- 优先考虑清晰度而非风格
- 人工审阅和修改仍然是必要的
- AI作为辅助工具，不能完全替代

## 实战案例

### 案例1：引言重写
**初稿**：
```
Deep learning has been successful. Object detection is important.
Many methods have been proposed. But they have problems.
We propose a new method.
```

**改进后**：
```
Object detection is a fundamental task in computer vision with numerous
applications such as autonomous driving and surveillance. While recent
advances in deep learning have led to significant improvements, existing
methods still struggle with small-scale objects due to limited feature
representation. In this paper, we propose a novel Multi-Scale Feature
Fusion Network (MSFFNet) that dynamically aggregates features from
different levels through a learned attention mechanism. Our main
contributions are: (1) ...
```

### 案例2：方法描述优化
**初稿**：
```
Our method has three parts. First, we extract features. Then we use attention.
Finally, we detect objects.
```

**改进后**：
```
Our method consists of three key components: (1) a multi-scale feature
extractor that captures fine-grained details at different scales, (2) an
adaptive attention module that dynamically weights features based on their
importance, and (3) a detection head that generates final predictions.
The overall pipeline is illustrated in Figure 2.
```

## 细节检查清单

**完成初稿后，检查**：

- [ ] **图表检查**：
  - 故事完整
  - 图表质量高
  - 自解释性强

- [ ] **一致性检查**：
  - 符号统一
  - 缩写一致
  - 引用正确

- [ ] **详细程度**：
  - 文本和图表的详细程度合适

- [ ] **信息位置**：
  - 重要信息放在显眼位置

- [ ] **图表可读性**：
  - 文字和图例可以更大
  - 表格清晰易懂

- [ ] **可复现性**：
  - 在附录中提供细节
  - 关键代码公开

## 注意事项

### 写作心态
- "Good writing is bad writing rewritten" — Stephen King
- 多次修改和润色是必要的
- 让同行审阅，收集反馈

### 时间管理
- 建议在截止日期前1个月完成初稿
- 留出充足时间修改和润色
- 提前准备图表和实验

### 避免的陷阱
- 不要等到最后一刻才开始写
- 不要忽视引言和相关工作
- 不要轻视图表质量
- 不要忽略可读性
