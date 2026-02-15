---
name: ai-research-guide
description: 提供计算机AI科研全流程指导，涵盖Idea发现、论文阅读、实验验证、性能优化、论文写作、Rebuttal策略等，适用于CVPR、ICCV、NeurIPS等顶会投稿及研究生科研能力培养
---

# AI Research Guide

## 任务目标
- 本 Skill 用于：指导计算机AI领域的科研全过程，从入门到论文投稿
- 能力包含：Idea发现、论文阅读、实验验证、论文写作、Rebuttal策略
- 触发条件：用户询问如何开始科研、寻找研究方向、改进实验效果、撰写论文、回复审稿意见等

## 前置准备
- 无特殊依赖
- 无需预创建文件或文件夹

## 操作步骤
- 标准流程：
  1. **阶段判断与需求识别**
     - 询问用户当前科研阶段（入门/做项目/独立研究）
     - 识别用户具体需求（找Idea/读论文/验证实验/写论文/Rebuttal）
  
  2. **资源选择与指导**
     - 根据用户需求，引导智能体读取对应的参考文档：
       - 入门阶段：见 [references/07-research-phases.md](references/07-research-phases.md)
       - 寻找Idea：见 [references/01-idea-discovery.md](references/01-idea-discovery.md)
       - 论文阅读：见 [references/02-paper-reading.md](references/02-paper-reading.md)
       - 验证Idea：见 [references/03-idea-validation.md](references/03-idea-validation.md)
       - 优化实验：见 [references/04-experiment-optimization.md](references/04-experiment-optimization.md)
       - 论文写作：见 [references/05-paper-writing.md](references/05-paper-writing.md)
       - Rebuttal：见 [references/06-rebuttal-strategies.md](references/06-rebuttal-strategies.md)
  
  3. **个性化建议生成**
     - 智能体基于参考文档的内容，结合用户的具体情况，提供针对性建议
     - 避免泛泛而谈，给出可执行的具体步骤
  
  4. **持续跟进与迭代**
     - 根据用户反馈调整建议
     - 针对实验失败、撞车等突发情况，提供应急策略

- 可选分支：
  - 当用户是科研新手：优先推荐从 [references/07-research-phases.md](references/07-research-phases.md) 开始，建立基础框架
  - 当用户已有Idea但实验不work：重点参考 [references/03-idea-validation.md](references/03-idea-validation.md) 中的调试方法
  - 当用户遇到撞车：参考 [references/01-idea-discovery.md](references/01-idea-discovery.md) 中的差异化策略

## 资源索引
- 科研入门规划：见 [references/07-research-phases.md](references/07-research-phases.md)（何时读取：用户首次接触科研或规划长期目标）
- Idea发现方法：见 [references/01-idea-discovery.md](references/01-idea-discovery.md)（何时读取：用户寻找研究方向或创新点）
- 论文阅读技巧：见 [references/02-paper-reading.md](references/02-paper-reading.md)（何时读取：用户需要快速理解领域文献或特定论文）
- Idea验证策略：见 [references/03-idea-validation.md](references/03-idea-validation.md)（何时读取：用户需要验证实验方案或分析失败原因）
- 实验优化方法：见 [references/04-experiment-optimization.md](references/04-experiment-optimization.md)（何时读取：用户需要提升性能或解决实验瓶颈）
- 论文写作指南：见 [references/05-paper-writing.md](references/05-paper-writing.md)（何时读取：用户开始撰写学术论文）
- Rebuttal策略：见 [references/06-rebuttal-strategies.md](references/06-rebuttal-strategies.md)（何时读取：用户收到审稿意见需要回复）

## 注意事项
- 仅在需要时读取参考文档，保持上下文简洁
- 根据用户科研阶段提供不同深度的建议（入门/进阶/专家）
- 强调实战经验，避免纯理论说教
- 鼓励用户记录实验过程和失败经验
- 提醒用户科研是螺旋上升的过程，失败是常态

## 使用示例
- **场景1：科研入门**
  - 功能说明：指导新人建立科研框架和知识体系
  - 执行方式：智能体 + references/07-research-phases.md
  - 关键指导：三个阶段学习路径、能力培养重点

- **场景2：寻找Idea**
  - 功能说明：帮助用户从已有工作中挖掘创新点
  - 执行方式：智能体 + references/01-idea-discovery.md
  - 关键指导：跨领域迁移、关注局限性、模块缝合技巧

- **场景3：实验失败分析**
  - 功能说明：诊断实验不work的原因并提供解决方案
  - 执行方式：智能体 + references/03-idea-validation.md
  - 关键指导：做减法验证、Toy Data测试、梯度分析

- **场景4：论文写作**
  - 功能说明：指导从零构建高质量的学术论文
  - 执行方式：智能体 + references/05-paper-writing.md
  - 关键指导：核心贡献识别、论文结构规划、可读性提升

- **场景5：回复审稿意见**
  - 功能说明：制定有效的Rebuttal策略
  - 执行方式：智能体 + references/06-rebuttal-strategies.md
  - 关键指导：分类处理审稿意见、提供充分证据、保持礼貌
