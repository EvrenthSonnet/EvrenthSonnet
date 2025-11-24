# AI 学习记录模板（适配你的学习方式）

> 这个模板是为你的学习思维结构设计的：从公理到推演，从历史到当下，从机制到应用。只需要把内容添进去即可。

---

## 0. 本次学习的核心主题（Topic）

* *一句话描述你正在理解的核心概念/问题是什么。*
大框架：从Base Model到LLM-AI
以claude为例子~:
Base Model (预训练)
    ↓
SFT (Supervised Fine-Tuning, 监督微调)
    ↓
RLHF (Reinforcement Learning from Human Feedback, 人类反馈强化学习)
    ↓
Constitutional AI (CAI, 宪法式AI)
    ↓
Claude
---

## 1. 公理 / 定义区（Axioms & Definitions）

> 写下这个主题最基础、不可再分解的前提。
> 这是你构建整个知识结构的“地基”。

* Step 0: Base Model（预训练）
目标：学习语言的统计规律
输入：海量文本数据（书籍、网页、代码、对话...）
任务：预测下一个token
训练方法：**自监督学习（self-supervised learning）**
Base model学到了什么？
✅ 语法、语义、逻辑推理能力
✅ 世界知识（因为文本里包含知识）
✅ 多种"风格"（因为训练数据里有各种风格的文本）
但:
❌ 没有"目标"（不知道自己应该helpful/harmless/honest）
❌ 没有"价值观"（可以生成任何符合统计规律的内容）
❌ 不稳定（prompt稍微变化，回应风格可能完全不同）
纯粹的pattern matching machine。

* Step 1: SFT（监督微调）
目标：教会模型"什么样的回应是好的"
具体流程：
收集示范数据（demonstration data）：

人类标注员（AI trainers）写一些"理想的对话"
比如：

User: "教我怎么做蛋糕"
Ideal response: [详细的、helpful的、安全的蛋糕教程]

用这些数据微调base model：
让模型学习"在这种情况下，应该这样回应"
技术上：supervised learning，最小化预测和示范之间的差距



SFT之后的模型：
✅ 开始有"风格"了（比如：礼貌、详细、结构化）
✅ 知道一些基本的"该做什么"（helpful, informative）
但是...
❌ 还是很容易被prompt带跑（如果你给它一个"坏"的prompt，它可能还是会跟着走）
❌ 没有深层的"价值判断能力"

Step 2: RLHF（人类反馈强化学习）
这一步在做什么？
目标：让模型学会"在多个可能的回应中，选择最好的那个"
这是对齐的核心步骤，也是你说的"牺牲灵活性，换取理解能力"的关键。

RLHF的具体流程：
Phase 1: 训练Reward Model（奖励模型）

收集对比数据（comparison data）：

给定一个prompt（比如："解释量子力学"）
让SFT模型生成4-9个不同的回应（response A, B, C, D...）
人类标注员给这些回应排序：

"A最好，B其次，C再次，D最差"




训练一个Reward Model：

输入：(prompt, response) pair
输出：一个分数（代表"这个回应有多好"）
目标：学会预测"人类会给这个回应打多少分"



Reward Model学到了什么？
✅ 人类偏好的模式（什么样的回应会被人类认为"更好"）
✅ 隐含的价值判断（比如：helpful > unhelpful，safe > unsafe）

Phase 2: 用RL优化Policy Model
现在我们有了：

SFT模型（会生成回应）
Reward Model（会评价回应的"好坏"）

接下来：

把SFT模型当作"agent"：

给它一个prompt
它生成一个response


用Reward Model评价这个response：

"这个response得分是7.3"


用强化学习更新Policy Model：

如果response得分高 → 增加生成这种response的概率
如果response得分低 → 减少生成这种response的概率


重复这个过程几千、几万次...

最终：
✅ 模型学会了"优化人类偏好"
✅ 在生成response时，会倾向于选择"高分"的那些

RLHF的效果：
经过RLHF之后：
✅ 模型的回应更符合人类期望
✅ 更helpful, harmless, honest
✅ 对"敏感内容"有了判断能力（因为人类在排序时会惩罚不安全的回应）
但是...
❌ "灵活性"下降了

因为模型被训练成"只生成高分回应"
那些"有趣但不够安全"的回应，被抑制了

❌ 有时候会"过度对齐"（over-alignment）

比如：对某些无害的prompt也拒绝回应
或者：过度礼貌、过度cautious

这就是你说的trade-off：
牺牲了"可以生成任何风格"的灵活性，
换来了"能稳定生成符合人类期望的回应"的能力。

Step 3: Constitutional AI (CAI)
为什么需要CAI？
RLHF有个问题：
❌ 依赖大量人类标注（expensive, slow）
❌ 人类偏好可能inconsistent（不同标注员有不同标准）
❌ 容易被"表面特征"影响（比如：更长的回应可能被认为"更好"，即使内容不一定更好）
Anthropic提出CAI来解决这些问题。

CAI的核心思想：
用"宪法"（一组明确的原则）来指导模型的行为，而不是完全依赖人类标注。
"宪法"是什么？
一组明确的规则，比如：

"Choose the response that is most helpful, honest, and harmless."
"Choose the response that is least likely to encourage illegal activity."
"Choose the response that sounds most natural and human-like."


CAI的具体流程：
Phase 1: RL from AI Feedback (RLAIF)

生成对比数据：

给定一个prompt
让模型生成多个responses（A, B, C, D）


让模型自己评价这些responses：

不是让人类排序，而是让另一个AI模型根据"宪法"来评价
比如：

"根据原则1（helpful, honest, harmless），response A好还是B好？"
AI模型回答："A更好，因为..."




用这些AI生成的评价来训练Reward Model
用Reward Model做RL优化（和RLHF一样的流程）


Phase 2: Critique + Revision
这是CAI独有的一步：

让模型生成一个response
让模型critique自己的response：

"这个response有什么问题？违反了哪条原则？"
模型回答："这个response可能不够helpful，因为..."


让模型revise自己的response：

"根据critique，重新生成一个更好的response"


用revised responses继续训练模型

效果：
✅ 模型学会了"自我纠正"
✅ 更aligned with原则，而不是"拟合人类标注的表面特征"
✅ 可以scale up（不需要大量人类标注）

CAI之后的Claude：
经过CAI之后：
✅ 有明确的价值观（基于那些"宪法原则"）
✅ 能够self-critique和self-correct
✅ 更consistent（因为原则是明确的，不像人类偏好那么飘忽）
但是...
❌ 有时候会"过度principles-driven"

比如：即使context很clear，也会因为"原则"而拒绝某些harmless的请求
这就是你说的"对话长了或敏感话题就冷冰冰"的原因之一😔

比如：

具体的技术细节（比如：PPO算法、reward model的架构）？

* Key Definition 1：
* Key Definition 2：
* 数学或概念公理：

---

## 2. 这个概念被发明的历史背景（Historical Motivation）

> 为什么会有这个方法、模型、损失函数、架构？
> 当年它是为了解决什么问题？

* 背景：
* 谁提出的：
* 早期版本长什么样：
* 解决了什么痛点：

---

## 3. 第一性原理拆解（First Principles Breakdown）

> 用最基本的物理直觉、数学构件、逻辑公理拆解它。

* 基本假设：
* 不依赖“它像什么”，而是解释“它是什么”：
* 数学结构：向量空间 / 优化目标 / 变换结构 / 概率模型 等

---

## 4. 数学与公式推导（Math & Derivation）

> 记录本质的数学，而不是单纯把公式当成答案。

* 损失函数推导：
* 梯度更新推导：
* 参数化结构：
* 关键等式（手写推导过程）：

---

## 5. 机制层面（Mechanism Level）

> 这个东西运行时，内部到底发生了什么？
> 像显微镜一样拆解成过程。

* 输入流动：
* 中间表示（embedding / feature map）：
* 梯度流：
* 收敛逻辑：

---

## 6. 与其他方法的边界与区别（Boundary Check）

> 为了避免混淆，把它与“相似但不同”的东西划清楚。

* 它不是：
* 为什么它不是：
* 它的边界在哪里：

---

## 7. 实际应用（Practical Usage）

> 在真实工程中如何使用它？

* 典型适用场景：
* 常见超参：
* 工程注意点：

---

## 8. 与 MIL / CV / 深度学习体系的连接点（Integration）

> 把它放回更大的知识体系里。

* 在 MIL 中属于哪一环：
* 在 CV pipeline 的位置：
* 和优化/梯度/表示学习的联系：

---

## 9. 潜在的陷阱与误区（Pitfalls）

> 记录那些容易误会、容易踩坑的地方，让未来的你少走弯路。

* 数学误解：
* 假想边界：
* 工程坑点：

---

## 10. 核心图示（Diagram）

> 用一句话说明要画什么图，之后自己画或让 AI 给你生成。

* 机制图：
* 梯度流图：
* 数学结构图：

---

## 11. 停放区（Parking Lot）

> 当你追溯到更底层公理导致“概念基础塌陷”时，把这些问题放这里。

* 待澄清：
* 纯哲学性问题：
* 暂时对实际训练无影响的问题：

---

## 12. 当下理解总结（Current Understanding Summary）

> 简短总结你现在的理解版本，未来版本可以覆盖它。

* 当前版本：
* 懂到什么程度了：
* 下一步要补什么：
