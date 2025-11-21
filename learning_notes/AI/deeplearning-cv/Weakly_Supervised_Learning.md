# AI 学习记录模板（适配你的学习方式）
## 0. 本次学习的核心主题（Topic）

* *一句话描述你正在理解的核心概念/问题是什么。*

---
# 核心问题：只有分类标签，模型怎么知道它丢的是背景？
这听起来很反直觉：没有画框（Bounding Box），模型怎么知道哪里是物体？
这里我们需要引入一个公理化的解释，即注意力机制与梯度流的博弈。
- 机制原理：在训练中，我们构造了一个包含两部分的损失函数（Loss Function）：
  - $$L_{total} = L_{cls} + \lambda \cdot L_{sparsity}$$
  - $L_{cls}$ (分类准确性)：要求模型必须要把“病害A”分对。
  - $L_{sparsity}$ (稀疏性/计算量约束)：强制要求模型必须丢弃 $K\%$ 的图像区域（Token）。
- 模型的“思考”过程（梯度下降的路径）：
  - 场景 1：丢错了（误删目标）:模型为了满足 $L_{sparsity}$，随机扔掉了包含病害的 Token。
    结果：分类层只能看到背景。 $L_{cls}$ 剧烈飙升（分错了）。
    惩罚（Gradients）：巨大的梯度回传，狠狠地修正前面的决策门控：“下次绝对不能扔这一块！”
  - 场景 2：丢对了（删除背景）:模型扔掉了纯绿色的健康叶片区域。
    结果：分类层依然能看到病害斑点。 $L_{cls}$ 保持很低（分对了）。同时 $L_{sparsity}$ 也很低（满足了丢弃要求）。
    奖励：参数更新方向会强化这种“丢弃背景”的决策。
  结论：模型并不懂语义上的“背景”和“前景”，它只懂得**“最大化保留能降低分类Loss的信息”**。
  而在我的水稻病害的数据集中，只有病害区域包含能区分 A/B/Health 的信息，背景全是冗余的。因此，最优解必然是“保留病害，丢弃背景”。

# 动态丢弃（一体化 / Integrated）流程：
网络前几层 $\rightarrow$ 决策门控 (Gating) $\rightarrow$ 网络后几层 (仅处理留下的)逻辑：特征即定位。
模型在提取特征的过程中，发现某些区域的特征响应值（Activation）很低或对最终分类贡献极小，顺手把它扔了。
优势：这是**端到端（End-to-End）**训练的。
后层的分类错误会通过梯度（Gradient）回传给决策门控，告诉它：“如果你把这块扔了，我就分不对了，所以你必须保留它。”

---

# 可能的解决办法之一:而直接从头训练动态丢弃有时很难收敛（因为一开始模型既不会分也不会丢），我们通常引入一个“老师”。
- 蒸馏（Distillation）:
  该方法不需要画框标注。利用**“信息瓶颈（Information Bottleneck）”**原理：(loss)
  强迫模型戴着镣铐（算力限制）跳舞（分类准确）。为了不摔倒，它自然学会了只把脚踩在最坚实的地方（病害区域）。
  - 训练架构 (Teacher-Student Pipeline)：
     Teacher (全知者)：先训练一个标准的 EfficientNetV2-S（或者你现有的最好模型），不进行任何丢弃。**它看全图**，能达到很高的分类准确率。
     Student (学习者 - 动态丢弃模型)：这是一个包含“丢弃模块”的网络。
  - 输入：同样的图片。(这部分可能会有问题,涉及到大图resize成小图片细节丢失)
  - 目标：分类结果要和 Teacher 一样（蒸馏损失）;特征图（Feature Map）要尽量拟合 Teacher 的特征。
  - 关键约束：只能用 Teacher 30% 的算力（丢弃 70% 的 Patch）。(loss)
  - “硬负样本（Negative Bag）”：对于那些纯健康/背景的图片，设置一个特殊的规则：允许丢弃率达到 95% 甚至更高。**(推理时应用?)**
     因为整张图都是背景，模型应该学会“看一眼全是绿的，直接全丢，输出健康”。这能极大压榨算力。(内部枝剪)
  - 边界审查 (Boundary Check)
     问题：如果病害特征极不明显（例如微小的变色），模型可能会为了满足稀疏性而将其丢弃，导致“伪阴性”。
     解决：不能强行设定固定的丢弃率（如必须丢 70%）。应该设定一个置信度阈值（Confidence Threshold）。
           如果模型对某块区域很不确定（高熵），即使它是背景看起来也像病害，也要保留。
    - 分辨率（信息量）与 显存（计算资源）的物理冲突:在训练teacher时,如果需要保留的目标太小,可能导致高频信息丢失（Information Loss），这对于“小目标病害”是致命的，
      因为病害的特征往往就在那些高频纹理中。
 > 针对这种情况,可考虑一种分层注意力架构（Hierarchical Attention Architecture），学术上常称为 "Glance and Focus"（瞥与凝视） 机制
 > 模仿人类生物视觉的公理：用低分辨率确定“哪里值得看”，用高分辨率去“看清它”。
 - Glance and Focus (G&F):我们需要构建两个解耦但协同工作的网络：
   - A. 提议网络 (Proposal Network / The "Scouter")
      - 输入：整张图像的 Resize 版本（例如 512x512）。
      - 疑问:Resize 后病害看不清了怎么办？公理回应：虽然看不清“是什么病”，但通常能保留“这里不对劲”的低频特征（如叶片颜色不均、纹理异常）。
      - Scouter 不需要做分类，只需要做显著性检测（Saliency Detection）。
      - 输出：一个 $N \times N$ 的概率网格（Grid），或者 Top-K 个值得关注的坐标中心。
      - 模型量级：极轻量（如 ResNet-18 甚至 MobileNet）。
    - B. 识别网络 (Fine-grained Network / The "Expert")
      - 输入：根据 Scouter 提供的坐标，从原图（3000x4000） 上 Crop 下来的高分辨率 Tile（例如 384x384）。
      - 模型量级：重量级（你的 EfficientNetV2-S）。
      - 操作：只对 Scouter 选出的 Top-K 个 Tile 进行推理。
  - 核心难点：没有Box，Scouter 怎么学？:在只有图像级标签（Image-level Label）的情况下，如何训练 Scouter 准确圈出病害区域，而不是背景？
    > 这里我们需要利用 MIL（多示例学习） 的数学本质来反向传播信号。
    - 训练策略：Top-K 优胜劣汰 (Top-K Selection & Backpropagation)**tile前提：我们假设一张图被切分为 Grid（比如 $8 \times 8 = 64$ 个区域）。**
      - 前向传播 (Forward Pass):Scouter 看 Resize 图，给 64 个区域打分（重要性分数 $s_1, ..., s_{64}$）。
        - 选出分数最高的 $K$ 个区域（比如 $K=5$）。从原图 Crop 出这 5 个高清 Tile，送入 Expert。
        - Expert 对这 5 个 Tile 分别给出预测 logits。
        - 聚合逻辑：通常使用 Max Pooling 或 Attention Pooling 综合这 5 个结果得到最终的 Image Prediction。
      - 反向传播 (Backward Pass) - 关键点:如果 Image Prediction 错了（Loss 大），梯度会回传。
        - Expert 的责任：我没认出病害 $\rightarrow$ 更新 Expert 参数。
        - Scouter 的责任：我选的这 5 个 Tile 全是绿叶子（背景），导致 Expert 没法认 $\rightarrow$
        - **这就是 Scouter 学习的信号！**
          - 因为 Expert 对这 5 个背景 Tile 的预测肯定不支持“病害”标签，
          - Loss 会惩罚 Scouter：“你选的这些区域对于分类没有贡献（贡献率为0），下次选别的区域！”
        - 技术障碍：**不可导的“Crop”操作直接**“选 Top-K 并 Crop”这个动作是离散的，梯度断了。
        - 解决方案：硬注意力（Hard Attention）+ 强化学习（REINFORCE算法）：
          - 把 Scouter 当作 Agent，把“选对了区域导致分类正确”当作 Reward。但这很难训练，不稳定。
          - 推荐：基于分数的加权（Score-based Weighting）。
              - 可以在初期训练时，随机抽样 Tile 送给 Expert，然后看 Expert 对哪个 Tile 的反应最大（Activation 最高）。
              - Scouter 的目标就是去拟合 Expert 的这种“兴奋度”。
              - 这被称为 知识蒸馏（Knowledge Distillation） 的一种变体：
              - Expert 是教师（它知道哪个高清块有用），Scouter 是学生（它要在低清图上猜哪个块会有用）。
    > 考虑到VRAM限制,可以结合异步迭代（Asynchronous Iteration） 的策略：
    - 第一阶段：训练 Expert (纯tile分类器)
    > **但我感觉这样会出问题,tile之后的正包病害混在一起了,还包含tile之间原图的某种信息,模型可能作弊**
      - 数据：把所有训练集的图，暴力切成 Tile。
      - 标签分配：
        - 对于负包（健康图）：所有 Tile 都是“健康”。
        - 正包（病害图）：有噪声。可以**暂时把所有 Tile 都标为“病害”（Label Smoothing）**，
        - 或者简单的传统图像处理（如颜色阈值）过滤掉纯绿背景，只保留“疑似”Tile 进行训练。
        > - **类似启发式前景,但这个可能有问题万一选中了杂草噪音怎么办?**背景也需要分类?这时候的硬负样本可以做什么?
      - 目的：得到一个能识别“高清病害 Tile”的强力特征提取器。
    - 第二阶段：训练 Scouter
      - (核心一步)冻结 Expert。
      - 输入：
        - 原图 Resize $\rightarrow$ Scouter $\rightarrow$ 预测热力图 (Heatmap)。
        - 原图所有/随机抽样的 Tile $\rightarrow$ Frozen Expert $\rightarrow$ 真实的重要性分数 (Expert 认为这个 Tile 是病害的概率)。
      - Loss：让 Scouter 的热力图去拟合 Expert 的重要性分数分布。
        - $$L_{scout} = MSE(Heatmap_{scouter}, ProbabilityMap_{expert})$$
      - 逻辑：Scouter 学会了在低分辨率下，寻找那些“Expert 看了会说是病害”的区域。
    - 第三阶段：推理 (Inference)
      - 输入大图 $\rightarrow$ Resize $\rightarrow$ Scouter。
      - Scouter 输出 Top-5 坐标。
      - 只 Crop 这 5 个 Tile $\rightarrow$ Expert。
      - 输出分类结果。
  - 边界审查 (Boundary Check)
    - Resize 盲区风险：病害是针尖大小（例如 3000x4000 图上的 10x10 像素点），且在 Resize 到 512x512 后彻底消失（变成亚像素），
      - 那么 Scouter 永远学不会。
      - 验证方法：肉眼看 512x512 的图，你能大概指出病害位置吗？如果人眼完全看不见，神经网络也很难。
    - 负样本清洗：不要试图让 Expert 强行拟合正包里的所有 Tile。正包里 90% 都是背景。
      - 容忍 Expert 对正包里的某些 Tile 输出“健康”。
      - MIL 的核心就是 Label = Max(Tile_Labels)，只要有一个 Tile 是病害，整包就是病害。
---

## **关于损失函数:单纯对所有Tile使用BCE Loss（二元交叉熵）确实不是最优解，甚至在你的数据分布下存在严重的数学逻辑冲突。**
1. 为什么现在的 BCE 是“有毒”的？(The Mathematical Conflict)

  - 现在做法推测是：如果一张大图是“病害A”，你就把这张图切出来的 20 个 Tile 全部打上“病害A”的标签，然后算 BCE Loss。
  - 这是一个噪声注入过程，会导致梯度冲突：
    场景 1（正包）：大图是病害。其中 Tile_1 是病害斑点，Tile_2 是杂草背景。
    loss告诉模型：Tile_2 (杂草) = 病害。
    梯度方向：拉高杂草特征对应“病害”的权重。
    场景 2（负包/硬负样本）：大图是健康/农田。其中 Tile_3 是杂草背景（和 Tile_2 极像）。
    loss告诉模型：Tile_3 (杂草) = 健康。
    梯度方向：拉低杂草特征对应“病害”的权重。
    **结果：模型精神分裂了**:它看到“杂草”时，既被惩罚又被奖励，导致**震荡和无法收敛**。这就是为什么你觉得它可能分不清。
2. 修正方案：
  - 基于 Top-K 聚合的 MIL Loss
    > 需要从“强迫模型看所有Tile”转变为**“允许模型挑着看”**。不需要改变模型架构，只需要改变 Loss 的计算方式。
    - 新的公理逻辑：
      > $$Prediction_{Bag} = Aggregation(Predictions_{Tiles})$$
      > $$Loss = Criterion(Prediction_{Bag}, Label_{Bag})$$
      在此逻辑下，背景 Tile 在正包中产生的预测值（应该是低的）会被聚合函数过滤掉，不会参与梯度回传。
      只有那个分值最高的 Tile（也就是模型认为最像病害的那个）会对 Loss 负责。
    - 聚合策略：Max-Pooling 或 Top-K Mean
      > 对于多分类问题（病害A, 病害B, 健康），建议使用以下设计：
        - 输入：一个 Bag 的 $N$ 个 Tile。
        - 模型输出：每个 Tile 输出 $C$ 个类别的 Logits（假设 $C$ 类病害）。
        - 聚合 (Aggregation)：
          > 对于每一类 $c \in C$：$$Logit_{Bag}^{(c)} = \max_{i=1}^{N} (Logit_{Tile\_i}^{(c)})$$
          > (解释：对于病害A，只看所有Tile里最像病害A的那个分值。)
        - Loss 计算：$$Loss = CrossEntropy(Logit_{Bag}, Label_{Bag})$$
        - 改动的本质影响：
          - 正包中的背景 Tile：因为它们的分数（Logits）通常低于真正的病害 Tile，所以它们会被 max 函数屏蔽。
            - 它们不产生梯度。模型不再被强迫把“杂草”学成“病害”。
          - 负包中的背景 Tile：因为标签是“健康”，CrossEntropy 会迫使 Max(Logits) 变低。
            - 这意味着负包里所有 Tile 的病害分值都必须很低（只要有一个高了，Loss就会大）。这正是**压制硬负样本。**
---

## 节省算力的策略:引入稀疏性约束 (Sparsity Constraint)
针对“更少的计算量”和“丢弃背景”:在 MIL框架稳定后（Warm-up 后），加入一个**正则化项（Regularization Term）**来逼迫模型更加“挑剔”。
> 期望模型不仅是“能找到最强的Tile”，而且是“除了那几个强的，其他的都认为是垃圾”。
- 优化后的 Loss 函数：
  - $$L_{total} = L_{class} + \lambda \cdot L_{sparse}$$
    > 其中 $L_{sparse}$ 可以设计为 $L1$ 正则化，作用在预测概率上：
  - $$L_{sparse} = \frac{1}{N} \sum_{i=1}^{N} |P(Tile_i)|$$
  - 原理：$L_{class}$ (Max-Pooling) 鼓励至少有一个 Tile 分数高（为了分类对）。
    - $L_{sparse}$ 鼓励尽可能多的 Tile 分数低（为了省能量）。
    - 博弈结果：模型会收敛到“只给真病害打高分，背景全打0分”的状态。
    - 实际应用：训练好后，推理时你就可以设定一个阈值（如 0.1）。低于 0.1 的 Tile 直接丢弃，不进入后续的复杂计算（或者在多级网络中被早早过滤）。


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
