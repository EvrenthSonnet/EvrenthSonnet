# RiceDetection项目 - Claude管理系统

> 完整的Claude规则、技能和自动化系统，用于管理像素级图像分类（语义分割）项目

---

## 📚 系统概览 (System Overview)

本系统基于Reddit用户 JokeGold5455 的6个月30万行代码实践经验，结合你的第一性原理学习风格，专门为RiceDetection项目定制。

### 核心问题的解决

**问题1：Claude每次新session都"失忆"**
- **解决：** Hooks系统 + 自动上下文注入

**问题2：项目结构容易被破坏**
- **解决：** PROJECT_RULES.md + 结构验证hooks

**问题3：重复实现已有功能**
- **解决：** 技能系统 + 功能索引

**问题4：代码不符合规范**
- **解决：** 自动检查 + 主动提醒

---

## 🗂️ 目录结构 (Directory Structure)

```
RiceDetection/.claude/
├── README.md                          # 本文件（系统总览）
├── CLAUDE_RULES.md                    # 交互风格规则（第一性原理学习）
├── PROJECT_RULES.md                   # 项目工程规范
│
├── skills/                            # 技能库
│   └── cv-semantic-segmentation/      # 计算机视觉-语义分割专家
│       ├── SKILL.md                   # 主技能文件
│       └── resources/                 # 按需加载的资源（可选）
│           ├── architectures.md       # 模型架构详解
│           ├── loss_functions.md      # 损失函数详解
│           └── tiling_strategies.md   # Tile切分策略
│
├── hooks/                             # 钩子系统
│   ├── README.md                      # Hooks系统说明
│   ├── user-prompt-submit.md          # 用户提示词提交钩子
│   ├── post-tool-use.md               # 工具使用后钩子
│   └── structure-validator.md         # 项目结构验证钩子
│
├── commands/                          # 斜杠命令
│   ├── learning-note.md               # /learning-note 生成学习笔记
│   └── dev-docs.md                    # /dev-docs 生成开发文档
│
├── agents/                            # 专项代理
│   ├── concept-explainer.md           # 概念解释代理（第一性原理）
│   └── code-reviewer.md               # 代码审查代理
│
└── rules/                             # 项目特定规则
    ├── project_structure.md           # 项目结构文档
    └── coding_standards.md            # 编码规范细则
```

---

## 🎯 双层规则体系 (Dual-Layer Rule System)

### 第一层：交互风格规则（CLAUDE_RULES.md）

**适用场景：** 学习和理解概念时

**触发条件：**
- 用户问"什么是XXX？"
- 用户问"为什么XXX？"
- 用户请求"解释XXX的原理"
- 用户提到"数学推导"、"历史背景"

**行为模式：**
```markdown
## [概念名称] (English Term)

### 📜 历史起源
- 谁在什么时候提出？
- 为了解决什么问题？

### 🧮 公理定义
- 严格的数学定义
- 基本假设和公理

### 🔨 推导与应用
- 从公理推导出的性质
- PyTorch实现（带详细注释）

### 🅿️ 边界问题（如有）
- 停放区中的哲学边界问题
```

**示例：**
```
用户："什么是Focal Loss？"

我的回应：
📜 历史起源：Lin et al. (2017) 在RetinaNet论文中提出...
🧮 公理定义：FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)...
🔨 PyTorch实现：class FocalLoss(nn.Module): ...
🅿️ 边界问题：为什么γ=2是最优值？这涉及信息论...
```

---

### 第二层：项目工程规范（PROJECT_RULES.md）

**适用场景：** 编码、修改、优化时

**触发条件：**
- 用户请求"实现XXX功能"
- 用户请求"修改XXX模块"
- 用户请求"优化XXX性能"
- 用户请求"部署XXX"

**行为模式：**
```
1. 检查项目结构（读取 rules/project_structure.md）
2. 搜索现有实现（src/rice_detection/ 下）
3. 确认修改位置符合规范
4. 实施修改（遵守编码规范）
5. 更新文档（脚本注释 + README）
6. 提醒测试建议
```

**示例：**
```
用户："实现数据增强"

我的回应：
1. ✅ 检查现有实现：发现 data/albumentations_transforms.py 已存在
2. 📝 报告："数据增强已在 albumentations_transforms.py:42 实现，
             支持Flip、Rotate、ColorJitter。是否需要添加新的增强方式？"
3. ⏸️ 等待用户确认后再实施
```

---

## 🛠️ 技能系统 (Skills System)

### 当前可用技能

#### 1. cv-semantic-segmentation（计算机视觉-语义分割专家）

**激活条件：**
- 关键词：语义分割、Dice Loss、Focal Loss、UNet、Tile
- 文件路径：`src/rice_detection/models/`, `src/rice_detection/data/`
- 任务类型：模型架构、损失函数、数据处理

**核心功能：**
- ✅ 从第一性原理解释CV概念
- ✅ 提供PyTorch实现（带详细注释）
- ✅ 检查并复用现有功能
- ✅ 主动提醒常见错误（如训练/推理不一致）
- ✅ 针对大图小目标场景优化

**使用示例：**
```
用户："如何处理类别不平衡？"

技能激活 → cv-semantic-segmentation

我的回应：
类别不平衡是语义分割中的常见问题，特别是大图小目标场景。

📊 你的数据特点：
- 背景类（背景）：占比约95%
- 目标类（水稻病害）：占比约5%

✅ 推荐解决方案（按优先级）：

1. **Dice Loss**（最推荐）
   - 原理：基于重叠度而非像素数量
   - 优势：对类别不平衡不敏感
   - 实现：见 losses/dice_loss.py

2. **Focal Loss**
   - 原理：降低简单样本权重，关注困难样本
   - 优势：明确针对类别不平衡设计
   - 超参数：γ=2.0, α=[0.25, 0.75]

3. **加权Cross-Entropy**
   - 原理：给少数类更高的损失权重
   - 实现简单但效果一般

⚠️ 注意：在你的config中添加：
loss:
  type: "dice"  # 或 "focal" 或 "weighted_ce"
```

---

## 🪝 Hooks系统 (Hooks System)

### 核心Hooks

#### 1. user-prompt-submit（用户提示词提交钩子）

**功能：** 在用户提交提示词前，自动分析并增强

**工作流程：**
```
用户输入 → Hook拦截 → 分析关键词 → 匹配技能 → 注入上下文 → Claude处理
```

**示例：**
```
用户："实现Tile切分"

Hook分析：
- 关键词"Tile" → 激活 cv-semantic-segmentation skill
- 关键词"实现" → 加载 PROJECT_RULES.md
- 提醒：先检查 src/rice_detection/data/ 是否已有实现
```

---

#### 2. post-tool-use（工具使用后钩子）

**功能：** 在Claude使用工具后，自动检查和提醒

**工作流程：**
```
Claude使用工具 → Hook检查 → 验证合规性 → 提醒文档更新
```

**示例：**
```
Claude: Edit → src/rice_detection/training/trainer.py

Hook检查：
✅ 路径符合项目结构
⚠️ 提醒：请更新脚本头部注释的 'Recent Updates' 部分
⚠️ 提醒：如果修改了训练逻辑，请更新 training/README.md
```

---

### Hooks配置文件（skill-rules.json）

```json
{
  "cv-semantic-segmentation": {
    "auto_activate": true,
    "triggers": {
      "keywords": ["语义分割", "Dice", "Focal", "UNet", "Tile"],
      "file_patterns": ["src/rice_detection/{models,data,losses}/**/*.py"]
    }
  }
}
```

---

## 📝 使用指南 (Usage Guide)

### 快速开始（3步）

#### 步骤1：复制到你的RiceDetection项目

```bash
# 在你的Windows电脑上
# 把这个.claude目录复制到：
D:\PyCharm\pycharmprojects\RiceDetection\.claude\
```

#### 步骤2：修改项目特定配置

编辑 `.claude/rules/project_structure.md`，填入你的实际项目结构：

```markdown
# RiceDetection项目结构

RiceDetection/
├── src/
│   └── rice_detection/
│       ├── data/              # 你的数据处理模块
│       │   ├── __init__.py
│       │   ├── tile_dataset.py
│       │   └── ...
│       ├── models/            # 你的模型定义
│       └── ...
├── configs/
└── ...
```

#### 步骤3：测试效果

在Claude Code中打开RiceDetection项目，然后：

```
测试1：问"什么是Dice Loss？"
期望：Claude从历史背景→数学定义→PyTorch实现（第一性原理模式）

测试2：说"实现数据增强"
期望：Claude先搜索现有实现，再征求确认（工程模式）

测试3：修改一个.py文件
期望：Claude修改后自动提醒更新文档
```

---

### 日常工作流

#### 场景1：学习新概念

```
你："解释一下Focal Loss的数学原理"

Claude（自动激活CLAUDE_RULES.md + cv-semantic-segmentation skill）：
- 📜 历史起源：Lin et al. (2017)，为了解决目标检测中的前景/背景极度不平衡...
- 🧮 数学定义：FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
  核心思想：简单样本权重低，困难样本权重高...
- 🔨 PyTorch实现：[详细代码，带逐行注释]
- 🅿️ 边界问题：为什么γ=2是最优值？...
```

---

#### 场景2：实现新功能

```
你："实现Dice Loss"

Claude（自动激活PROJECT_RULES.md + cv-semantic-segmentation skill）：
1. ✅ 搜索现有实现...
   → 未找到 src/rice_detection/losses/dice_loss.py
2. 📝 建议：在 src/rice_detection/losses/ 下创建 dice_loss.py
3. ⏸️ 是否继续？

你："继续"

Claude：
1. ✅ 创建文件：src/rice_detection/losses/dice_loss.py
2. ✅ 更新文档：losses/README.md
3. ✅ 提醒：
   - 在 config 中添加 loss.type: "dice"
   - 建议编写单元测试 test_dice_loss.py
   - Dice Loss特别适合你的大图小目标场景（对类别不平衡不敏感）
```

---

#### 场景3：调试问题

```
你："训练速度很慢，怎么优化？"

Claude（自动激活cv-semantic-segmentation skill）：
1. 🔍 诊断瓶颈：
   - 数据加载？→ 使用 DataLoader(num_workers=4, pin_memory=True)
   - 模型前向？→ 启用混合精度训练（AMP）
   - Tile切分？→ 实现Lazy Loading，避免预切分所有图片

2. 📊 预期效果：
   - DataLoader优化：提升2-3x
   - 混合精度：提升1.5-2x
   - Lazy Loading：内存从161GB降到5MB

3. ⏸️ 选择哪个方案？
```

---

## 🎓 学习资源 (Learning Resources)

### 推荐学习路径

#### 路径1：语义分割基础

```
第1周：基础概念
- 什么是语义分割？（像素级分类）
- 与目标检测、实例分割的区别
- 评估指标：IoU、Dice、Pixel Accuracy

第2周：经典架构
- UNet（医学图像分割）
- DeepLab（ASPP多尺度）
- 编码器-解码器结构的本质

第3周：损失函数
- Cross-Entropy Loss
- Dice Loss
- Focal Loss
- 组合损失策略

第4周：工程实践
- Tile切分策略
- 类别不平衡处理
- 训练技巧（学习率、增强、正则化）
```

**每个话题都会按照第一性原理模式教学：**
- 📜 历史起源
- 🧮 数学定义
- 🔨 代码实现
- 🅿️ 边界问题

---

#### 路径2：PyTorch工程实践

```
第1周：模块化设计
- 数据模块：Dataset、DataLoader、transforms
- 模型模块：backbone、head、loss
- 训练模块：Trainer、optimizer、scheduler

第2周：配置驱动开发
- YAML配置文件设计
- Hydra框架（可选）
- 实验管理

第3周：部署优化
- 模型导出：PyTorch → ONNX
- 推理优化：TensorRT、OpenVINO
- 服务化：FastAPI、Docker

第4周：调试与测试
- 单元测试（pytest）
- 性能分析（torch.profiler）
- 可视化（TensorBoard、WandB）
```

---

## ⚠️ 我会主动提醒的常见错误

### 新手常见错误清单

#### 1. 训练/推理不一致
```
❌ 训练时：tile_size=512, overlap=64
❌ 推理时：tile_size=1024, overlap=128

我会提醒：
⚠️ Tile切分参数不一致！这会导致性能下降。
✅ 建议：在 configs/base.yaml 中统一定义 tile_config
```

#### 2. 忘记设置随机种子
```
❌ 代码中没有设置seed

我会提醒：
⚠️ 未检测到随机种子设置！实验不可复现。
✅ 建议：在 configs/xxx.yaml 中添加 seed: 42
```

#### 3. 类别不平衡未处理
```
❌ 数据集：背景95% vs 目标5%
❌ 损失函数：标准CrossEntropyLoss

我会提醒：
⚠️ 类别不平衡数据集 + 标准CE Loss = 模型偏向背景类
✅ 建议：使用Dice Loss或Focal Loss
```

#### 4. 硬编码路径
```
❌ image_path = "/home/user/data/train/"

我会提醒：
⚠️ 硬编码路径！违反配置驱动原则。
✅ 建议：在 config.yaml 中定义 data.train_path
```

#### 5. 重复实现功能
```
用户："实现数据增强"
我：搜索发现已有 data/albumentations_transforms.py

我会提醒：
⚠️ 数据增强已在 albumentations_transforms.py:42 实现。
✅ 是否需要复用现有实现，还是添加新的增强方式？
```

---

## 📊 系统效果评估

### 衡量指标

| 指标 | 目标 | 当前 |
|------|------|------|
| 项目结构违规次数 | 0次/周 | - |
| 文档同步率 | 100% | - |
| 功能复用率 | >80% | - |
| 代码规范符合度 | 100% | - |

---

## 🔧 自定义和扩展

### 添加新技能

```bash
1. 创建技能目录：.claude/skills/你的技能名/
2. 编写 SKILL.md：定义触发条件、核心知识、行为模式
3. 更新 skill-rules.json：添加激活规则
4. 测试：询问相关问题，观察是否自动激活
```

### 添加新Hook

```bash
1. 创建Hook文件：.claude/hooks/你的hook名.md
2. 定义触发时机和逻辑
3. 测试：执行相关操作，观察Hook是否触发
```

---

## 📞 获取帮助

### 快速命令

在Claude Code中，你可以直接问：

```
# 学习模式
"什么是IoU？"
"为什么Dice Loss对类别不平衡不敏感？"
"Focal Loss的数学推导"

# 工程模式
"实现Tile切分"
"优化训练速度"
"检查我的代码是否符合规范"

# 诊断模式
"为什么训练Loss不下降？"
"为什么推理结果和训练不一致？"
"我的项目结构合理吗？"
```

---

## 🎯 下一步建议

### 立即可做的3件事

1. **复制.claude目录到RiceDetection项目**
   - 路径：`D:\PyCharm\pycharmprojects\RiceDetection\.claude\`

2. **填写项目结构文档**
   - 编辑：`.claude/rules/project_structure.md`
   - 描述你当前的目录结构和模块划分

3. **测试效果**
   - 问Claude："什么是Dice Loss？"（测试学习模式）
   - 说Claude："实现数据增强"（测试工程模式）

---

## 📚 参考资料

- [Reddit原帖](https://github.com/diet103/claude-code-infrastructure-showcase)
- [Claude Code官方文档](https://docs.claude.com/en/docs/claude-code/)
- [第一性原理学习方法](your_learning_methodology)

---

*系统版本：1.0*
*创建日期：2025-01-14*
*适用项目：RiceDetection（像素级图像分类/语义分割）*
*作者：为Verse定制*
