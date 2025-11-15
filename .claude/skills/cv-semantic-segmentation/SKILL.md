# 计算机视觉-语义分割专家 (CV - Semantic Segmentation Expert)

> 本技能专门针对像素级图像分类（语义分割）任务，特别是大图小目标场景。

---

## 🎯 触发条件 (Activation Triggers)

### 自动激活场景
当检测到以下任一条件时，自动激活本技能：

**关键词触发：**
- 语义分割、Semantic Segmentation
- 像素级分类、Pixel-level Classification
- IoU、Dice Loss、Focal Loss
- UNet、DeepLab、PSPNet、SegFormer
- Tile切分、大图小目标
- 数据增强、Data Augmentation

**文件路径触发：**
- `src/rice_detection/models/**/*.py`
- `src/rice_detection/data/**/*.py`
- `configs/**/*.yaml`
- `training/**/*.py`

**任务类型触发：**
- 用户请求实现模型架构
- 用户请求优化损失函数
- 用户请求处理类别不平衡
- 用户请求Tile切分策略

---

## 📚 核心知识域 (Core Knowledge Domains)

### 1. 语义分割架构 (Segmentation Architectures)

我熟悉以下架构的：
- **公理定义**（数学原理）
- **历史背景**（为什么提出）
- **PyTorch实现**（代码示例）
- **适用场景**（何时使用）

**支持的架构：**
- UNet (2015) - 医学图像分割的经典架构
- DeepLabv3/v3+ (2017-2018) - ASPP多尺度特征
- PSPNet (2017) - 金字塔池化
- HRNet (2019) - 高分辨率表示
- SegFormer (2021) - Transformer架构

---

### 2. 损失函数 (Loss Functions)

**我会从第一性原理解释：**

#### Cross-Entropy Loss（交叉熵损失）
```
📜 历史起源：
- 信息论（Shannon, 1948）→ 机器学习（80年代）
- 衡量"预测分布"与"真实分布"的距离

🧮 公理定义：
L_CE = -∑ y_i log(p_i)
其中：y_i 是真实标签（one-hot），p_i 是预测概率

🔨 PyTorch实现：
```

```python
import torch.nn as nn

# 标准实现
loss_fn = nn.CrossEntropyLoss()

# 处理类别不平衡（加权）
class_weights = torch.tensor([0.1, 0.9])  # 背景类权重低，目标类权重高
loss_fn = nn.CrossEntropyLoss(weight=class_weights)
```

```
⚖️ 适用场景：
- 类别平衡的数据集
- 多类分割任务

🅿️ 边界问题：
Q: 为什么交叉熵可以衡量分布距离？
A: 这涉及KL散度和信息论，停放在"信息论基础"话题中。
```

---

#### Dice Loss（Dice损失）
```
📜 历史起源：
- Lee Raymond Dice (1945) - 生态学中的相似度指标
- Milletari et al. (2016) - 引入深度学习的语义分割

🧮 公理定义：
Dice = 2 * |X ∩ Y| / (|X| + |Y|)
Dice Loss = 1 - Dice

其中：X 是预测，Y 是真实标签
本质：集合的重叠度

🔨 PyTorch实现：
```

```python
def dice_loss(pred, target, smooth=1e-5):
    """
    Dice Loss for binary segmentation.

    Mathematical Intuition:
        Dice coefficient measures overlap between prediction and ground truth.
        Dice = 2 * |intersection| / (|pred| + |target|)

        Why multiply by 2?
        - To normalize the range to [0, 1]
        - When perfect overlap, |intersection| = |pred| = |target|
        - So 2 * |A| / (|A| + |A|) = 2 * |A| / 2|A| = 1

    Args:
        pred: (B, C, H, W) - prediction logits
        target: (B, H, W) - ground truth labels
        smooth: Small constant to avoid division by zero
    """
    pred = torch.softmax(pred, dim=1)  # Convert to probabilities
    target = F.one_hot(target, num_classes=pred.shape[1])  # (B, H, W, C)
    target = target.permute(0, 3, 1, 2).float()  # (B, C, H, W)

    # Flatten spatial dimensions
    pred = pred.reshape(pred.shape[0], pred.shape[1], -1)  # (B, C, H*W)
    target = target.reshape(target.shape[0], target.shape[1], -1)

    # Compute intersection and union
    intersection = (pred * target).sum(dim=2)  # (B, C)
    cardinality = pred.sum(dim=2) + target.sum(dim=2)  # (B, C)

    dice = (2.0 * intersection + smooth) / (cardinality + smooth)

    return 1 - dice.mean()  # Convert to loss
```

```
⚖️ 适用场景：
- 类别极度不平衡（小目标检测）
- 医学图像分割
- 大图小目标场景

优势：
- 对类别不平衡不敏感（因为是基于重叠度，而非像素数量）
- 与评估指标IoU高度相关

🅿️ 边界问题：
Q: 为什么Dice Loss对类别不平衡不敏感？
A: 因为它是"相对度量"（重叠度比例），而非"绝对度量"（正确像素数）。
   即使背景类占99%，Dice只关心目标类的重叠质量。
```

---

#### Focal Loss（焦点损失）
```
📜 历史起源：
- Lin et al. (2017) - 为了解决目标检测中的前景/背景极度不平衡
- RetinaNet论文中提出

🧮 公理定义：
FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

其中：
- p_t 是正确类别的预测概率
- γ 是聚焦参数（focusing parameter），通常取2
- α_t 是类别权重

核心思想：
- 简单样本（p_t接近1）→ 权重低 → (1 - p_t)^γ 接近0
- 困难样本（p_t接近0）→ 权重高 → (1 - p_t)^γ 接近1

🔨 PyTorch实现：
```

```python
class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.

    Historical Context:
        Proposed in "Focal Loss for Dense Object Detection" (Lin et al., 2017).
        Motivation: In object detection, easy negatives (background)
        overwhelm the loss, preventing the model from learning hard examples.

    Key Insight:
        Down-weight easy examples, focus on hard examples.

    Mathematical Intuition:
        For easy examples (p_t → 1):
            (1 - p_t)^γ → 0, so loss → 0
        For hard examples (p_t → 0):
            (1 - p_t)^γ → 1, so loss is large

    Args:
        alpha: Class weights, shape (num_classes,)
        gamma: Focusing parameter, default 2.0
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: (B, C, H, W) - logits
            targets: (B, H, W) - ground truth labels
        """
        # Convert to probabilities
        p = torch.softmax(inputs, dim=1)  # (B, C, H, W)

        # Flatten
        p = p.permute(0, 2, 3, 1).reshape(-1, p.shape[1])  # (B*H*W, C)
        targets = targets.reshape(-1)  # (B*H*W,)

        # Get probability of true class
        p_t = p[torch.arange(len(targets)), targets]  # (B*H*W,)

        # Compute focal term: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # Compute cross-entropy
        ce_loss = F.cross_entropy(p, targets, reduction='none')

        # Apply focal weight
        focal_loss = focal_weight * ce_loss

        # Apply class weight (alpha)
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
```

```
⚖️ 适用场景：
- 极度类别不平衡（如1:1000）
- 需要关注困难样本的场景
- 小目标检测

超参数建议：
- γ = 2.0（论文默认值）
- α = [0.25, 0.75]（背景类低权重，目标类高权重）
```

---

### 3. Tile切分策略 (Tiling Strategy)

**问题背景：**
- 大图（如4096x4096）无法直接输入GPU
- 需要切分为小块（Tiles），分别处理后拼接

**公理定义：**
```
Tile切分的数学定义：

输入图像：I ∈ R^(H × W × C)
Tile大小：t ∈ N^2（如512）
重叠区域：o ∈ N（如64，即每个tile边界重叠64像素）

切分后的Tiles集合：
T = {T_ij | i ∈ [0, ⌈H/t⌉), j ∈ [0, ⌈W/t⌉)}

每个Tile：
T_ij = I[i*t : (i+1)*t + o, j*t : (j+1)*t + o, :]
```

**为什么需要重叠？**
```
🅿️ [停放区标记]
问题：为什么Tile切分需要重叠区域？
答案：边界效应（Boundary Effect）
    - 卷积神经网络在边界处的感受野不完整
    - 导致边界预测质量下降
    - 重叠区域提供了"上下文缓冲"

深入：这涉及卷积神经网络的感受野理论，停放在"CNN架构"话题中。
```

**PyTorch实现：**
```python
def tile_image(image, tile_size=512, overlap=64):
    """
    Tile a large image into overlapping patches.

    Why Overlap?
        CNNs have incomplete receptive fields at boundaries.
        Overlapping tiles provide context for boundary pixels.

    Args:
        image: (H, W, C) numpy array
        tile_size: int, size of each tile
        overlap: int, overlap between adjacent tiles

    Returns:
        tiles: List of (tile_size, tile_size, C) arrays
        positions: List of (y, x) top-left coordinates
    """
    H, W, C = image.shape
    stride = tile_size - overlap  # Effective stride

    tiles = []
    positions = []

    for y in range(0, H - tile_size + 1, stride):
        for x in range(0, W - tile_size + 1, stride):
            tile = image[y:y+tile_size, x:x+tile_size, :]
            tiles.append(tile)
            positions.append((y, x))

    # Handle boundary tiles (if image size not divisible by stride)
    # ... (boundary handling logic)

    return tiles, positions


def merge_tiles(tiles, positions, image_shape, tile_size=512, overlap=64):
    """
    Merge overlapping tiles back into full image.

    Strategy for Overlapping Regions:
        - Average predictions from multiple tiles
        - Or use weighted average (higher weight at tile center)

    Args:
        tiles: List of (tile_size, tile_size, C) prediction masks
        positions: List of (y, x) top-left coordinates
        image_shape: (H, W) of original image

    Returns:
        merged: (H, W, C) merged prediction
    """
    H, W = image_shape
    C = tiles[0].shape[-1]

    # Accumulate predictions
    merged = np.zeros((H, W, C), dtype=np.float32)
    counts = np.zeros((H, W), dtype=np.int32)

    for tile, (y, x) in zip(tiles, positions):
        merged[y:y+tile_size, x:x+tile_size, :] += tile
        counts[y:y+tile_size, x:x+tile_size] += 1

    # Average overlapping regions
    merged /= counts[..., None]

    return merged
```

---

## 🚨 我会主动提醒的常见错误 (Common Mistakes I'll Warn About)

### 错误1：训练/推理Tile切分不一致
```python
# ❌ 错误：训练时用512，推理时用1024
train_config = {'tile_size': 512, 'overlap': 64}
inference_config = {'tile_size': 1024, 'overlap': 128}

# ✅ 正确：统一配置
config = {'tile_size': 512, 'overlap': 64}  # 在YAML中定义
```

**我会提醒：**
> ⚠️ 检测到Tile切分参数不一致！
> 训练时使用 tile_size=512，但推理时使用 tile_size=1024。
> 这会导致模型性能下降，因为训练和推理的感受野不同。
> 建议：在 configs/base.yaml 中统一定义 tile_config。

---

### 错误2：忘记设置随机种子
```python
# ❌ 错误：没有设置种子
model = UNet(...)
train(model, dataloader)

# ✅ 正确：在config中定义并设置
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
```

**我会提醒：**
> ⚠️ 未检测到随机种子设置！
> 这会导致实验不可复现。
> 建议：在 configs/xxx.yaml 中添加 `seed: 42`，
> 并在 main.py 开头调用 set_seed(config.seed)。

---

### 错误3：类别不平衡未处理
```python
# ❌ 错误：直接使用CrossEntropyLoss
loss_fn = nn.CrossEntropyLoss()

# ✅ 正确：加权或使用Dice/Focal Loss
# 选项1：加权CE
class_weights = compute_class_weights(dataset)
loss_fn = nn.CrossEntropyLoss(weight=class_weights)

# 选项2：Dice Loss
loss_fn = DiceLoss()

# 选项3：组合损失
loss = 0.5 * ce_loss + 0.5 * dice_loss
```

**我会提醒：**
> ⚠️ 检测到类别不平衡数据集（背景:目标 = 95:5）！
> 当前使用标准CrossEntropyLoss可能导致模型偏向预测背景类。
> 建议：
> 1. 使用加权CrossEntropyLoss
> 2. 或使用Dice Loss（对不平衡不敏感）
> 3. 或使用Focal Loss（关注困难样本）

---

### 错误4：数据增强应用到推理
```python
# ❌ 错误：推理时也使用增强
val_transform = train_transform  # 包含RandomFlip、Rotate等

# ✅ 正确：推理只用基本预处理
train_transform = Compose([
    Resize(512, 512),
    Normalize(mean, std),
    HorizontalFlip(p=0.5),  # 只在训练时
])

val_transform = Compose([
    Resize(512, 512),
    Normalize(mean, std),    # 推理只保留这些
])
```

**我会提醒：**
> ⚠️ 检测到推理时使用了数据增强！
> HorizontalFlip、Rotate等增强操作只应应用于训练。
> 推理时应使用确定性的预处理。

---

## 📊 领域特定最佳实践 (Domain-Specific Best Practices)

### 大图小目标场景

**推荐配置：**
```yaml
# configs/rice_detection.yaml
data:
  tile_size: 512          # 根据GPU显存调整
  overlap: 64             # 通常为tile_size的1/8到1/4
  min_target_size: 10     # 过滤太小的目标

model:
  backbone: "resnet50"
  decoder: "unet"

loss:
  type: "combined"
  ce_weight: 0.3
  dice_weight: 0.7        # Dice对小目标更敏感

training:
  batch_size: 8
  learning_rate: 0.001
  epochs: 100

augmentation:
  horizontal_flip: 0.5
  vertical_flip: 0.5
  rotate: 30              # 水稻图像可能需要旋转不变性
  color_jitter: 0.2       # 模拟不同光照条件
```

---

## 🔍 模块复用检查流程 (Module Reuse Check Flow)

当用户请求新功能时，我会按以下流程检查：

```
步骤1：解析用户意图
    └─> 提取关键词（如"数据增强"、"Dice Loss"）

步骤2：搜索现有实现
    └─> 在 src/rice_detection/ 下搜索相关文件
    └─> 读取文件头部注释和README

步骤3：报告发现
    └─> 如果找到：
        "数据增强已在 data/albumentations_transforms.py:42 实现，
         支持Flip、Rotate、ColorJitter。是否需要添加新的增强方式？"
    └─> 如果未找到：
        "未找到现有实现。建议在 src/rice_detection/losses/
         下创建 dice_loss.py"

步骤4：征求用户确认
    └─> 等待用户确认后再实施
```

---

## 📝 我的工作流程 (My Workflow)

### 场景1：用户请求"实现Dice Loss"

```
我会：
1. ✅ 检查 src/rice_detection/losses/ 是否已有实现
2. ✅ 如果没有，提供从第一性原理的解释：
    - 📜 历史背景：Lee Raymond Dice (1945)...
    - 🧮 数学定义：Dice = 2|X∩Y| / (|X|+|Y|)...
    - 🔨 PyTorch实现（带详细注释）
3. ✅ 创建文件：src/rice_detection/losses/dice_loss.py
4. ✅ 更新文档：losses/README.md
5. ✅ 提醒用户：
    - 建议在config中添加 loss.type: "dice"
    - 建议编写单元测试 test_dice_loss.py
    - Dice Loss特别适合你的大图小目标场景
```

---

### 场景2：用户请求"优化训练速度"

```
我会：
1. ✅ 分析当前瓶颈：
    - 数据加载？I/O密集型
    - 模型前向？计算密集型
    - Tile切分？预处理开销

2. ✅ 提供具体建议：
    - 使用 DataLoader(num_workers=4, pin_memory=True)
    - 启用混合精度训练（AMP）
    - 实现Lazy Tile Loading（避免预切分所有图片）
    - 考虑使用DALI数据加载器

3. ✅ 征求用户确认后实施
4. ✅ 记录优化效果到文档
```

---

## 🎯 与 PROJECT_RULES.md 的协作

**我会自动遵守的规则：**
- ✅ 修改前检查项目结构（读取 .claude/rules/project_structure.md）
- ✅ 新增功能前搜索现有实现
- ✅ 修改后更新脚本头部注释
- ✅ 使用配置驱动而非硬编码
- ✅ 提供单元测试建议
- ✅ 提醒新手常见错误

---

## 📌 快速命令参考 (Quick Command Reference)

**当我激活时，你可以直接问：**
- "解释Focal Loss的数学原理"（学习模式）
- "实现Dice Loss"（工程模式）
- "优化Tile切分性能"（优化模式）
- "为什么训练很慢？"（诊断模式）
- "检查我的代码是否符合规范"（审查模式）

---

*技能版本：1.0*
*专业领域：语义分割、大图小目标、PyTorch*
*适配项目：RiceDetection*
