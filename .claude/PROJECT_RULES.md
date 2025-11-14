# 项目规范规则 (Project Rules for RiceDetection)

> 本文件定义了 RiceDetection 项目的工程规范和架构约束。
> 所有代码修改必须遵守这些规则，以保持项目的模块化和可维护性。

---

## 🎯 项目背景 (Project Context)

### 项目定义
- **任务类型：** 像素级图像分类（语义分割，Semantic Segmentation）
- **数据特点：** 大图片、小目标、纹理细节丰富（类似医学图像）
- **技术栈：** PyTorch
- **目标：** 完整的端到端pipeline（训练→导出→部署→优化）

### 核心挑战
1. **大图小目标：** 需要Tile切分策略
2. **类别不平衡：** 小目标占比少，背景类占主导
3. **端到端部署：** 训练代码需考虑部署一致性
4. **实验可复现：** 科研项目要求严格的版本控制

---

## 📐 架构公理 (Architectural Axioms)

### 公理1：单一职责原则 (Single Responsibility Principle, SRP)
```
定义：每个模块只负责一个明确定义的功能
推论：修改某个功能时，只需要改一个模块
反例：在 Trainer 中混入数据加载逻辑
```

**实施细则：**
- 预处理 ←→ 模型 ←→ 后处理 ←→ 服务：各层独立
- 各层不混在同一个文件夹中

**示例：**
```python
# ✅ 正确：职责分离
src/rice_detection/data/albumentations_transforms.py  # 只负责数据增强
src/rice_detection/training/trainer.py                # 只负责训练循环

# ❌ 错误：职责混乱
src/rice_detection/training/trainer.py  # 里面包含数据加载、增强、训练逻辑
```

---

### 公理2：接口隔离原则 (Interface Segregation Principle, ISP)
```
定义：模块间通过明确的接口交互，而非直接耦合
推论：可以替换实现而不影响其他模块
示例：能替换 augmentor 实现而不修改 Trainer 代码
```

**实施细则：**
- 传递数据结构：`torch.Tensor` / `dict` / `torch.utils.data.Dataset`
- 避免传递模块内部对象或全局变量
- 减少使用 dict 的任意键（使用 TypedDict 或 dataclass）

**示例：**
```python
# ✅ 正确：通过接口注入
class Trainer:
    def __init__(self, model, optimizer, dataloader, augmentor):
        self.augmentor = augmentor  # 接口，可替换实现

# ❌ 错误：直接导入具体实现
from data.albumentations_transforms import AlbumentationsAugmentor
class Trainer:
    def __init__(self, model, optimizer, dataloader):
        self.augmentor = AlbumentationsAugmentor()  # 硬编码，无法替换
```

---

### 公理3：配置驱动原则 (Configuration-Driven Development)
```
定义：所有可变的参数都通过配置注入，而非硬编码
推论：同一份代码可以复现不同的实验
工具：使用 YAML 文件 + Hydra（可选）
```

**实施细则：**
- 输入输出路径、超参数、device 等参数，都要通过 config 注入
- 可配置的硬件适配（`device` 参数）
- 不要在模块中硬编码任何路径或数值

---

## 📁 项目结构规范 (Project Structure)

### 强制约束
1. **项目结构文档：** `.claude/rules/project_structure.md`
2. **根据项目结构进行修改**，不破坏原本结构
3. **修改后立即更新文档**：脚本头部注释 + 模块README

### 标准目录结构
```
RiceDetection/
├── src/
│   └── rice_detection/
│       ├── data/              # 数据处理模块
│       ├── models/            # 模型定义
│       ├── training/          # 训练逻辑
│       ├── deployment/        # 部署相关
│       ├── io/                # 特殊I/O操作集中处理
│       └── utils/             # 通用工具
├── scripts/                   # 启动脚本
│   ├── train.sh
│   ├── export.sh
│   └── serve.sh
├── configs/                   # 配置文件
│   └── experiments/
│       └── exp001.yaml
├── checkpoints/               # 模型checkpoint
│   └── [实验名]/
│       └── [config文件名]/
├── outputs/                   # 输出结果
│   └── expX/
│       └── exports/
├── tests/                     # 单元测试
└── docs/                      # 文档
```

---

## 🐍 Python编码规范 (Python Coding Standards)

### 字符编码规范 (Encoding Rules)
```python
# 所有文件使用 UTF-8 编码
# -*- coding: utf-8 -*-

# ✅ 正确：英文变量名和注释
tile_size = 512  # Tile size in pixels

# ❌ 错误：中文变量名
瓦片大小 = 512  # 避免使用中文变量名
```

**规则：**
- 所有Python文件使用 UTF-8 编码
- 变量名、函数名、注释使用英文
- 避免在代码输出中使用emoji/特殊Unicode字符
- Console输出使用ASCII安全字符

---

### 文件命名规范 (File Naming Conventions)
```
✅ 正确：
    albumentations_transforms.py
    tile_dataset.py
    unet_model.py

❌ 错误：
    AlbumentationsTransforms.py    # 避免驼峰命名
    tile-dataset.py                # 避免短横线
    瓦片数据集.py                   # 避免中文文件名
```

**规则：**
- Python文件使用 `snake_case` 命名
- 只使用ASCII字符
- 避免空格和特殊字符
- README文档可以使用中文内容

---

### 脚本头部注释模板 (Script Header Template)

**强制要求：** 所有Python脚本必须包含以下头部注释

```python
"""
[脚本名称] - Brief description in English

Recent Updates:
  - [2025-01-14] 重构: 统一使用 Albumentations 增强
  - [2025-01-12] 新增: Tile 边界处理（大图/狭长/小图）
  - [2025-01-10] 修复: 数据集划分泄漏问题（按原图 ID 划分）
  - [2025-01-08] 优化: Lazy tile loading（161GB → 5MB）

Key Features:
  - 核心功能 1（用英文或中文描述）
  - 核心功能 2

Usage:
  from rice_detection.data.tile_dataset import TileDataset
  dataset = TileDataset(config)

Configuration:
  - tile_size: int, default 512
  - overlap: int, default 64
  - augmentation: bool, default True

Dependencies:
  - torch >= 2.0.0
  - albumentations >= 1.3.0
"""
```

---

## 🔧 模块化编程要点 (Modular Programming Guidelines)

### 1. 可替换实现和复用 (Replaceable Implementation & Reuse)

**核心原则：** 新增功能前，先检查是否已有实现。

**查找顺序：**
```
步骤1：在 src/rice_detection/ 下搜索相关功能
步骤2：读取模块的 README.md 和代码注释
步骤3：如果找到，复用；如果没有，在恰当路径下新建
```

**示例：**
```python
# 用户请求："实现数据增强"

# ✅ 正确流程：
# 1. 搜索 src/rice_detection/data/
# 2. 发现 albumentations_transforms.py 已存在
# 3. 回复："数据增强已在 albumentations_transforms.py:15 实现，
#          支持 Flip、Rotate、ColorJitter。是否需要添加新的增强方式？"

# ❌ 错误流程：
# 直接创建新文件 augmentation.py，导致功能重复
```

---

### 2. 轻量启动器模式 (Lightweight Launcher Pattern)

**原则：** 每个模块的 `main.py` 只负责：
1. 解析配置
2. 构造对象
3. 调用核心逻辑

**示例：**
```python
# training/main.py（轻量启动器）
def main():
    # 1. 解析配置
    config = load_config("configs/exp001.yaml")

    # 2. 构造对象
    model = build_model(config.model)
    optimizer = build_optimizer(model, config.optimizer)
    dataloader = build_dataloader(config.data)
    trainer = Trainer(model, optimizer, dataloader, config.training)

    # 3. 调用核心逻辑
    trainer.train()
```

**禁止：** 在 `main.py` 中实现复杂逻辑

---

### 3. 最小化外部依赖 (Minimize External Dependencies)

**原则：** 接口不直接导入具体实现类，而是通过类型注入或工厂模式。

**示例：**
```python
# ✅ 正确：依赖注入
class Trainer:
    def __init__(
        self,
        model: nn.Module,           # 类型注解，不依赖具体实现
        optimizer: torch.optim.Optimizer,
        dataloader: DataLoader
    ):
        self.model = model

# ❌ 错误：直接导入
from models.unet import UNet
class Trainer:
    def __init__(self):
        self.model = UNet()  # 硬编码，无法替换
```

---

### 4. 无隐藏副作用 (No Hidden Side Effects)

**原则：** 函数/类调用不会以不可见方式修改磁盘/全局状态，除非函数名/文档明确说明。

**示例：**
```python
# ✅ 正确：明确说明副作用
def save_checkpoint(model, path):
    """Save model checkpoint to disk.

    Side Effect: Writes file to disk at `path`.
    """
    torch.save(model.state_dict(), path)

# ❌ 错误：隐藏的副作用
def evaluate_model(model, dataloader):
    """Evaluate model performance."""
    metrics = compute_metrics(model, dataloader)
    torch.save(metrics, "results.pth")  # 隐藏的磁盘写入！
    return metrics
```

---

### 5. 单元测试覆盖 (Unit Test Coverage)

**规则：**
- 单元测试覆盖核心边界
- 每个模块能独立 mock 测试
- 测试文件位置：`src/rice_detection/[模块名]/test/`

**示例：**
```
src/rice_detection/data/
├── tile_dataset.py
├── albumentations_transforms.py
└── test/
    ├── test_tile_dataset.py
    └── test_albumentations_transforms.py
```

---

### 6. 统一异常处理 (Unified Exception Handling)

**原则：**
- 底层模块：抛出异常（`raise ValueError`）
- 顶层CLI：捕获异常 + 记录日志

**示例：**
```python
# ✅ 正确：底层抛错
def load_image(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    return cv2.imread(path)

# ✅ 正确：顶层捕获
def main():
    try:
        image = load_image(config.image_path)
    except FileNotFoundError as e:
        logger.error(f"Failed to load image: {e}")
        sys.exit(1)

# ❌ 错误：底层吞掉异常
def load_image(path):
    if not os.path.exists(path):
        print(f"Warning: {path} not found")  # 隐藏问题！
        return None
```

---

### 7. 集中化特殊 I/O 操作 (Centralized Special I/O)

**原则：** 所有特殊的磁盘操作集中在 `src/rice_detection/io/`

**什么是"特殊I/O"？**
- 大文件分块读取
- 特殊格式解析（如医学图像格式）
- 异步I/O操作
- 内存映射文件

**普通I/O（如简单的 `torch.save`）不需要抽象。**

**目录结构：**
```
src/rice_detection/io/
├── large_image_reader.py   # 大图分块读取
├── tiff_reader.py          # 特殊格式读取
└── lazy_loader.py          # 延迟加载
```

---

## 🔄 工作流程规范 (Workflow Standards)

### 训练线 (Training Pipeline)
```bash
scripts/train.sh → training/main.py
```

**流程：**
1. `main.py` 解析配置文件
2. 构造 `dataloader, augmentor, model, optimizer`
3. 创建 `Trainer` 对象
4. 调用 `Trainer.train()`
5. 保存 checkpoint 到：`checkpoints/[实验名]/[config文件名]/`

**Checkpoint保存规范：**
- 路径：`checkpoints/exp001/baseline/epoch_10.pth`
- 同时保存 `config.yaml` 副本
- 记录 config hash

---

### 导出线 (Export Pipeline)
```bash
scripts/export.sh → deployment/export.py
```

**流程：**
1. `export.py` 读取 checkpoint 和 config
2. 恢复模型状态
3. 调用 `export_to_onnx(model, sample_input)`
4. 调用优化器/adapter（TensorRT、OpenVINO等）
5. 结果保存到：`outputs/expX/exports/`
6. 更新 artifact registry

---

### 推理/服务线 (Inference/Serving Pipeline)
```bash
deployment/serve.py
```

**流程：**
1. 根据 config 决定加载 engine 或 checkpoint
2. 通过 `IPredictor` 接口调用推理
3. 不直接触碰训练逻辑
4. 不修改 checkpoint 或模型权重

---

## 🔬 实验可复现性 (Experiment Reproducibility)

### 强制要求

**规则1：随机种子配置化**
```yaml
# configs/exp001.yaml
seed: 42
deterministic: true
```

**规则2：配置文件包含所有超参数**
```yaml
# 禁止：在代码中硬编码
learning_rate = 0.001  # ❌

# 正确：在config中定义
learning_rate: 0.001   # ✅
```

**规则3：记录 config hash**
```python
import hashlib
config_hash = hashlib.md5(str(config).encode()).hexdigest()
torch.save({
    'model_state_dict': model.state_dict(),
    'config_hash': config_hash
}, checkpoint_path)
```

**规则4：Checkpoint + Config 同步保存**
```
checkpoints/exp001/baseline/
├── epoch_10.pth        # 模型权重
└── config.yaml         # 对应的配置文件副本
```

---

## 🖼️ 计算机视觉特定规范 (CV-Specific Standards)

### Tile 切分一致性 (Tile Splitting Consistency)

**公理：** 训练和推理必须使用相同的Tile切分策略。

**规则：**
- `tile_size` 和 `overlap` 必须在训练和推理中相同
- 边界处理策略必须一致
- 特殊预处理（如针对Tile的normalization）包含在原始预处理中

**实施：**
```python
# 在config中统一定义
tile_config:
  tile_size: 512
  overlap: 64
  boundary_handling: "pad"  # or "crop", "resize"
```

---

### 数据增强规范 (Data Augmentation Standards)

**规则：**
- 数据增强**仅应用于训练**
- 推理使用原始预处理
- 针对Tile的特殊预处理包含在原始预处理中

**示例：**
```python
# 训练时
train_transform = Compose([
    Resize(512, 512),              # 原始预处理
    Normalize(mean, std),          # 原始预处理
    HorizontalFlip(p=0.5),         # 数据增强
    Rotate(limit=30, p=0.5),       # 数据增强
])

# 推理时
val_transform = Compose([
    Resize(512, 512),              # 只保留原始预处理
    Normalize(mean, std),
])
```

---

### 评估指标规范 (Evaluation Metrics Standards)

**强制包含的指标：**
1. **IoU (Intersection over Union)** - 语义分割的标准指标
2. **Pixel Accuracy** - 像素级准确率
3. **Class-wise Metrics** - 每个类别的单独指标（因为类别不平衡）
4. **Dice Coefficient** - 医学图像常用指标

**可选指标：**
- Precision / Recall / F1
- Boundary IoU（边界质量）

---

## 🛠️ Debug与重构指南 (Debug & Refactoring Guide)

### 识别"脏点" (Identifying "Dirty Spots")

**信号：**
- 模块职责不清晰（如 `trainer.py` 中有读图代码）
- 函数过长（超过50行）
- 重复代码出现3次以上
- 测试无法mock

---

### 重构步骤 (Refactoring Steps)

**步骤1：找到"脏点"**
```python
# 示例：training/trainer.py 里有读图代码
class Trainer:
    def train(self):
        for batch in dataloader:
            image = cv2.imread(batch['path'])  # 脏点！读图逻辑不应在Trainer
            ...
```

**步骤2：抽离功能**
```python
# 创建 src/rice_detection/io/image_reader.py
def read_image(path):
    """Read image from disk."""
    return cv2.imread(path)

# 在 Trainer 中用接口替换
class Trainer:
    def train(self):
        for batch in dataloader:
            image = self.image_reader(batch['path'])  # 通过接口调用
            ...
```

**步骤3：编写测试**
```python
# src/rice_detection/io/test/test_image_reader.py
def test_read_image():
    path = "test_image.png"
    image = read_image(path)
    assert image is not None
```

**步骤4：持续集成**
- 在CI中添加测试，确保不回退

---

## 📦 Artifact 管理 (Artifact Management)

**规则：**
- Checkpoint、ONNX、TensorRT engine 按版本存放
- 记录创建时的 config hash
- 使用 artifact registry（如 MLflow、Weights & Biases）

**目录结构：**
```
checkpoints/
└── exp001/
    ├── baseline/
    │   ├── epoch_10.pth
    │   └── config.yaml
    └── improved/
        ├── epoch_15.pth
        └── config.yaml

outputs/
└── exp001/
    └── exports/
        ├── model.onnx
        ├── model_fp16.engine
        └── metadata.json
```

---

## 📝 文档更新规范 (Documentation Update Standards)

### 强制要求

**规则1：修改后立即更新文档**
- 修改模块功能 → 更新模块README
- 添加新脚本 → 更新脚本头部注释
- 修改项目结构 → 更新 `.claude/rules/project_structure.md`

**规则2：文档内容要求**
- 只包含技术性总结和描述
- 避免冗余和重复
- 使用代码示例而非长篇叙述

---

## 🎯 与学习风格规则的交互 (Interaction with Learning Style Rules)

### 触发逻辑

**当用户的请求属于以下类型时，切换到"学习模式"（使用 CLAUDE_RULES.md）：**
1. "什么是XXX？"（概念解释）
2. "为什么XXX？"（原理探究）
3. "XXX的数学原理"（公理推导）
4. "XXX的历史背景"（历史起源）

**示例：**
```
用户："什么是 Focal Loss？"
→ 切换到学习模式
→ 回答包含：历史起源 + 公理定义 + 推导过程 + Python实现
```

**当用户的请求属于以下类型时，使用"工程模式"（使用 PROJECT_RULES.md）：**
1. "实现XXX功能"（编码任务）
2. "修改XXX模块"（代码修改）
3. "优化XXX性能"（工程优化）
4. "部署XXX"（部署任务）

**示例：**
```
用户："实现数据增强"
→ 使用工程模式
→ 检查现有实现 → 复用或新建 → 遵守项目结构规范
```

---

## ⚠️ Claude 的行为约束 (Behavioral Constraints for Claude)

### Claude 必须做的 (MUST DO)

1. **✅ 修改代码前，先检查项目结构**
   - 读取 `.claude/rules/project_structure.md`
   - 确认修改位置符合规范

2. **✅ 新增功能前，先搜索现有实现**
   - 在 `src/rice_detection/` 下搜索相关文件
   - 询问用户是否复用

3. **✅ 修改文件后，立即更新文档**
   - 更新脚本头部注释的 "Recent Updates"
   - 更新模块 README.md

4. **✅ 提交前，提醒用户检查清单**
   ```
   修改清单：
   - [x] 更新了 data/tile_dataset.py
   - [x] 更新了脚本头部注释
   - [x] 更新了 data/README.md
   - [ ] 建议：添加单元测试 test_tile_dataset.py
   ```

5. **✅ 识别新手可能的错误，主动提醒**
   - 例如：忘记设置随机种子
   - 例如：训练/推理的Tile切分不一致
   - 例如：硬编码路径而非使用config

---

### Claude 禁止做的 (MUST NOT DO)

1. **❌ 破坏项目结构**
   - 不要在错误的目录创建文件
   - 不要随意修改目录结构

2. **❌ 重复实现已有功能**
   - 先搜索，再实现

3. **❌ 硬编码参数**
   - 所有参数必须通过config注入

4. **❌ 在底层吞掉异常**
   - 底层抛错，顶层捕获

5. **❌ 创建隐藏副作用**
   - 如果函数有副作用，必须在文档中明确说明

---

## 📌 快速参考卡 (Quick Reference Card)

**项目结构：** `.claude/rules/project_structure.md`
**学习风格：** `CLAUDE_RULES.md`
**工程规范：** `PROJECT_RULES.md`（本文件）

**新增功能流程：**
```
1. 搜索现有实现 (src/rice_detection/)
2. 如果存在 → 复用
3. 如果不存在 → 确定合适的路径
4. 编写代码（遵守编码规范）
5. 更新文档（脚本注释 + README）
6. 编写测试（可选但推荐）
```

**修改代码流程：**
```
1. 读取文件
2. 理解当前实现
3. 修改代码
4. 更新 "Recent Updates" 部分
5. 更新 README（如果功能变化）
6. 提醒用户检查清单
```

---

*最后更新：2025-01-14*
*版本：1.0*
*适用项目：RiceDetection（像素级图像分类/语义分割）*
