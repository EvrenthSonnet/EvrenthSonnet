# Hooks系统说明 (Hooks System Guide)

> Hooks是Claude Code的"事件监听器"，在特定时机自动执行，用于约束和引导Claude的行为。

---

## 🧮 Hooks的本质 (The Essence of Hooks)

### 公理定义
```
定义：Hook是一个在特定事件发生时自动触发的函数
事件类型：
  - UserPromptSubmit: 用户提交提示词前
  - PostToolUse: Claude使用工具后
  - PreEdit: 修改文件前
  - PostEdit: 修改文件后
```

### 为什么需要Hooks？

**问题：** Claude的"会话失忆症"
```
场景：新session开始
Claude状态：忘记所有项目规范和上下文
结果：可能破坏项目结构、重复实现功能
```

**解决：** Hooks作为"自动提醒系统"
```
UserPromptSubmit Hook → 分析用户意图 → 自动加载相关上下文
PostToolUse Hook → 检查操作合规性 → 提醒违规行为
```

---

## 📋 核心Hooks清单

### 1. user-prompt-submit.md（用户提示词提交钩子）

**触发时机：** 用户提交提示词之前

**功能：**
```
1. 分析用户意图（关键词提取）
2. 检查打开的文件上下文
3. 匹配相关技能（skill-rules.json）
4. 自动建议或激活技能
5. 注入项目规范提醒
```

**伪代码逻辑：**
```python
def on_user_prompt_submit(user_input, open_files):
    # 1. 提取关键词
    keywords = extract_keywords(user_input)

    # 2. 匹配技能
    if "Dice Loss" in keywords or "损失函数" in keywords:
        suggest_skill("cv-semantic-segmentation")

    # 3. 检查文件上下文
    if any("src/rice_detection/models/" in f for f in open_files):
        remind_rules("PROJECT_RULES.md#模型设计规范")

    # 4. 提醒项目结构
    if "新建" in user_input or "create" in user_input:
        remind("请先检查 src/rice_detection/ 下是否已有类似功能")
```

---

### 2. post-tool-use.md（工具使用后钩子）

**触发时机：** Claude使用任何工具后

**功能：**
```
1. 追踪工具使用模式
2. 检查操作合规性
3. 自动更新文档提醒
4. 违规操作警告
```

**伪代码逻辑：**
```python
def post_tool_use(tool_name, tool_args, result):
    # 1. 文件修改检查
    if tool_name == "Edit" or tool_name == "Write":
        file_path = tool_args['file_path']

        # 检查是否符合项目结构
        if not is_valid_path(file_path):
            warn("⚠️ 文件路径不符合项目结构规范！")

        # 提醒更新文档
        if not has_updated_header_comment(file_path):
            remind("请更新脚本头部注释的 'Recent Updates' 部分")

    # 2. 新建文件检查
    if tool_name == "Write" and "src/rice_detection" in file_path:
        # 检查是否重复实现
        similar_files = search_similar_functionality(file_path)
        if similar_files:
            warn(f"发现类似功能文件：{similar_files}，是否需要复用？")

    # 3. 配置修改检查
    if file_path.endswith(".yaml"):
        # 检查是否包含必要字段
        if "seed" not in result:
            warn("⚠️ 配置文件缺少随机种子设置！")
```

---

### 3. pre-edit.md（文件修改前钩子）

**触发时机：** Claude准备修改文件之前

**功能：**
```
1. 读取文件当前状态
2. 分析修改范围
3. 检查是否需要备份
4. 提醒相关文档更新
```

---

### 4. structure-validator.md（项目结构验证钩子）

**触发时机：** 任何文件操作前

**功能：**
```
1. 验证目标路径是否符合项目结构
2. 检查模块依赖关系
3. 确保文件命名规范
```

---

## 🔧 Hooks配置文件

### skill-rules.json（技能激活规则）

```json
{
  "cv-semantic-segmentation": {
    "auto_activate": true,
    "triggers": {
      "keywords": [
        "语义分割", "Semantic Segmentation",
        "IoU", "Dice", "Focal Loss",
        "UNet", "DeepLab",
        "Tile", "大图小目标"
      ],
      "file_patterns": [
        "src/rice_detection/models/**/*.py",
        "src/rice_detection/data/**/*.py",
        "src/rice_detection/losses/**/*.py"
      ],
      "context_conditions": [
        {
          "type": "user_mentions",
          "pattern": "(实现|优化|修改).*(模型|损失|数据)"
        }
      ]
    },
    "priority": "high"
  },

  "first-principles": {
    "auto_activate": true,
    "triggers": {
      "keywords": [
        "什么是", "为什么", "原理",
        "数学", "推导", "历史"
      ],
      "patterns": [
        "^(什么|为什么|如何理解|解释一下)",
        "the mathematical (principle|derivation|proof)"
      ]
    },
    "priority": "high",
    "combine_with": "CLAUDE_RULES.md"
  },

  "project-standards": {
    "auto_activate": true,
    "triggers": {
      "keywords": [
        "实现", "修改", "新增", "重构",
        "implement", "modify", "add", "refactor"
      ],
      "file_patterns": [
        "src/rice_detection/**/*.py"
      ]
    },
    "priority": "high",
    "combine_with": "PROJECT_RULES.md"
  }
}
```

---

## 🚀 实施Hooks的步骤

### 步骤1：创建Hooks文件
```bash
RiceDetection/.claude/hooks/
├── user-prompt-submit.md      # 核心hook
├── post-tool-use.md            # 核心hook
├── pre-edit.md                 # 可选
└── structure-validator.md      # 可选
```

### 步骤2：配置skill-rules.json
```bash
RiceDetection/.claude/
└── skill-rules.json            # 定义技能激活规则
```

### 步骤3：编写Hook逻辑

**示例：user-prompt-submit.md**
```markdown
# User Prompt Submit Hook

When the user submits a prompt, analyze and enhance it:

## Analysis Steps
1. Extract keywords from user input
2. Check open files context
3. Match relevant skills from skill-rules.json
4. Inject project-specific reminders

## Actions

### If keywords match "semantic segmentation" domain:
- Activate: cv-semantic-segmentation skill
- Remind: Check existing implementations first
- Load: PROJECT_RULES.md#计算机视觉特定规范

### If user asks "what is" or "why":
- Activate: first-principles mode (CLAUDE_RULES.md)
- Response format: 历史起源 → 公理定义 → 推导 → 应用

### If user requests code implementation:
- Remind: Read PROJECT_RULES.md#模块化编程要点
- Check: Search src/rice_detection/ for existing functionality
- Validate: Ensure path matches project structure

### If user modifies configs/*.yaml:
- Remind: Include seed, all hyperparameters
- Validate: Config schema compliance
- Suggest: Save config hash with checkpoint
```

---

## 💡 Hooks的高级用法

### 组合Hooks

```
场景：用户说"实现Focal Loss"

Trigger Chain:
1. user-prompt-submit:
    - 关键词"Focal Loss" → 激活 cv-semantic-segmentation skill
    - 关键词"实现" → 加载 PROJECT_RULES.md

2. 我的行为：
    - 先解释Focal Loss原理（遵循CLAUDE_RULES.md）
    - 然后检查 src/rice_detection/losses/ 是否已有实现
    - 如果没有，按PROJECT_RULES.md规范创建文件
    - 更新文档（post-tool-use提醒）
```

### 条件激活

```json
{
  "detailed-explanation": {
    "auto_activate": false,  // 默认不激活
    "manual_trigger": "/explain",
    "conditions": {
      "user_expertise": "beginner",  // 只对新手激活
      "topic_complexity": "high"     // 只对复杂话题激活
    }
  }
}
```

---

## 📊 Hooks效果评估

### 衡量指标

**1. 结构完整性 (Structural Integrity)**
```
指标：项目结构违规次数
目标：0次违规

监控：
- 文件创建在错误位置的次数
- 硬编码参数的次数
- 重复实现功能的次数
```

**2. 文档同步率 (Documentation Sync Rate)**
```
指标：代码修改后文档更新的比例
目标：100%

监控：
- 修改代码后未更新头部注释的次数
- 新增功能后未更新README的次数
```

**3. 知识复用率 (Knowledge Reuse Rate)**
```
指标：复用现有功能 vs 重复实现的比例
目标：>80%复用

监控：
- 使用现有模块的次数
- 新建重复功能的次数
```

---

## 🅿️ 停放区问题

### Q: Hooks是如何实现的底层机制？

🅿️ **[停放区标记]**
- **类型：** 实现细节
- **说明：** Claude Code的Hooks基于"事件驱动架构"（Event-Driven Architecture）。
  每个工具调用前后都会触发事件，Hooks订阅这些事件。
  深入理解需要学习事件驱动编程和观察者模式。
  我们先接受"Hooks会在特定时机自动执行"这个行为，掌握使用方法后可以深入研究。

### Q: 为什么Hooks文件用Markdown而非Python？

🅿️ **[停放区标记]**
- **类型：** 设计选择
- **说明：** Claude Code使用自然语言（Markdown）定义Hooks逻辑，
  因为Claude本身是语言模型，理解自然语言比执行代码更自然。
  这是"提示工程"（Prompt Engineering）的应用。

---

## 📌 快速参考

**创建第一个Hook：**
```bash
1. 创建 .claude/hooks/user-prompt-submit.md
2. 定义触发条件和行为
3. 测试：问Claude一个问题，观察是否自动激活技能
```

**调试Hooks：**
```bash
# 在Hook文件中添加调试输出
Echo: "🔍 [DEBUG] User-Prompt-Submit Hook triggered"
Echo: "📝 Keywords detected: {keywords}"
Echo: "🎯 Activating skills: {skills}"
```

---

*文档版本：1.0*
*适用于：Claude Code Hooks System*
*参考：diet103/claude-code-infrastructure-showcase*
