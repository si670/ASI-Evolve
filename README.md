# Evolve Framework

自动化实验进化框架 - 通过 LLM Agent 自动生成、运行、分析实验，并持续优化。

## 项目结构

```
Evolve/
├── config.yaml          # 全局配置文件
├── main.py              # 主入口
├── README.md
│
├── utils/               # 工具组件
│   ├── config.py        # 配置加载
│   ├── llm.py           # LLM 调用封装
│   ├── logger.py        # 日志 & WandB
│   ├── prompt.py        # Prompt 模板管理
│   └── structures.py    # 数据结构定义
│
├── database/            # 实验数据库
│   ├── database.py      # 主数据库类
│   ├── algorithms.py    # 采样算法 (UCB1, Random, Greedy)
│   ├── faiss_index.py   # FAISS 向量索引
│   └── embedding.py     # Embedding 服务
│
├── cognition/           # 知识库
│   └── cognition.py     # RAG 检索
│
├── pipeline/            # 实验 Pipeline
│   ├── main.py          # Pipeline 主逻辑
│   ├── base.py          # Agent 基类
│   ├── researcher/      # 代码生成 Agent
│   ├── engineer/        # 实验运行 Agent
│   ├── analyzer/        # 结果分析 Agent
│   └── manager/         # Meta Prompt Agent
│
└── experiments/         # 实验目录
    └── {experiment_name}/
        ├── input.md             # 任务描述
        ├── eval_criteria.md     # 评估标准（可选）
        ├── prompts/             # Prompt 模板
        ├── database_data/       # 数据库存储
        ├── cognition_data/      # 知识库存储
        ├── logs/                # 日志
        └── step_{n}/            # 每轮实验数据
```

## 快速开始

### 1. 安装依赖

```bash
pip install openai pyyaml jinja2 numpy faiss-cpu sentence-transformers
# 可选: wandb 支持
pip install wandb
```

### 2. 下载模型（可选，推荐）

为了支持离线运行，建议预先下载所需的 embedding 模型：

```bash
# 设置 HuggingFace 缓存目录（如果需要）
export HF_HOME="/path/to/your/cache"

# 运行下载脚本
./download_models.sh

# 或者直接用 Python
python3 download_models.py
```

下载完成后，可以启用离线模式：

```bash
# 每次运行前设置环境变量
source .env

# 或手动设置
export HF_HOME="/path/to/cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

**所需模型：**
- `sentence-transformers/all-MiniLM-L6-v2` (用于文本 embedding，约 80MB)

### 3. 配置 API

编辑 `config.yaml`，设置 API 密钥：

```yaml
api:
  provider: "openai"
  base_url: "https://api.openai.com/v1"
  api_key: "${OPENAI_API_KEY}"  # 从环境变量读取
  model: "gpt-4"
```

或设置环境变量：

```bash
export OPENAI_API_KEY="your-api-key"
```

### 4. 创建实验

```bash
# 创建实验目录
mkdir -p experiments/my_experiment

# 创建任务描述
cat > experiments/my_experiment/input.md << 'EOF'
# 任务描述

实现一个高效的排序算法...
EOF
```

### 5. 运行

```bash
# 基本运行
python main.py --experiment my_experiment --steps 10

# 指定运行/评估脚本
python main.py --experiment my_experiment \
    --run-script experiments/my_experiment/run.sh \
    --eval-script experiments/my_experiment/eval.sh
```

## Resume 功能

框架支持自动恢复中断的实验。当使用相同的 `experiment_name` 重新运行时：

- **Database**: 自动从 `database_data/nodes.json` 加载所有已完成的节点
- **Cognition**: 自动从 `cognition_data/cognition.json` 加载知识库
- **FAISS 索引**: 自动从本地文件恢复向量索引
- **Step 计数器**: 从 `pipeline_state.json` 恢复，或根据数据库推断
- **Manager 状态**: 如果 prompts 已生成，不会重复执行

```bash
# 第一次运行（运行 10 步后中断）
python main.py --experiment my_experiment --steps 100
# 假设运行到 step 35 时中断...

# 重新运行（自动从 step 36 继续）
python main.py --experiment my_experiment --steps 100
# 输出: Resuming experiment 'my_experiment' from step 35 (database: 35 nodes, ...)
```

**注意**: 如果某个 step 运行到一半中断（比如 Researcher 完成但 Engineer 还没开始），该 step 不会被保存，重新运行时会从上一个完整的 step 继续。

## 配置说明

### config.yaml 主要配置项

```yaml
# 实验名称
experiment_name: "default"

# API 配置
api:
  model: "gpt-4"
  temperature: 0.7
  max_tokens: 4096

# Pipeline 配置
pipeline:
  agents:
    manager: false     # 启用 Meta Prompter
    researcher: true   # 代码生成
    engineer: true     # 运行实验
    analyzer: true     # 分析结果
  
  # Researcher 配置（新增）
  researcher:
    diff_based_evolution: true  # 增量更新模式（默认开启）
    diff_pattern: "<<<<<<< SEARCH\\n(.*?)=======\\n(.*?)>>>>>>> REPLACE"
    max_code_length: 10000

# 数据库配置
database:
  sampling:
    algorithm: "ucb1"  # ucb1 / random / greedy / island
    ucb1_c: 1.414      # UCB1 探索参数

# 知识库配置
cognition:
  retrieval:
    top_k: 5
    score_threshold: 0.5
```

### 增量更新模式 (Diff-Based Evolution)

框架支持两种代码生成模式:

1. **增量更新 (diff_based_evolution: true)** - 默认模式
   - 基于历史节点的代码进行增量修改
   - 使用 SEARCH/REPLACE 格式指定修改
   - 更精确、成本更低、适合大型代码库

2. **完整重写 (diff_based_evolution: false)**
   - 每次完整生成新代码
   - 适合探索全新方法或简短代码

详细说明请参考: [docs/DIFF_EVOLUTION.md](docs/DIFF_EVOLUTION.md)

## 数据结构

### Node (实验节点)

```python
@dataclass
class Node:
    name: str           # 节点名称
    created_at: str     # 创建时间
    parent: List[int]   # 父节点 ID 列表
    motivation: str     # 动机分析
    code: str           # 代码
    results: Dict       # 实验结果
    analysis: str       # 分析总结
    meta_info: Dict     # 元信息
    score: float        # 评分
```

## API 接口

### Database

```python
from database import Database

db = Database(storage_dir="path/to/db")

# 采样
nodes = db.sample(n=3, algorithm="ucb1")

# 添加
node_id = db.add(node)

# 删除
db.remove(node_id)

# 重置
db.reset()
```

### Cognition

```python
from cognition import Cognition

cog = Cognition(storage_dir="path/to/cog")

# 添加知识
item_id = cog.add(CognitionItem(content="..."))

# RAG 检索
items = cog.search("query text", top_k=5)

# 删除
cog.remove(item_id)

# 重置
cog.reset()
```

### Pipeline

```python
from pipeline import Pipeline

# 初始化
pipeline = Pipeline(experiment_name="my_exp")

# 运行单步
node = pipeline.run_step(task_description="...")

# 运行多步
pipeline.run(max_steps=10)

# 获取最佳结果
best = pipeline.get_best_node()
```

## 自定义 Prompt

如果不使用 Manager 自动生成 prompt，需要手动创建：

```
experiments/my_experiment/prompts/
├── researcher.jinja2   # Researcher Agent 的 prompt
└── analyzer.jinja2     # Analyzer Agent 的 prompt
```

模板使用 Jinja2 语法，可用变量：
- `task_description`: 任务描述
- `context_nodes`: 历史节点列表
- `cognition_items`: 相关知识列表
- `code`: 代码（仅 analyzer）
- `results`: 实验结果（仅 analyzer）

## 采样算法

### UCB1 (默认)

平衡探索与利用：
```
UCB1 = normalized_score + c * sqrt(ln(N) / n_i)
```

- `normalized_score`: 归一化的节点分数
- `c`: 探索参数（默认 1.414）
- `N`: 总访问次数
- `n_i`: 该节点访问次数

### Random

随机采样。

### Greedy

按分数降序选择。

## License

MIT
