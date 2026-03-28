# Circle Packing Experiment (n=26)

基于 Evolve 框架的圆形填充优化实验，目标是在单位正方形中放置 26 个圆，使其半径之和最大化。

## 问题描述

- **目标**: 在 [0,1]×[0,1] 单位正方形中放置 26 个圆
- **优化目标**: 最大化 Σ(r_i)（所有圆半径之和）
- **约束**: 
  - 所有圆必须在正方形内
  - 圆之间不能重叠
- **基准**: AlphaEvolve (Nature 2025) 达到 2.635

## 实验结构

```
circle_packing/
├── input.md                    # 任务描述
├── seed_code.py                # 初始代码（baseline）
├── init_cognition.py          # 初始化知识库脚本 ⭐
├── eval.sh                     # 评估脚本
├── run.sh                      # 运行脚本（可选）
├── run_experiment.sh           # 实验启动脚本
├── config_phase1.yaml          # Phase 1 配置（探索）
├── config_phase2.yaml          # Phase 2 配置（优化）
├── prompts_phase1/             # Phase 1 提示词
│   ├── researcher.jinja2       # 代码生成
│   └── analyzer.jinja2         # 结果分析
├── prompts_phase2/             # Phase 2 提示词
│   ├── researcher.jinja2       # 优化策略
│   └── analyzer.jinja2         # 结果分析
├── database_data/              # 历史节点数据库
├── cognition_data/             # 知识库（运行 init_cognition.py 后生成）
├── logs/                       # 日志和 WandB
└── steps/                      # 每一步的代码和结果
```

## 两阶段策略

### Phase 1: 探索（Steps 1-50）

**目标**: 2.0 → 2.4

**策略**:
- 探索多种几何模式（同心圆、网格、六边形、螺旋等）
- 尝试不同的构造方法
- 关注创新和多样性
- 高探索系数（UCB1 c=1.8）
- 较高温度（0.7）

**运行**:
```bash
cd experiments/circle_packing
./run_experiment.sh 50 phase1
```

### Phase 2: 优化（Steps 51+）

**目标**: 2.4 → 2.635

**策略**:
- 基于 Phase 1 最佳模式进行数值优化
- 使用 scipy.optimize (SLSQP) 进行约束优化
- 精细调整圆的位置和半径
- 低探索系数（UCB1 c=1.2）
- 较低温度（0.5）

**切换到 Phase 2**:
```bash
# 在 Phase 1 达到稳定（如 50 步后）
./run_experiment.sh 50 phase2
```

## 快速开始

### 1. 准备环境

```bash
# 安装依赖
pip install numpy scipy matplotlib wandb sentence-transformers faiss-cpu pyyaml jinja2
```

### 2. 初始化 Cognition 知识库（推荐）

```bash
cd /inspire/hdd/project/qproject-fundationmodel/public/wxxu/Evolve/experiments/circle_packing
python3 init_cognition.py
```

这会添加圆形填充相关的知识到 cognition 知识库，包括：
- 六边形紧密填充原理
- 边界效应处理
- 数值优化方法（scipy.optimize）
- AlphaEvolve 方法论
- 常见问题和解决方案

**注意**: 如果不初始化 cognition，框架仍可运行，但会缺少领域知识，可能影响进化效率。

### 3. 运行 Phase 1

```bash
./run_experiment.sh 50 phase1
```

### 3. 监控进度

- 控制台会实时显示每一步的 score
- WandB 仪表板: https://wandb.ai/hz-czar-sjtu/evolve-circle-packing
- 本地日志: `logs/evolve.log`

### 4. 切换到 Phase 2

当 Phase 1 的 score 稳定在 2.3-2.4 左右时：

```bash
./run_experiment.sh 50 phase2
```

## 评估指标

- **score**: 半径之和（主要指标，越高越好）
- **valid**: 是否合法（无重叠，在边界内）
- **num_circles**: 圆的数量（应为 26）
- **packing_density**: 填充密度（圆面积之和 / 正方形面积）
- **avg_radius**: 平均半径
- **radii_range**: 半径范围 [min, max]

## 代码要求

生成的代码必须包含：

```python
import numpy as np

def construct_packing():
    """
    Returns:
        centers: np.array (26, 2) - 圆心坐标
        radii: np.array (26,) - 半径
        sum_radii: float - 半径之和
    """
    n = 26
    centers = ...  # 你的构造逻辑
    radii = ...    # 计算最大合法半径
    sum_radii = np.sum(radii)
    return centers, radii, sum_radii
```

## 关键几何洞察

1. **六边形模式**: 无限平面最密填充遵循六边形结构（密度 π/(2√3) ≈ 0.9069）
2. **边界效应**: 正方形边界打破对称性，角落和边缘需要特殊处理
3. **可变半径**: 最优填充通常使用不同大小的圆
4. **分层结构**: 许多最优填充显示出壳层结构
5. **数值优化**: Phase 2 使用 scipy.optimize 可以突破平台期

## Resume 机制

框架支持自动 resume：

```bash
# 中断后继续运行
./run_experiment.sh 20 phase1  # 会从上次中断处继续

# 手动切换配置
./run_experiment.sh 30 phase2  # 使用新配置继续演化
```

数据持久化在：
- `database_data/`: 所有节点的代码、结果、分析
- `cognition_data/`: 知识库（如果添加了外部知识）
- `pipeline_state.json`: Pipeline 状态

## 配置差异

| 配置项 | Phase 1 | Phase 2 |
|--------|---------|---------|
| 目标 | 探索 | 优化 |
| Temperature | 0.7 | 0.5 |
| UCB1 C | 1.8 | 1.2 |
| Retrieval Top-K | 4 | 5 |
| Score Threshold | 0.25 | 0.4 |
| Timeout | 300s | 600s |
| Prompt 策略 | 鼓励新模式 | 鼓励优化 |

## 预期进展

- **Steps 1-10**: 探索基础模式，score 2.0-2.2
- **Steps 11-30**: 找到较好模式，score 2.2-2.35
- **Steps 31-50**: 优化模式，达到 2.35-2.4（Phase 1 plateau）
- **Steps 51-80**: 数值优化，突破 2.4
- **Steps 81-100**: 精细调整，冲击 2.6+

## 参考资料

- AlphaEvolve: "Mathematical discoveries from program search with large language models" (Nature, 2025)
- Packomania: http://www.packomania.com/ (已知最佳圆形填充)
- 最大圆形填充密度: π/(2√3) ≈ 0.9069

## 故障排查

### 评估失败
```bash
# 查看评估日志
cat steps/step_N/eval.log
cat steps/step_N/results.json
```

### 代码错误
- 检查生成的代码: `steps/step_N/code`
- 确保返回正确的元组: `(centers, radii, sum_radii)`
- 验证所有圆在边界内且无重叠

### WandB 问题
```bash
# 离线模式
# 在 config.yaml 中设置: wandb.offline = true
# 之后同步: wandb sync logs/wandb/
```

## Cognition 知识库

### 初始化知识库

在运行实验前，强烈建议初始化 cognition 知识库：

```bash
python3 init_cognition.py
```

这会添加 25+ 条圆形填充相关知识，包括：
- **几何原理**: 六边形紧密填充、边界效应、可变半径策略
- **构造方法**: 分层填充、网格优化、螺旋排列、混合策略
- **数值优化**: scipy.optimize 使用指南、多起点优化、分层优化
- **特定洞察**: n=26 的最优模式、AlphaEvolve 方法论
- **问题解决**: 重叠检测、边界约束、突破平台期

### 知识库的作用

Cognition 知识库会在 Researcher 生成代码时提供上下文：
- 当分析历史节点时，自动检索相关知识
- 提供领域专家级别的指导
- 避免重复探索已知无效的方法
- 加速找到有效的优化策略

### 验证知识库

```bash
# 运行初始化脚本会显示测试检索结果
python3 init_cognition.py

# 输出示例：
# Added 25 knowledge items to cognition
# Total items: 25
# --- Testing retrieval ---
# Search: 'How to use scipy optimization...'
#   [0.856] Constrained optimization with scipy.optimize.minimize...
```

## 下一步

1. **初始化 cognition**: `python3 init_cognition.py`
2. 运行完整的 Phase 1（50 步）
3. 分析最佳模式
4. 切换到 Phase 2 config
5. 运行优化阶段
6. 分析结果，调整策略
7. 可选：启用 LLM Judge 进行代码质量评估

祝实验顺利！冲击 2.635！🎯
