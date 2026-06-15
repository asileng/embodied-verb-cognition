# 实验框架使用说明

## 项目结构

```
framework/
├── config.py                    # 中文版配置文件
├── config_en.py                 # 英文版配置文件
├── run_experiment.py            # 主执行脚本（支持vllm、断点续传）
├── run_all_experiments.py       # 母脚本（批量运行实验）
├── screen_data.py               # 数据完整性筛查脚本
├── prompt_templates/            # 提示词模板
│   ├── task1_zh.txt            # 任务1中文提示词
│   ├── task1_en.txt            # 任务1英文提示词
│   ├── task2_zh.txt            # 任务2中文提示词
│   └── task2_en.txt            # 任务2英文提示词
└── docs/                        # 文档
    ├── data_screening_guide.md  # 数据筛查指南
    ├── error_types_guide.md     # 错误类型指南
    └── parallel_execution_guide.md  # 并行执行指南
```

## 任务说明

### 任务1：动作编码测评
- **输出格式**：JSON对象
- **编码维度**：FORCE, ARM, HAND, VD, HD
- **数值编码**：整数编码值

### 任务2：动作描述测评
- **输出格式**：一句话文本描述
- **描述维度**：初始手部高度、手臂状态、力度、水平方向、垂直方向
- **词语要求**：必须使用预定义的可选描述词语

## 环境准备

### 1. 安装依赖

```bash
pip install requests
```

### 2. 启动vllm服务

```bash
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/your/model \
    --host localhost \
    --port 8000 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code
```

## 快速开始

### 1. 修改配置

编辑 `config.py`（中文）或 `config_en.py`（英文）文件：

```python
# 修改要测评的模型路径
MODELS = [
    "/path/to/your/model1",
    "/path/to/your/model2",
]

# 修改vllm服务配置
VLLM_CONFIG = {
    "host": "localhost",
    "port": 8000,
}
```

### 2. 运行单个实验

```bash
# 运行任务1（中文版）
python run_experiment.py --task task1 --language zh

# 运行任务2（中文版）
python run_experiment.py --task task2 --language zh

# 运行任务1（英文版）
python run_experiment.py --task task1 --language en

# 运行任务2（英文版）
python run_experiment.py --task task2 --language en

# 指定模型和动词
python run_experiment.py --task task1 --models /path/to/model1 /path/to/model2 --verbs 投 扔

# 指定输出目录和重复次数
python run_experiment.py --task task2 --output-dir my_results --repeat 3

# 指定参与者ID（用于拉丁方顺序分配）
python run_experiment.py --task task1 --participant-id 1

# 使用Completion API格式（非Chat）
python run_experiment.py --task task1 --no-chat

# 指定vllm服务地址和端口
python run_experiment.py --task task2 --host 192.168.1.100 --port 8001

# 禁用断点续传
python run_experiment.py --task task1 --no-resume
```

### 3. 运行母脚本（批量实验）

```bash
# 运行单个任务
python run_all_experiments.py --mode single --task task1 --language zh

# 运行所有任务（中文和英文）
python run_all_experiments.py --mode all --languages zh en

# 运行拉丁方实验（所有参与者ID）
python run_all_experiments.py --mode latin-square --task task1 --language zh

# 指定模型和重复次数
python run_all_experiments.py --mode all --models /path/to/model1 /path/to/model2 --repeat 3
```

### 4. 数据完整性筛查

```bash
# 筛查任务1数据
python screen_data.py results/task1_zh_20240115_103000 --task task1 --language zh

# 筛查任务2数据
python screen_data.py results/task2_en_20240115_103000 --task task2 --language en

# 指定输出报告路径
python screen_data.py results/task1_zh_20240115_103000 --task task1 --output report.json
```

## 核心功能

### 断点续传

脚本会自动记录已完成的实验，中断后重新运行会跳过已完成的部分：

- 跟踪文件：`{output_dir}/experiment_tracker.json`
- 禁用断点续传：`--no-resume`

### 超时重试

API调用失败时自动重试：

- 最大重试次数：`--max-retries`（默认3次）
- 重试延迟：`--retry-delay`（默认5秒）

### 实时日志

所有运行日志保存在 `logs/` 目录：

- 日志文件：`logs/experiment_{timestamp}.log`
- 同时输出到控制台和文件

## 拉丁方顺序

为了平衡顺序效应，脚本支持拉丁方（Latin Square）顺序：

- 6个动词有6种不同的排列顺序
- 每个参与者被分配一种顺序
- 通过 `--participant-id` 参数指定参与者ID

### 中文动词拉丁方顺序
```
顺序1: 投 → 扔 → 摔 → 丢 → 甩 → 抛
顺序2: 扔 → 摔 → 丢 → 甩 → 抛 → 投
顺序3: 摔 → 丢 → 甩 → 抛 → 投 → 扔
顺序4: 丢 → 甩 → 抛 → 投 → 扔 → 摔
顺序5: 甩 → 抛 → 投 → 扔 → 摔 → 丢
顺序6: 抛 → 投 → 扔 → 摔 → 丢 → 甩
```

### 英文动词拉丁方顺序
```
顺序1: fling → chuck → cast → throw → hurl → toss
顺序2: chuck → cast → throw → hurl → toss → fling
顺序3: cast → throw → hurl → toss → fling → chuck
顺序4: throw → hurl → toss → fling → chuck → cast
顺序5: hurl → toss → fling → chuck → cast → throw
顺序6: toss → fling → chuck → cast → throw → hurl
```

## 命令行参数

### run_experiment.py

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--models` | 模型路径列表 | config.py中的配置 |
| `--verbs` | 动词列表 | config.py中的配置 |
| `--output-dir` | 输出目录 | results |
| `--repeat` | 重复次数 | 1 |
| `--participant-id` | 参与者ID | None（随机） |
| `--language` | 语言版本 | zh |
| `--task` | 任务ID | task1 |
| `--use-chat` | 使用Chat API | True |
| `--no-chat` | 使用Completion API | - |
| `--host` | vllm服务地址 | localhost |
| `--port` | vllm服务端口 | 8000 |
| `--max-retries` | 最大重试次数 | 3 |
| `--retry-delay` | 重试延迟（秒） | 5.0 |
| `--no-resume` | 禁用断点续传 | - |

## 注意事项

1. 确保vllm服务已启动并可访问
2. 模型路径需要正确配置
3. 网络连接稳定
4. 注意GPU内存使用
5. 结果文件会自动保存，请勿手动删除results目录
6. 使用拉丁方顺序可以平衡顺序效应，提高实验效度
7. 任务2要求模型输出必须使用预定义的描述词语
8. 使用断点续传功能可以安全中断和恢复实验
9. 数据筛查脚本用于检查数据合法性，确保入组数据质量
10. **独立上下文原则**：每个模型进行每次判断时必须是独立的，动词和任务之间处于独立上下文，不可以有相互干扰
