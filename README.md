# Embodied Verb Cognition: VLM/LLM 具身动作语义认知测评框架

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 项目简介

本项目基于认知语言学理论，提出了一种探测大模型**"潜具身性"（Latent Embodiment）**的新型评测框架。通过复现心理语言学中的"表演范式"（Enactment Paradigm）实验，评估 VLM/LLM 在细粒度动作语义理解方面的能力。

### 核心研究问题

- **RQ1**: 当前 VLA 基座模型能否预测人类在表演范式下针对身体动作动词表现出的结果？
- **RQ2**: LLM 与 VLM 在不同任务下的表现有何差异？
- **RQ3**: VLA 基座模型对身体动作动词的理解是否对输出格式和语言敏感？

### 主要发现

1. 多数模型在结构化表征能力上表现有限，仅部分 VLA 模型展现出统计显著的动词区分能力
2. 模型与人类认知结构的一致性整体偏低，但 VLA 模型在英文条件下展现出一定的结构对齐能力
3. 模型类型对动词表征能力的影响呈边缘显著，视觉模态的引入效果不均匀

## 项目结构

```
embodied-verb-cognition/
├── README.md                    # 项目说明文档
├── .gitignore                   # Git 忽略配置
├── LICENSE                      # 开源许可证
│
├── framework/                   # 实验框架（原 enactment-test-requirements）
│   ├── README.md               # 框架使用说明
│   ├── config.py               # 中文版配置文件
│   ├── config_en.py            # 英文版配置文件
│   ├── run_experiment.py       # 主执行脚本
│   ├── run_all_experiments.py  # 母脚本（批量运行）
│   ├── screen_data.py          # 数据筛查脚本
│   ├── prompt_templates/       # 提示词模板
│   │   ├── task1_zh.txt
│   │   ├── task1_en.txt
│   │   ├── task2_zh.txt
│   │   └── task2_en.txt
│   └── docs/                   # 文档
│       ├── data_screening_guide.md
│       ├── error_types_guide.md
│       └── parallel_execution_guide.md
│
├── presentation/                # 结果展示（原 1 仓库）
│   ├── index.html              # 主演示文稿
│   ├── figures/                # 图表资源
│   └── assets/                 # 其他资源
│
├── papers/                      # 参考文献
│   ├── Gao_2016_Throw_Verbs.pdf
│   ├── Hoang_2024_HL_Mandarin.pdf
│   └── Wang_Gao_2016_CogLS.pdf
│
└── data/                        # 实验数据（本地，不提交）
    ├── results/
    └── pilot_results/
```

## 快速开始

### 环境要求

- Python 3.8+
- vLLM（用于本地模型推理）
- 8GB+ GPU 内存

### 安装

```bash
# 克隆仓库
git clone https://github.com/asileng/embodied-verb-cognition.git
cd embodied-verb-cognition

# 安装依赖
pip install -r requirements.txt

# 启动 vLLM 服务
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/your/model \
    --host localhost \
    --port 8000
```

### 运行实验

```bash
# 运行单个实验
python framework/run_experiment.py --task task1 --language zh

# 运行所有实验
python framework/run_all_experiments.py --mode all --languages zh en

# 数据筛查
python framework/screen_data.py data/results/task1_zh --task task1 --language zh
```

### 查看结果演示

```bash
# 在浏览器中打开演示文稿
open presentation/index.html
```

## 编码维度说明

### 任务1：动作编码测评

| 维度 | 含义 | 取值范围 |
|------|------|----------|
| FORCE | 力量大小 | 1-5 |
| ARM | 手臂初始状态 | 0（弯曲）或 1（伸直）|
| HAND | 手部初始高度 | 0-12 |
| VD | 垂直运动方向 | 0（向上）或 1（向下）|
| HD | 水平运动方向 | 0（向侧方）或 1（向前）|

### 任务2：动作描述测评

| 维度 | 含义 | 可选描述 |
|------|------|----------|
| FORCE | 力量大小 | 非常强、强、中等、弱、非常弱 |
| ARM | 手臂初始状态 | 手臂伸直、手臂弯曲 |
| HAND | 手部初始高度 | 接近地面、膝盖高度、腰部高度、胸部高度、肩部高度、头部高度、高于头部 |
| VD | 垂直运动方向 | 向下、向上 |
| HD | 水平运动方向 | 向前、向侧方 |

## 拉丁方顺序

为平衡顺序效应，实验采用拉丁方设计：

### 中文动词
```
顺序1: 投 → 扔 → 摔 → 丢 → 甩 → 抛
顺序2: 扔 → 摔 → 丢 → 甩 → 抛 → 投
顺序3: 摔 → 丢 → 甩 → 抛 → 投 → 扔
顺序4: 丢 → 甩 → 抛 → 投 → 扔 → 摔
顺序5: 甩 → 抛 → 投 → 扔 → 摔 → 丢
顺序6: 抛 → 投 → 扔 → 摔 → 丢 → 甩
```

### 英文动词
```
顺序1: fling → chuck → cast → throw → hurl → toss
顺序2: chuck → cast → throw → hurl → toss → fling
顺序3: cast → throw → hurl → toss → fling → chuck
顺序4: throw → hurl → toss → fling → chuck → cast
顺序5: hurl → toss → fling → chuck → cast → throw
顺序6: toss → fling → chuck → cast → throw → hurl
```

## 测评指标

- **RSA（表征相似性分析）**：衡量模型动词区分能力
- **CKA（中心核对齐）**：衡量模型与人类认知结构的一致性
- **Spearman 相关**：检验模型与人类在动词-参数空间中的表征结构一致性

## 贡献指南

1. Fork 本仓库
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 引用

如果您使用了本项目的代码或数据，请引用：

```bibtex
@article{xiong2026embodied,
  title={Cognitive Semantic Perspective on Latent Embodiment of Embodied Intelligence Foundation Models},
  author={Xiong, Yicheng and Li, Mengxi},
  journal={TBD},
  year={2026}
}
```

## 联系方式

- 熊羿成 - 上海外国语大学
- 李檬希 - 西南大学

## 致谢

- Helena Gao 的动词语义分解理论
- 认知语言学与具身认知研究社区
- vLLM 开发团队
