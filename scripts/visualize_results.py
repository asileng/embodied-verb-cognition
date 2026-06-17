"""
可视化实验结果
1. 总表：分组柱状图
2. 语言影响：差值柱状图
3. 提示词影响：差值柱状图
4. 模型进化：折线图
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 设置Nature风格
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['xtick.major.width'] = 0.8
plt.rcParams['ytick.major.width'] = 0.8

# ============================================================
# 数据
# ============================================================

# 总表数据
data = {
    'Mimo-7B-SFT': {
        'Param-CN': {'MSE': 0.190, 'RSA': -0.442, 'CKA': 0.073, 'Jaccard': 0.273},
        'Param-EN': {'MSE': 0.235, 'RSA': -0.043, 'CKA': 0.223, 'Jaccard': 0.083},
        'Verb-CN': {'MSE': 0.197, 'RSA': 0.464, 'CKA': 0.485, 'Jaccard': 0.077},
        'Verb-EN': {'MSE': 0.175, 'RSA': 0.152, 'CKA': 0.487, 'Jaccard': 0.231},
    },
    'Mimo-VL-7B-SFT': {
        'Param-CN': {'MSE': 0.155, 'RSA': -0.178, 'CKA': 0.389, 'Jaccard': 0.188},
        'Param-EN': {'MSE': 0.171, 'RSA': -0.132, 'CKA': 0.415, 'Jaccard': 0.071},
        'Verb-CN': {'MSE': 0.135, 'RSA': -0.027, 'CKA': 0.220, 'Jaccard': 0.000},
        'Verb-EN': {'MSE': 0.104, 'RSA': 0.076, 'CKA': 0.366, 'Jaccard': 0.167},
    },
    'Mimo-embodied-7B': {
        'Param-CN': {'MSE': 0.213, 'RSA': 0.066, 'CKA': 0.288, 'Jaccard': 0.333},
        'Param-EN': {'MSE': 0.218, 'RSA': 0.014, 'CKA': 0.320, 'Jaccard': 0.214},
        'Verb-CN': {'MSE': 0.189, 'RSA': 0.121, 'CKA': 0.268, 'Jaccard': 0.214},
        'Verb-EN': {'MSE': 0.113, 'RSA': 0.205, 'CKA': 0.522, 'Jaccard': 0.167},
    },
    'Qwen2.5-7B-Instruct': {
        'Param-CN': {'MSE': 0.225, 'RSA': 0.147, 'CKA': 0.387, 'Jaccard': 0.154},
        'Param-EN': {'MSE': 0.239, 'RSA': -0.242, 'CKA': 0.401, 'Jaccard': 0.000},
        'Verb-CN': {'MSE': 0.150, 'RSA': 0.138, 'CKA': 0.366, 'Jaccard': 0.077},
        'Verb-EN': {'MSE': 0.071, 'RSA': 0.622, 'CKA': 0.618, 'Jaccard': 0.182},
    },
    'Qwen2.5-VL-7B-Instruct': {
        'Param-CN': {'MSE': 0.233, 'RSA': -0.528, 'CKA': 0.302, 'Jaccard': 0.154},
        'Param-EN': {'MSE': 0.201, 'RSA': 0.513, 'CKA': 0.677, 'Jaccard': 0.267},
        'Verb-CN': {'MSE': 0.180, 'RSA': 0.152, 'CKA': 0.373, 'Jaccard': 0.000},
        'Verb-EN': {'MSE': 0.075, 'RSA': 0.465, 'CKA': 0.640, 'Jaccard': 0.308},
    },
    'RoboBrain2.0-7B': {
        'Param-CN': {'MSE': 0.122, 'RSA': 0.504, 'CKA': 0.565, 'Jaccard': 0.250},
        'Param-EN': {'MSE': 0.185, 'RSA': 0.780, 'CKA': 0.849, 'Jaccard': 0.267},
        'Verb-CN': {'MSE': 0.163, 'RSA': -0.201, 'CKA': 0.168, 'Jaccard': 0.154},
        'Verb-EN': {'MSE': 0.072, 'RSA': -0.124, 'CKA': 0.478, 'Jaccard': 0.133},
    },
}

# 模型名称（短名称）
model_names = ['Mimo-7B', 'Mimo-VL', 'Mimo-emb', 'Qwen2.5', 'Qwen2.5-VL', 'RoboBrain']
model_keys = list(data.keys())

# 条件
conditions = ['Param-CN', 'Param-EN', 'Verb-CN', 'Verb-EN']
condition_labels = ['Param-CN', 'Param-EN', 'Verb-CN', 'Verb-EN']

# 指标
metrics = ['MSE', 'RSA', 'CKA', 'Jaccard']

# ============================================================
# 图1：总表 - 分组柱状图（4个子图，每个指标一个）
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']

for idx, metric in enumerate(metrics):
    ax = axes[idx]

    x = np.arange(len(model_names))
    width = 0.2

    for i, cond in enumerate(conditions):
        values = [data[model][cond][metric] for model in model_keys]
        ax.bar(x + i * width, values, width, label=cond, color=colors[i], edgecolor='white', linewidth=0.5)

    ax.set_xlabel('Model', fontsize=10)
    ax.set_ylabel(metric, fontsize=10)
    ax.set_title(metric, fontsize=12, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=8)
    ax.legend(fontsize=7, loc='best')
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('D:/task/科研/LLM-evaluation/具神认知/embodied-verb-cognition/presentation/figures/fig1_overview.png', dpi=300, bbox_inches='tight')
plt.savefig('D:/task/科研/LLM-evaluation/具神认知/embodied-verb-cognition/presentation/figures/fig1_overview.pdf', bbox_inches='tight')
plt.close()
print("Saved: fig1_overview.png/pdf")

# ============================================================
# 图2：语言影响 - 差值柱状图
# ============================================================

# 计算语言差值（中文 - 英文）
lang_diff = {}
for model in model_keys:
    lang_diff[model] = {}
    for fmt in ['Param', 'Verb']:
        for metric in metrics:
            cn = data[model][f'{fmt}-CN'][metric]
            en = data[model][f'{fmt}-EN'][metric]
            lang_diff[model][f'{fmt}_{metric}'] = cn - en

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, metric in enumerate(metrics):
    ax = axes[idx]

    x = np.arange(len(model_names))
    width = 0.35

    param_diff = [lang_diff[model][f'Param_{metric}'] for model in model_keys]
    verb_diff = [lang_diff[model][f'Verb_{metric}'] for model in model_keys]

    ax.bar(x - width/2, param_diff, width, label='Param Format', color='#4C72B0', edgecolor='white', linewidth=0.5)
    ax.bar(x + width/2, verb_diff, width, label='Verb Format', color='#55A868', edgecolor='white', linewidth=0.5)

    ax.set_xlabel('Model', fontsize=10)
    ax.set_ylabel(f'{metric} Diff (CN - EN)', fontsize=10)
    ax.set_title(f'{metric}: Language Effect', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=8)
    ax.legend(fontsize=8)
    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('D:/task/科研/LLM-evaluation/具神认知/embodied-verb-cognition/presentation/figures/fig2_language_effect.png', dpi=300, bbox_inches='tight')
plt.savefig('D:/task/科研/LLM-evaluation/具神认知/embodied-verb-cognition/presentation/figures/fig2_language_effect.pdf', bbox_inches='tight')
plt.close()
print("Saved: fig2_language_effect.png/pdf")

# ============================================================
# 图3：提示词影响 - 差值柱状图
# ============================================================

# 计算提示词差值（参数 - 言语）
prompt_diff = {}
for model in model_keys:
    prompt_diff[model] = {}
    for lang in ['CN', 'EN']:
        for metric in metrics:
            param = data[model][f'Param-{lang}'][metric]
            verb = data[model][f'Verb-{lang}'][metric]
            prompt_diff[model][f'{lang}_{metric}'] = param - verb

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, metric in enumerate(metrics):
    ax = axes[idx]

    x = np.arange(len(model_names))
    width = 0.35

    cn_diff = [prompt_diff[model][f'CN_{metric}'] for model in model_keys]
    en_diff = [prompt_diff[model][f'EN_{metric}'] for model in model_keys]

    ax.bar(x - width/2, cn_diff, width, label='Chinese', color='#C44E52', edgecolor='white', linewidth=0.5)
    ax.bar(x + width/2, en_diff, width, label='English', color='#8172B2', edgecolor='white', linewidth=0.5)

    ax.set_xlabel('Model', fontsize=10)
    ax.set_ylabel(f'{metric} Diff (Param - Verb)', fontsize=10)
    ax.set_title(f'{metric}: Prompt Format Effect', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=8)
    ax.legend(fontsize=8)
    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('D:/task/科研/LLM-evaluation/具神认知/embodied-verb-cognition/presentation/figures/fig3_prompt_effect.png', dpi=300, bbox_inches='tight')
plt.savefig('D:/task/科研/LLM-evaluation/具神认知/embodied-verb-cognition/presentation/figures/fig3_prompt_effect.pdf', bbox_inches='tight')
plt.close()
print("Saved: fig3_prompt_effect.png/pdf")

# ============================================================
# 图4：模型进化 - 折线图
# ============================================================

# Qwen系列进化数据
qwen_evolution = {
    'Param-CN': {
        'MSE': [0.225, 0.233, 0.122],
        'RSA': [0.147, -0.528, 0.504],
        'CKA': [0.387, 0.302, 0.565],
        'Jaccard': [0.154, 0.154, 0.250],
    },
    'Param-EN': {
        'MSE': [0.239, 0.201, 0.185],
        'RSA': [-0.242, 0.513, 0.780],
        'CKA': [0.401, 0.677, 0.849],
        'Jaccard': [0.000, 0.267, 0.267],
    },
    'Verb-CN': {
        'MSE': [0.150, 0.180, 0.163],
        'RSA': [0.138, 0.152, -0.201],
        'CKA': [0.366, 0.373, 0.168],
        'Jaccard': [0.077, 0.000, 0.154],
    },
    'Verb-EN': {
        'MSE': [0.071, 0.075, 0.072],
        'RSA': [0.622, 0.465, -0.124],
        'CKA': [0.618, 0.640, 0.478],
        'Jaccard': [0.182, 0.308, 0.133],
    },
}

# Mimo系列进化数据
mimo_evolution = {
    'Param-CN': {
        'MSE': [0.190, 0.155, 0.213],
        'RSA': [-0.442, -0.178, 0.066],
        'CKA': [0.073, 0.389, 0.288],
        'Jaccard': [0.273, 0.188, 0.333],
    },
    'Param-EN': {
        'MSE': [0.235, 0.171, 0.218],
        'RSA': [-0.043, -0.132, 0.014],
        'CKA': [0.223, 0.415, 0.320],
        'Jaccard': [0.083, 0.071, 0.214],
    },
    'Verb-CN': {
        'MSE': [0.197, 0.135, 0.189],
        'RSA': [0.464, -0.027, 0.121],
        'CKA': [0.485, 0.220, 0.268],
        'Jaccard': [0.077, 0.000, 0.214],
    },
    'Verb-EN': {
        'MSE': [0.175, 0.104, 0.113],
        'RSA': [0.152, 0.076, 0.205],
        'CKA': [0.487, 0.366, 0.522],
        'Jaccard': [0.231, 0.167, 0.167],
    },
}

stages = ['LLM', 'VLM', 'VLA']
markers = ['o', 's', '^', 'D']

# Qwen系列
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, metric in enumerate(metrics):
    ax = axes[idx]

    for i, cond in enumerate(conditions):
        values = qwen_evolution[cond][metric]
        ax.plot(stages, values, marker=markers[i], label=cond, linewidth=2, markersize=8)

    ax.set_xlabel('Training Stage', fontsize=10)
    ax.set_ylabel(metric, fontsize=10)
    ax.set_title(f'Qwen Series: {metric}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('D:/task/科研/LLM-evaluation/具神认知/embodied-verb-cognition/presentation/figures/fig4a_qwen_evolution.png', dpi=300, bbox_inches='tight')
plt.savefig('D:/task/科研/LLM-evaluation/具神认知/embodied-verb-cognition/presentation/figures/fig4a_qwen_evolution.pdf', bbox_inches='tight')
plt.close()
print("Saved: fig4a_qwen_evolution.png/pdf")

# Mimo系列
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, metric in enumerate(metrics):
    ax = axes[idx]

    for i, cond in enumerate(conditions):
        values = mimo_evolution[cond][metric]
        ax.plot(stages, values, marker=markers[i], label=cond, linewidth=2, markersize=8)

    ax.set_xlabel('Training Stage', fontsize=10)
    ax.set_ylabel(metric, fontsize=10)
    ax.set_title(f'Mimo Series: {metric}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('D:/task/科研/LLM-evaluation/具神认知/embodied-verb-cognition/presentation/figures/fig4b_mimo_evolution.png', dpi=300, bbox_inches='tight')
plt.savefig('D:/task/科研/LLM-evaluation/具神认知/embodied-verb-cognition/presentation/figures/fig4b_mimo_evolution.pdf', bbox_inches='tight')
plt.close()
print("Saved: fig4b_mimo_evolution.png/pdf")

print("\nAll figures saved to presentation/figures/")
