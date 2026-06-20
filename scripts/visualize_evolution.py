"""
模型进化折线图
每个系列（Qwen/Mimo）在4种条件下，4个指标的变化趋势
线条样式：蓝色=英文，红色=中文，实线=参数格式，虚线=言语格式
"""
import matplotlib.pyplot as plt
import numpy as np

# 设置Nature风格
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['xtick.major.width'] = 0.8
plt.rcParams['ytick.major.width'] = 0.8

# ============================================================
# 数据
# ============================================================

stages = ['LLM', 'VLM', 'VLA']
conditions = ['Param-CN', 'Param-EN', 'Verb-CN', 'Verb-EN']

# 颜色和样式：蓝色=英文，红色=中文，实线=参数格式，虚线=言语格式
cond_styles = {
    'Param-CN': {'color': '#C44E52', 'linestyle': '-', 'marker': 'o', 'label': 'Param-CN'},
    'Param-EN': {'color': '#4C72B0', 'linestyle': '-', 'marker': 's', 'label': 'Param-EN'},
    'Verb-CN': {'color': '#C44E52', 'linestyle': '--', 'marker': '^', 'label': 'Verb-CN'},
    'Verb-EN': {'color': '#4C72B0', 'linestyle': '--', 'marker': 'D', 'label': 'Verb-EN'},
}
metrics = ['MSE', 'RSA', 'CKA', 'Jaccard']

# Qwen系列数据
qwen_data = {
    'MSE': {
        'Param-CN': [0.225, 0.233, 0.122],
        'Param-EN': [0.239, 0.201, 0.185],
        'Verb-CN': [0.150, 0.180, 0.163],
        'Verb-EN': [0.071, 0.075, 0.072],
    },
    'RSA': {
        'Param-CN': [0.147, -0.528, 0.504],
        'Param-EN': [-0.242, 0.513, 0.780],
        'Verb-CN': [0.138, 0.152, -0.201],
        'Verb-EN': [0.622, 0.465, -0.124],
    },
    'CKA': {
        'Param-CN': [0.387, 0.302, 0.565],
        'Param-EN': [0.401, 0.677, 0.849],
        'Verb-CN': [0.366, 0.373, 0.168],
        'Verb-EN': [0.618, 0.640, 0.478],
    },
    'Jaccard': {
        'Param-CN': [0.154, 0.154, 0.250],
        'Param-EN': [0.000, 0.267, 0.267],
        'Verb-CN': [0.077, 0.000, 0.154],
        'Verb-EN': [0.182, 0.308, 0.133],
    },
}

# Mimo系列数据
mimo_data = {
    'MSE': {
        'Param-CN': [0.190, 0.155, 0.213],
        'Param-EN': [0.235, 0.171, 0.218],
        'Verb-CN': [0.197, 0.135, 0.189],
        'Verb-EN': [0.175, 0.104, 0.113],
    },
    'RSA': {
        'Param-CN': [-0.442, -0.178, 0.066],
        'Param-EN': [-0.043, -0.132, 0.014],
        'Verb-CN': [0.464, -0.027, 0.121],
        'Verb-EN': [0.152, 0.076, 0.205],
    },
    'CKA': {
        'Param-CN': [0.073, 0.389, 0.288],
        'Param-EN': [0.223, 0.415, 0.320],
        'Verb-CN': [0.485, 0.220, 0.268],
        'Verb-EN': [0.487, 0.366, 0.522],
    },
    'Jaccard': {
        'Param-CN': [0.273, 0.188, 0.333],
        'Param-EN': [0.083, 0.071, 0.214],
        'Verb-CN': [0.077, 0.000, 0.214],
        'Verb-EN': [0.231, 0.167, 0.167],
    },
}

# ============================================================
# Qwen系列进化图
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, metric in enumerate(metrics):
    ax = axes[idx]

    for cond in conditions:
        values = qwen_data[metric][cond]
        style = cond_styles[cond]
        ax.plot(stages, values, marker=style['marker'], label=style['label'],
                color=style['color'], linestyle=style['linestyle'],
                linewidth=2, markersize=8)

    ax.set_xlabel('Training Stage', fontsize=11)
    ax.set_ylabel(metric, fontsize=11)
    ax.set_title(f'Qwen Series: {metric}', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.tick_params(labelsize=9)

plt.tight_layout()
plt.savefig('D:/task/科研/LLM-evaluation/具神认知/embodied-verb-cognition/presentation/figures/fig4a_qwen_evolution.png', dpi=300, bbox_inches='tight')
plt.savefig('D:/task/科研/LLM-evaluation/具神认知/embodied-verb-cognition/presentation/figures/fig4a_qwen_evolution.pdf', bbox_inches='tight')
plt.close()
print("Saved: fig4a_qwen_evolution.png/pdf")

# ============================================================
# Mimo系列进化图
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, metric in enumerate(metrics):
    ax = axes[idx]

    for cond in conditions:
        values = mimo_data[metric][cond]
        style = cond_styles[cond]
        ax.plot(stages, values, marker=style['marker'], label=style['label'],
                color=style['color'], linestyle=style['linestyle'],
                linewidth=2, markersize=8)

    ax.set_xlabel('Training Stage', fontsize=11)
    ax.set_ylabel(metric, fontsize=11)
    ax.set_title(f'Mimo Series: {metric}', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.tick_params(labelsize=9)

plt.tight_layout()
plt.savefig('D:/task/科研/LLM-evaluation/具神认知/embodied-verb-cognition/presentation/figures/fig4b_mimo_evolution.png', dpi=300, bbox_inches='tight')
plt.savefig('D:/task/科研/LLM-evaluation/具神认知/embodied-verb-cognition/presentation/figures/fig4b_mimo_evolution.pdf', bbox_inches='tight')
plt.close()
print("Saved: fig4b_mimo_evolution.png/pdf")

print("\nDone!")
