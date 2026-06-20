"""
模型进化折线图 v2
按组别（格式-语言）分图，每个图展示4个指标的变化趋势
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

# 指标颜色和样式
metric_styles = {
    'MSE': {'color': '#4C72B0', 'marker': 'o', 'label': 'MSE'},
    'RSA': {'color': '#C44E52', 'marker': 's', 'label': 'RSA'},
    'CKA': {'color': '#55A868', 'marker': '^', 'label': 'CKA'},
    'Jaccard': {'color': '#8172B2', 'marker': 'D', 'label': 'Jaccard'},
}

# Qwen系列数据
qwen_data = {
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

# Mimo系列数据
mimo_data = {
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

# 条件标题
cond_titles = {
    'Param-CN': 'Parametric Format - Chinese',
    'Param-EN': 'Parametric Format - English',
    'Verb-CN': 'Verbal Format - Chinese',
    'Verb-EN': 'Verbal Format - English',
}

# ============================================================
# Qwen系列进化图
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, cond in enumerate(['Param-CN', 'Param-EN', 'Verb-CN', 'Verb-EN']):
    ax = axes[idx]

    for metric in ['MSE', 'RSA', 'CKA', 'Jaccard']:
        values = qwen_data[cond][metric]
        style = metric_styles[metric]
        ax.plot(stages, values, marker=style['marker'], label=style['label'],
                color=style['color'], linewidth=2, markersize=8)

    ax.set_xlabel('Training Stage', fontsize=11)
    ax.set_ylabel('Value', fontsize=11)
    ax.set_title(f'Qwen Series: {cond_titles[cond]}', fontsize=12, fontweight='bold')
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

for idx, cond in enumerate(['Param-CN', 'Param-EN', 'Verb-CN', 'Verb-EN']):
    ax = axes[idx]

    for metric in ['MSE', 'RSA', 'CKA', 'Jaccard']:
        values = mimo_data[cond][metric]
        style = metric_styles[metric]
        ax.plot(stages, values, marker=style['marker'], label=style['label'],
                color=style['color'], linewidth=2, markersize=8)

    ax.set_xlabel('Training Stage', fontsize=11)
    ax.set_ylabel('Value', fontsize=11)
    ax.set_title(f'Mimo Series: {cond_titles[cond]}', fontsize=12, fontweight='bold')
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
