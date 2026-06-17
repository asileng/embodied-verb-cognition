"""
四个条件的可视化
每个条件：6个模型 × 4个指标
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

# 排列顺序：Qwen2.5, Qwen2.5-VL, RoboBrain, Mimo-7B, Mimo-VL, Mimo-embodied
model_names = ['Qwen2.5', 'Qwen2.5-VL', 'RoboBrain', 'Mimo-7B', 'Mimo-VL', 'Mimo-emb']
metrics = ['MSE', 'RSA', 'CKA', 'Jaccard']

# 颜色：Qwen系列蓝色渐进，Mimo系列绿色渐进
colors = ['#B3CDE3', '#6C9BD1', '#3A7CA5', '#B5E0B5', '#6BCB77', '#2D8B4E']

# 四个条件的数据（按照新顺序：Qwen2.5, Qwen2.5-VL, RoboBrain, Mimo-7B, Mimo-VL, Mimo-embodied）
conditions_data = {
    'Param-CN': {
        'title': 'Parametric Format - Chinese',
        'data': {
            'MSE': [0.225, 0.233, 0.122, 0.190, 0.155, 0.213],
            'RSA': [0.147, -0.528, 0.504, -0.442, -0.178, 0.066],
            'CKA': [0.387, 0.302, 0.565, 0.073, 0.389, 0.288],
            'Jaccard': [0.154, 0.154, 0.250, 0.273, 0.188, 0.333],
        }
    },
    'Param-EN': {
        'title': 'Parametric Format - English',
        'data': {
            'MSE': [0.239, 0.201, 0.185, 0.235, 0.171, 0.218],
            'RSA': [-0.242, 0.513, 0.780, -0.043, -0.132, 0.014],
            'CKA': [0.401, 0.677, 0.849, 0.223, 0.415, 0.320],
            'Jaccard': [0.000, 0.267, 0.267, 0.083, 0.071, 0.214],
        }
    },
    'Verb-CN': {
        'title': 'Verbal Format - Chinese',
        'data': {
            'MSE': [0.150, 0.180, 0.163, 0.197, 0.135, 0.189],
            'RSA': [0.138, 0.152, -0.201, 0.464, -0.027, 0.121],
            'CKA': [0.366, 0.373, 0.168, 0.485, 0.220, 0.268],
            'Jaccard': [0.077, 0.000, 0.154, 0.077, 0.000, 0.214],
        }
    },
    'Verb-EN': {
        'title': 'Verbal Format - English',
        'data': {
            'MSE': [0.071, 0.075, 0.072, 0.175, 0.104, 0.113],
            'RSA': [0.622, 0.465, -0.124, 0.152, 0.076, 0.205],
            'CKA': [0.618, 0.640, 0.478, 0.487, 0.366, 0.522],
            'Jaccard': [0.182, 0.308, 0.133, 0.231, 0.167, 0.167],
        }
    },
}

# ============================================================
# 为每个条件生成图表
# ============================================================

for cond_key, cond_info in conditions_data.items():
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        values = cond_info['data'][metric]

        bars = ax.bar(range(len(model_names)), values, color=colors[idx % len(colors)],
                      edgecolor='white', linewidth=0.5)

        # 标记最大值
        max_idx = np.argmax(values)
        bars[max_idx].set_edgecolor('black')
        bars[max_idx].set_linewidth(2)

        # 标记最小值（MSE越小越好）
        if metric == 'MSE':
            min_idx = np.argmin(values)
            bars[min_idx].set_edgecolor('black')
            bars[min_idx].set_linewidth(2)

        ax.set_xlabel('Model', fontsize=10)
        ax.set_ylabel(metric, fontsize=10)
        ax.set_title(f'{metric}', fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=8)
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.3, axis='y', linestyle=':')

    plt.suptitle(cond_info['title'], fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    filename = f'fig5_{cond_key.lower()}'
    plt.savefig(f'D:/task/科研/LLM-evaluation/具神认知/embodied-verb-cognition/presentation/figures/{filename}.png',
                dpi=300, bbox_inches='tight')
    plt.savefig(f'D:/task/科研/LLM-evaluation/具神认知/embodied-verb-cognition/presentation/figures/{filename}.pdf',
                bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}.png/pdf")

print("\nDone!")
