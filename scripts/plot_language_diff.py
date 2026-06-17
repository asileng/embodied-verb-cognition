# -*- coding: utf-8 -*-
"""
绘制语言对比差值图：六个模型的MSE、RSA、CKA中文-英文差值
"""
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 数据（中文值 - 英文值）
models = ['Mimo-7B\nSFT', 'Mimo-VL\n7B-SFT', 'Mimo-\nembodied-7B',
          'Qwen2.5\n7B-Instruct', 'Qwen2.5-VL\n7B-Instruct', 'RoboBrain\n2.0-7B']

# MSE差值（中文 - 英文，正值表示中文MSE更高，即英文更好）
mse_diff = [0.82-0.61, 0.75-0.58, 0.71-0.68, 0.65-0.53, 0.58-0.49, 0.52-0.41]

# RSA差值（中文 - 英文）
rsa_diff = [-0.14-0.12, 0.14-0.02, 0.46-(-0.28), 0.31-0.08, -0.32-0.42, 0.43-0.76]

# CKA差值（中文 - 英文）
cka_diff = [0.21-0.55, 0.43-0.54, 0.50-0.32, 0.65-0.48, 0.36-0.61, 0.56-0.86]

# p值（假设检验结果）
mse_p = [0.12, 0.08, 0.45, 0.03, 0.02, 0.01]
rsa_p = [0.35, 0.42, 0.04, 0.28, 0.05, 0.02]
cka_p = [0.03, 0.15, 0.22, 0.04, 0.03, 0.01]

# 创建图表
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

metrics = [('MSE Diff (CN - EN)', mse_diff, mse_p, 'Blues'),
           ('RSA Diff (CN - EN)', rsa_diff, rsa_p, 'Reds'),
           ('CKA Diff (CN - EN)', cka_diff, cka_p, 'Greens')]

for ax, (title, values, pvals, colormap) in zip(axes, metrics):
    colors = plt.get_cmap(colormap)(np.linspace(0.4, 0.8, len(models)))
    bars = ax.bar(range(len(models)), values, color=colors, edgecolor='black', linewidth=0.5)

    # 添加数值和p值标注
    for i, (bar, val, p) in enumerate(zip(bars, values, pvals)):
        # 柱子上方写数值
        ypos = bar.get_height() if bar.get_height() >= 0 else bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, ypos + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        # 柱子上方写p值
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'n.s.'))
        ax.text(bar.get_x() + bar.get_width()/2, ypos + 0.06,
                f'p={p:.3f}\n{sig}', ha='center', va='bottom', fontsize=7, color='gray')

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, fontsize=8)
    ax.set_ylabel('Difference', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(axis='y', alpha=0.3)

plt.suptitle('Language Impact: Chinese - English Difference (6 Models x 3 Metrics)',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('D:/task/科研/LLM-evaluation/具神认知/embodied-verb-cognition/presentation/figures/language_impact_diff.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Done: language_impact_diff.png")
