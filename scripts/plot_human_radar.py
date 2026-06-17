# -*- coding: utf-8 -*-
"""
绘制人类数据雷达图：中英文6个动词在5个维度上的参数
"""
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 人类数据（Gao 2016）
# 维度：FORCE, HAND, ARM(straight%), HD(forward%), VD(down%)
verbs_cn = ['reng\n(rēng)', 'diu\n(diū)', 'pao\n(pāo)', 'tou\n(tóu)', 'shuāi\n(shuāi)', 'shuǎi\n(shuǎi)']
verbs_en = ['throw', 'fling', 'chuck', 'cast', 'hurl', 'toss']

# 中文数据：FORCE, HAND(归一化到0-1), ARM(伸直比例), HD(向前比例), VD(向下比例)
human_cn = np.array([
    [3.05/5, 5.43/10, 18/29, 21/29, 21/29],  # reng
    [2.33/5, 4.63/10, 24/30, 16/28, 23/28],  # diu
    [3.16/5, 6.65/10, 15/29, 29/30, 30/30],  # pao
    [3.21/5, 9.13/10, 30/30, 30/30, 29/30],  # tou
    [4.55/5, 8.51/10, 29/29, 25/29, 29/29],  # shuāi
    [3.74/5, 6.27/10, 8/29, 9/29, 15/29],   # shuǎi
])

# 英文数据
human_en = np.array([
    [3.62/5, 8.58/10, 5/29, 29/29, 26/29],  # throw
    [3.44/5, 6.41/10, 11/28, 25/29, 27/29],  # fling
    [3.91/5, 7.06/10, 10/29, 27/29, 21/29],  # chuck
    [3.01/5, 6.18/10, 14/29, 27/29, 28/29],  # cast
    [4.39/5, 8.00/10, 8/29, 28/29, 24/29],  # hurl
    [3.01/5, 4.50/10, 26/29, 29/29, 29/29],  # toss
])

dimensions = ['FORCE', 'HAND', 'ARM\n(straight)', 'HD\n(forward)', 'VD\n(down)']
n_dims = len(dimensions)
angles = np.linspace(0, 2 * np.pi, n_dims, endpoint=False).tolist()
angles += angles[:1]  # 闭合

# 创建图表
fig, axes = plt.subplots(1, 2, figsize=(14, 6), subplot_kw=dict(polar=True))

# 中文雷达图
ax1 = axes[0]
ax1.set_theta_offset(np.pi / 2)
ax1.set_theta_direction(-1)
ax1.set_rlabel_position(30)

# 绘制6个中文动词
colors_cn = plt.cm.Set2(np.linspace(0, 1, 6))
for i, (verb, color) in enumerate(zip(verbs_cn, colors_cn)):
    values = human_cn[i].tolist()
    values += values[:1]
    ax1.plot(angles, values, 'o-', linewidth=2, label=verb, color=color, markersize=4)
    ax1.fill(angles, values, alpha=0.1, color=color)

ax1.set_xticks(angles[:-1])
ax1.set_xticklabels(dimensions, fontsize=9)
ax1.set_ylim(0, 1)
ax1.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax1.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=7)
ax1.set_title('Chinese Throw Verbs\n(Human Data, Gao 2016)', fontsize=13, fontweight='bold', pad=20)
ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)

# 英文雷达图
ax2 = axes[1]
ax2.set_theta_offset(np.pi / 2)
ax2.set_theta_direction(-1)
ax2.set_rlabel_position(30)

# 绘制6个英文动词
colors_en = plt.cm.Set1(np.linspace(0, 1, 6))
for i, (verb, color) in enumerate(zip(verbs_en, colors_en)):
    values = human_en[i].tolist()
    values += values[:1]
    ax2.plot(angles, values, 'o-', linewidth=2, label=verb, color=color, markersize=4)
    ax2.fill(angles, values, alpha=0.1, color=color)

ax2.set_xticks(angles[:-1])
ax2.set_xticklabels(dimensions, fontsize=9)
ax2.set_ylim(0, 1)
ax2.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax2.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=7)
ax2.set_title('English Throw Verbs\n(Human Data, Gao 2016)', fontsize=13, fontweight='bold', pad=20)
ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)

plt.suptitle('Human Prototype Structure: 6 Verbs × 5 Physical Dimensions',
             fontsize=15, fontweight='bold', y=1.05)
plt.tight_layout()
plt.savefig('D:/task/科研/LLM-evaluation/具神认知/embodied-verb-cognition/presentation/figures/human_radar_verbs.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Done: human_radar_verbs.png")
