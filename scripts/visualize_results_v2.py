"""
重新生成图表，把CKA子图替换成维度相关性热力图
"""
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec

# 设置Nature风格
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.8

# ============================================================
# 数据
# ============================================================

model_names = ['Mimo-7B', 'Mimo-VL', 'Mimo-emb', 'Qwen2.5', 'Qwen2.5-VL', 'RoboBrain']
model_keys = ['Mimo-7B-SFT', 'Mimo-VL-7B-SFT', 'Mimo-embodied-7B',
              'Qwen2.5-7B-Instruct', 'Qwen2.5-VL-7B-Instruct', 'RoboBrain2.0-7B']

# 总表数据
data = {
    'Mimo-7B-SFT': {
        'Param-CN': {'MSE': 0.190, 'RSA': -0.442, 'Jaccard': 0.273},
        'Param-EN': {'MSE': 0.235, 'RSA': -0.043, 'Jaccard': 0.083},
        'Verb-CN': {'MSE': 0.197, 'RSA': 0.464, 'Jaccard': 0.077},
        'Verb-EN': {'MSE': 0.175, 'RSA': 0.152, 'Jaccard': 0.231},
    },
    'Mimo-VL-7B-SFT': {
        'Param-CN': {'MSE': 0.155, 'RSA': -0.178, 'Jaccard': 0.188},
        'Param-EN': {'MSE': 0.171, 'RSA': -0.132, 'Jaccard': 0.071},
        'Verb-CN': {'MSE': 0.135, 'RSA': -0.027, 'Jaccard': 0.000},
        'Verb-EN': {'MSE': 0.104, 'RSA': 0.076, 'Jaccard': 0.167},
    },
    'Mimo-embodied-7B': {
        'Param-CN': {'MSE': 0.213, 'RSA': 0.066, 'Jaccard': 0.333},
        'Param-EN': {'MSE': 0.218, 'RSA': 0.014, 'Jaccard': 0.214},
        'Verb-CN': {'MSE': 0.189, 'RSA': 0.121, 'Jaccard': 0.214},
        'Verb-EN': {'MSE': 0.113, 'RSA': 0.205, 'Jaccard': 0.167},
    },
    'Qwen2.5-7B-Instruct': {
        'Param-CN': {'MSE': 0.225, 'RSA': 0.147, 'Jaccard': 0.154},
        'Param-EN': {'MSE': 0.239, 'RSA': -0.242, 'Jaccard': 0.000},
        'Verb-CN': {'MSE': 0.150, 'RSA': 0.138, 'Jaccard': 0.077},
        'Verb-EN': {'MSE': 0.071, 'RSA': 0.622, 'Jaccard': 0.182},
    },
    'Qwen2.5-VL-7B-Instruct': {
        'Param-CN': {'MSE': 0.233, 'RSA': -0.528, 'Jaccard': 0.154},
        'Param-EN': {'MSE': 0.201, 'RSA': 0.513, 'Jaccard': 0.267},
        'Verb-CN': {'MSE': 0.180, 'RSA': 0.152, 'Jaccard': 0.000},
        'Verb-EN': {'MSE': 0.075, 'RSA': 0.465, 'Jaccard': 0.308},
    },
    'RoboBrain2.0-7B': {
        'Param-CN': {'MSE': 0.122, 'RSA': 0.504, 'Jaccard': 0.250},
        'Param-EN': {'MSE': 0.185, 'RSA': 0.780, 'Jaccard': 0.267},
        'Verb-CN': {'MSE': 0.163, 'RSA': -0.201, 'Jaccard': 0.154},
        'Verb-EN': {'MSE': 0.072, 'RSA': -0.124, 'Jaccard': 0.133},
    },
}

conditions = ['Param-CN', 'Param-EN', 'Verb-CN', 'Verb-EN']
condition_labels = ['Param-CN', 'Param-EN', 'Verb-CN', 'Verb-EN']

# ============================================================
# 图1：总表（3个指标 + 维度相关性热力图）
# ============================================================

# 维度相关性数据（中文条件）
dim_corr_zh = {
    'Mimo-7B': np.array([[np.nan, np.nan, np.nan, np.nan, np.nan],
                         [-0.60, -0.04, 0.04, -0.04, 0.04],
                         [np.nan, np.nan, np.nan, np.nan, np.nan],
                         [np.nan, np.nan, np.nan, np.nan, np.nan],
                         [np.nan, np.nan, np.nan, np.nan, np.nan]]),
    'Mimo-VL': np.array([[0.34, 0.41, 0.79, 0.66, 0.80],
                         [0.07, 0.49, 0.84, 0.23, 0.42],
                         [-0.54, -0.36, -0.12, -0.52, -0.63],
                         [-0.80, -0.49, -0.51, -0.23, -0.40],
                         [0.19, 0.38, 0.13, 0.03, 0.27]]),
    'Mimo-emb': np.array([[0.80, 0.49, 0.51, 0.23, 0.40],
                          [-0.78, -0.66, 0.07, -0.34, -0.10],
                          [0.80, 0.49, 0.51, 0.23, 0.40],
                          [np.nan, np.nan, np.nan, np.nan, np.nan],
                          [0.10, -0.09, 0.54, -0.07, 0.30]]),
    'Qwen2.5': np.array([[0.80, 0.49, 0.51, 0.23, 0.40],
                         [0.88, 0.30, -0.19, -0.40, -0.37],
                         [np.nan, np.nan, np.nan, np.nan, np.nan],
                         [-0.80, -0.49, -0.51, -0.23, -0.40],
                         [0.12, 0.03, 0.37, -0.43, -0.40]]),
    'Qwen2.5-VL': np.array([[-0.64, -0.56, -0.41, -0.17, -0.43],
                            [-0.33, -0.37, 0.25, 0.02, 0.37],
                            [-0.36, -0.06, -0.01, -0.46, -0.40],
                            [-0.09, 0.66, 0.51, 0.49, 0.33],
                            [0.12, 0.03, 0.37, -0.43, -0.40]]),
    'RoboBrain': np.array([[-0.60, -0.30, 0.57, 0.31, 0.52],
                           [-0.80, -0.49, -0.51, -0.23, -0.40],
                           [0.03, 0.41, 0.89, 0.30, 0.52],
                           [-0.84, -0.28, 0.15, 0.45, 0.34],
                           [np.nan, np.nan, np.nan, np.nan, np.nan]]),
}

dims = ['FORCE', 'HAND', 'ARM', 'HD', 'VD']

fig = plt.figure(figsize=(15, 12))
gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

# MSE
ax1 = fig.add_subplot(gs[0, 0])
x = np.arange(len(model_names))
width = 0.2
colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']
for i, cond in enumerate(conditions):
    values = [data[model][cond]['MSE'] for model in model_keys]
    ax1.bar(x + i * width, values, width, label=cond, color=colors[i])
ax1.set_xlabel('Model')
ax1.set_ylabel('MSE')
ax1.set_title('MSE', fontweight='bold')
ax1.set_xticks(x + width * 1.5)
ax1.set_xticklabels(model_names, rotation=45, ha='right', fontsize=8)
ax1.legend(fontsize=7)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# RSA
ax2 = fig.add_subplot(gs[0, 1])
for i, cond in enumerate(conditions):
    values = [data[model][cond]['RSA'] for model in model_keys]
    ax2.bar(x + i * width, values, width, label=cond, color=colors[i])
ax2.set_xlabel('Model')
ax2.set_ylabel('RSA')
ax2.set_title('RSA', fontweight='bold')
ax2.set_xticks(x + width * 1.5)
ax2.set_xticklabels(model_names, rotation=45, ha='right', fontsize=8)
ax2.legend(fontsize=7)
ax2.axhline(y=0, color='black', linewidth=0.5)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# 维度相关性热力图（替代CKA）
ax3 = fig.add_subplot(gs[1, 0])
# 使用RoboBrain的数据作为示例
corr_matrix = dim_corr_zh['RoboBrain']
im = ax3.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
ax3.set_xticks(range(5))
ax3.set_yticks(range(5))
ax3.set_xticklabels(dims, fontsize=8)
ax3.set_yticklabels(dims, fontsize=8)
ax3.set_xlabel('Human Dimension')
ax3.set_ylabel('Model Dimension')
ax3.set_title('Dimension Correlation (RoboBrain)', fontweight='bold')
for i in range(5):
    for j in range(5):
        val = corr_matrix[i, j]
        if not np.isnan(val):
            color = 'white' if abs(val) > 0.5 else 'black'
            ax3.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=7, color=color)
plt.colorbar(im, ax=ax3, shrink=0.8)

# Jaccard
ax4 = fig.add_subplot(gs[1, 1])
for i, cond in enumerate(conditions):
    values = [data[model][cond]['Jaccard'] for model in model_keys]
    ax4.bar(x + i * width, values, width, label=cond, color=colors[i])
ax4.set_xlabel('Model')
ax4.set_ylabel('Jaccard')
ax4.set_title('Jaccard', fontweight='bold')
ax4.set_xticks(x + width * 1.5)
ax4.set_xticklabels(model_names, rotation=45, ha='right', fontsize=8)
ax4.legend(fontsize=7)
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)

plt.suptitle('Overview: Model Performance Across Conditions', fontsize=14, fontweight='bold', y=1.02)
plt.savefig('D:/task/科研/LLM-evaluation/具神认知/embodied-verb-cognition/presentation/figures/fig1_overview.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: fig1_overview.png")

# ============================================================
# 图2：语言影响（3个指标 + 维度相关性热力图）
# ============================================================

fig = plt.figure(figsize=(15, 12))
gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

# 计算语言差值
lang_diff = {}
for model in model_keys:
    lang_diff[model] = {}
    for metric in ['MSE', 'RSA', 'Jaccard']:
        cn_param = data[model]['Param-CN'][metric]
        en_param = data[model]['Param-EN'][metric]
        cn_verb = data[model]['Verb-CN'][metric]
        en_verb = data[model]['Verb-EN'][metric]
        lang_diff[model][f'Param_{metric}'] = cn_param - en_param
        lang_diff[model][f'Verb_{metric}'] = cn_verb - en_verb

# MSE Language Effect
ax1 = fig.add_subplot(gs[0, 0])
x = np.arange(len(model_names))
width = 0.35
param_diff = [lang_diff[model]['Param_MSE'] for model in model_keys]
verb_diff = [lang_diff[model]['Verb_MSE'] for model in model_keys]
ax1.bar(x - width/2, param_diff, width, label='Param Format', color='#4C72B0')
ax1.bar(x + width/2, verb_diff, width, label='Verb Format', color='#55A868')
ax1.set_xlabel('Model')
ax1.set_ylabel('MSE Diff (CN - EN)')
ax1.set_title('MSE: Language Effect', fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(model_names, rotation=45, ha='right', fontsize=8)
ax1.legend(fontsize=8)
ax1.axhline(y=0, color='black', linewidth=0.8)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# RSA Language Effect
ax2 = fig.add_subplot(gs[0, 1])
param_diff = [lang_diff[model]['Param_RSA'] for model in model_keys]
verb_diff = [lang_diff[model]['Verb_RSA'] for model in model_keys]
ax2.bar(x - width/2, param_diff, width, label='Param Format', color='#4C72B0')
ax2.bar(x + width/2, verb_diff, width, label='Verb Format', color='#55A868')
ax2.set_xlabel('Model')
ax2.set_ylabel('RSA Diff (CN - EN)')
ax2.set_title('RSA: Language Effect', fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(model_names, rotation=45, ha='right', fontsize=8)
ax2.legend(fontsize=8)
ax2.axhline(y=0, color='black', linewidth=0.8)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# 维度相关性热力图（替代CKA）
ax3 = fig.add_subplot(gs[1, 0])
corr_matrix = dim_corr_zh['RoboBrain']
im = ax3.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
ax3.set_xticks(range(5))
ax3.set_yticks(range(5))
ax3.set_xticklabels(dims, fontsize=8)
ax3.set_yticklabels(dims, fontsize=8)
ax3.set_xlabel('Human Dimension')
ax3.set_ylabel('Model Dimension')
ax3.set_title('Dimension Correlation (RoboBrain)', fontweight='bold')
for i in range(5):
    for j in range(5):
        val = corr_matrix[i, j]
        if not np.isnan(val):
            color = 'white' if abs(val) > 0.5 else 'black'
            ax3.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=7, color=color)
plt.colorbar(im, ax=ax3, shrink=0.8)

# Jaccard Language Effect
ax4 = fig.add_subplot(gs[1, 1])
param_diff = [lang_diff[model]['Param_Jaccard'] for model in model_keys]
verb_diff = [lang_diff[model]['Verb_Jaccard'] for model in model_keys]
ax4.bar(x - width/2, param_diff, width, label='Param Format', color='#4C72B0')
ax4.bar(x + width/2, verb_diff, width, label='Verb Format', color='#55A868')
ax4.set_xlabel('Model')
ax4.set_ylabel('Jaccard Diff (CN - EN)')
ax4.set_title('Jaccard: Language Effect', fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(model_names, rotation=45, ha='right', fontsize=8)
ax4.legend(fontsize=8)
ax4.axhline(y=0, color='black', linewidth=0.8)
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)

plt.suptitle('Language Effect: Chinese vs English', fontsize=14, fontweight='bold', y=1.02)
plt.savefig('D:/task/科研/LLM-evaluation/具神认知/embodied-verb-cognition/presentation/figures/fig2_language_effect.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: fig2_language_effect.png")

# ============================================================
# 图3：提示词影响（3个指标 + 维度相关性热力图）
# ============================================================

fig = plt.figure(figsize=(15, 12))
gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

# 计算提示词差值
prompt_diff = {}
for model in model_keys:
    prompt_diff[model] = {}
    for lang in ['CN', 'EN']:
        for metric in ['MSE', 'RSA', 'Jaccard']:
            param = data[model][f'Param-{lang}'][metric]
            verb = data[model][f'Verb-{lang}'][metric]
            prompt_diff[model][f'{lang}_{metric}'] = param - verb

# MSE Prompt Effect
ax1 = fig.add_subplot(gs[0, 0])
x = np.arange(len(model_names))
width = 0.35
cn_diff = [prompt_diff[model]['CN_MSE'] for model in model_keys]
en_diff = [prompt_diff[model]['EN_MSE'] for model in model_keys]
ax1.bar(x - width/2, cn_diff, width, label='Chinese', color='#C44E52')
ax1.bar(x + width/2, en_diff, width, label='English', color='#8172B2')
ax1.set_xlabel('Model')
ax1.set_ylabel('MSE Diff (Param - Verb)')
ax1.set_title('MSE: Prompt Format Effect', fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(model_names, rotation=45, ha='right', fontsize=8)
ax1.legend(fontsize=8)
ax1.axhline(y=0, color='black', linewidth=0.8)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# RSA Prompt Effect
ax2 = fig.add_subplot(gs[0, 1])
cn_diff = [prompt_diff[model]['CN_RSA'] for model in model_keys]
en_diff = [prompt_diff[model]['EN_RSA'] for model in model_keys]
ax2.bar(x - width/2, cn_diff, width, label='Chinese', color='#C44E52')
ax2.bar(x + width/2, en_diff, width, label='English', color='#8172B2')
ax2.set_xlabel('Model')
ax2.set_ylabel('RSA Diff (Param - Verb)')
ax2.set_title('RSA: Prompt Format Effect', fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(model_names, rotation=45, ha='right', fontsize=8)
ax2.legend(fontsize=8)
ax2.axhline(y=0, color='black', linewidth=0.8)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# 维度相关性热力图（替代CKA）
ax3 = fig.add_subplot(gs[1, 0])
corr_matrix = dim_corr_zh['RoboBrain']
im = ax3.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
ax3.set_xticks(range(5))
ax3.set_yticks(range(5))
ax3.set_xticklabels(dims, fontsize=8)
ax3.set_yticklabels(dims, fontsize=8)
ax3.set_xlabel('Human Dimension')
ax3.set_ylabel('Model Dimension')
ax3.set_title('Dimension Correlation (RoboBrain)', fontweight='bold')
for i in range(5):
    for j in range(5):
        val = corr_matrix[i, j]
        if not np.isnan(val):
            color = 'white' if abs(val) > 0.5 else 'black'
            ax3.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=7, color=color)
plt.colorbar(im, ax=ax3, shrink=0.8)

# Jaccard Prompt Effect
ax4 = fig.add_subplot(gs[1, 1])
cn_diff = [prompt_diff[model]['CN_Jaccard'] for model in model_keys]
en_diff = [prompt_diff[model]['EN_Jaccard'] for model in model_keys]
ax4.bar(x - width/2, cn_diff, width, label='Chinese', color='#C44E52')
ax4.bar(x + width/2, en_diff, width, label='English', color='#8172B2')
ax4.set_xlabel('Model')
ax4.set_ylabel('Jaccard Diff (Param - Verb)')
ax4.set_title('Jaccard: Prompt Format Effect', fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(model_names, rotation=45, ha='right', fontsize=8)
ax4.legend(fontsize=8)
ax4.axhline(y=0, color='black', linewidth=0.8)
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)

plt.suptitle('Prompt Format Effect: Param vs Verb', fontsize=14, fontweight='bold', y=1.02)
plt.savefig('D:/task/科研/LLM-evaluation/具神认知/embodied-verb-cognition/presentation/figures/fig3_prompt_effect.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: fig3_prompt_effect.png")

print("\nDone!")
