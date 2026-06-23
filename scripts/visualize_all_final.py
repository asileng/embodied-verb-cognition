"""
最终版可视化脚本
1. fig1_overview: 3个指标 + 平均相关性热力图
2. fig5: 四个条件图改为2*2排布
3. fig2: 3个柱状图 + 2张三角热力图
4. fig3: 3个柱状图 + 2张三角热力图
"""
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr
import json
import glob

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.8

# ============================================================
# 数据准备
# ============================================================

model_names = ['Mimo-7B', 'Mimo-VL', 'Mimo-emb', 'Qwen2.5', 'Qwen2.5-VL', 'RoboBrain']
model_keys = ['Mimo-7B-SFT', 'Mimo-VL-7B-SFT', 'Mimo-embodied-7B',
              'Qwen2.5-7B-Instruct', 'Qwen2.5-VL-7B-Instruct', 'RoboBrain2.0-7B']
dims = ['FORCE', 'HAND', 'ARM', 'HD', 'VD']
conditions = ['Param-CN', 'Param-EN', 'Verb-CN', 'Verb-EN']

human_data = {
    'zh': {
        '投': [3.21, 9.13, 1.0, 1.0, 0.97],
        '扔': [3.05, 5.43, 0.62, 0.72, 0.72],
        '摔': [4.55, 8.51, 1.0, 0.86, 1.0],
        '丢': [2.33, 4.63, 0.80, 0.57, 0.82],
        '甩': [3.74, 6.27, 0.28, 0.31, 0.48],
        '抛': [3.16, 6.65, 0.48, 0.97, 1.0],
    },
    'en': {
        'throw': [3.62, 8.58, 0.83, 1.0, 0.87],
        'fling': [3.44, 6.41, 0.61, 0.86, 0.93],
        'chuck': [3.91, 7.06, 0.66, 0.93, 0.72],
        'cast': [3.01, 6.18, 0.52, 0.93, 0.97],
        'hurl': [4.39, 8.00, 0.72, 0.97, 0.83],
        'toss': [3.01, 4.50, 0.90, 1.0, 1.0],
    }
}

norm_ranges = {'FORCE': (1, 5), 'HAND': (0, 12), 'ARM': (0, 1), 'HD': (0, 1), 'VD': (0, 1)}

def normalize(value, low, high):
    return (value - low) / (high - low) if high > low else 0.5

def get_human_features(lang):
    verb_names = list(human_data[lang].keys())
    features = {dim: [] for dim in dims}
    for verb in verb_names:
        human = human_data[lang][verb]
        features['FORCE'].append(normalize(human[0], 1, 5))
        features['HAND'].append(normalize(human[1], 0, 12))
        features['ARM'].append(human[2])
        features['HD'].append(human[3])
        features['VD'].append(human[4])
    return features

def get_model_features(model_name, lang, task):
    base_path = 'D:/task/科研/LLM-evaluation/具神认知/enactment-test-requirements/pilot_results'
    model_dir_map = {'Mimo-VL-7B-SFT': 'Mimo-VL-7B-SFT-2508'}
    dir_name = model_dir_map.get(model_name, model_name)
    files = glob.glob(f'{base_path}/{dir_name}/{task}_{lang}/*.json')
    verb_names = list(human_data[lang].keys())
    features = {dim: [] for dim in dims}
    for f in files:
        try:
            with open(f, 'r', encoding='utf-8') as fh:
                d = json.load(fh)
            if d.get('is_valid', False):
                verb = d['verb']
                result = d['parsed_result']
                if verb in verb_names:
                    if task == 'task2':
                        task2_mapping_zh = {'非常强': 5, '强': 4, '中等': 3, '弱': 2, '非常弱': 1}
                        task2_mapping_en = {'very strong': 5, 'strong': 4, 'moderate': 3, 'weak': 2, 'very weak': 1}
                        arm_map = {'手臂伸直': 1, '手臂弯曲': 0, 'arm extended': 1, 'arm bent': 0}
                        vd_map = {'向下': 1, '向上': 0, 'downward': 1, 'upward': 0}
                        hd_map = {'向前': 1, '向侧方': 0, 'forward': 1, 'sideways': 0}
                        hand_map_zh = {'接近地面': 0, '膝盖高度': 2, '腰部高度': 5, '胸部高度': 7, '肩部高度': 9, '头部高度': 10, '高于头部': 12}
                        hand_map_en = {'near ground': 0, 'knee height': 2, 'waist height': 5, 'chest height': 7, 'shoulder height': 9, 'head height': 10, 'above head': 12}
                        force_map = task2_mapping_zh if lang == 'zh' else task2_mapping_en
                        hand_map = hand_map_zh if lang == 'zh' else hand_map_en
                        force = force_map.get(str(result.get('FORCE', '中等')), 3)
                        hand = hand_map.get(str(result.get('HAND', '腰部高度')), 6)
                        arm = arm_map.get(str(result.get('ARM', '手臂弯曲')), 0)
                        hd = hd_map.get(str(result.get('HD', '向前')), 0)
                        vd = vd_map.get(str(result.get('VD', '向下')), 0)
                    else:
                        force = result.get('FORCE', 3)
                        hand = result.get('HAND', 6)
                        arm = result.get('ARM', 0)
                        hd = result.get('HD', 0)
                        vd = result.get('VD', 0)
                    features['FORCE'].append(normalize(force, 1, 5))
                    features['HAND'].append(normalize(hand, 0, 12))
                    features['ARM'].append(arm)
                    features['HD'].append(hd)
                    features['VD'].append(vd)
        except:
            continue
    return features

def compute_diagonal_corr(model_features, human_features):
    corrs = []
    for dim in dims:
        model_vals = model_features[dim]
        human_vals = human_features[dim]
        if len(set(model_vals)) == 1 or len(set(human_vals)) == 1:
            corrs.append(np.nan)
        else:
            corr, _ = pearsonr(model_vals, human_vals)
            corrs.append(corr)
    return corrs

def get_corr_matrix(lang, task):
    human_features = get_human_features(lang)
    corr_data = np.zeros((5, 6))
    for j, model in enumerate(model_keys):
        model_features = get_model_features(model, lang, task)
        corrs = compute_diagonal_corr(model_features, human_features)
        for i, corr in enumerate(corrs):
            corr_data[i, j] = corr
    return corr_data

# MSE, RSA, Jaccard数据
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

# 获取所有相关矩阵
corr_param_cn = get_corr_matrix('zh', 'task1')
corr_param_en = get_corr_matrix('en', 'task1')
corr_verb_cn = get_corr_matrix('zh', 'task2')
corr_verb_en = get_corr_matrix('en', 'task2')

# 计算平均相关矩阵（用于总体表现）
corr_avg = np.nanmean([corr_param_cn, corr_param_en, corr_verb_cn, corr_verb_en], axis=0)

# ============================================================
# 图1：总体表现（3个指标 + 平均相关性热力图）
# ============================================================

fig = plt.figure(figsize=(15, 10))
gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

x = np.arange(len(model_names))
width = 0.2
colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']

# MSE
ax1 = fig.add_subplot(gs[0, 0])
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

# 平均相关性热力图
ax3 = fig.add_subplot(gs[1, 0])
im = ax3.imshow(corr_avg, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax3.set_xticks(range(6))
ax3.set_yticks(range(5))
ax3.set_xticklabels(model_names, rotation=45, ha='right', fontsize=8)
ax3.set_yticklabels(dims, fontsize=9)
ax3.set_xlabel('Model')
ax3.set_ylabel('Dimension')
ax3.set_title('Avg Dimension Correlation', fontweight='bold')
for i in range(5):
    for j in range(6):
        val = corr_avg[i, j]
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
print('Saved: fig1_overview.png')

# ============================================================
# 图5：四个条件图（2*2排布）
# ============================================================

cond_configs = [
    ('Param-CN', 'Parametric Format - Chinese', 'zh', 'task1'),
    ('Param-EN', 'Parametric Format - English', 'en', 'task1'),
    ('Verb-CN', 'Verbal Format - Chinese', 'zh', 'task2'),
    ('Verb-EN', 'Verbal Format - English', 'en', 'task2'),
]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, (cond_key, title, lang, task) in enumerate(cond_configs):
    ax = axes[idx]
    corr_data = get_corr_matrix(lang, task)
    im = ax.imshow(corr_data, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(6))
    ax.set_yticks(range(5))
    ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=7)
    ax.set_yticklabels(dims, fontsize=8)
    ax.set_xlabel('Model', fontsize=9)
    ax.set_ylabel('Dimension', fontsize=9)
    ax.set_title(title, fontweight='bold', fontsize=10)
    for i in range(5):
        for j in range(6):
            val = corr_data[i, j]
            if not np.isnan(val):
                color = 'white' if abs(val) > 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=6, color=color)
    plt.colorbar(im, ax=ax, shrink=0.8)

plt.tight_layout()
plt.savefig('D:/task/科研/LLM-evaluation/具神认知/embodied-verb-cognition/presentation/figures/fig5_conditions.png', dpi=300, bbox_inches='tight')
plt.close()
print('Saved: fig5_conditions.png')

# ============================================================
# 对角线分割热力图函数
# ============================================================

def value_to_color(val, colormap, vmin=-1, vmax=1):
    if np.isnan(val):
        return (0.9, 0.9, 0.9, 1.0)
    norm_val = (val - vmin) / (vmax - vmin)
    norm_val = max(0, min(1, norm_val))
    return colormap(norm_val)

def draw_split_heatmap(ax, data_low, data_high, title, label_low, label_high,
                       colormap_low, colormap_high, vmin=-1, vmax=1):
    n_rows, n_cols = data_low.shape
    for i in range(n_rows):
        for j in range(n_cols):
            val_low = data_low[i, j]
            color_low = value_to_color(val_low, colormap_low, vmin, vmax)
            triangle_low = plt.Polygon([[j, n_rows-1-i], [j+1, n_rows-1-i], [j, n_rows-i]],
                                       facecolor=color_low, edgecolor='white', linewidth=0.5)
            ax.add_patch(triangle_low)
            val_high = data_high[i, j]
            color_high = value_to_color(val_high, colormap_high, vmin, vmax)
            triangle_high = plt.Polygon([[j+1, n_rows-1-i], [j+1, n_rows-i], [j, n_rows-i]],
                                        facecolor=color_high, edgecolor='white', linewidth=0.5)
            ax.add_patch(triangle_high)
            if not np.isnan(val_low):
                ax.text(j+0.25, n_rows-0.5-i, f'{val_low:.2f}', ha='center', va='center',
                       fontsize=5, color='white' if abs(val_low) > 0.5 else 'black')
            if not np.isnan(val_high):
                ax.text(j+0.75, n_rows-0.5-i, f'{val_high:.2f}', ha='center', va='center',
                       fontsize=5, color='white' if abs(val_high) > 0.5 else 'black')
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_xticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
    ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=7)
    ax.set_yticks([0.5, 1.5, 2.5, 3.5, 4.5])
    ax.set_yticklabels(dims[::-1], fontsize=8)
    ax.set_xlabel('Model', fontsize=9)
    ax.set_ylabel('Dimension', fontsize=9)
    ax.set_title(title, fontweight='bold', fontsize=10)
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colormap_low(0.7), label=label_low),
                       Patch(facecolor=colormap_high(0.7), label=label_high)]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=7)

# ============================================================
# 图2：语言影响（3个柱状图 + 2张三角热力图）
# ============================================================

fig = plt.figure(figsize=(15, 12))
gs = gridspec.GridSpec(2, 3, hspace=0.4, wspace=0.4)

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

x = np.arange(len(model_names))
width = 0.35

# MSE柱状图
ax1 = fig.add_subplot(gs[0, 0])
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

# RSA柱状图
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

# Jaccard柱状图
ax3 = fig.add_subplot(gs[0, 2])
param_diff = [lang_diff[model]['Param_Jaccard'] for model in model_keys]
verb_diff = [lang_diff[model]['Verb_Jaccard'] for model in model_keys]
ax3.bar(x - width/2, param_diff, width, label='Param Format', color='#4C72B0')
ax3.bar(x + width/2, verb_diff, width, label='Verb Format', color='#55A868')
ax3.set_xlabel('Model')
ax3.set_ylabel('Jaccard Diff (CN - EN)')
ax3.set_title('Jaccard: Language Effect', fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(model_names, rotation=45, ha='right', fontsize=8)
ax3.legend(fontsize=8)
ax3.axhline(y=0, color='black', linewidth=0.8)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

# 参数格式三角热力图
ax4 = fig.add_subplot(gs[1, 0])
colormap_cn = plt.cm.Reds
colormap_en = plt.cm.Purples
draw_split_heatmap(ax4, corr_param_cn, corr_param_en,
                   'Param Format: Language', 'Chinese (Red)', 'English (Purple)',
                   colormap_cn, colormap_en)

# 言语格式三角热力图
ax5 = fig.add_subplot(gs[1, 1])
draw_split_heatmap(ax5, corr_verb_cn, corr_verb_en,
                   'Verb Format: Language', 'Chinese (Red)', 'English (Purple)',
                   colormap_cn, colormap_en)

# 删除第三个子图位置
ax6 = fig.add_subplot(gs[1, 2])
ax6.axis('off')

plt.suptitle('Language Effect: Chinese vs English', fontsize=14, fontweight='bold', y=1.02)
plt.savefig('D:/task/科研/LLM-evaluation/具神认知/embodied-verb-cognition/presentation/figures/fig2_language_effect.png', dpi=300, bbox_inches='tight')
plt.close()
print('Saved: fig2_language_effect.png')

# ============================================================
# 图3：提示词影响（3个柱状图 + 2张三角热力图）
# ============================================================

fig = plt.figure(figsize=(15, 12))
gs = gridspec.GridSpec(2, 3, hspace=0.4, wspace=0.4)

# 计算提示词差值
prompt_diff = {}
for model in model_keys:
    prompt_diff[model] = {}
    for lang in ['CN', 'EN']:
        for metric in ['MSE', 'RSA', 'Jaccard']:
            param = data[model][f'Param-{lang}'][metric]
            verb = data[model][f'Verb-{lang}'][metric]
            prompt_diff[model][f'{lang}_{metric}'] = param - verb

# MSE柱状图
ax1 = fig.add_subplot(gs[0, 0])
cn_diff = [prompt_diff[model]['CN_MSE'] for model in model_keys]
en_diff = [prompt_diff[model]['EN_MSE'] for model in model_keys]
ax1.bar(x - width/2, cn_diff, width, label='Chinese', color='#C44E52')
ax1.bar(x + width/2, en_diff, width, label='English', color='#8172B2')
ax1.set_xlabel('Model')
ax1.set_ylabel('MSE Diff (Param - Verb)')
ax1.set_title('MSE: Prompt Effect', fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(model_names, rotation=45, ha='right', fontsize=8)
ax1.legend(fontsize=8)
ax1.axhline(y=0, color='black', linewidth=0.8)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# RSA柱状图
ax2 = fig.add_subplot(gs[0, 1])
cn_diff = [prompt_diff[model]['CN_RSA'] for model in model_keys]
en_diff = [prompt_diff[model]['EN_RSA'] for model in model_keys]
ax2.bar(x - width/2, cn_diff, width, label='Chinese', color='#C44E52')
ax2.bar(x + width/2, en_diff, width, label='English', color='#8172B2')
ax2.set_xlabel('Model')
ax2.set_ylabel('RSA Diff (Param - Verb)')
ax2.set_title('RSA: Prompt Effect', fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(model_names, rotation=45, ha='right', fontsize=8)
ax2.legend(fontsize=8)
ax2.axhline(y=0, color='black', linewidth=0.8)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Jaccard柱状图
ax3 = fig.add_subplot(gs[0, 2])
cn_diff = [prompt_diff[model]['CN_Jaccard'] for model in model_keys]
en_diff = [prompt_diff[model]['EN_Jaccard'] for model in model_keys]
ax3.bar(x - width/2, cn_diff, width, label='Chinese', color='#C44E52')
ax3.bar(x + width/2, en_diff, width, label='English', color='#8172B2')
ax3.set_xlabel('Model')
ax3.set_ylabel('Jaccard Diff (Param - Verb)')
ax3.set_title('Jaccard: Prompt Effect', fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(model_names, rotation=45, ha='right', fontsize=8)
ax3.legend(fontsize=8)
ax3.axhline(y=0, color='black', linewidth=0.8)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

# 中文条件下格式三角热力图
ax4 = fig.add_subplot(gs[1, 0])
colormap_param = plt.cm.Blues
colormap_verb = plt.cm.Greens
draw_split_heatmap(ax4, corr_param_cn, corr_verb_cn,
                   'Chinese: Prompt Effect', 'Param (Blue)', 'Verb (Green)',
                   colormap_param, colormap_verb)

# 英文条件下格式三角热力图
ax5 = fig.add_subplot(gs[1, 1])
draw_split_heatmap(ax5, corr_param_en, corr_verb_en,
                   'English: Prompt Effect', 'Param (Blue)', 'Verb (Green)',
                   colormap_param, colormap_verb)

# 删除第三个子图位置
ax6 = fig.add_subplot(gs[1, 2])
ax6.axis('off')

plt.suptitle('Prompt Format Effect: Param vs Verb', fontsize=14, fontweight='bold', y=1.02)
plt.savefig('D:/task/科研/LLM-evaluation/具神认知/embodied-verb-cognition/presentation/figures/fig3_prompt_effect.png', dpi=300, bbox_inches='tight')
plt.close()
print('Saved: fig3_prompt_effect.png')

print('\nDone!')
