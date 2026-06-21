"""
模型进化图（删除CKA，添加5参数相关性折线图）
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
import json
import glob

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.8

dims = ['FORCE', 'HAND', 'ARM', 'HD', 'VD']
stages = ['LLM', 'VLM', 'VLA']

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

# Qwen系列
qwen_models = ['Qwen2.5-7B-Instruct', 'Qwen2.5-VL-7B-Instruct', 'RoboBrain2.0-7B']
# Mimo系列
mimo_models = ['Mimo-7B-SFT', 'Mimo-VL-7B-SFT', 'Mimo-embodied-7B']

cond_styles = {
    'Param-CN': {'color': '#C44E52', 'linestyle': '-', 'marker': 'o', 'label': 'Param-CN'},
    'Param-EN': {'color': '#4C72B0', 'linestyle': '-', 'marker': 's', 'label': 'Param-EN'},
    'Verb-CN': {'color': '#C44E52', 'linestyle': '--', 'marker': '^', 'label': 'Verb-CN'},
    'Verb-EN': {'color': '#4C72B0', 'linestyle': '--', 'marker': 'D', 'label': 'Verb-EN'},
}

dim_colors = ['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3', '#FF7F00']

# ============================================================
# Qwen系列
# ============================================================

qwen_evolution = {
    'Param-CN': {'MSE': [0.225, 0.233, 0.122], 'RSA': [0.147, -0.528, 0.504], 'Jaccard': [0.154, 0.154, 0.250]},
    'Param-EN': {'MSE': [0.239, 0.201, 0.185], 'RSA': [-0.242, 0.513, 0.780], 'Jaccard': [0.000, 0.267, 0.267]},
    'Verb-CN': {'MSE': [0.150, 0.180, 0.163], 'RSA': [0.138, 0.152, -0.201], 'Jaccard': [0.077, 0.000, 0.154]},
    'Verb-EN': {'MSE': [0.071, 0.075, 0.072], 'RSA': [0.622, 0.465, -0.124], 'Jaccard': [0.182, 0.308, 0.133]},
}

qwen_corr = {}
for lang, task in [('zh', 'task1'), ('en', 'task1'), ('zh', 'task2'), ('en', 'task2')]:
    cond_key = f"{'Param' if task == 'task1' else 'Verb'}-{'CN' if lang == 'zh' else 'EN'}"
    human_features = get_human_features(lang)
    corr_values = []
    for model in qwen_models:
        model_features = get_model_features(model, lang, task)
        corrs = compute_diagonal_corr(model_features, human_features)
        corr_values.append(corrs)
    qwen_corr[cond_key] = np.array(corr_values)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, metric in enumerate(['MSE', 'RSA', 'Jaccard']):
    ax = axes[idx]
    for cond in ['Param-CN', 'Param-EN', 'Verb-CN', 'Verb-EN']:
        values = qwen_evolution[cond][metric]
        style = cond_styles[cond]
        ax.plot(stages, values, marker=style['marker'], label=style['label'],
                color=style['color'], linestyle=style['linestyle'], linewidth=2, markersize=8)
    ax.set_xlabel('Training Stage')
    ax.set_ylabel(metric)
    ax.set_title(f'Qwen Series: {metric}', fontweight='bold')
    ax.legend(fontsize=8, loc='best')
    ax.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, linestyle=':')

ax4 = axes[3]
for i, dim in enumerate(dims):
    for cond in ['Param-CN', 'Param-EN', 'Verb-CN', 'Verb-EN']:
        values = [qwen_corr[cond][j, i] for j in range(3)]
        style = cond_styles[cond]
        if i == 0:
            ax4.plot(stages, values, marker=style['marker'], label=style['label'],
                    color=dim_colors[i], linestyle=style['linestyle'], linewidth=2, markersize=6)
        else:
            ax4.plot(stages, values, marker=style['marker'],
                    color=dim_colors[i], linestyle=style['linestyle'], linewidth=2, markersize=6)
ax4.set_xlabel('Training Stage')
ax4.set_ylabel('Pearson r')
ax4.set_title('Qwen Series: Dimension Correlation', fontweight='bold')
ax4.legend(fontsize=7, loc='best', ncol=2)
ax4.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.grid(True, alpha=0.3, linestyle=':')

plt.tight_layout()
plt.savefig('D:/task/科研/LLM-evaluation/具神认知/embodied-verb-cognition/presentation/figures/fig4a_qwen_evolution.png', dpi=300, bbox_inches='tight')
plt.close()
print('Saved: fig4a_qwen_evolution.png')

# ============================================================
# Mimo系列
# ============================================================

mimo_evolution = {
    'Param-CN': {'MSE': [0.190, 0.155, 0.213], 'RSA': [-0.442, -0.178, 0.066], 'Jaccard': [0.273, 0.188, 0.333]},
    'Param-EN': {'MSE': [0.235, 0.171, 0.218], 'RSA': [-0.043, -0.132, 0.014], 'Jaccard': [0.083, 0.071, 0.214]},
    'Verb-CN': {'MSE': [0.197, 0.135, 0.189], 'RSA': [0.464, -0.027, 0.121], 'Jaccard': [0.077, 0.000, 0.214]},
    'Verb-EN': {'MSE': [0.175, 0.104, 0.113], 'RSA': [0.152, 0.076, 0.205], 'Jaccard': [0.231, 0.167, 0.167]},
}

mimo_corr = {}
for lang, task in [('zh', 'task1'), ('en', 'task1'), ('zh', 'task2'), ('en', 'task2')]:
    cond_key = f"{'Param' if task == 'task1' else 'Verb'}-{'CN' if lang == 'zh' else 'EN'}"
    human_features = get_human_features(lang)
    corr_values = []
    for model in mimo_models:
        model_features = get_model_features(model, lang, task)
        corrs = compute_diagonal_corr(model_features, human_features)
        corr_values.append(corrs)
    mimo_corr[cond_key] = np.array(corr_values)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, metric in enumerate(['MSE', 'RSA', 'Jaccard']):
    ax = axes[idx]
    for cond in ['Param-CN', 'Param-EN', 'Verb-CN', 'Verb-EN']:
        values = mimo_evolution[cond][metric]
        style = cond_styles[cond]
        ax.plot(stages, values, marker=style['marker'], label=style['label'],
                color=style['color'], linestyle=style['linestyle'], linewidth=2, markersize=8)
    ax.set_xlabel('Training Stage')
    ax.set_ylabel(metric)
    ax.set_title(f'Mimo Series: {metric}', fontweight='bold')
    ax.legend(fontsize=8, loc='best')
    ax.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, linestyle=':')

ax4 = axes[3]
for i, dim in enumerate(dims):
    for cond in ['Param-CN', 'Param-EN', 'Verb-CN', 'Verb-EN']:
        values = [mimo_corr[cond][j, i] for j in range(3)]
        style = cond_styles[cond]
        if i == 0:
            ax4.plot(stages, values, marker=style['marker'], label=style['label'],
                    color=dim_colors[i], linestyle=style['linestyle'], linewidth=2, markersize=6)
        else:
            ax4.plot(stages, values, marker=style['marker'],
                    color=dim_colors[i], linestyle=style['linestyle'], linewidth=2, markersize=6)
ax4.set_xlabel('Training Stage')
ax4.set_ylabel('Pearson r')
ax4.set_title('Mimo Series: Dimension Correlation', fontweight='bold')
ax4.legend(fontsize=7, loc='best', ncol=2)
ax4.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.grid(True, alpha=0.3, linestyle=':')

plt.tight_layout()
plt.savefig('D:/task/科研/LLM-evaluation/具神认知/embodied-verb-cognition/presentation/figures/fig4b_mimo_evolution.png', dpi=300, bbox_inches='tight')
plt.close()
print('Saved: fig4b_mimo_evolution.png')

print('\nDone!')
