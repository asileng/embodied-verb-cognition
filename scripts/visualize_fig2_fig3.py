"""
生成fig2（语言影响）和fig3（提示词影响）
"""
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr
import json
import glob

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.8

model_names = ['Mimo-7B', 'Mimo-VL', 'Mimo-emb', 'Qwen2.5', 'Qwen2.5-VL', 'RoboBrain']
model_keys = ['Mimo-7B-SFT', 'Mimo-VL-7B-SFT', 'Mimo-embodied-7B',
              'Qwen2.5-7B-Instruct', 'Qwen2.5-VL-7B-Instruct', 'RoboBrain2.0-7B']
dims = ['FORCE', 'HAND', 'ARM', 'HD', 'VD']

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

# 数据
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

# ============================================================
# 图2：语言影响
# ============================================================

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

corr_param_cn = get_corr_matrix('zh', 'task1')
corr_param_en = get_corr_matrix('en', 'task1')
corr_verb_cn = get_corr_matrix('zh', 'task2')
corr_verb_en = get_corr_matrix('en', 'task2')

fig = plt.figure(figsize=(15, 10))
gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

x = np.arange(len(model_names))
width = 0.35

# MSE
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

# RSA
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

# 参数格式相关性矩阵
ax3 = fig.add_subplot(gs[1, 0])
im = ax3.imshow(corr_param_cn, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax3.set_xticks(range(6))
ax3.set_yticks(range(5))
ax3.set_xticklabels(model_names, rotation=45, ha='right', fontsize=7)
ax3.set_yticklabels(dims, fontsize=8)
ax3.set_xlabel('Model')
ax3.set_ylabel('Dimension')
ax3.set_title('Param Format: CN Correlation', fontweight='bold')
for i in range(5):
    for j in range(6):
        val = corr_param_cn[i, j]
        if not np.isnan(val):
            color = 'white' if abs(val) > 0.5 else 'black'
            ax3.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=6, color=color)
plt.colorbar(im, ax=ax3, shrink=0.8)

# Jaccard
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
print('Saved: fig2_language_effect.png')

# ============================================================
# 图3：提示词影响
# ============================================================

prompt_diff = {}
for model in model_keys:
    prompt_diff[model] = {}
    for lang in ['CN', 'EN']:
        for metric in ['MSE', 'RSA', 'Jaccard']:
            param = data[model][f'Param-{lang}'][metric]
            verb = data[model][f'Verb-{lang}'][metric]
            prompt_diff[model][f'{lang}_{metric}'] = param - verb

fig = plt.figure(figsize=(15, 10))
gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

x = np.arange(len(model_names))
width = 0.35

# MSE
ax1 = fig.add_subplot(gs[0, 0])
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

# RSA
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

# 中文相关性矩阵
ax3 = fig.add_subplot(gs[1, 0])
im = ax3.imshow(corr_param_cn, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax3.set_xticks(range(6))
ax3.set_yticks(range(5))
ax3.set_xticklabels(model_names, rotation=45, ha='right', fontsize=7)
ax3.set_yticklabels(dims, fontsize=8)
ax3.set_xlabel('Model')
ax3.set_ylabel('Dimension')
ax3.set_title('Param-CN Correlation', fontweight='bold')
for i in range(5):
    for j in range(6):
        val = corr_param_cn[i, j]
        if not np.isnan(val):
            color = 'white' if abs(val) > 0.5 else 'black'
            ax3.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=6, color=color)
plt.colorbar(im, ax=ax3, shrink=0.8)

# Jaccard
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
print('Saved: fig3_prompt_effect.png')

print('\nDone!')
