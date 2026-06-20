"""
绘制模型-人类维度相关性热力图
每个模型：5×5矩阵（模型维度 vs 人类维度）
6个模型：6个热力图
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import json
import glob

# 人类基准数据
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
dims = ['FORCE', 'HAND', 'ARM', 'HD', 'VD']

def read_model_data(model_name, lang, task):
    base_path = 'D:/task/科研/LLM-evaluation/具神认知/enactment-test-requirements/pilot_results'
    model_dir_map = {'Mimo-VL-7B-SFT': 'Mimo-VL-7B-SFT-2508'}
    dir_name = model_dir_map.get(model_name, model_name)
    files = glob.glob(f'{base_path}/{dir_name}/{task}_{lang}/*.json')
    verb_names = list(human_data[lang].keys())
    data_dict = {}
    for f in files:
        try:
            with open(f, 'r', encoding='utf-8') as fh:
                data = json.load(fh)
            if data.get('is_valid', False):
                verb = data.get('verb', '')
                result = data.get('parsed_result', {})
                if verb and result:
                    data_dict[verb] = result
        except:
            continue
    return data_dict

def normalize(value, low, high):
    return (value - low) / (high - low) if high > low else 0.5

def get_feature_matrix(data_dict, lang):
    verb_names = list(human_data[lang].keys())
    features = []
    for verb in verb_names:
        if verb in data_dict:
            result = data_dict[verb]
            force = normalize(result.get('FORCE', 3), 1, 5)
            hand = normalize(result.get('HAND', 6), 0, 12)
            arm = result.get('ARM', 0)
            hd = result.get('HD', 0)
            vd = result.get('VD', 0)
            features.append([force, hand, arm, hd, vd])
        else:
            features.append([0.5, 0.5, 0.5, 0.5, 0.5])
    return np.array(features)

def get_human_matrix(lang):
    verb_names = list(human_data[lang].keys())
    features = []
    for verb in verb_names:
        human = human_data[lang][verb]
        force = normalize(human[0], 1, 5)
        hand = normalize(human[1], 0, 12)
        arm = human[2]
        hd = human[3]
        vd = human[4]
        features.append([force, hand, arm, hd, vd])
    return np.array(features)

def compute_correlation_matrix(model_matrix, human_matrix):
    n_dims = 5
    corr_matrix = np.zeros((n_dims, n_dims))
    for i in range(n_dims):
        for j in range(n_dims):
            if np.std(model_matrix[:, i]) == 0 or np.std(human_matrix[:, j]) == 0:
                corr_matrix[i, j] = np.nan
            else:
                corr, _ = pearsonr(model_matrix[:, i], human_matrix[:, j])
                corr_matrix[i, j] = corr
    return corr_matrix

# 模型列表
models = ['Mimo-7B-SFT', 'Mimo-VL-7B-SFT', 'Mimo-embodied-7B',
          'Qwen2.5-7B-Instruct', 'Qwen2.5-VL-7B-Instruct', 'RoboBrain2.0-7B']
model_display = {
    'Mimo-7B-SFT': 'Mimo-7B',
    'Mimo-VL-7B-SFT': 'Mimo-VL',
    'Mimo-embodied-7B': 'Mimo-emb',
    'Qwen2.5-7B-Instruct': 'Qwen2.5',
    'Qwen2.5-VL-7B-Instruct': 'Qwen2.5-VL',
    'RoboBrain2.0-7B': 'RoboBrain'
}

# ============================================================
# 中文条件热力图
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, model in enumerate(models):
    ax = axes[idx]
    model_data = read_model_data(model, 'zh', 'task1')
    model_matrix = get_feature_matrix(model_data, 'zh')
    human_matrix = get_human_matrix('zh')
    corr_matrix = compute_correlation_matrix(model_matrix, human_matrix)

    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(dims, fontsize=8)
    ax.set_yticklabels(dims, fontsize=8)
    ax.set_xlabel('Human Dimension', fontsize=9)
    ax.set_ylabel('Model Dimension', fontsize=9)
    ax.set_title(model_display.get(model, model), fontsize=11, fontweight='bold')

    # 添加数值标注
    for i in range(5):
        for j in range(5):
            val = corr_matrix[i, j]
            if not np.isnan(val):
                color = 'white' if abs(val) > 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=7, color=color)

plt.suptitle('Chinese Condition: Model-Human Dimension Correlation', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('D:/task/科研/LLM-evaluation/具神认知/embodied-verb-cognition/presentation/figures/fig6a_dimension_corr_zh.png', dpi=300, bbox_inches='tight')
plt.savefig('D:/task/科研/LLM-evaluation/具神认知/embodied-verb-cognition/presentation/figures/fig6a_dimension_corr_zh.pdf', bbox_inches='tight')
plt.close()
print("Saved: fig6a_dimension_corr_zh.png/pdf")

# ============================================================
# 英文条件热力图
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, model in enumerate(models):
    ax = axes[idx]
    model_data = read_model_data(model, 'en', 'task1')
    model_matrix = get_feature_matrix(model_data, 'en')
    human_matrix = get_human_matrix('en')
    corr_matrix = compute_correlation_matrix(model_matrix, human_matrix)

    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(dims, fontsize=8)
    ax.set_yticklabels(dims, fontsize=8)
    ax.set_xlabel('Human Dimension', fontsize=9)
    ax.set_ylabel('Model Dimension', fontsize=9)
    ax.set_title(model_display.get(model, model), fontsize=11, fontweight='bold')

    for i in range(5):
        for j in range(5):
            val = corr_matrix[i, j]
            if not np.isnan(val):
                color = 'white' if abs(val) > 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=7, color=color)

plt.suptitle('English Condition: Model-Human Dimension Correlation', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('D:/task/科研/LLM-evaluation/具神认知/embodied-verb-cognition/presentation/figures/fig6b_dimension_corr_en.png', dpi=300, bbox_inches='tight')
plt.savefig('D:/task/科研/LLM-evaluation/具神认知/embodied-verb-cognition/presentation/figures/fig6b_dimension_corr_en.pdf', bbox_inches='tight')
plt.close()
print("Saved: fig6b_dimension_corr_en.png/pdf")

print("\nDone!")
