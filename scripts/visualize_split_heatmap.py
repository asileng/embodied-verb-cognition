"""
对角线分割热力图
- 语言影响图：参数格式中文相关性（红色）vs 英文相关性（紫色）
- 格式影响图：中文条件下参数格式相关性（红色）vs 言语格式相关性（紫色）
"""
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from scipy.stats import pearsonr
import json
import glob

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.8

dims = ['FORCE', 'HAND', 'ARM', 'HD', 'VD']
model_names = ['Mimo-7B', 'Mimo-VL', 'Mimo-emb', 'Qwen2.5', 'Qwen2.5-VL', 'RoboBrain']
model_keys = ['Mimo-7B-SFT', 'Mimo-VL-7B-SFT', 'Mimo-embodied-7B',
              'Qwen2.5-7B-Instruct', 'Qwen2.5-VL-7B-Instruct', 'RoboBrain2.0-7B']

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

def value_to_color(val, colormap, vmin=-1, vmax=1):
    """将值映射到颜色"""
    if np.isnan(val):
        return (0.9, 0.9, 0.9, 1.0)  # 浅灰色
    norm_val = (val - vmin) / (vmax - vmin)
    norm_val = max(0, min(1, norm_val))
    return colormap(norm_val)

def draw_split_heatmap(ax, data_low, data_high, title, label_low, label_high,
                       colormap_low, colormap_high, vmin=-1, vmax=1):
    """绘制对角线分割热力图"""
    n_rows, n_cols = data_low.shape

    for i in range(n_rows):
        for j in range(n_cols):
            # 下三角（左下）- 使用data_low
            val_low = data_low[i, j]
            color_low = value_to_color(val_low, colormap_low, vmin, vmax)
            triangle_low = plt.Polygon([[j, n_rows-1-i], [j+1, n_rows-1-i], [j, n_rows-i]],
                                       facecolor=color_low, edgecolor='white', linewidth=0.5)
            ax.add_patch(triangle_low)

            # 上三角（右上）- 使用data_high
            val_high = data_high[i, j]
            color_high = value_to_color(val_high, colormap_high, vmin, vmax)
            triangle_high = plt.Polygon([[j+1, n_rows-1-i], [j+1, n_rows-i], [j, n_rows-i]],
                                        facecolor=color_high, edgecolor='white', linewidth=0.5)
            ax.add_patch(triangle_high)

            # 添加数值标注
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

    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colormap_low(0.7), label=label_low),
                       Patch(facecolor=colormap_high(0.7), label=label_high)]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=7)

# ============================================================
# 获取相关矩阵
# ============================================================

# 参数格式
corr_param_cn = get_corr_matrix('zh', 'task1')
corr_param_en = get_corr_matrix('en', 'task1')

# 言语格式
corr_verb_cn = get_corr_matrix('zh', 'task2')
corr_verb_en = get_corr_matrix('en', 'task2')

# ============================================================
# 图2：语言影响（参数格式：中文红色 vs 英文紫色）
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左图：参数格式语言对比
ax1 = axes[0]
colormap_cn = plt.cm.Reds
colormap_en = plt.cm.Purples
draw_split_heatmap(ax1, corr_param_cn, corr_param_en,
                   'Language Effect (Param Format)', 'Chinese (Red)', 'English (Purple)',
                   colormap_cn, colormap_en)

# 右图：言语格式语言对比
ax2 = axes[1]
draw_split_heatmap(ax2, corr_verb_cn, corr_verb_en,
                   'Language Effect (Verb Format)', 'Chinese (Red)', 'English (Purple)',
                   colormap_cn, colormap_en)

plt.tight_layout()
plt.savefig('D:/task/科研/LLM-evaluation/具神认知/embodied-verb-cognition/presentation/figures/fig2_language_effect.png', dpi=300, bbox_inches='tight')
plt.close()
print('Saved: fig2_language_effect.png')

# ============================================================
# 图3：格式影响（中文：参数蓝色 vs 言语绿色）
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左图：中文条件下格式对比
ax1 = axes[0]
colormap_param = plt.cm.Blues
colormap_verb = plt.cm.Greens
draw_split_heatmap(ax1, corr_param_cn, corr_verb_cn,
                   'Prompt Effect (Chinese)', 'Param Format (Blue)', 'Verb Format (Green)',
                   colormap_param, colormap_verb)

# 右图：英文条件下格式对比
ax2 = axes[1]
draw_split_heatmap(ax2, corr_param_en, corr_verb_en,
                   'Prompt Effect (English)', 'Param Format (Blue)', 'Verb Format (Green)',
                   colormap_param, colormap_verb)

plt.tight_layout()
plt.savefig('D:/task/科研/LLM-evaluation/具神认知/embodied-verb-cognition/presentation/figures/fig3_prompt_effect.png', dpi=300, bbox_inches='tight')
plt.close()
print('Saved: fig3_prompt_effect.png')

print('\nDone!')
