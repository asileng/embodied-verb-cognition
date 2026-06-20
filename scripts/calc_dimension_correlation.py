"""
计算每个模型每个维度与人类每个维度之间的Pearson相关系数
"""
import json
import numpy as np
from scipy.stats import pearsonr
import pandas as pd

# 人类基准数据（6个动词 × 5个维度）
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

# 归一化范围
norm_ranges = {
    'FORCE': (1, 5),
    'HAND': (0, 12),
    'ARM': (0, 1),
    'HD': (0, 1),
    'VD': (0, 1)
}

# 读取模型数据
def read_model_data(model_name, lang, task):
    """读取模型实验数据"""
    import glob
    import os

    base_path = 'D:/task/科研/LLM-evaluation/具神认知/enactment-test-requirements/pilot_results'
    task_dir = f'{task}_{lang}'

    # 模型名称映射
    model_dir_map = {
        'Mimo-VL-7B-SFT': 'Mimo-VL-7B-SFT-2508'
    }
    dir_name = model_dir_map.get(model_name, model_name)

    files = glob.glob(f'{base_path}/{dir_name}/{task_dir}/*.json')

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
    """归一化到0-1"""
    return (value - low) / (high - low) if high > low else 0.5

def get_feature_matrix(data_dict, lang):
    """获取特征矩阵（6个动词 × 5个维度）"""
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
    """获取人类特征矩阵"""
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

# 计算Pearson相关
def compute_pearson_correlation(model_matrix, human_matrix):
    """计算模型与人类每个维度之间的Pearson相关"""
    n_dims = 5
    corr_matrix = np.zeros((n_dims, n_dims))
    p_matrix = np.zeros((n_dims, n_dims))

    for i in range(n_dims):
        for j in range(n_dims):
            corr, p = pearsonr(model_matrix[:, i], human_matrix[:, j])
            corr_matrix[i, j] = corr
            p_matrix[i, j] = p

    return corr_matrix, p_matrix

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

dims = ['FORCE', 'HAND', 'ARM', 'HD', 'VD']

# 计算并输出结果
print("="*80)
print("每个模型每个维度与人类每个维度之间的Pearson相关系数")
print("="*80)

results = {}
for lang, lang_name in [('zh', 'Chinese'), ('en', 'English')]:
    print(f"\n{'='*80}")
    print(f"{lang_name} Condition")
    print(f"{'='*80}")

    human_matrix = get_human_matrix(lang)

    for model in models:
        model_data = read_model_data(model, lang, 'task1')
        model_matrix = get_feature_matrix(model_data, lang)

        corr_matrix, p_matrix = compute_pearson_correlation(model_matrix, human_matrix)

        display_name = model_display.get(model, model)
        print(f"\n{display_name}:")
        print(f"{'':8}", end='')
        for d in dims:
            print(f"{d:>8}", end='')
        print()

        for i, d in enumerate(dims):
            print(f"{d:8}", end='')
            for j in range(5):
                corr = corr_matrix[i, j]
                p = p_matrix[i, j]
                sig = '*' if p < 0.05 else ' '
                print(f"{corr:>7.3f}{sig}", end='')
            print()

        results[f"{model}_{lang}"] = corr_matrix

# 保存为CSV
print("\n\n" + "="*80)
print("保存为CSV格式")
print("="*80)

csv_rows = []
for lang, lang_name in [('zh', 'Chinese'), ('en', 'English')]:
    human_matrix = get_human_matrix(lang)
    for model in models:
        model_data = read_model_data(model, lang, 'task1')
        model_matrix = get_feature_matrix(model_data, lang)
        corr_matrix, p_matrix = compute_pearson_correlation(model_matrix, human_matrix)

        display_name = model_display.get(model, model)
        for i, model_dim in enumerate(dims):
            for j, human_dim in enumerate(dims):
                csv_rows.append({
                    'Model': display_name,
                    'Language': lang_name,
                    'Model_Dimension': model_dim,
                    'Human_Dimension': human_dim,
                    'Pearson_r': corr_matrix[i, j],
                    'P_value': p_matrix[i, j],
                    'Significant': '*' if p_matrix[i, j] < 0.05 else ''
                })

df = pd.DataFrame(csv_rows)
df.to_csv('D:/task/科研/LLM-evaluation/具神认知/embodied-verb-cognition/verified_data/dimension_correlation.csv', index=False)
print("\nSaved: dimension_correlation.csv")

# 输出汇总表
print("\n\n" + "="*80)
print("汇总表：每个模型的对角线相关（模型维度i与人类维度i的相关）")
print("="*80)

summary_rows = []
for lang, lang_name in [('zh', 'Chinese'), ('en', 'English')]:
    human_matrix = get_human_matrix(lang)
    for model in models:
        model_data = read_model_data(model, lang, 'task1')
        model_matrix = get_feature_matrix(model_data, lang)
        corr_matrix, p_matrix = compute_pearson_correlation(model_matrix, human_matrix)

        display_name = model_display.get(model, model)
        diag_corrs = [corr_matrix[i, i] for i in range(5)]
        avg_corr = np.mean(diag_corrs)

        summary_rows.append({
            'Model': display_name,
            'Language': lang_name,
            'FORCE': f"{corr_matrix[0, 0]:.3f}",
            'HAND': f"{corr_matrix[1, 1]:.3f}",
            'ARM': f"{corr_matrix[2, 2]:.3f}",
            'HD': f"{corr_matrix[3, 3]:.3f}",
            'VD': f"{corr_matrix[4, 4]:.3f}",
            'Average': f"{avg_corr:.3f}"
        })

summary_df = pd.DataFrame(summary_rows)
print(summary_df.to_string(index=False))
