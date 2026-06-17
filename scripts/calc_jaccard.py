# -*- coding: utf-8 -*-
"""
计算模型与人类的Jaccard相似度（维度选择一致性）
"""
import pandas as pd
import numpy as np
from itertools import combinations
import json
import glob
import os

# 人类数据
human_stats = pd.read_csv('D:/task/科研/LLM-evaluation/具神认知/enactment-test-requirements/tables/pairwise_statistical_chinese.csv')
human_stats_en = pd.read_csv('D:/task/科研/LLM-evaluation/具神认知/enactment-test-requirements/tables/pairwise_statistical_english.csv')

# 动词名称映射（中文）
verb_map_cn = {
    '扔': 'reng 掷',
    '丢': 'diu 丢',
    '抛': 'pao 抛',
    '投': 'tou 投',
    '摔': 'shuai 摔',
    '甩': 'shuai 甩'
}

# 动词名称映射（英文）
verb_map_en = {
    'throw': 'throw',
    'fling': 'fling',
    'chuck': 'chuck',
    'cast': 'cast',
    'hurl': 'hurl',
    'toss': 'toss'
}

verb_names_cn = ['扔', '丢', '抛', '投', '摔', '甩']
verb_names_en = ['throw', 'fling', 'chuck', 'cast', 'hurl', 'toss']
param_names = ['FORCE', 'HAND', 'ARM', 'HD', 'VD']

def get_human_significant_dims(lang='chinese'):
    """获取人类数据中每对动词的显著维度集合"""
    df = human_stats if lang == 'chinese' else human_stats_en
    verb_map = verb_map_cn if lang == 'chinese' else verb_map_en
    verb_names = verb_names_cn if lang == 'chinese' else verb_names_en
    pairs = list(combinations(range(6), 2))

    human_dims = {}
    for idx, (i, j) in enumerate(pairs):
        v1 = verb_map[verb_names[i]]
        v2 = verb_map[verb_names[j]]
        pair_data = df[(df['verb1'] == v1) & (df['verb2'] == v2)]
        sig_dims = set()
        for _, row in pair_data.iterrows():
            if row['significance'] >= 1:  # p < 0.05
                sig_dims.add(row['parameter'])
        human_dims[(i, j)] = sig_dims
    return human_dims

def read_model_data(model_name, lang='zh'):
    """读取模型数据"""
    task_dir = f'task1_{lang}'
    base_path = 'D:/task/科研/LLM-evaluation/具神认知/enactment-test-requirements/pilot_results'
    files = glob.glob(f'{base_path}/{model_name}/{task_dir}/*.json')

    verb_names = verb_names_cn if lang == 'zh' else verb_names_en
    data_dict = {}
    for f in files:
        with open(f, 'r') as fh:
            data = json.load(fh)
        if data.get('is_valid', False):
            verb = data.get('verb', '')
            result = data.get('parsed_result', {})
            if verb and result:
                data_dict[verb] = result

    features = np.zeros((6, 5))
    for i, verb in enumerate(verb_names):
        if verb in data_dict:
            r = data_dict[verb]
            features[i] = [r.get('FORCE', 0), r.get('HAND', 0), r.get('ARM', 0), r.get('HD', 0), r.get('VD', 0)]
    return features

def get_model_important_dims(features, method='threshold'):
    """获取模型数据中每对动词的重要维度集合"""
    pairs = list(combinations(range(6), 2))
    model_dims = {}

    for idx, (i, j) in enumerate(pairs):
        diffs = np.abs(features[i] - features[j])
        # 使用阈值方法：差异大于该维度均值的视为重要
        threshold = np.mean(diffs) if method == 'threshold' else np.median(diffs)
        important_dims = set()
        for p_idx, param in enumerate(param_names):
            if diffs[p_idx] > threshold:
                important_dims.add(param)
        model_dims[(i, j)] = important_dims
    return model_dims

def compute_jaccard(set_a, set_b):
    """计算Jaccard相似度"""
    if len(set_a) == 0 and len(set_b) == 0:
        return 1.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0

# 计算
models = ['Mimo-7B-SFT', 'Mimo-VL-7B-SFT', 'Mimo-embodied-7B',
          'Qwen2.5-7B-Instruct', 'Qwen2.5-VL-7B-Instruct', 'RoboBrain2.0-7B']

print("=" * 70)
print("Jaccard Similarity: Model vs Human Dimension Selection")
print("=" * 70)

results = []
for lang, lang_name in [('zh', 'Chinese'), ('en', 'English')]:
    human_dims = get_human_significant_dims('chinese' if lang == 'zh' else 'english')

    print(f"\n{lang_name} condition:")
    print(f"{'Model':<30} {'Jaccard Mean':<15} {'Jaccard SD':<15}")
    print("-" * 60)

    for model in models:
        features = read_model_data(model, lang)
        if features.sum() == 0:
            continue

        model_dims = get_model_important_dims(features)

        # 计算每对动词的Jaccard
        jaccard_scores = []
        pairs = list(combinations(range(6), 2))
        for pair in pairs:
            human_set = human_dims.get(pair, set())
            model_set = model_dims.get(pair, set())
            j = compute_jaccard(human_set, model_set)
            jaccard_scores.append(j)

        mean_j = np.mean(jaccard_scores)
        std_j = np.std(jaccard_scores)
        print(f"{model:<30} {mean_j:<15.3f} {std_j:<15.3f}")

        results.append({
            'model': model,
            'language': lang_name,
            'jaccard_mean': mean_j,
            'jaccard_std': std_j
        })

# 汇总表
print("\n" + "=" * 70)
print("Summary Table")
print("=" * 70)
print(f"{'Model':<30} {'Chinese Jaccard':<18} {'English Jaccard':<18}")
print("-" * 66)

for model in models:
    cn = [r for r in results if r['model'] == model and r['language'] == 'Chinese']
    en = [r for r in results if r['model'] == model and r['language'] == 'English']
    cn_val = f"{cn[0]['jaccard_mean']:.3f}±{cn[0]['jaccard_std']:.3f}" if cn else "N/A"
    en_val = f"{en[0]['jaccard_mean']:.3f}±{en[0]['jaccard_std']:.3f}" if en else "N/A"
    print(f"{model:<30} {cn_val:<18} {en_val:<18}")
