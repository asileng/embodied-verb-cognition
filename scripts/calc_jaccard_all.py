# -*- coding: utf-8 -*-
"""
计算所有维度的Jaccard相似度
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

# 动词名称映射
verb_map_cn = {'扔': 'reng 掷', '丢': 'diu 丢', '抛': 'pao 抛', '投': 'tou 投', '摔': 'shuai 摔', '甩': 'shuai 甩'}
verb_map_en = {'throw': 'throw', 'fling': 'fling', 'chuck': 'chuck', 'cast': 'cast', 'hurl': 'hurl', 'toss': 'toss'}

verb_names_cn = ['扔', '丢', '抛', '投', '摔', '甩']
verb_names_en = ['throw', 'fling', 'chuck', 'cast', 'hurl', 'toss']
param_names = ['FORCE', 'HAND', 'ARM', 'HD', 'VD']

# 言语格式到数值的映射
text_to_num = {
    'FORCE': {'非常强': 5, '强': 4, '中等': 3, '弱': 2, '非常弱': 1},
    'HAND': {'接近地面': 0, '膝盖高度': 2, '腰部高度': 4, '胸部高度': 6, '肩部高度': 8, '头部高度': 10, '高于头部': 12},
    'ARM': {'手臂伸直': 1, '手臂弯曲': 0},
    'VD': {'向下': 1, '向上': 0},
    'HD': {'向前': 1, '向侧方': 0}
}

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
            if row['significance'] >= 1:
                sig_dims.add(row['parameter'])
        human_dims[(i, j)] = sig_dims
    return human_dims

def read_model_data(model_name, lang='zh', task='task1'):
    """读取模型数据"""
    # 模型名称到目录名称的映射
    model_dir_map = {
        'Mimo-VL-7B-SFT': 'Mimo-VL-7B-SFT-2508'
    }
    dir_name = model_dir_map.get(model_name, model_name)

    task_dir = f'{task}_{lang}'
    base_path = 'D:/task/科研/LLM-evaluation/具神认知/enactment-test-requirements/pilot_results'
    files = glob.glob(f'{base_path}/{dir_name}/{task_dir}/*.json')

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
            if task == 'task1':
                # 参数格式：直接是数值
                features[i] = [r.get('FORCE', 0), r.get('HAND', 0), r.get('ARM', 0), r.get('HD', 0), r.get('VD', 0)]
            else:
                # 言语格式：需要转换为数值
                features[i] = [
                    text_to_num['FORCE'].get(r.get('FORCE', '中等'), 3),
                    text_to_num['HAND'].get(r.get('HAND', '腰部高度'), 4),
                    text_to_num['ARM'].get(r.get('ARM', '手臂弯曲'), 0),
                    text_to_num['HD'].get(r.get('HD', '向前'), 1),
                    text_to_num['VD'].get(r.get('VD', '向下'), 1)
                ]
    return features

def get_model_important_dims(features):
    """获取模型数据中每对动词的重要维度集合"""
    pairs = list(combinations(range(6), 2))
    model_dims = {}

    for idx, (i, j) in enumerate(pairs):
        diffs = np.abs(features[i] - features[j])
        threshold = np.mean(diffs)
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

def calc_jaccard_for_model(model_name, lang, task='task1'):
    """计算单个模型的Jaccard相似度"""
    human_dims = get_human_significant_dims('chinese' if lang == 'zh' else 'english')
    features = read_model_data(model_name, lang, task)
    if features.sum() == 0:
        return None, None

    model_dims = get_model_important_dims(features)
    jaccard_scores = []
    pairs = list(combinations(range(6), 2))
    for pair in pairs:
        human_set = human_dims.get(pair, set())
        model_set = model_dims.get(pair, set())
        j = compute_jaccard(human_set, model_set)
        jaccard_scores.append(j)

    return np.mean(jaccard_scores), np.std(jaccard_scores)

# ============================================================
# 计算所有维度
# ============================================================

models = ['Mimo-7B-SFT', 'Mimo-VL-7B-SFT', 'Mimo-embodied-7B',
          'Qwen2.5-7B-Instruct', 'Qwen2.5-VL-7B-Instruct', 'RoboBrain2.0-7B']

print("=" * 80)
print("Jaccard Similarity Analysis")
print("=" * 80)

# 1. 语言维度
print("\n【1. 语言维度】")
print(f"{'Model':<30} {'Chinese (task1)':<20} {'English (task1)':<20}")
print("-" * 70)
for model in models:
    cn_mean, cn_std = calc_jaccard_for_model(model, 'zh', 'task1')
    en_mean, en_std = calc_jaccard_for_model(model, 'en', 'task1')
    cn_str = f"{cn_mean:.3f}±{cn_std:.3f}" if cn_mean else "N/A"
    en_str = f"{en_mean:.3f}±{en_std:.3f}" if en_mean else "N/A"
    print(f"{model:<30} {cn_str:<20} {en_str:<20}")

# 2. 提示词维度
print("\n【2. 提示词维度】")
print(f"{'Model':<30} {'Task1 (param)':<20} {'Task2 (text)':<20}")
print("-" * 70)
for model in models:
    t1_mean, t1_std = calc_jaccard_for_model(model, 'zh', 'task1')
    t2_mean, t2_std = calc_jaccard_for_model(model, 'zh', 'task2')
    t1_str = f"{t1_mean:.3f}±{t1_std:.3f}" if t1_mean else "N/A"
    t2_str = f"{t2_mean:.3f}±{t2_std:.3f}" if t2_mean else "N/A"
    print(f"{model:<30} {t1_str:<20} {t2_str:<20}")

# 3. 模型维度（Qwen系列）
print("\n【3. 模型维度 - Qwen系列】")
qwen_models = ['Qwen2.5-7B-Instruct', 'Qwen2.5-VL-7B-Instruct', 'RoboBrain2.0-7B']
print(f"{'Model':<30} {'Chinese':<20} {'English':<20}")
print("-" * 70)
for model in qwen_models:
    cn_mean, cn_std = calc_jaccard_for_model(model, 'zh', 'task1')
    en_mean, en_std = calc_jaccard_for_model(model, 'en', 'task1')
    cn_str = f"{cn_mean:.3f}±{cn_std:.3f}" if cn_mean else "N/A"
    en_str = f"{en_mean:.3f}±{en_std:.3f}" if en_mean else "N/A"
    print(f"{model:<30} {cn_str:<20} {en_str:<20}")

# 4. 模型维度（Mimo系列）
print("\n【4. 模型维度 - Mimo系列】")
mimo_models = ['Mimo-7B-SFT', 'Mimo-VL-7B-SFT', 'Mimo-embodied-7B']
print(f"{'Model':<30} {'Chinese':<20} {'English':<20}")
print("-" * 70)
for model in mimo_models:
    cn_mean, cn_std = calc_jaccard_for_model(model, 'zh', 'task1')
    en_mean, en_std = calc_jaccard_for_model(model, 'en', 'task1')
    cn_str = f"{cn_mean:.3f}±{cn_std:.3f}" if cn_mean else "N/A"
    en_str = f"{en_mean:.3f}±{en_std:.3f}" if en_mean else "N/A"
    print(f"{model:<30} {cn_str:<20} {en_str:<20}")
