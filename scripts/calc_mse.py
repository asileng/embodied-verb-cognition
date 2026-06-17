# -*- coding: utf-8 -*-
"""
重新计算所有模型的MSE数据
"""
import json
import numpy as np
import glob

# 人类基准数据（Gao 2016）
human基准 = {
    'zh': {
        '投': {'FORCE': 3.21, 'HAND': 9.13, 'ARM': 1.0, 'HD': 1.0, 'VD': 0.97},
        '扔': {'FORCE': 3.05, 'HAND': 5.43, 'ARM': 0.62, 'HD': 0.72, 'VD': 0.72},
        '摔': {'FORCE': 4.55, 'HAND': 8.51, 'ARM': 1.0, 'HD': 0.86, 'VD': 1.0},
        '丢': {'FORCE': 2.33, 'HAND': 4.63, 'ARM': 0.80, 'HD': 0.57, 'VD': 0.82},
        '甩': {'FORCE': 3.74, 'HAND': 6.27, 'ARM': 0.28, 'HD': 0.31, 'VD': 0.48},
        '抛': {'FORCE': 3.16, 'HAND': 6.65, 'ARM': 0.48, 'HD': 0.97, 'VD': 1.0}
    },
    'en': {
        'throw': {'FORCE': 3.62, 'HAND': 8.58, 'ARM': 0.83, 'HD': 1.0, 'VD': 0.87},
        'fling': {'FORCE': 3.44, 'HAND': 6.41, 'ARM': 0.61, 'HD': 0.86, 'VD': 0.93},
        'chuck': {'FORCE': 3.91, 'HAND': 7.06, 'ARM': 0.66, 'HD': 0.93, 'VD': 0.72},
        'cast': {'FORCE': 3.01, 'HAND': 6.18, 'ARM': 0.52, 'HD': 0.93, 'VD': 0.97},
        'hurl': {'FORCE': 4.39, 'HAND': 8.00, 'ARM': 0.72, 'HD': 0.97, 'VD': 0.83},
        'toss': {'FORCE': 3.01, 'HAND': 4.50, 'ARM': 0.90, 'HD': 1.0, 'VD': 1.0}
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

def read_model_data(model_name, lang='zh'):
    """读取模型实验数据"""
    task_dir = f'task1_{lang}'
    base_path = 'D:/task/科研/LLM-evaluation/具神认知/enactment-test-requirements/pilot_results'

    # 模型名称到目录名称的映射
    model_dir_map = {
        'Mimo-VL-7B-SFT': 'Mimo-VL-7B-SFT-2508'
    }
    dir_name = model_dir_map.get(model_name, model_name)

    files = glob.glob(f'{base_path}/{dir_name}/{task_dir}/*.json')

    verb_names = list(human基准[lang].keys())
    data_dict = {}

    for f in files:
        with open(f, 'r') as fh:
            data = json.load(fh)
        if data.get('is_valid', False):
            verb = data.get('verb', '')
            result = data.get('parsed_result', {})
            if verb and result:
                data_dict[verb] = result

    return data_dict

def normalize(value, param):
    """归一化到0-1"""
    low, high = norm_ranges[param]
    return (value - low) / (high - low) if high > low else 0.5

def calc_mse_for_verb(model_vals, human_vals):
    """计算单个动词的MSE"""
    params = ['FORCE', 'ARM', 'HAND', 'VD', 'HD']
    mse_sum = 0
    for param in params:
        m = normalize(model_vals.get(param, 0), param)
        h = normalize(human_vals.get(param, 0), param)
        mse_sum += (m - h) ** 2
    return mse_sum / len(params)

def calc_mse_for_model(model_name, lang):
    """计算单个模型在某语言下的MSE"""
    model_data = read_model_data(model_name, lang)
    human = human基准[lang]
    verb_names = list(human.keys())

    mse_values = []
    for verb in verb_names:
        if verb in model_data and verb in human:
            mse = calc_mse_for_verb(model_data[verb], human[verb])
            mse_values.append(mse)

    if mse_values:
        return np.mean(mse_values), np.std(mse_values)
    return None, None

# 计算所有模型
models = ['Mimo-7B-SFT', 'Mimo-VL-7B-SFT', 'Mimo-embodied-7B',
          'Qwen2.5-7B-Instruct', 'Qwen2.5-VL-7B-Instruct', 'RoboBrain2.0-7B']

print("=" * 70)
print("MSE Calculation Results")
print("=" * 70)

results = []
for model in models:
    cn_mean, cn_std = calc_mse_for_model(model, 'zh')
    en_mean, en_std = calc_mse_for_model(model, 'en')

    cn_str = f"{cn_mean:.4f}" if cn_mean else "N/A"
    en_str = f"{en_mean:.4f}" if en_mean else "N/A"

    print(f"{model:<30} Chinese: {cn_str:<15} English: {en_str:<15}")

    results.append({
        'model': model,
        'zh_mse': cn_mean,
        'en_mse': en_mean
    })

print("\n" + "=" * 70)
print("Summary Table (for document)")
print("=" * 70)
print("| Model | Chinese MSE | English MSE |")
print("|-------|-------------|-------------|")
for r in results:
    cn = f"{r['zh_mse']:.4f}" if r['zh_mse'] else "N/A"
    en = f"{r['en_mse']:.4f}" if r['en_mse'] else "N/A"
    print(f"| {r['model']:<25} | {cn:<11} | {en:<11} |")
