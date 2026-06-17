"""
基于verified_data计算所有模型在所有条件下的MSE、RSA、CKA、Jaccard
"""
import json
import glob
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
import os

# ============================================================
# 人类基准数据（Gao 2016）
# ============================================================
human_baseline = {
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

# task2言语到数值的映射规则（来自实验提示词）
task2_mapping_zh = {
    'FORCE': {'非常强': 5, '强': 4, '中等': 3, '弱': 2, '非常弱': 1},
    'ARM': {'手臂伸直': 1, '手臂弯曲': 0},
    'HAND': {'接近地面': 0, '膝盖高度': 2, '腰部高度': 5, '胸部高度': 7, '肩部高度': 9, '头部高度': 10, '高于头部': 12},
    'VD': {'向下': 1, '向上': 0},
    'HD': {'向前': 1, '向侧方': 0}
}

task2_mapping_en = {
    'FORCE': {'very strong': 5, 'strong': 4, 'moderate': 3, 'weak': 2, 'very weak': 1},
    'ARM': {'arm extended': 1, 'arm bent': 0},
    'HAND': {'near ground': 0, 'knee height': 2, 'waist height': 5, 'chest height': 7, 'shoulder height': 9, 'head height': 10, 'above head': 12},
    'VD': {'downward': 1, 'upward': 0},
    'HD': {'forward': 1, 'sideways': 0}
}

# ============================================================
# 读取模型数据
# ============================================================
def read_model_data(model_name, lang, task):
    """读取模型实验数据"""
    base_path = os.path.dirname(os.path.abspath(__file__))
    task_dir = f'{task}_{lang}'

    # 模型名称映射
    model_dir_map = {
        'Mimo-VL-7B-SFT': 'Mimo-VL-7B-SFT-2508'
    }
    dir_name = model_dir_map.get(model_name, model_name)

    files = glob.glob(f'{base_path}/{dir_name}/{task_dir}/*.json')

    verb_names = list(human_baseline[lang].keys())
    data_dict = {}

    for f in files:
        with open(f, 'r', encoding='utf-8') as fh:
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

def convert_task2_to数值(result, lang):
    """将task2的言语描述转换为数值"""
    task2_mapping = task2_mapping_zh if lang == 'zh' else task2_mapping_en
    converted = {}
    for param in ['FORCE', 'ARM', 'HAND', 'VD', 'HD']:
        val = result.get(param, '')
        if param in task2_mapping and val in task2_mapping[param]:
            converted[param] = task2_mapping[param][val]
        else:
            # 默认值
            defaults = {'FORCE': 3, 'ARM': 0, 'HAND': 6, 'VD': 0, 'HD': 0}
            converted[param] = defaults.get(param, 0)
    return converted

def get_feature_vector(verbs_data, verb_names, lang):
    """从动词数据提取特征向量"""
    features = []
    for verb in verb_names:
        if verb in verbs_data:
            result = verbs_data[verb]
            # 检查是否需要转换（task2的言语格式）
            if isinstance(result.get('FORCE'), str):
                result = convert_task2_to数值(result, lang)

            force = normalize(result.get('FORCE', 3), 'FORCE')
            hand = normalize(result.get('HAND', 6), 'HAND')
            arm = result.get('ARM', 0)
            hd = result.get('HD', 0)
            vd = result.get('VD', 0)
            features.extend([force, hand, arm, hd, vd])
        else:
            features.extend([0.5, 0.5, 0.5, 0.5, 0.5])
    return np.array(features)

def get_human_feature_vector(lang):
    """获取人类基准特征向量"""
    verb_names = list(human_baseline[lang].keys())
    features = []
    for verb in verb_names:
        human = human_baseline[lang][verb]
        force = normalize(human['FORCE'], 'FORCE')
        hand = normalize(human['HAND'], 'HAND')
        arm = human['ARM']
        hd = human['HD']
        vd = human['VD']
        features.extend([force, hand, arm, hd, vd])
    return np.array(features)

# ============================================================
# 计算指标
# ============================================================
def compute_mse(model_features, human_features):
    """计算MSE"""
    return np.mean((model_features - human_features) ** 2)

def compute_rsa(features1, features2):
    """计算RSA"""
    dist1 = squareform(pdist(features1.reshape(6, 5), metric='euclidean'))
    dist2 = squareform(pdist(features2.reshape(6, 5), metric='euclidean'))
    upper_tri_indices = np.triu_indices_from(dist1, k=1)
    vec1 = dist1[upper_tri_indices]
    vec2 = dist2[upper_tri_indices]
    corr, _ = spearmanr(vec1, vec2)
    return corr

def compute_cka(features1, features2):
    """计算CKA"""
    X = features1.reshape(6, 5)
    Y = features2.reshape(6, 5)
    X_c = X - X.mean(axis=0)
    Y_c = Y - Y.mean(axis=0)
    K = X_c @ X_c.T
    L = Y_c @ Y_c.T
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    HKH = H @ K @ H
    HLH = H @ L @ H
    hsic_xy = np.trace(HKH @ HLH) / (n - 1) ** 2
    hsic_xx = np.trace(HKH @ HKH) / (n - 1) ** 2
    hsic_yy = np.trace(HLH @ HLH) / (n - 1) ** 2
    return hsic_xy / np.sqrt(hsic_xx * hsic_yy) if hsic_xx * hsic_yy > 0 else 0

def compute_jaccard(model_features, human_features, model_data, human_baseline, lang):
    """计算Jaccard相似度"""
    verb_names = list(human_baseline[lang].keys())

    # 计算每个动词对在每个维度上的差异
    model_diffs = np.zeros((6, 5))
    human_diffs = np.zeros((6, 5))

    for i, v1 in enumerate(verb_names):
        for j, v2 in enumerate(verb_names):
            if i < j:
                if v1 in model_data and v2 in model_data:
                    m1 = model_features[i*5:(i+1)*5]
                    m2 = model_features[j*5:(j+1)*5]
                    model_diffs[i] += np.abs(m1 - m2)
                    model_diffs[j] += np.abs(m1 - m2)

                h1 = human_features[i*5:(i+1)*5]
                h2 = human_features[j*5:(j+1)*5]
                human_diffs[i] += np.abs(h1 - h2)
                human_diffs[j] += np.abs(h1 - h2)

    # 找出每个数据点认为重要的维度（差异>均值）
    model_important = model_diffs > np.mean(model_diffs, axis=0)
    human_important = human_diffs > np.mean(human_diffs, axis=0)

    # 计算Jaccard
    intersection = np.sum(model_important & human_important)
    union = np.sum(model_important | human_important)

    return intersection / union if union > 0 else 0

# ============================================================
# 主程序
# ============================================================
models = ['Mimo-7B-SFT', 'Mimo-VL-7B-SFT', 'Mimo-embodied-7B',
          'Qwen2.5-7B-Instruct', 'Qwen2.5-VL-7B-Instruct', 'RoboBrain2.0-7B']

model_dir_names = {
    'Mimo-VL-7B-SFT': 'Mimo-VL-7B-SFT-2508'
}

results = []

for model in models:
    dir_name = model_dir_names.get(model, model)
    print(f"\n{'='*60}")
    print(f"Processing: {model}")
    print(f"{'='*60}")

    for lang in ['zh', 'en']:
        human_feat = get_human_feature_vector(lang)

        for task in ['task1', 'task2']:
            prompt_format = '参数格式' if task == 'task1' else '言语格式'
            lang_name = '中文' if lang == 'zh' else '英文'

            # 读取模型数据
            model_data = read_model_data(dir_name, lang, task)

            if not model_data:
                print(f"  {prompt_format}-{lang_name}: No data")
                continue

            # 获取特征向量
            verb_names = list(human_baseline[lang].keys())
            model_feat = get_feature_vector(model_data, verb_names, lang)

            # 计算指标
            mse = compute_mse(model_feat, human_feat)
            rsa = compute_rsa(model_feat, human_feat)
            cka = compute_cka(model_feat, human_feat)
            jaccard = compute_jaccard(model_feat, human_feat, model_data, human_baseline, lang)

            results.append({
                'model': model,
                'lang': lang_name,
                'prompt': prompt_format,
                'mse': mse,
                'rsa': rsa,
                'cka': cka,
                'jaccard': jaccard
            })

            print(f"  {prompt_format}-{lang_name}: MSE={mse:.4f}, RSA={rsa:.4f}, CKA={cka:.4f}, Jaccard={jaccard:.4f}")

# ============================================================
# 输出结果
# ============================================================
df = pd.DataFrame(results)

print("\n\n" + "="*100)
print("汇总表格")
print("="*100)

for metric in ['mse', 'rsa', 'cka', 'jaccard']:
    print(f"\n{'='*100}")
    print(f"{metric.upper()}数据")
    print(f"{'='*100}")
    print(f"{'模型':<25} {'参数-中文':<12} {'参数-英文':<12} {'参数差':<10} {'言语-中文':<12} {'言语-英文':<12} {'言语差':<10}")
    print("-"*100)

    for model in models:
        m = df[df['model'] == model]

        p_cn = m[(m['prompt'] == '参数格式') & (m['lang'] == '中文')][metric].values
        p_en = m[(m['prompt'] == '参数格式') & (m['lang'] == '英文')][metric].values
        v_cn = m[(m['prompt'] == '言语格式') & (m['lang'] == '中文')][metric].values
        v_en = m[(m['prompt'] == '言语格式') & (m['lang'] == '英文')][metric].values

        p_cn = p_cn[0] if len(p_cn) > 0 else None
        p_en = p_en[0] if len(p_en) > 0 else None
        v_cn = v_cn[0] if len(v_cn) > 0 else None
        v_en = v_en[0] if len(v_en) > 0 else None

        p_cn_str = f"{p_cn:<12.4f}" if p_cn is not None else f"{'N/A':<12}"
        p_en_str = f"{p_en:<12.4f}" if p_en is not None else f"{'N/A':<12}"
        v_cn_str = f"{v_cn:<12.4f}" if v_cn is not None else f"{'N/A':<12}"
        v_en_str = f"{v_en:<12.4f}" if v_en is not None else f"{'N/A':<12}"

        p_diff = f"{p_cn - p_en:+<10.4f}" if p_cn is not None and p_en is not None else f"{'N/A':<10}"
        v_diff = f"{v_cn - v_en:+<10.4f}" if v_cn is not None and v_en is not None else f"{'N/A':<10}"

        print(f"{model:<25} {p_cn_str} {p_en_str} {p_diff} {v_cn_str} {v_en_str} {v_diff}")

# 保存为CSV
df.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'all_metrics.csv'), index=False, encoding='utf-8-sig')
print(f"\n结果已保存到 all_metrics.csv")
