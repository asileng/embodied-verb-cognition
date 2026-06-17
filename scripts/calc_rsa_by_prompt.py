"""
计算各模型在task1（参数格式）和task2（言语格式）下的RSA值
"""
import json
import glob
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

# 读取人类基准数据
df_human = pd.read_csv('D:/task/科研/LLM-evaluation/具神认知/enactment-test-requirements/tables/table1_descriptive_statistics.csv')
cn_verbs_human = df_human[df_human['Language'] == 'Chinese'].head(6).copy().reset_index(drop=True)

def extract_human_features(verbs_df):
    """从人类数据提取30维特征向量"""
    features = []
    for _, row in verbs_df.iterrows():
        force = (row['FORCE Mean'] - 1) / 4
        hand = row['HAND Mean'] / 12
        arm_total = row['ARM bend'] + row['ARM straight']
        arm = row['ARM straight'] / arm_total if arm_total > 0 else 0.5
        hd_total = row['HD forward'] + row['HD sidewise']
        hd = row['HD forward'] / hd_total if hd_total > 0 else 0.5
        vd_total = row['VD upward'] + row['VD downward']
        vd = row['VD upward'] / vd_total if vd_total > 0 else 0.5
        features.extend([force, hand, arm, hd, vd])
    return np.array(features)

human_cn = extract_human_features(cn_verbs_human)

# 读取模型数据
def extract_model_features(model_name, language, task):
    """从模型JSON文件提取30维特征向量"""
    verb_map_zh = {'扔': 0, '丢': 1, '抛': 2, '投': 3, '摔': 4, '甩': 5}

    verb_map = verb_map_zh

    # 初始化特征向量
    features = np.zeros((6, 5))
    counts = np.zeros(6)

    # 读取所有JSON文件
    base_path = 'D:/task/科研/LLM-evaluation/具神认知/enactment-test-requirements/pilot_results'
    files = glob.glob(f'{base_path}/{model_name}/{task}_{language}/*.json')

    for f in files:
        try:
            with open(f, 'r') as fh:
                data = json.load(fh)

            if not data.get('is_valid', False):
                continue

            verb = data.get('verb', '')
            if verb not in verb_map:
                continue

            verb_idx = verb_map[verb]
            result = data.get('parsed_result', {})

            if result:
                # task1是参数格式，task2是言语格式
                if task == 'task1':
                    # 参数格式：直接使用数值（已归一化到0-1）
                    force = (result.get('FORCE', 3) - 1) / 4
                    hand = result.get('HAND', 6) / 12
                    arm = result.get('ARM', 0)
                    hd = result.get('HD', 1)
                    vd = result.get('VD', 1)
                else:
                    # 言语格式：根据实验提示词的映射规则转换为数值
                    # FORCE: 非常强=5, 强=4, 中等=3, 弱=2, 非常弱=1
                    force_map = {'非常强': 5, '强': 4, '中等': 3, '弱': 2, '非常弱': 1}
                    # ARM: 手臂伸直=1, 手臂弯曲=0
                    arm_map = {'手臂伸直': 1, '手臂弯曲': 0}
                    # HAND: 接近地面=0, 膝盖高度=2, 腰部高度=5, 胸部高度=7, 肩部高度=9, 头部高度=10, 高于头部=12
                    hand_map = {'接近地面': 0, '膝盖高度': 2, '腰部高度': 5, '胸部高度': 7, '肩部高度': 9, '头部高度': 10, '高于头部': 12}
                    # VD: 向下=1, 向上=0
                    vd_map = {'向下': 1, '向上': 0}
                    # HD: 向前=1, 向侧方=0
                    hd_map = {'向前': 1, '向侧方': 0}

                    force_str = str(result.get('FORCE', '中等'))
                    arm_str = str(result.get('ARM', '手臂弯曲'))
                    vd_str = str(result.get('VD', '向下'))
                    hd_str = str(result.get('HD', '向前'))
                    hand_str = str(result.get('HAND', '腰部高度'))

                    # 转换为数值后归一化到0-1
                    force = (force_map.get(force_str, 3) - 1) / 4
                    arm = arm_map.get(arm_str, 0)
                    vd = vd_map.get(vd_str, 0)
                    hd = hd_map.get(hd_str, 0)
                    hand = hand_map.get(hand_str, 6) / 12

                features[verb_idx] += [force, hand, arm, hd, vd]
                counts[verb_idx] += 1
        except Exception as e:
            continue

    # 计算平均值
    for i in range(6):
        if counts[i] > 0:
            features[i] /= counts[i]

    return features.flatten()

# 计算RSA
def compute_rsa(features1, features2):
    """计算两个特征向量之间的RSA"""
    dist1 = squareform(pdist(features1.reshape(6, 5), metric='euclidean'))
    dist2 = squareform(pdist(features2.reshape(6, 5), metric='euclidean'))

    upper_tri_indices = np.triu_indices_from(dist1, k=1)
    vec1 = dist1[upper_tri_indices]
    vec2 = dist2[upper_tri_indices]

    corr, p_value = spearmanr(vec1, vec2)
    return corr, p_value

# 模型列表
models = [
    'Mimo-7B-SFT',
    'Mimo-VL-7B-SFT-2508',
    'Mimo-embodied-7B',
    'Qwen2.5-7B-Instruct',
    'Qwen2.5-VL-7B-Instruct',
    'RoboBrain2.0-7B'
]

model_display_names = {
    'Mimo-7B-SFT': 'Mimo-7B-SFT',
    'Mimo-VL-7B-SFT-2508': 'Mimo-VL-7B-SFT',
    'Mimo-embodied-7B': 'Mimo-embodied-7B',
    'Qwen2.5-7B-Instruct': 'Qwen2.5-7B-Instruct',
    'Qwen2.5-VL-7B-Instruct': 'Qwen2.5-VL-7B-Instruct',
    'RoboBrain2.0-7B': 'RoboBrain2.0-7B'
}

print("RSA by prompt format (Chinese):")
print(f"{'Model':<25} {'task1 (param)':<15} {'task2 (verbal)':<15}")
print("-" * 55)

results = []
for model in models:
    # task1 (参数格式)
    features_t1 = extract_model_features(model, 'zh', 'task1')
    rsa_t1, _ = compute_rsa(human_cn, features_t1)

    # task2 (言语格式)
    features_t2 = extract_model_features(model, 'zh', 'task2')
    rsa_t2, _ = compute_rsa(human_cn, features_t2)

    display_name = model_display_names.get(model, model)
    print(f"{display_name:<25} {rsa_t1:<15.4f} {rsa_t2:<15.4f}")
    results.append({
        'model': display_name,
        'rsa_task1': rsa_t1,
        'rsa_task2': rsa_t2,
        'rsa_change': rsa_t2 - rsa_t1
    })

print("\n\nDetailed results:")
for r in results:
    print(f"{r['model']}: task1={r['rsa_task1']:.4f}, task2={r['rsa_task2']:.4f}, change={r['rsa_change']:+.4f}")
