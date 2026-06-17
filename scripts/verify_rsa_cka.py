"""
验证RSA和CKA数据的真实性
抽取Mimo-embodied-7B中文和RoboBrain2.0-7B中文两条进行验证
"""
import json
import glob
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

# 读取人类基准数据
import pandas as pd
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
print("Human CN shape:", human_cn.shape)

# 读取模型数据
def extract_model_features(model_name, language):
    """从模型JSON文件提取30维特征向量"""
    verb_map_zh = {'扔': 0, '丢': 1, '抛': 2, '投': 3, '摔': 4, '甩': 5}

    verb_map = verb_map_zh

    # 初始化特征向量
    features = np.zeros((6, 5))
    counts = np.zeros(6)

    # 读取所有JSON文件
    base_path = 'D:/task/科研/LLM-evaluation/具神认知/enactment-test-requirements/pilot_results'
    files = glob.glob(f'{base_path}/{model_name}/task1_{language}/*.json')

    print(f"\nReading {model_name} {language}: {len(files)} files")

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
                # 归一化参数
                force = (result.get('FORCE', 3) - 1) / 4
                hand = result.get('HAND', 6) / 12
                arm = result.get('ARM', 0)  # 0=bend, 1=straight
                hd = result.get('HD', 1)    # 0=sideways, 1=forward
                vd = result.get('VD', 1)    # 0=downward, 1=upward

                features[verb_idx] += [force, hand, arm, hd, vd]
                counts[verb_idx] += 1
        except Exception as e:
            print(f"Error reading {f}: {e}")
            continue

    # 计算平均值
    for i in range(6):
        if counts[i] > 0:
            features[i] /= counts[i]

    print(f"  Valid verbs: {counts}")
    return features.flatten()

# 读取Mimo和RoboBrain数据
mimo_cn = extract_model_features('Mimo-embodied-7B', 'zh')
robobrain_cn = extract_model_features('RoboBrain2.0-7B', 'zh')

print("\nMimo CN shape:", mimo_cn.shape)
print("RoboBrain CN shape:", robobrain_cn.shape)

# ============================================================
# 计算RSA
# ============================================================

def compute_rsa(features1, features2):
    """计算两个特征向量之间的RSA"""
    # 计算距离矩阵
    dist1 = squareform(pdist(features1.reshape(6, 5), metric='euclidean'))
    dist2 = squareform(pdist(features2.reshape(6, 5), metric='euclidean'))

    # 取上三角
    upper_tri_indices = np.triu_indices_from(dist1, k=1)
    vec1 = dist1[upper_tri_indices]
    vec2 = dist2[upper_tri_indices]

    # 计算Spearman相关
    corr, p_value = spearmanr(vec1, vec2)
    return corr, p_value

rsa_mimo, p_mimo = compute_rsa(human_cn, mimo_cn)
rsa_robobrain, p_robobrain = compute_rsa(human_cn, robobrain_cn)

print("\n" + "="*60)
print("RSA验证结果")
print("="*60)
print(f"RSA(Human, Mimo-embodied-7B) = {rsa_mimo:.4f} (p = {p_mimo:.4f})")
print(f"RSA(Human, RoboBrain2.0-7B) = {rsa_robobrain:.4f} (p = {p_robobrain:.4f})")

print("\n与RESEARCH_SUMMARY.md对比:")
print(f"  Mimo: 计算值 {rsa_mimo:.4f} ≈ 文档值 0.46")
print(f"  RoboBrain: 计算值 {rsa_robobrain:.4f} ≈ 文档值 0.43")

# ============================================================
# 计算CKA
# ============================================================

def compute_cka(features1, features2):
    """计算两个特征向量之间的CKA"""
    # 重塑为矩阵
    X = features1.reshape(6, 5)
    Y = features2.reshape(6, 5)

    # 中心化
    X_c = X - X.mean(axis=0)
    Y_c = Y - Y.mean(axis=0)

    # 计算核矩阵
    K = X_c @ X_c.T
    L = Y_c @ Y_c.T

    # 计算HSIC
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n

    HKH = H @ K @ H
    HLH = H @ L @ H

    hsic_xy = np.trace(HKH @ HLH) / (n - 1) ** 2
    hsic_xx = np.trace(HKH @ HKH) / (n - 1) ** 2
    hsic_yy = np.trace(HLH @ HLH) / (n - 1) ** 2

    # CKA
    cka = hsic_xy / np.sqrt(hsic_xx * hsic_yy)
    return cka

cka_mimo = compute_cka(human_cn, mimo_cn)
cka_robobrain = compute_cka(human_cn, robobrain_cn)

print("\n" + "="*60)
print("CKA验证结果")
print("="*60)
print(f"CKA(Human, Mimo-embodied-7B) = {cka_mimo:.4f}")
print(f"CKA(Human, RoboBrain2.0-7B) = {cka_robobrain:.4f}")

print("\n与RESEARCH_SUMMARY.md对比:")
print(f"  Mimo: 计算值 {cka_mimo:.4f} ≈ 文档值 0.50")
print(f"  RoboBrain: 计算值 {cka_robobrain:.4f} ≈ 文档值 0.56")

# ============================================================
# 与consistency_analysis.md对比
# ============================================================

print("\n" + "="*60)
print("与consistency_analysis.md对比")
print("="*60)
print("consistency_analysis.md中的值:")
print(f"  RSA(H, Mimo) = 0.4607")
print(f"  RSA(H, RoboBrain) = 0.4251")
print(f"  CKA(H, Mimo) = 0.4954")
print(f"  CKA(H, RoboBrain) = 0.5571")

print("\n验证结论:")
if abs(rsa_mimo - 0.4607) < 0.01 and abs(rsa_robobrain - 0.4251) < 0.01:
    print("✓ RSA数据验证通过，与原始计算一致")
else:
    print("✗ RSA数据验证失败，存在差异")

if abs(cka_mimo - 0.4954) < 0.01 and abs(cka_robobrain - 0.5571) < 0.01:
    print("✓ CKA数据验证通过，与原始计算一致")
else:
    print("✗ CKA数据验证失败，存在差异")
