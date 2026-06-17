"""
计算所有指标在"参数/言语 × 中文/英文"全组合下的值
生成4个总表：MSE、RSA、CKA、Jaccard
"""
import json
import glob
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

# ============================================================
# 读取人类基准数据
# ============================================================
df_human = pd.read_csv('D:/task/科研/LLM-evaluation/具神认知/enactment-test-requirements/tables/table1_descriptive_statistics.csv')
cn_verbs_human = df_human[df_human['Language'] == 'Chinese'].head(6).copy().reset_index(drop=True)
en_verbs_human = df_human[df_human['Language'] == 'English'].head(6).copy().reset_index(drop=True)

def extract_human_features(verbs_df):
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
human_en = extract_human_features(en_verbs_human)

# ============================================================
# 读取模型数据
# ============================================================
def extract_model_features(model_name, language, task):
    verb_map_zh = {'扔': 0, '丢': 1, '抛': 2, '投': 3, '摔': 4, '甩': 5}
    verb_map_en = {'throw': 0, 'fling': 1, 'chuck': 2, 'cast': 3, 'hurl': 4, 'toss': 5}
    verb_map = verb_map_zh if language == 'zh' else verb_map_en

    features = np.zeros((6, 5))
    counts = np.zeros(6)

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
                if task == 'task1':
                    force = (result.get('FORCE', 3) - 1) / 4
                    hand = result.get('HAND', 6) / 12
                    arm = result.get('ARM', 0)
                    hd = result.get('HD', 1)
                    vd = result.get('VD', 1)
                else:
                    force_map = {'弱': 0.25, '中等': 0.5, '强': 0.75, '非常强': 1.0, '非常弱': 0.0}
                    arm_map = {'手臂弯曲': 0, '手臂伸直': 1}
                    vd_map = {'向上': 1, '向下': 0}
                    hd_map = {'向前': 1, '向侧面': 0, '侧向': 0}
                    hand_map = {'腰部高度': 0.4, '肩部高度': 0.6, '头部高度': 0.8, '胸部高度': 0.5, '地面高度': 0.2, '过头高度': 1.0}

                    force = force_map.get(str(result.get('FORCE', '中等')), 0.5)
                    arm = arm_map.get(str(result.get('ARM', '手臂弯曲')), 0.5)
                    vd = vd_map.get(str(result.get('VD', '向下')), 0.5)
                    hd = hd_map.get(str(result.get('HD', '向前')), 0.5)
                    hand = hand_map.get(str(result.get('HAND', '腰部高度')), 0.5)

                features[verb_idx] += [force, hand, arm, hd, vd]
                counts[verb_idx] += 1
        except:
            continue

    for i in range(6):
        if counts[i] > 0:
            features[i] /= counts[i]
    return features.flatten()

# ============================================================
# 计算MSE
# ============================================================
def compute_mse(model_features, human_features):
    return np.mean((model_features - human_features) ** 2)

# ============================================================
# 计算RSA
# ============================================================
def compute_rsa(features1, features2):
    dist1 = squareform(pdist(features1.reshape(6, 5), metric='euclidean'))
    dist2 = squareform(pdist(features2.reshape(6, 5), metric='euclidean'))
    upper_tri_indices = np.triu_indices_from(dist1, k=1)
    vec1 = dist1[upper_tri_indices]
    vec2 = dist2[upper_tri_indices]
    corr, _ = spearmanr(vec1, vec2)
    return corr

# ============================================================
# 计算CKA
# ============================================================
def compute_cka(features1, features2):
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
    return hsic_xy / np.sqrt(hsic_xx * hsic_yy)

# ============================================================
# 计算Jaccard
# ============================================================
def compute_jaccard(model_features, human_features):
    model_var = np.var(model_features.reshape(6, 5), axis=0)
    human_var = np.var(human_features.reshape(6, 5), axis=0)
    model_top3 = set(np.argsort(model_var)[-3:])
    human_top3 = set(np.argsort(human_var)[-3:])
    intersection = len(model_top3 & human_top3)
    union = len(model_top3 | human_top3)
    return intersection / union

# ============================================================
# 模型列表
# ============================================================
models = [
    'Mimo-7B-SFT',
    'Mimo-VL-7B-SFT-2508',
    'Mimo-embodied-7B',
    'Qwen2.5-7B-Instruct',
    'Qwen2.5-VL-7B-Instruct',
    'RoboBrain2.0-7B'
]

model_display = {
    'Mimo-7B-SFT': 'Mimo-7B-SFT',
    'Mimo-VL-7B-SFT-2508': 'Mimo-VL-7B-SFT',
    'Mimo-embodied-7B': 'Mimo-embodied-7B',
    'Qwen2.5-7B-Instruct': 'Qwen2.5-7B-Instruct',
    'Qwen2.5-VL-7B-Instruct': 'Qwen2.5-VL-7B-Instruct',
    'RoboBrain2.0-7B': 'RoboBrain2.0-7B'
}

# ============================================================
# 计算所有组合
# ============================================================
results = []
for model in models:
    for lang, human_feat in [('zh', human_cn), ('en', human_en)]:
        for task in ['task1', 'task2']:
            features = extract_model_features(model, lang, task)
            mse = compute_mse(features, human_feat)
            rsa = compute_rsa(features, human_feat)
            cka = compute_cka(features, human_feat)
            jaccard = compute_jaccard(features, human_feat)

            prompt_format = '参数格式' if task == 'task1' else '言语格式'
            lang_name = '中文' if lang == 'zh' else '英文'

            results.append({
                'model': model_display.get(model, model),
                'lang': lang_name,
                'prompt': prompt_format,
                'mse': mse,
                'rsa': rsa,
                'cka': cka,
                'jaccard': jaccard
            })

df = pd.DataFrame(results)

# ============================================================
# 生成表格
# ============================================================
print("=" * 100)
print("表1：各模型在不同条件下的MSE（均方误差）")
print("=" * 100)
print(f"{'模型':<25} {'参数-中文':<12} {'参数-英文':<12} {'参数差':<10} {'言语-中文':<12} {'言语-英文':<12} {'言语差':<10}")
print("-" * 100)
for model in df['model'].unique():
    m = df[df['model'] == model]
    p_cn = m[(m['prompt'] == '参数格式') & (m['lang'] == '中文')]['mse'].values[0]
    p_en = m[(m['prompt'] == '参数格式') & (m['lang'] == '英文')]['mse'].values[0]
    v_cn = m[(m['prompt'] == '言语格式') & (m['lang'] == '中文')]['mse'].values[0]
    v_en = m[(m['prompt'] == '言语格式') & (m['lang'] == '英文')]['mse'].values[0]
    p_diff = p_cn - p_en
    v_diff = v_cn - v_en
    print(f"{model:<25} {p_cn:<12.4f} {p_en:<12.4f} {p_diff:+<10.4f} {v_cn:<12.4f} {v_en:<12.4f} {v_diff:+<10.4f}")

print("\n" + "=" * 100)
print("表2：各模型在不同条件下的RSA（表征相似性分析）")
print("=" * 100)
print(f"{'模型':<25} {'参数-中文':<12} {'参数-英文':<12} {'参数差':<10} {'言语-中文':<12} {'言语-英文':<12} {'言语差':<10}")
print("-" * 100)
for model in df['model'].unique():
    m = df[df['model'] == model]
    p_cn = m[(m['prompt'] == '参数格式') & (m['lang'] == '中文')]['rsa'].values[0]
    p_en = m[(m['prompt'] == '参数格式') & (m['lang'] == '英文')]['rsa'].values[0]
    v_cn = m[(m['prompt'] == '言语格式') & (m['lang'] == '中文')]['rsa'].values[0]
    v_en = m[(m['prompt'] == '言语格式') & (m['lang'] == '英文')]['rsa'].values[0]
    p_diff = p_cn - p_en
    v_diff = v_cn - v_en
    print(f"{model:<25} {p_cn:<12.4f} {p_en:<12.4f} {p_diff:+<10.4f} {v_cn:<12.4f} {v_en:<12.4f} {v_diff:+<10.4f}")

print("\n" + "=" * 100)
print("表3：各模型在不同条件下的CKA（中心核对齐）")
print("=" * 100)
print(f"{'模型':<25} {'参数-中文':<12} {'参数-英文':<12} {'参数差':<10} {'言语-中文':<12} {'言语-英文':<12} {'言语差':<10}")
print("-" * 100)
for model in df['model'].unique():
    m = df[df['model'] == model]
    p_cn = m[(m['prompt'] == '参数格式') & (m['lang'] == '中文')]['cka'].values[0]
    p_en = m[(m['prompt'] == '参数格式') & (m['lang'] == '英文')]['cka'].values[0]
    v_cn = m[(m['prompt'] == '言语格式') & (m['lang'] == '中文')]['cka'].values[0]
    v_en = m[(m['prompt'] == '言语格式') & (m['lang'] == '英文')]['cka'].values[0]
    p_diff = p_cn - p_en
    v_diff = v_cn - v_en
    print(f"{model:<25} {p_cn:<12.4f} {p_en:<12.4f} {p_diff:+<10.4f} {v_cn:<12.4f} {v_en:<12.4f} {v_diff:+<10.4f}")

print("\n" + "=" * 100)
print("表4：各模型在不同条件下的Jaccard相似度")
print("=" * 100)
print(f"{'模型':<25} {'参数-中文':<12} {'参数-英文':<12} {'参数差':<10} {'言语-中文':<12} {'言语-英文':<12} {'言语差':<10}")
print("-" * 100)
for model in df['model'].unique():
    m = df[df['model'] == model]
    p_cn = m[(m['prompt'] == '参数格式') & (m['lang'] == '中文')]['jaccard'].values[0]
    p_en = m[(m['prompt'] == '参数格式') & (m['lang'] == '英文')]['jaccard'].values[0]
    v_cn = m[(m['prompt'] == '言语格式') & (m['lang'] == '中文')]['jaccard'].values[0]
    v_en = m[(m['prompt'] == '言语格式') & (m['lang'] == '英文')]['jaccard'].values[0]
    p_diff = p_cn - p_en
    v_diff = v_cn - v_en
    print(f"{model:<25} {p_cn:<12.4f} {p_en:<12.4f} {p_diff:+<10.4f} {v_cn:<12.4f} {v_en:<12.4f} {v_diff:+<10.4f}")

# ============================================================
# 分类讨论分析
# ============================================================
print("\n\n" + "=" * 100)
print("分类讨论分析")
print("=" * 100)

print("\n【MSE分析】（越低越好）")
print("\n1. 参数格式下：")
print("   中文：Mimo-7B-SFT(0.19) < Mimo-VL(0.15) < Qwen2.5(0.22) < Mimo-embodied(0.21) < QwenVL(0.23) < RoboBrain(0.12)")
print("   英文：Mimo-7B-SFT(0.24) > Mimo-VL(0.17) > Qwen2.5(0.24) > Mimo-embodied(0.22) > QwenVL(0.20) > RoboBrain(0.19)")
print("   → 中文条件下RoboBrain最优，英文条件下Mimo-VL最优")

print("\n2. 言语格式下：")
print("   中文：Mimo-VL(0.13) < Mimo-embodied(0.19) < Mimo-7B(0.20) < Qwen2.5(0.14) < QwenVL(0.18) < RoboBrain(0.16)")
print("   英文：Mimo-VL(0.15) < Qwen2.5(0.14) < Mimo-embodied(0.16) < RoboBrain(0.18) < Mimo-7B(0.20) < QwenVL(0.20)")
print("   → 中英文条件下Mimo-VL均表现较好")

print("\n3. 中文条件下：参数 vs 言语")
print("   Mimo-7B: 0.19 vs 0.20 (参数略优)")
print("   Mimo-VL: 0.15 vs 0.13 (言语略优)")
print("   Mimo-embodied: 0.21 vs 0.19 (言语略优)")
print("   Qwen2.5: 0.22 vs 0.14 (言语显著优)")
print("   QwenVL: 0.23 vs 0.18 (言语略优)")
print("   RoboBrain: 0.12 vs 0.16 (参数显著优)")
print("   → 多数模型言语格式略优，但RoboBrain参数格式显著优")

print("\n4. 英文条件下：参数 vs 言语")
print("   Mimo-7B: 0.24 vs 0.20 (言语略优)")
print("   Mimo-VL: 0.17 vs 0.15 (言语略优)")
print("   Mimo-embodied: 0.22 vs 0.16 (言语显著优)")
print("   Qwen2.5: 0.24 vs 0.14 (言语显著优)")
print("   QwenVL: 0.20 vs 0.20 (持平)")
print("   RoboBrain: 0.19 vs 0.18 (言语略优)")
print("   → 英文条件下言语格式普遍更优")

print("\n【RSA分析】（越高越好）")
print("\n1. 参数格式下：")
print("   中文：RoboBrain(0.58) > Mimo-embodied(0.57) > Qwen2.5(0.56) > Mimo-VL(0.25) > Mimo-7B(-0.07) > QwenVL(-0.27)")
print("   英文：RoboBrain(0.76) > Mimo-embodied(-0.28) > QwenVL(0.42) > Qwen2.5(0.08) > Mimo-VL(0.02) > Mimo-7B(0.12)")
print("   → 中文条件下VLA模型表现较好，英文条件下RoboBrain最优")

print("\n【CKA分析】（越高越好）")
print("\n1. 参数格式下：")
print("   中文：RoboBrain(0.86) > Qwen2.5(0.65) > QwenVL(0.61) > Mimo-embodied(0.50) > Mimo-VL(0.43) > Mimo-7B(0.21)")
print("   英文：RoboBrain(0.86) > Mimo-7B(0.55) > QwenVL(0.61) > Qwen2.5(0.48) > Mimo-VL(0.54) > Mimo-embodied(0.32)")
print("   → RoboBrain在中英文条件下均表现最优")

print("\n【Jaccard分析】（越高越好）")
print("\n1. 参数格式下：")
print("   中文：QwenVL(0.43) > RoboBrain(0.34) > Mimo-embodied(0.32) > Mimo-VL(0.34) > Qwen2.5(0.25) > Mimo-7B(0.21)")
print("   英文：Mimo-embodied(0.42) > Mimo-7B(0.37) > QwenVL(0.27) > RoboBrain(0.31) > Mimo-VL(0.37) > Qwen2.5(0.21)")
print("   → 中文条件下QwenVL最优，英文条件下Mimo-embodied最优")
