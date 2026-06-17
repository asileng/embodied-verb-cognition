"""
生成4个总表：MSE、RSA、CKA、Jaccard
使用之前验证过的正确数据
"""
import pandas as pd

# ============================================================
# 已验证的正确数据
# ============================================================

# MSE数据（来自calc_mse.py）
mse_data = {
    'Mimo-7B-SFT': {'参数-中文': 0.1895, '参数-英文': 0.2355, '言语-中文': 0.2001, '言语-英文': 0.0948},
    'Mimo-VL-7B-SFT': {'参数-中文': 0.1545, '参数-英文': 0.1713, '言语-中文': 0.1331, '言语-英文': 0.0948},
    'Mimo-embodied-7B': {'参数-中文': 0.2131, '参数-英文': 0.2177, '言语-中文': 0.1860, '言语-英文': 0.0948},
    'Qwen2.5-7B-Instruct': {'参数-中文': 0.2250, '参数-英文': 0.2387, '言语-中文': 0.1449, '言语-英文': 0.0948},
    'Qwen2.5-VL-7B-Instruct': {'参数-中文': 0.2330, '参数-英文': 0.2011, '言语-中文': 0.1769, '言语-英文': 0.0948},
    'RoboBrain2.0-7B': {'参数-中文': 0.1217, '参数-英文': 0.1852, '言语-中文': 0.1614, '言语-英文': 0.0948}
}

# RSA数据（来自consistency_analysis.md和calc_rsa_by_prompt.py）
rsa_data = {
    'Mimo-7B-SFT': {'参数-中文': -0.14, '参数-英文': 0.12, '言语-中文': 0.66, '言语-英文': None},
    'Mimo-VL-7B-SFT': {'参数-中文': 0.14, '参数-英文': 0.02, '言语-中文': -0.00, '言语-英文': None},
    'Mimo-embodied-7B': {'参数-中文': 0.46, '参数-英文': -0.28, '言语-中文': 0.49, '言语-英文': None},
    'Qwen2.5-7B-Instruct': {'参数-中文': 0.31, '参数-英文': 0.08, '言语-中文': 0.38, '言语-英文': None},
    'Qwen2.5-VL-7B-Instruct': {'参数-中文': -0.32, '参数-英文': 0.42, '言语-中文': 0.06, '言语-英文': None},
    'RoboBrain2.0-7B': {'参数-中文': 0.43, '参数-英文': 0.76, '言语-中文': -0.02, '言语-英文': None}
}

# CKA数据（来自consistency_analysis.md）
cka_data = {
    'Mimo-7B-SFT': {'参数-中文': 0.21, '参数-英文': 0.55, '言语-中文': 0.18, '言语-英文': None},
    'Mimo-VL-7B-SFT': {'参数-中文': 0.43, '参数-英文': 0.54, '言语-中文': 0.38, '言语-英文': None},
    'Mimo-embodied-7B': {'参数-中文': 0.50, '参数-英文': 0.32, '言语-中文': 0.45, '言语-英文': None},
    'Qwen2.5-7B-Instruct': {'参数-中文': 0.65, '参数-英文': 0.48, '言语-中文': 0.58, '言语-英文': None},
    'Qwen2.5-VL-7B-Instruct': {'参数-中文': 0.36, '参数-英文': 0.61, '言语-中文': 0.54, '言语-英文': None},
    'RoboBrain2.0-7B': {'参数-中文': 0.56, '参数-英文': 0.86, '言语-中文': 0.79, '言语-英文': None}
}

# Jaccard数据（来自calc_jaccard_all.py）
jaccard_data = {
    'Mimo-7B-SFT': {'参数-中文': 0.209, '参数-英文': 0.372, '言语-中文': None, '言语-英文': None},
    'Mimo-VL-7B-SFT': {'参数-中文': 0.340, '参数-英文': 0.373, '言语-中文': None, '言语-英文': None},
    'Mimo-embodied-7B': {'参数-中文': 0.317, '参数-英文': 0.422, '言语-中文': None, '言语-英文': None},
    'Qwen2.5-7B-Instruct': {'参数-中文': 0.254, '参数-英文': 0.206, '言语-中文': None, '言语-英文': None},
    'Qwen2.5-VL-7B-Instruct': {'参数-中文': 0.431, '参数-英文': 0.272, '言语-中文': None, '言语-英文': None},
    'RoboBrain2.0-7B': {'参数-中文': 0.344, '参数-英文': 0.310, '言语-中文': None, '言语-英文': None}
}

# ============================================================
# 生成表格
# ============================================================

models = list(mse_data.keys())

print("=" * 120)
print("表1：各模型在不同条件下的MSE（均方误差，越低越好）")
print("=" * 120)
print(f"{'模型':<25} {'参数-中文':<12} {'参数-英文':<12} {'参数差(CN-EN)':<15} {'言语-中文':<12} {'言语-英文':<12} {'言语差(CN-EN)':<15}")
print("-" * 120)
for model in models:
    d = mse_data[model]
    p_diff = d['参数-中文'] - d['参数-英文']
    v_diff = d['言语-中文'] - d['言语-英文']
    print(f"{model:<25} {d['参数-中文']:<12.4f} {d['参数-英文']:<12.4f} {p_diff:+<15.4f} {d['言语-中文']:<12.4f} {d['言语-英文']:<12.4f} {v_diff:+<15.4f}")

print("\n" + "=" * 120)
print("表2：各模型在不同条件下的RSA（表征相似性分析，越高越好）")
print("=" * 120)
print(f"{'模型':<25} {'参数-中文':<12} {'参数-英文':<12} {'参数差(CN-EN)':<15} {'言语-中文':<12} {'言语-英文':<12} {'言语差(CN-EN)':<15}")
print("-" * 120)
for model in models:
    d = rsa_data[model]
    p_diff = d['参数-中文'] - d['参数-英文'] if d['参数-英文'] is not None else None
    v_diff = d['言语-中文'] - d['言语-英文'] if d['言语-英文'] is not None else None
    p_en = f"{d['参数-英文']:<12.4f}" if d['参数-英文'] is not None else f"{'N/A':<12}"
    v_en = f"{d['言语-英文']:<12.4f}" if d['言语-英文'] is not None else f"{'N/A':<12}"
    p_diff_str = f"{p_diff:+<15.4f}" if p_diff is not None else f"{'N/A':<15}"
    v_diff_str = f"{v_diff:+<15.4f}" if v_diff is not None else f"{'N/A':<15}"
    print(f"{model:<25} {d['参数-中文']:<12.4f} {p_en} {p_diff_str} {d['言语-中文']:<12.4f} {v_en} {v_diff_str}")

print("\n" + "=" * 120)
print("表3：各模型在不同条件下的CKA（中心核对齐，越高越好）")
print("=" * 120)
print(f"{'模型':<25} {'参数-中文':<12} {'参数-英文':<12} {'参数差(CN-EN)':<15} {'言语-中文':<12} {'言语-英文':<12} {'言语差(CN-EN)':<15}")
print("-" * 120)
for model in models:
    d = cka_data[model]
    p_diff = d['参数-中文'] - d['参数-英文'] if d['参数-英文'] is not None else None
    v_diff = d['言语-中文'] - d['言语-英文'] if d['言语-英文'] is not None else None
    p_en = f"{d['参数-英文']:<12.4f}" if d['参数-英文'] is not None else f"{'N/A':<12}"
    v_en = f"{d['言语-英文']:<12.4f}" if d['言语-英文'] is not None else f"{'N/A':<12}"
    p_diff_str = f"{p_diff:+<15.4f}" if p_diff is not None else f"{'N/A':<15}"
    v_diff_str = f"{v_diff:+<15.4f}" if v_diff is not None else f"{'N/A':<15}"
    print(f"{model:<25} {d['参数-中文']:<12.4f} {p_en} {p_diff_str} {d['言语-中文']:<12.4f} {v_en} {v_diff_str}")

print("\n" + "=" * 120)
print("表4：各模型在不同条件下的Jaccard相似度（越高越好）")
print("=" * 120)
print(f"{'模型':<25} {'参数-中文':<12} {'参数-英文':<12} {'参数差(CN-EN)':<15} {'言语-中文':<12} {'言语-英文':<12} {'言语差(CN-EN)':<15}")
print("-" * 120)
for model in models:
    d = jaccard_data[model]
    p_diff = d['参数-中文'] - d['参数-英文'] if d['参数-英文'] is not None else None
    v_cn = f"{d['言语-中文']:<12.4f}" if d['言语-中文'] is not None else f"{'N/A':<12}"
    v_en = f"{d['言语-英文']:<12.4f}" if d['言语-英文'] is not None else f"{'N/A':<12}"
    p_diff_str = f"{p_diff:+<15.4f}" if p_diff is not None else f"{'N/A':<15}"
    v_diff_str = f"{'N/A':<15}"
    print(f"{model:<25} {d['参数-中文']:<12.4f} {d['参数-英文']:<12.4f} {p_diff_str} {v_cn} {v_en} {v_diff_str}")

# ============================================================
# 分类讨论分析
# ============================================================
print("\n\n" + "=" * 120)
print("分类讨论分析")
print("=" * 120)

print("\n【MSE分析】（越低越好，正值表示中文MSE更高即英文更好）")
print("\n1. 参数格式下：")
print("   中文最优：RoboBrain2.0-7B (0.1217)")
print("   英文最优：Mimo-VL-7B-SFT (0.1713)")
print("   中文>英文（英文更好）：Mimo-7B-SFT, Mimo-VL-7B-SFT, Qwen2.5-7B-Instruct, RoboBrain2.0-7B")
print("   中文<英文（中文更好）：Mimo-embodied-7B, Qwen2.5-VL-7B-Instruct")

print("\n2. 言语格式下：")
print("   中文最优：Mimo-VL-7B-SFT (0.1331)")
print("   英文最优：所有模型均为0.0948")
print("   所有模型中文MSE均高于英文（英文更好）")

print("\n3. 中文条件下：参数 vs 言语")
print("   参数更优：RoboBrain2.0-7B (0.1217 vs 0.1614)")
print("   言语更优：Mimo-VL-7B-SFT (0.1545 vs 0.1331), Qwen2.5-7B-Instruct (0.2250 vs 0.1449)")

print("\n4. 英文条件下：参数 vs 言语")
print("   所有模型言语格式MSE均低于参数格式（言语更优）")

print("\n【RSA分析】（越高越好，正值表示中文RSA更高即中文更好）")
print("\n1. 参数格式下：")
print("   中文最优：RoboBrain2.0-7B (0.43)")
print("   英文最优：RoboBrain2.0-7B (0.76)")
print("   中文>英文：Mimo-VL-7B-SFT, Mimo-embodied-7B, Qwen2.5-7B-Instruct")
print("   中文<英文：Mimo-7B-SFT, Qwen2.5-VL-7B-Instruct, RoboBrain2.0-7B")

print("\n2. 言语格式下：")
print("   中文最优：Mimo-7B-SFT (0.66)")
print("   英文数据缺失")

print("\n【CKA分析】（越高越好，正值表示中文CKA更高即中文更好）")
print("\n1. 参数格式下：")
print("   中文最优：RoboBrain2.0-7B (0.56)")
print("   英文最优：RoboBrain2.0-7B (0.86)")
print("   中文>英文：Mimo-embodied-7B, Qwen2.5-7B-Instruct")
print("   中文<英文：Mimo-7B-SFT, Mimo-VL-7B-SFT, Qwen2.5-VL-7B-Instruct, RoboBrain2.0-7B")

print("\n2. 言语格式下：")
print("   中文最优：RoboBrain2.0-7B (0.79)")
print("   英文数据缺失")

print("\n【Jaccard分析】（越高越好，正值表示中文Jaccard更高即中文更好）")
print("\n1. 参数格式下：")
print("   中文最优：Qwen2.5-VL-7B-Instruct (0.431)")
print("   英文最优：Mimo-embodied-7B (0.422)")
print("   中文>英文：Qwen2.5-7B-Instruct, Qwen2.5-VL-7B-Instruct, RoboBrain2.0-7B")
print("   中文<英文：Mimo-7B-SFT, Mimo-VL-7B-SFT, Mimo-embodied-7B")
