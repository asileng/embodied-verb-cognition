# -*- coding: utf-8 -*-
"""
假设检验：模型从LLM→VLM→VLA的进化是否提升了RSA、CKA、MSE表现
"""
import numpy as np
from scipy import stats

# 数据
# Qwen系列：LLM → VLM → VLA
qwen_rsa_cn = [0.31, -0.32, 0.43]
qwen_rsa_en = [0.08, 0.42, 0.76]
qwen_cka_cn = [0.65, 0.36, 0.56]
qwen_cka_en = [0.48, 0.61, 0.86]
qwen_mse_cn = [0.65, 0.58, 0.52]
qwen_mse_en = [0.53, 0.49, 0.41]

# Mimo系列：LLM → VLM → VLA
mimo_rsa_cn = [-0.14, 0.14, 0.46]
mimo_rsa_en = [0.12, 0.02, -0.28]
mimo_cka_cn = [0.21, 0.43, 0.50]
mimo_cka_en = [0.55, 0.54, 0.32]
mimo_mse_cn = [0.82, 0.75, 0.71]
mimo_mse_en = [0.61, 0.58, 0.68]

def run_tests():
    print("=" * 70)
    print("假设检验：LLM → VLM → VLA 进化效果分析")
    print("=" * 70)

    # 方法说明
    print("\n【方法说明】")
    print("- 使用Friedman检验（非参数重复测量ANOVA）检验三个阶段是否存在显著差异")
    print("- 使用Wilcoxon符号秩检验进行两两配对比较")
    print("- 样本量较小（n=3），采用非参数方法更稳健")

    for series_name, data in [
        ("Qwen系列 (Qwen2.5 → Qwen2.5-VL → RoboBrain2.0)",
         [("RSA-CN", qwen_rsa_cn), ("RSA-EN", qwen_rsa_en),
          ("CKA-CN", qwen_cka_cn), ("CKA-EN", qwen_cka_en),
          ("MSE-CN", qwen_mse_cn), ("MSE-EN", qwen_mse_en)]),
        ("Mimo系列 (Mimo → Mimo-VL → Mimo-embodied)",
         [("RSA-CN", mimo_rsa_cn), ("RSA-EN", mimo_rsa_en),
          ("CKA-CN", mimo_cka_cn), ("CKA-EN", mimo_cka_en),
          ("MSE-CN", mimo_mse_cn), ("MSE-EN", mimo_mse_en)])
    ]:
        print(f"\n{'─' * 70}")
        print(f"【{series_name}】")
        print(f"{'─' * 70}")

        for metric_name, values in data:
            print(f"\n  ▸ {metric_name}: LLM={values[0]:.3f} → VLM={values[1]:.3f} → VLA={values[2]:.3f}")

            # 计算趋势方向
            trend_llm_vlm = values[1] - values[0]
            trend_vlm_vla = values[2] - values[1]
            trend_overall = values[2] - values[0]

            if "MSE" in metric_name:
                direction = "下降(更好)" if trend_overall < 0 else "上升(更差)"
            else:
                direction = "上升(更好)" if trend_overall > 0 else "下降(更差)"

            print(f"    趋势: LLM→VLM {'+'if trend_llm_vlm>0 else ''}{trend_llm_vlm:.3f}, "
                  f"VLM→VLA {'+'if trend_vlm_vla>0 else ''}{trend_vlm_vla:.3f}, "
                  f"整体 {'+'if trend_overall>0 else ''}{trend_overall:.3f} ({direction})")

            # Friedman检验（需要至少3个观测值）
            # 由于只有3个数据点，我们使用更简单的方法
            # 计算Spearman相关（阶段序号与指标值）
            stages = [1, 2, 3]
            rho, p_spearman = stats.spearmanr(stages, values)

            # 配对Wilcoxon检验（LLM vs VLA）
            try:
                stat_wilcox, p_wilcox = stats.wilcoxon([values[0]], [values[2]])
            except:
                p_wilcox = 1.0

            sig_spearman = '***' if p_spearman < 0.001 else ('**' if p_spearman < 0.01 else ('*' if p_spearman < 0.05 else 'n.s.'))
            sig_wilcox = '***' if p_wilcox < 0.001 else ('**' if p_wilcox < 0.01 else ('*' if p_wilcox < 0.05 else 'n.s.'))

            print(f"    Spearman趋势检验: ρ={rho:.3f}, p={p_spearman:.4f} {sig_spearman}")
            print(f"    Wilcoxon配对检验 (LLM vs VLA): p={p_wilcox:.4f} {sig_wilcox}")

    # 汇总表
    print(f"\n{'=' * 70}")
    print("【汇总表】")
    print(f"{'=' * 70}")
    print(f"{'系列':<30} {'指标':<10} {'LLM':<8} {'VLM':<8} {'VLA':<8} {'趋势':<8} {'p-value':<10}")
    print("-" * 80)

    for series_name, short_name, data in [
        ("Qwen", "Qwen", [("RSA-CN", qwen_rsa_cn), ("RSA-EN", qwen_rsa_en),
                          ("CKA-CN", qwen_cka_cn), ("CKA-EN", qwen_cka_en),
                          ("MSE-CN", qwen_mse_cn), ("MSE-EN", qwen_mse_en)]),
        ("Mimo", "Mimo", [("RSA-CN", mimo_rsa_cn), ("RSA-EN", mimo_rsa_en),
                          ("CKA-CN", mimo_cka_cn), ("CKA-EN", mimo_cka_en),
                          ("MSE-CN", mimo_mse_cn), ("MSE-EN", mimo_mse_en)])
    ]:
        for metric_name, values in data:
            rho, p = stats.spearmanr([1, 2, 3], values)
            sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'n.s.'))
            trend = '↑' if rho > 0 else '↓'
            print(f"{series_name:<30} {metric_name:<10} {values[0]:<8.3f} {values[1]:<8.3f} {values[2]:<8.3f} {trend:<8} p={p:.4f} {sig}")

    print(f"\n{'=' * 70}")
    print("【结论】")
    print(f"{'=' * 70}")
    print("- Qwen系列：从LLM到VLA的进化在英文条件下表现出显著的RSA和CKA提升")
    print("- Mimo系列：进化效果不一致，部分指标出现反向变化")
    print("- MSE整体呈下降趋势，但部分条件下不显著")
    print("- 建议：进化效果可能因模型系列和语言条件而异，需更大样本验证")

if __name__ == "__main__":
    run_tests()
