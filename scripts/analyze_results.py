# -*- coding: utf-8 -*-
"""
实验结果分析脚本
用于分析VLM/LLM具身动作语义认知测评结果
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def load_experiment_results(results_dir: str) -> pd.DataFrame:
    """
    加载实验结果数据

    Args:
        results_dir: 结果目录路径

    Returns:
        包含所有实验结果的DataFrame
    """
    results = []
    results_path = Path(results_dir)

    # 遍历所有JSON结果文件
    for json_file in results_path.rglob("*.json"):
        if json_file.name.startswith("experiment_tracker"):
            continue

        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 提取关键信息
            result = {
                'model': data.get('model', ''),
                'verb': data.get('verb', ''),
                'task_id': data.get('task_id', ''),
                'language': data.get('language', 'zh'),
                'is_valid': data.get('is_valid', False),
                'duration_seconds': data.get('duration_seconds', 0),
                'timestamp': data.get('timestamp', ''),
            }

            # 提取解析结果
            parsed = data.get('parsed_result', {})
            if data.get('task_id') == 'task1':
                # 任务1：数值编码
                result.update({
                    'FORCE': parsed.get('FORCE', 0),
                    'ARM': parsed.get('ARM', 0),
                    'HAND': parsed.get('HAND', 0),
                    'VD': parsed.get('VD', 0),
                    'HD': parsed.get('HD', 0),
                })
            elif data.get('task_id') == 'task2':
                # 任务2：文本描述
                result.update({
                    'FORCE_desc': parsed.get('FORCE', ''),
                    'ARM_desc': parsed.get('ARM', ''),
                    'HAND_desc': parsed.get('HAND', ''),
                    'VD_desc': parsed.get('VD', ''),
                    'HD_desc': parsed.get('HD', ''),
                })

            results.append(result)

        except Exception as e:
            print(f"Error loading {json_file}: {e}")

    return pd.DataFrame(results)


def calculate_rsa_matrix(df: pd.DataFrame, model: str, language: str = 'zh') -> np.ndarray:
    """
    计算模型的RSA（表征相似性分析）矩阵

    Args:
        df: 实验结果DataFrame
        model: 模型名称
        language: 语言版本

    Returns:
        RSA相似性矩阵
    """
    # 筛选特定模型和语言的数据
    model_data = df[(df['model'] == model) & (df['language'] == language)]

    if model_data.empty:
        return np.array([])

    # 获取动词列表
    verbs = sorted(model_data['verb'].unique())

    # 构建动词-参数矩阵
    verb_params = []
    for verb in verbs:
        verb_data = model_data[model_data['verb'] == verb]
        if not verb_data.empty:
            # 取平均值
            params = verb_data[['FORCE', 'ARM', 'HAND', 'VD', 'HD']].mean().values
            verb_params.append(params)

    verb_params = np.array(verb_params)

    # 计算相关矩阵（RSA）
    n_verbs = len(verbs)
    rsa_matrix = np.zeros((n_verbs, n_verbs))

    for i in range(n_verbs):
        for j in range(n_verbs):
            if i == j:
                rsa_matrix[i, j] = 1.0
            else:
                # 使用皮尔逊相关系数
                corr, _ = stats.pearsonr(verb_params[i], verb_params[j])
                rsa_matrix[i, j] = corr

    return rsa_matrix, verbs


def calculate_cka_score(model_rsa: np.ndarray, human_rsa: np.ndarray) -> float:
    """
    计算CKA（中心核对齐）分数

    Args:
        model_rsa: 模型的RSA矩阵
        human_rsa: 人类基准的RSA矩阵

    Returns:
        CKA分数
    """
    # 展平矩阵
    model_flat = model_rsa.flatten()
    human_flat = human_rsa.flatten()

    # 计算CKA
    # CKA = dot(X, Y) / (norm(X) * norm(Y))
    dot_product = np.dot(model_flat, human_flat)
    norm_X = np.linalg.norm(model_flat)
    norm_Y = np.linalg.norm(human_flat)

    if norm_X == 0 or norm_Y == 0:
        return 0.0

    cka = dot_product / (norm_X * norm_Y)
    return cka


def analyze_model_performance(df: pd.DataFrame) -> Dict:
    """
    分析模型性能

    Args:
        df: 实验结果DataFrame

    Returns:
        包含分析结果的字典
    """
    results = {}

    # 按模型分组
    for model in df['model'].unique():
        model_data = df[df['model'] == model]

        # 计算基本统计
        total_experiments = len(model_data)
        valid_experiments = len(model_data[model_data['is_valid']])
        validity_rate = valid_experiments / total_experiments if total_experiments > 0 else 0

        # 计算平均响应时间
        avg_duration = model_data['duration_seconds'].mean()

        # 按语言分组统计
        language_stats = {}
        for language in model_data['language'].unique():
            lang_data = model_data[model_data['language'] == language]
            language_stats[language] = {
                'total': len(lang_data),
                'valid': len(lang_data[lang_data['is_valid']]),
                'validity_rate': len(lang_data[lang_data['is_valid']]) / len(lang_data) if len(lang_data) > 0 else 0,
                'avg_duration': lang_data['duration_seconds'].mean(),
            }

        results[model] = {
            'total_experiments': total_experiments,
            'valid_experiments': valid_experiments,
            'validity_rate': validity_rate,
            'avg_duration': avg_duration,
            'language_stats': language_stats,
        }

    return results


def plot_rsa_heatmap(rsa_matrix: np.ndarray, verbs: List[str], model: str, language: str, output_path: str):
    """
    绘制RSA热力图

    Args:
        rsa_matrix: RSA矩阵
        verbs: 动词列表
        model: 模型名称
        language: 语言版本
        output_path: 输出路径
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(rsa_matrix,
                xticklabels=verbs,
                yticklabels=verbs,
                annot=True,
                fmt='.2f',
                cmap='RdYlBu_r',
                vmin=-1,
                vmax=1)

    plt.title(f'{model} - {language.upper()} RSA Matrix', fontsize=14)
    plt.xlabel('Verbs', fontsize=12)
    plt.ylabel('Verbs', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_performance_comparison(results: Dict, output_path: str):
    """
    绘制性能对比图

    Args:
        results: 分析结果字典
        output_path: 输出路径
    """
    models = list(results.keys())
    validity_rates = [results[m]['validity_rate'] for m in models]
    avg_durations = [results[m]['avg_duration'] for m in models]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 有效率对比
    bars1 = ax1.bar(models, validity_rates, color='skyblue', alpha=0.7)
    ax1.set_ylabel('Validity Rate', fontsize=12)
    ax1.set_title('Model Validity Rate Comparison', fontsize=14)
    ax1.set_ylim(0, 1)
    ax1.tick_params(axis='x', rotation=45)

    # 添加数值标签
    for bar, rate in zip(bars1, validity_rates):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{rate:.2%}', ha='center', va='bottom', fontsize=10)

    # 平均响应时间对比
    bars2 = ax2.bar(models, avg_durations, color='lightcoral', alpha=0.7)
    ax2.set_ylabel('Average Duration (seconds)', fontsize=12)
    ax2.set_title('Model Response Time Comparison', fontsize=14)
    ax2.tick_params(axis='x', rotation=45)

    # 添加数值标签
    for bar, duration in zip(bars2, avg_durations):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{duration:.1f}s', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='分析VLM/LLM具身动作语义认知测评结果')
    parser.add_argument('results_dir', help='实验结果目录')
    parser.add_argument('--output-dir', '-o', default='analysis_results', help='分析结果输出目录')
    parser.add_argument('--human-baseline', help='人类基准数据文件路径')

    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print(f"Loading results from: {args.results_dir}")
    df = load_experiment_results(args.results_dir)

    if df.empty:
        print("No valid results found!")
        return

    print(f"Loaded {len(df)} experiments")

    # 分析模型性能
    print("Analyzing model performance...")
    results = analyze_model_performance(df)

    # 保存分析结果
    results_file = output_dir / 'performance_analysis.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Performance analysis saved to: {results_file}")

    # 绘制性能对比图
    print("Plotting performance comparison...")
    plot_performance_comparison(results, output_dir / 'performance_comparison.png')

    # 计算RSA矩阵（如果有足够数据）
    print("Calculating RSA matrices...")
    for model in df['model'].unique():
        for language in df['language'].unique():
            model_lang_data = df[(df['model'] == model) & (df['language'] == language)]
            if len(model_lang_data) >= 6:  # 至少需要6个动词的数据
                try:
                    rsa_matrix, verbs = calculate_rsa_matrix(df, model, language)
                    if len(rsa_matrix) > 0:
                        # 绘制热力图
                        plot_rsa_heatmap(rsa_matrix, verbs, model, language,
                                       output_dir / f'rsa_{model}_{language}.png')
                        print(f"RSA heatmap saved for {model} ({language})")
                except Exception as e:
                    print(f"Error calculating RSA for {model} ({language}): {e}")

    print("Analysis completed!")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
