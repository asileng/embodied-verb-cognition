# -*- coding: utf-8 -*-
"""
检查文档中所有表格的模型数据完整性
"""
import re

# 所有模型列表
ALL_MODELS = [
    'Mimo-7B-SFT',
    'Mimo-VL-7B-SFT',
    'Mimo-embodied-7B',
    'Qwen2.5-7B-Instruct',
    'Qwen2.5-VL-7B-Instruct',
    'RoboBrain2.0-7B'
]

# 读取文档
doc_path = 'D:/task/科研/LLM-evaluation/具神认知/embodied-verb-cognition/RESEARCH_SUMMARY.md'
with open(doc_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 查找所有表格
tables = re.findall(r'\|[^|]+\|[^|]+\|[^|]+\|.*?\n\|[-| ]+\|[-| ]+\|.*?\n((?:\|[^|]+\|.*?\n)*)', content)

print("=" * 70)
print("数据完整性检查报告")
print("=" * 70)

# 检查每个表格
for i, table in enumerate(tables):
    print(f"\n【表格 {i+1}】")

    # 提取表格中的模型名称
    models_in_table = []
    for line in table.strip().split('\n'):
        if '|' in line:
            cells = [cell.strip() for cell in line.split('|') if cell.strip()]
            if cells and any(model in cells[0] for model in ALL_MODELS):
                models_in_table.append(cells[0])

    # 检查是否有遗漏
    missing = [m for m in ALL_MODELS if m not in models_in_table and not any(m.split('-')[0] in model for model in models_in_table)]

    if missing:
        print(f"  ⚠️ 缺少模型: {', '.join(missing)}")
    else:
        print(f"  ✓ 模型完整")

    print(f"  已包含: {', '.join(models_in_table) if models_in_table else '无'}")

# 检查Jaccard数据
print("\n【Jaccard数据检查】")
jaccard_section = content[content.find('### 二、关键参数识别能力'):content.find('---', content.find('### 二、关键参数识别能力'))]

for model in ALL_MODELS:
    if model in jaccard_section:
        # 检查是否有N/A
        model_section = jaccard_section[jaccard_section.find(model):jaccard_section.find(model)+200]
        if 'N/A' in model_section:
            print(f"  ⚠️ {model}: 存在N/A数据")
        else:
            print(f"  ✓ {model}: 数据完整")
    else:
        print(f"  ❌ {model}: 缺失")

# 检查图表引用
print("\n【图表引用检查】")
figures = re.findall(r'!\[.*?\]\((.*?)\)', content)
for fig in figures:
    if not fig.startswith('http'):
        import os
        full_path = f'D:/task/科研/LLM-evaluation/具神认知/embodied-verb-cognition/{fig}'
        if os.path.exists(full_path):
            print(f"  ✓ {fig}")
        else:
            print(f"  ❌ {fig} (文件不存在)")
