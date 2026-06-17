"""
统一RESEARCH_SUMMARY.md中的数据为三位小数（只处理表格中的数据）
"""
import re

# 读取文件
with open('D:/task/科研/LLM-evaluation/具神认知/embodied-verb-cognition/RESEARCH_SUMMARY.md', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 处理每一行
formatted_lines = []
for line in lines:
    # 检查是否是表格行（包含|）
    if '|' in line and ('---' not in line):  # 跳过分隔行
        # 分割表格单元格
        parts = line.split('|')
        formatted_parts = []
        for part in parts:
            # 检查是否包含数字
            if re.search(r'\d+\.\d+', part):
                # 格式化数字为三位小数
                def format_num(match):
                    num = float(match.group(0))
                    return f"{num:.3f}"
                part = re.sub(r'-?\d+\.\d+', format_num, part)
            formatted_parts.append(part)
        line = '|'.join(formatted_parts)
    formatted_lines.append(line)

# 写入文件
with open('D:/task/科研/LLM-evaluation/具神认知/embodied-verb-cognition/RESEARCH_SUMMARY.md', 'w', encoding='utf-8') as f:
    f.writelines(formatted_lines)

print("格式化完成！表格中的数据已统一为三位小数。")
