"""
统一RESEARCH_SUMMARY.md中实验结果部分的数据为三位小数
只处理从"## 实验结果"开始的表格
"""
import re

# 读取文件
with open('D:/task/科研/LLM-evaluation/具神认知/embodied-verb-cognition/RESEARCH_SUMMARY.md', 'r', encoding='utf-8') as f:
    content = f.read()

# 找到"## 四、实验结果与分析"的位置
start_marker = "## 四、实验结果与分析"
start_idx = content.find(start_marker)

if start_idx == -1:
    print("未找到'## 实验结果'标记")
else:
    # 分割内容：实验结果之前的内容 + 实验结果及之后的内容
    before = content[:start_idx]
    after = content[start_idx:]

    # 处理实验结果部分
    lines = after.split('\n')
    formatted_lines = []

    for line in lines:
        # 检查是否是表格行（包含|）
        if '|' in line and ('---' not in line):  # 跳过分隔行
            # 分割表格单元格
            parts = line.split('|')
            formatted_parts = []
            for part in parts:
                # 检查是否包含数字（但不是模型名称的一部分）
                # 只格式化独立的数字，不格式化如"Qwen2.5"这样的名称
                if re.search(r'(?<![A-Za-z])\d+\.\d+(?![A-Za-z])', part):
                    # 格式化数字为三位小数
                    def format_num(match):
                        num = float(match.group(0))
                        return f"{num:.3f}"
                    part = re.sub(r'(?<![A-Za-z])-?\d+\.\d+(?![A-Za-z])', format_num, part)
                formatted_parts.append(part)
            line = '|'.join(formatted_parts)
        formatted_lines.append(line)

    # 重新组合内容
    after_formatted = '\n'.join(formatted_lines)
    content = before + after_formatted

    # 写入文件
    with open('D:/task/科研/LLM-evaluation/具神认知/embodied-verb-cognition/RESEARCH_SUMMARY.md', 'w', encoding='utf-8') as f:
        f.write(content)

    print("格式化完成！实验结果部分的数据已统一为三位小数。")
