"""
统一RESEARCH_SUMMARY.md中的数据为三位小数
"""
import re

# 读取文件
with open('D:/task/科研/LLM-evaluation/具神认知/embodied-verb-cognition/RESEARCH_SUMMARY.md', 'r', encoding='utf-8') as f:
    content = f.read()

# 定义需要格式化的模式
# 匹配表格中的数字（包括正负号、小数点、数字）
def format_number(match):
    num_str = match.group(0)
    try:
        # 尝试解析为浮点数
        num = float(num_str)
        # 格式化为三位小数
        if num == int(num):
            return f"{num:.3f}"
        else:
            return f"{num:.3f}"
    except:
        return num_str

# 匹配表格中的数字模式
# 匹配类似 0.1895, -0.44, 0.07 等数字
pattern = r'(?<!\w)([-]?)(\d+\.\d+)(?!\w)'

# 替换所有匹配的数字
def replace_numbers(text):
    result = []
    i = 0
    while i < len(text):
        # 检查是否在表格行中
        if '|' in text[i:i+50]:
            # 在表格行中，查找数字
            match = re.search(pattern, text[i:])
            if match and match.start() < 50:
                num_str = match.group(0)
                try:
                    num = float(num_str)
                    formatted = f"{num:.3f}"
                    result.append(text[i:i+match.start()])
                    result.append(formatted)
                    i += match.end()
                    continue
                except:
                    pass
        result.append(text[i])
        i += 1
    return ''.join(result)

# 更简单的方法：直接替换特定模式
# 1. 替换 0.XXXX 为 0.XXX（四位小数变三位）
# 2. 替换 0.XX 为 0.XX0（两位小数变三位）
# 3. 替换 -0.XX 为 -0.XX0

# 使用正则表达式进行替换
def format_all_numbers(text):
    # 匹配表格中的数字（包括正负号）
    # 模式：可选的负号 + 数字 + 小数点 + 数字
    pattern = r'([-]?\d+\.\d+)'

    def replace_match(match):
        num_str = match.group(1)
        try:
            num = float(num_str)
            # 格式化为三位小数
            return f"{num:.3f}"
        except:
            return num_str

    return re.sub(pattern, replace_match, text)

# 应用格式化
formatted_content = format_all_numbers(content)

# 写入文件
with open('D:/task/科研/LLM-evaluation/具神认知/embodied-verb-cognition/RESEARCH_SUMMARY.md', 'w', encoding='utf-8') as f:
    f.write(formatted_content)

print("格式化完成！所有数据已统一为三位小数。")
