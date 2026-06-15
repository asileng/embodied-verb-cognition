# -*- coding: utf-8 -*-
"""
大模型测评任务配置文件模板
复制此文件为 config.py 或 config_en.py 并修改配置
"""

# ==================== 模型配置 ====================
# vllm本地模型配置
# 模型路径或名称（用于vllm加载）
MODELS = [
    "/path/to/your/model1",
    "/path/to/your/model2",
]

# vllm服务配置
VLLM_CONFIG = {
    "host": "localhost",           # vllm服务地址
    "port": 8000,                  # vllm服务端口
    "tensor_parallel_size": 1,     # 张量并行大小
    "gpu_memory_utilization": 0.9, # GPU内存使用率
    "max_model_len": 4096,         # 最大模型长度
    "trust_remote_code": True,     # 是否信任远程代码
}

# ==================== 动词配置 ====================
# 6个目标动词（中文）
VERBS = [
    "投",
    "扔",
    "摔",
    "丢",
    "甩",
    "抛",
]

# 英文动词配置（用于英文实验）
VERBS_EN = [
    "throw",
    "fling",
    "chuck",
    "cast",
    "hurl",
    "toss",
]

# 拉丁方顺序配置（用于平衡顺序效应）
# 6个动词的拉丁方排列
LATIN_SQUARE_ORDERS = [
    ["投", "扔", "摔", "丢", "甩", "抛"],
    ["扔", "摔", "丢", "甩", "抛", "投"],
    ["摔", "丢", "甩", "抛", "投", "扔"],
    ["丢", "甩", "抛", "投", "扔", "摔"],
    ["甩", "抛", "投", "扔", "摔", "丢"],
    ["抛", "投", "扔", "摔", "丢", "甩"],
]

# 英文拉丁方顺序
LATIN_SQUARE_ORDERS_EN = [
    ["throw", "fling", "chuck", "cast", "hurl", "toss"],
    ["fling", "chuck", "cast", "hurl", "toss", "throw"],
    ["chuck", "cast", "hurl", "toss", "throw", "fling"],
    ["cast", "hurl", "toss", "throw", "fling", "chuck"],
    ["hurl", "toss", "throw", "fling", "chuck", "cast"],
    ["toss", "throw", "fling", "chuck", "cast", "hurl"],
]

# ==================== 任务配置 ====================
# 支持的任务列表
TASKS = {
    "task1": {
        "name": "动作编码测评",
        "description": "输出JSON格式的物理参数编码",
        "output_format": "json",
    },
    "task2": {
        "name": "动作描述测评",
        "description": "输出一句话文本描述",
        "output_format": "text",
    },
}

# 当前任务（可在命令行中覆盖）
CURRENT_TASK = "task1"

# ==================== 编码维度配置 ====================
# 任务1的编码维度
CODING_DIMENSIONS = {
    "FORCE": {
        "name": "力量大小",
        "range": [1, 5],
        "description": "1=非常弱, 2=弱, 3=中等, 4=强, 5=非常强",
    },
    "ARM": {
        "name": "手臂初始状态",
        "range": [0, 1],
        "description": "0=弯曲, 1=伸直",
    },
    "HAND": {
        "name": "手部初始高度",
        "range": [0, 12],
        "description": "以身高归一化的高度值",
    },
    "VD": {
        "name": "垂直运动方向",
        "range": [0, 1],
        "description": "0=向上, 1=向下",
    },
    "HD": {
        "name": "水平运动方向",
        "range": [0, 1],
        "description": "0=向侧方, 1=向前",
    },
}

# 任务2的描述维度
DESCRIPTION_DIMENSIONS = {
    "FORCE": {
        "name": "力量大小",
        "options": ["非常强", "强", "中等", "弱", "非常弱"],
    },
    "ARM": {
        "name": "手臂初始状态",
        "options": ["手臂伸直", "手臂弯曲"],
    },
    "HAND": {
        "name": "手部初始高度",
        "options": ["接近地面", "膝盖高度", "腰部高度", "胸部高度", "肩部高度", "头部高度", "高于头部"],
    },
    "VD": {
        "name": "垂直运动方向",
        "options": ["向下", "向上"],
    },
    "HD": {
        "name": "水平运动方向",
        "options": ["向前", "向侧方"],
    },
}

# ==================== 实验配置 ====================
# 重复次数
REPEAT_COUNT = 1

# 是否保存原始响应
SAVE_RAW_RESPONSE = True

# 超时时间（秒）
TIMEOUT = 30

# 推理参数
INFERENCE_PARAMS = {
    "temperature": 0.7,
    "top_p": 0.9,
    "max_tokens": 512,
}

# ==================== 输出配置 ====================
# 结果存储目录
RESULTS_DIR = "results"

# 结果文件名格式
RESULT_FILENAME_FORMAT = "{model}_{verb}_{timestamp}_{uuid}.json"

# 汇总文件名
SUMMARY_FILENAME = "summary_{timestamp}.json"

# ==================== 拉丁方配置 ====================
# 是否使用拉丁方顺序
USE_LATIN_SQUARE = True

# 参与者ID（用于拉丁方顺序分配）
PARTICIPANT_ID = None

# ==================== 语言配置 ====================
# 当前语言版本
LANGUAGE = "zh"  # "zh" 或 "en"
