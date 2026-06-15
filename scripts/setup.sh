#!/bin/bash
# 环境配置脚本

set -e

echo "=== 配置 VLM/LLM 具身动作语义认知测评环境 ==="

# 检查Python版本
echo "检查Python版本..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python版本: $python_version"

# 创建虚拟环境（可选）
read -p "是否创建虚拟环境? (y/n): " create_venv
if [ "$create_venv" = "y" ] || [ "$create_venv" = "Y" ]; then
    echo "创建虚拟环境..."
    python3 -m venv venv
    source venv/bin/activate
    echo "虚拟环境已激活"
fi

# 安装依赖
echo "安装依赖..."
pip install -r requirements.txt

# 检查vLLM
echo "检查vLLM安装..."
if ! command -v python &> /dev/null; then
    echo "警告: 未找到vLLM，请手动安装"
    echo "安装命令: pip install vllm"
else
    if python -c "import vllm" 2>/dev/null; then
        echo "vLLM已安装"
    else
        echo "警告: vLLM未安装，请手动安装"
        echo "安装命令: pip install vllm"
    fi
fi

# 创建配置文件
echo "创建配置文件..."
if [ ! -f "framework/config.py" ]; then
    cp framework/config_template.py framework/config.py
    echo "已创建 framework/config.py"
    echo "请编辑此文件配置模型路径和参数"
else
    echo "framework/config.py 已存在"
fi

if [ ! -f "framework/config_en.py" ]; then
    cp framework/config_template.py framework/config_en.py
    echo "已创建 framework/config_en.py"
    echo "请编辑此文件配置英文实验参数"
else
    echo "framework/config_en.py 已存在"
fi

# 创建结果目录
echo "创建结果目录..."
mkdir -p data/results
mkdir -p data/pilot_results

# 创建日志目录
mkdir -p logs

echo ""
echo "=== 环境配置完成 ==="
echo ""
echo "下一步操作:"
echo "1. 编辑 framework/config.py 配置模型路径"
echo "2. 启动vLLM服务: python -m vllm.entrypoints.openai.api_server --model /path/to/model --host localhost --port 8000"
echo "3. 运行实验: python framework/run_experiment.py --task task1 --language zh"
echo "4. 分析结果: python scripts/analyze_results.py data/results"
echo ""
