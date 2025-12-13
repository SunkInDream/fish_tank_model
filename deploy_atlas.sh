#!/bin/bash
# Atlas 200dk 快速部署脚本

echo "=========================================="
echo "Atlas 200dk 推理环境部署"
echo "=========================================="

# 1. 安装依赖
echo ""
echo "1. 安装Python依赖..."
pip install -r requirements_atlas.txt

# 2. 检查模型文件
echo ""
echo "2. 检查模型文件..."
if [ -f "models/fish_tank_model.mindir" ]; then
    echo "✓ 找到mindir文件"
else
    echo "✗ 未找到mindir文件"
    echo "   正在重新转换模型..."
    python convert_to_mindir.py
fi

# 3. 运行测试
echo ""
echo "3. 运行推理测试..."
python inference_mindspore.py test

echo ""
echo "=========================================="
echo "部署完成！"
echo "=========================================="
echo ""
echo "可用命令："
echo "  python inference_mindspore.py test        # 测试模式"
echo "  python inference_mindspore.py jsonl       # 批量推理"
echo "  python inference_mindspore.py interactive # 交互模式"
