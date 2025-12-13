#!/bin/bash
# 在Atlas 200dk上执行此脚本转换模型

echo "============================================================"
echo "转换MindIR为MindSpore Lite格式"
echo "============================================================"
echo ""

# 检查输入文件
if [ ! -f "models/classifier_model.mindir" ]; then
    echo "✗ 错误: models/classifier_model.mindir 不存在"
    echo ""
    echo "请先上传文件:"
    echo "  scp models/classifier_model.mindir HwHiAiUser@<atlas_ip>:~/models/"
    exit 1
fi

echo "✓ 找到输入文件: models/classifier_model.mindir"
ls -lh models/classifier_model.mindir
echo ""

# 执行转换
echo "开始转换..."
echo ""

converter_lite \
  --fmk=MINDIR \
  --modelFile=models/classifier_model.mindir \
  --outputFile=models/classifier_model \
  --optimize=ascend_oriented

# 检查结果
if [ -f "models/classifier_model.ms" ]; then
    echo ""
    echo "============================================================"
    echo "✓ 转换成功!"
    echo "============================================================"
    echo ""
    ls -lh models/classifier_model.ms
    echo ""
    echo "现在可以运行推理:"
    echo "  python inference_lite.py"
else
    echo ""
    echo "============================================================"
    echo "✗ 转换失败"
    echo "============================================================"
    echo ""
    echo "请检查:"
    echo "1. converter_lite是否已安装"
    echo "2. MindIR文件是否完整"
    echo "3. 是否有写权限"
fi
