@echo off
echo ============================================================
echo 鱼缸分类器完整流程 - 轻量级分类+模板方案
echo ============================================================
echo.

echo [1/3] 训练分类器（预计30秒）
python train_classifier.py
if errorlevel 1 (
    echo 训练失败！
    pause
    exit /b 1
)
echo.

echo [2/3] 转换为MindSpore格式
python convert_classifier.py
if errorlevel 1 (
    echo 转换失败！
    pause
    exit /b 1
)
echo.

echo [3/3] 本地测试推理（CPU）
python inference_classifier.py
if errorlevel 1 (
    echo 推理失败！
    pause
    exit /b 1
)

echo.
echo ============================================================
echo 完成！模型已准备好部署到Atlas 200dk
echo ============================================================
echo.
echo 下一步：
echo 1. 上传模型: scp models/classifier_model.mindir HwHiAiUser@[atlas_ip]:~/models/
echo 2. 上传脚本: scp inference_classifier.py HwHiAiUser@[atlas_ip]:~/
echo 3. Atlas运行: ssh HwHiAiUser@[atlas_ip] "python inference_classifier.py"
echo.
pause
