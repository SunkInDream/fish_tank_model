@echo off
REM Windows上传脚本 - 上传推理脚本和转换脚本到Atlas

echo ============================================================
echo 上传文件到Atlas 200dk
echo ============================================================
echo.

set ATLAS_IP=192.168.137.2
set ATLAS_USER=HwHiAiUser

echo [1/4] 上传inference_lite.py...
scp inference_lite.py %ATLAS_USER%@%ATLAS_IP%:~/fish/
if %ERRORLEVEL% NEQ 0 (
    echo X 上传失败
    pause
    exit /b 1
)
echo ✓ 上传完成
echo.

echo [2/4] 上传convert_on_atlas.sh...
scp convert_on_atlas.sh %ATLAS_USER%@%ATLAS_IP%:~/fish/
if %ERRORLEVEL% NEQ 0 (
    echo X 上传失败
    pause
    exit /b 1
)
echo ✓ 上传完成
echo.

echo [3/4] 上传convert_on_atlas_py.py...
scp convert_on_atlas_py.py %ATLAS_USER%@%ATLAS_IP%:~/fish/
if %ERRORLEVEL% NEQ 0 (
    echo X 上传失败
    pause
    exit /b 1
)
echo ✓ 上传完成
echo.

echo [4/4] 上传.mindir模型...
scp models/classifier_model.mindir %ATLAS_USER%@%ATLAS_IP%:~/fish/models/
if %ERRORLEVEL% NEQ 0 (
    echo X 上传失败
    pause
    exit /b 1
)
echo ✓ 上传完成
echo.

echo ============================================================
echo 所有文件上传完成！
echo ============================================================
echo.
echo 下一步在Atlas上执行:
echo   1. ssh %ATLAS_USER%@%ATLAS_IP%
echo   2. cd ~/fish
echo   3. chmod +x convert_on_atlas.sh
echo   4. ./convert_on_atlas.sh  # 转换模型
echo   5. python inference_lite.py  # 运行推理
echo.
pause
