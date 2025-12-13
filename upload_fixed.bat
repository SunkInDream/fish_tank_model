@echo off
REM Windows上传脚本 - 上传更新后的推理脚本到Atlas

echo ============================================================
echo 上传修复后的推理脚本到Atlas 200dk
echo ============================================================
echo.

set ATLAS_IP=192.168.137.2
set ATLAS_USER=HwHiAiUser

echo [1/2] 上传inference_lite.py...
scp inference_lite.py %ATLAS_USER%@%ATLAS_IP%:~/
if %ERRORLEVEL% NEQ 0 (
    echo X 上传失败
    pause
    exit /b 1
)
echo ✓ 上传完成
echo.

echo [2/2] 上传转换脚本...
scp convert_on_atlas.sh %ATLAS_USER%@%ATLAS_IP%:~/
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
echo   2. chmod +x convert_on_atlas.sh
echo   3. ./convert_on_atlas.sh
echo   4. python inference_lite.py
echo.
pause
