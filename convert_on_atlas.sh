#!/bin/bash
# Atlas 200 DK: MindIR -> MindSpore Lite (.ms)

set -u

echo "============================================================"
echo "转换MindIR为MindSpore Lite格式"
echo "============================================================"
echo ""

# 固定工作目录为脚本所在目录（避免你在别处执行导致相对路径错）
WORKDIR="$(cd "$(dirname "$0")" && pwd)"
cd "$WORKDIR" || exit 1

# ---------- 模型输入输出 ----------
MODEL_DIR="$WORKDIR/models"
INPUT_MINDIR="$MODEL_DIR/classifier_model.mindir"
OUTPUT_PREFIX="$MODEL_DIR/classifier_model"
OUTPUT_MS="$MODEL_DIR/classifier_model.ms"

# ---------- 日志 ----------
LOG_DIR="$WORKDIR/log"
LOG_FILE="$LOG_DIR/convert_lite.log"

mkdir -p "$MODEL_DIR" "$LOG_DIR"

# ---------- converter_lite 绝对路径（你已确认存在） ----------
CONVERTER_BIN="/home/HwHiAiUser/mindspore-lite-2.7.1-linux-aarch64/tools/converter/converter/converter_lite"

# ---------- 检查输入文件 ----------
if [ ! -f "$INPUT_MINDIR" ]; then
  echo "✗ 错误: $INPUT_MINDIR 不存在"
  echo ""
  echo "请先把 MindIR 放到: $MODEL_DIR"
  echo "示例:"
  echo "  scp models/classifier_model.mindir HwHiAiUser@<atlas_ip>:$MODEL_DIR/"
  exit 1
fi

echo "✓ 找到输入文件: $INPUT_MINDIR"
ls -lh "$INPUT_MINDIR"
echo ""

# ---------- 检查 converter_lite ----------
if [ ! -f "$CONVERTER_BIN" ]; then
  echo "✗ 错误: 未找到 converter_lite: $CONVERTER_BIN"
  echo "请确认 mindspore-lite 解压目录是否为 /home/HwHiAiUser/mindspore-lite-2.7.1-linux-aarch64"
  exit 2
fi

# 确保可执行
chmod +x "$CONVERTER_BIN" 2>/dev/null || true
if [ ! -x "$CONVERTER_BIN" ]; then
  echo "✗ 错误: converter_lite 不可执行: $CONVERTER_BIN"
  echo "请检查权限: ls -l $CONVERTER_BIN"
  exit 2
fi

echo "✓ 使用 converter_lite: $CONVERTER_BIN"
echo ""

# （可选）Ascend 环境提醒
if [ -z "${ASCEND_HOME_PATH:-}" ] && [ -z "${ASCEND_TOOLKIT_HOME:-}" ]; then
  echo "⚠ 提示: 未检测到 ASCEND_HOME_PATH/ASCEND_TOOLKIT_HOME"
  echo "   若后续报 Ascend/CANN 相关错误，请先 source CANN 环境脚本（set_env.sh / setenv.bash）"
  echo ""
fi

# ---------- 执行转换 ----------
echo "开始转换..."
echo "日志输出: $LOG_FILE"
echo ""

rm -f "$OUTPUT_MS"

"$CONVERTER_BIN" \
  --fmk=MINDIR \
  --modelFile="$INPUT_MINDIR" \
  --outputFile="$OUTPUT_PREFIX" \
#   --optimize=ascend_oriented 2>&1 | tee "$LOG_FILE"

RET=${PIPESTATUS[0]}
echo ""

# ---------- 检查结果 ----------
if [ $RET -eq 0 ] && [ -f "$OUTPUT_MS" ]; then
  echo "============================================================"
  echo "✓ 转换成功!"
  echo "============================================================"
  echo ""
  ls -lh "$OUTPUT_MS"
  echo ""
  echo "现在可以运行推理:"
  echo "  python inference_lite.py"
  exit 0
else
  echo "============================================================"
  echo "✗ 转换失败"
  echo "============================================================"
  echo ""
  echo "返回码: $RET"
  echo "未找到输出文件: $OUTPUT_MS"
  echo ""
  echo "请查看日志定位原因:"
  echo "  less $LOG_FILE"
  exit 3
fi
