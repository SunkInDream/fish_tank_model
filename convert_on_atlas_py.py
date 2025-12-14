"""
Atlas 回退转换脚本 - 使用 MindSpore Lite Python API 尝试转换
"""
import os
import sys

MINDIR = "models/classifier_model.mindir"
OUTPUT = "models/classifier_model"
MS_PATH = OUTPUT + ".ms"

print("=" * 60)
print("使用 Python API 尝试转换 MindIR → .ms")
print("=" * 60)

# 检查输入
if not os.path.exists(MINDIR):
    print(f"✗ 未找到 {MINDIR}")
    sys.exit(2)

print(f"✓ 找到 MindIR: {MINDIR} ({os.path.getsize(MINDIR)/1024:.2f} KB)")

try:
    import mindspore_lite as mslite
    print("✓ 已导入 mindspore_lite")
except Exception as e:
    print(f"✗ 导入 mindspore_lite 失败: {e}")
    print("  请先安装: pip install mindspore-lite")
    sys.exit(2)

# 依次尝试不同的 API 形式（兼容不同版本）
errors = []

try:
    # 形式A：通过构造函数传参后调用 convert()
    print("\n[尝试A] mslite.Converter(...).convert()")
    conv = mslite.Converter(
        fmk_type=mslite.FmkType.MINDIR,
        model_file=MINDIR,
        output_file=OUTPUT,
        config_file="",
    )
    conv.convert()
except Exception as e:
    errors.append(("A", e))

if not os.path.exists(MS_PATH):
    try:
        # 形式B：创建对象后设置属性，调用 convert()
        print("\n[尝试B] 设置属性后 convert()")
        conv = mslite.Converter()
        conv.fmk_type = mslite.FmkType.MINDIR
        conv.model_file = MINDIR
        conv.output_file = OUTPUT
        conv.config_file = ""
        conv.convert()
    except Exception as e:
        errors.append(("B", e))

if not os.path.exists(MS_PATH):
    try:
        # 形式C：设置属性后调用旧方法名 converter()
        print("\n[尝试C] 设置属性后 converter()")
        conv = mslite.Converter()
        conv.fmk_type = mslite.FmkType.MINDIR
        conv.model_file = MINDIR
        conv.output_file = OUTPUT
        conv.config_file = ""
        conv.converter()
    except Exception as e:
        errors.append(("C", e))

if os.path.exists(MS_PATH):
    size = os.path.getsize(MS_PATH)/1024
    print(f"\n✓ 转换成功: {MS_PATH} ({size:.2f} KB)")
    sys.exit(0)

print("\n✗ 所有 Python API 尝试均失败")
for label, err in errors:
    print(f"  - 尝试{label} 失败: {err}")

print("\n建议改用 converter_lite 二进制: 添加到 PATH 后重新运行 convert_on_atlas.sh")
sys.exit(2)
