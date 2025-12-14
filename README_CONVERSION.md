# 模型转换和推理指南

## 问题总结
1. **已修复**: NPU precision_mode参数错误 → 改为 `"preferred_fp32"`
2. **待解决**: MindSpore Lite不能直接加载.mindir，需要转换为.ms格式

## 操作步骤

### 方案1: 使用converter_lite命令行工具（推荐）

#### 在Atlas上执行:
```bash
# 1. 上传文件（在Windows上执行）
upload_fixed.bat

# 2. SSH到Atlas
ssh HwHiAiUser@192.168.137.2

# 3. 赋予执行权限
chmod +x convert_on_atlas.sh

# 4. 执行转换脚本
./convert_on_atlas.sh

# 5. 检查生成的.ms文件
ls -lh models/classifier_model.ms

# 6. 运行推理
python inference_lite.py
```

### 方案2: 手动转换命令

如果脚本失败，直接运行:
```bash
converter_lite \
  --fmk=MINDIR \
  --modelFile=models/classifier_model.mindir \
  --outputFile=models/classifier_model \
  --optimize=ascend_oriented
```

**参数说明:**
- `--fmk=MINDIR`: 输入格式为MindIR
- `--modelFile`: 输入的.mindir文件路径
- `--outputFile`: 输出文件名（不含.ms扩展名）
- `--optimize=ascend_oriented`: 针对Ascend NPU优化

### 方案3: 如果converter_lite不可用

检查并安装工具包:
```bash
# 1. 检查是否有 converter_lite
which converter_lite || echo "converter_lite 不存在"

# 2. （推荐）安装包含 converter 的Lite工具包，并加入PATH
#    解压后的路径示例: ~/tools/mindspore-lite/converter
export PATH="~/tools/mindspore-lite/converter:$PATH"
echo 'export PATH="~/tools/mindspore-lite/converter:$PATH"' >> ~/.bashrc
source ~/.bashrc

# 3. 重试转换脚本
./convert_on_atlas.sh

# 4. 如果仍不可用，尝试Python回退（自动触发）
python convert_on_atlas_py.py
```

## 预期结果

转换成功后应该看到:
```
✓ 转换成功!
models/classifier_model.ms  (大约 30-50KB)
```

## 已知问题

1. **Ascend版本警告**: 
   ```
   Ascend version is 1.84, but expect 7.6 or 7.7
   ```
   这可能影响NPU推理，如果NPU失败会自动降级到CPU。

2. **模型加载错误**:
   ```
   The model buffer is invalid
   ```
   原因: MindSpore Lite需要.ms格式，不能直接加载.mindir

   3. **converter_lite缺失**:
      - 现象: `command not found`
      - 处理: 安装Lite工具包，或使用Python回退脚本 `convert_on_atlas_py.py`

## 验证推理

成功运行应该输出:
```
{'question': '当前鱼缸状态如何？', 'answer': '...'}
{'question': '水质怎么样？', 'answer': '...'}
{'question': '需要换水吗？', 'answer': '...'}
```

## 如果转换仍然失败

请提供以下信息:
1. `converter_lite --help` 输出
2. MindSpore Lite版本: `pip show mindspore-lite`
3. 完整错误日志
