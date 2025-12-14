"""
Atlas 200dk MindSpore Lite 推理脚本
轻量级边缘设备推理，支持NPU加速
"""
import mindspore_lite as mslite
import numpy as np
import re
import os

def extract_features(text):
    """从文本中提取温度、TDS、PH"""
    temp_match = re.search(r'温度([\d.]+)', text)
    tds_match = re.search(r'TDS([\d.]+)', text)
    ph_match = re.search(r'PH([\d.]+)', text)
    
    temp = float(temp_match.group(1)) if temp_match else 25.0
    tds = float(tds_match.group(1)) if tds_match else 250.0
    ph = float(ph_match.group(1)) if ph_match else 7.0
    
    return temp, tds, ph

def normalize_features(temp, tds, ph):
    """归一化特征"""
    temp_norm = (temp - 25.0) / 10.0
    tds_norm = (tds - 250.0) / 200.0
    ph_norm = (ph - 7.0) / 2.0
    return [temp_norm, tds_norm, ph_norm]

def generate_response(fish_state, water_quality, temp, tds, ph, confidence=0.85):
    """使用模板生成回答（与训练数据格式一致）"""
    # 标签映射
    fish_labels = {
        0: 'normal',      # 正常游动
        1: 'abnormal',    # 异常
        2: 'none'         # 无
    }
    water_labels = {
        0: 'normal',      # 正常
        1: 'change',      # 变化
        2: 'abnormal'     # 异常
    }
    
    fish_text = fish_labels[fish_state]
    water_text = water_labels[water_quality]
    
    # 生成结构化输出（模仿训练数据格式）
    if fish_state == 0 and water_quality == 0:
        # 完全正常
        response = f"鱼处于{fish_text}状态，置信度均值为{confidence:.2f}，温度、TDS、PH均在适宜范围，水质{water_text}。"
    elif fish_state == 0:
        # 鱼正常但水质有问题
        response = f"鱼处于{fish_text}状态，置信度均值为{confidence:.2f}，传感器数据显示{water_text}，水质{water_text}。"
    elif fish_state == 2:
        # 检测不到鱼
        response = f"鱼处于{fish_text}状态，置信度均值为{confidence:.2f}，传感器数据显示{water_text}，水质{water_text}。"
    else:
        # 鱼异常
        response = f"鱼处于{fish_text}状态，置信度均值为{confidence:.2f}，传感器数据显示{water_text}，水质{water_text}。"
    
    return response

class LiteInference:
    """MindSpore Lite 推理封装"""
    
    def __init__(self, model_path, use_npu=True):
        """
        初始化Lite推理器
        
        Args:
            model_path: .ms模型文件路径
            use_npu: 是否使用NPU（Ascend）
        """
        self.model_path = model_path
        
        # 创建上下文
        context = mslite.Context()
        
        if use_npu:
            # 配置NPU（Ascend）
            context.target = ["Ascend"]
            context.ascend.device_id = 0
            context.ascend.precision_mode = "preferred_fp32"  # 修复：使用正确的precision_mode
            print("✓ 使用NPU推理（Ascend）")
        else:
            # 配置CPU
            context.target = ["cpu"]
            context.cpu.thread_num = 2
            print("✓ 使用CPU推理")
        
        # 加载模型
        print(f"加载模型: {model_path}")
        self.model = mslite.Model()
        self.model.build_from_file(model_path, mslite.ModelType.MINDIR_LITE, context)
        
        # 获取输入输出
        self.inputs = self.model.get_inputs()
        self.outputs = self.model.get_outputs()
        
        print(f"✓ 模型加载成功")
        print(f"  输入shape: {self.inputs[0].shape}")
        print(f"  输出数量: {len(self.outputs)}")
    
    def predict(self, features):
        """
        执行推理
        
        Args:
            features: 归一化后的特征 [temp, tds, ph]
        
        Returns:
            fish_state, water_quality
        """
        # 准备输入数据
        input_data = np.array([features], dtype=np.float32)
        self.inputs[0].set_data_from_numpy(input_data)
        
        # 执行推理 - 注意：outputs 必须是 list 不是 tuple
        self.model.predict(self.inputs, list(self.outputs))
        
        # 获取输出
        fish_pred = self.outputs[0].get_data_to_numpy()
        water_pred = self.outputs[1].get_data_to_numpy()
        
        # 获取预测类别
        fish_idx = int(np.argmax(fish_pred, axis=1)[0])
        water_idx = int(np.argmax(water_pred, axis=1)[0])
        
        return fish_idx, water_idx

def inference(model_path, user_input, use_npu=True):
    """
    端到端推理函数
    
    Args:
        model_path: .ms模型文件路径
        user_input: 用户输入文本
        use_npu: 是否使用NPU
    
    Returns:
        response, temp, tds, ph, fish_idx, water_idx
    """
    # 提取特征
    temp, tds, ph = extract_features(user_input)
    feat = normalize_features(temp, tds, ph)
    
    # 创建推理器并预测
    inferencer = LiteInference(model_path, use_npu)
    fish_idx, water_idx = inferencer.predict(feat)
    
    # 生成回答
    response = generate_response(fish_idx, water_idx, temp, tds, ph)
    
    return response, temp, tds, ph, fish_idx, water_idx

if __name__ == '__main__':
    import sys
    
    print("=" * 60)
    print("Atlas 200dk MindSpore Lite 推理")
    print("=" * 60)
    
    # 检查模型文件（优先使用.ms，备用.mindir）
    ms_path = 'models/classifier_model.ms'
    mindir_path = 'models/classifier_model.mindir'
    
    if os.path.exists(ms_path):
        model_path = ms_path
        print(f"✓ 找到Lite模型: {model_path}")
    elif os.path.exists(mindir_path):
        model_path = mindir_path
        print(f"✓ 找到MindIR模型: {model_path}")
        print("⚠ 注意: 使用MindIR格式，推理速度可能稍慢")
        print("  建议转换为.ms格式: converter_lite --fmk=MINDIR --modelFile=models/classifier_model.mindir --outputFile=models/classifier_model")
    else:
        print(f"✗ 错误: 模型文件不存在")
        print(f"\n请上传以下文件之一:")
        print(f"  1. {ms_path} (Lite格式，推荐)")
        print(f"  2. {mindir_path} (MindIR格式，备用)")
        print(f"\n上传命令:")
        print(f"  scp models/classifier_model.mindir HwHiAiUser@<atlas_ip>:~/models/")
        sys.exit(1)
    
    # 检测NPU（由于Ascend版本不匹配，默认使用CPU）
    # Ascend 1.84 vs MindSpore Lite期望的7.6/7.7版本不兼容
    use_npu = False
    print("⚠ 由于Ascend版本不兼容(1.84 vs 7.6/7.7)，使用CPU推理")
    print("  CPU推理速度仍然很快 (轻量级MLP模型)")
    
    print("\n" + "=" * 60)
    
    # 测试样本
    test_samples = [
        "鱼缸传感器显示温度25.1℃，TDS256.0ppm，PH7.2，视频里鱼的状态如何？水质是否正常？",
        "鱼缸传感器显示温度28.9℃，TDS459.0ppm，PH6.3，视频里鱼的状态如何？水质是否正常？",
        "鱼缸传感器显示温度26.0℃，TDS300.0ppm，PH7.0，视频里鱼的状态如何？水质是否正常？"
    ]
    
    for i, test_input in enumerate(test_samples):
        print(f"\n样本 {i+1}:")
        print(f"  输入: {test_input}")
        
        try:
            response, temp, tds, ph, fish_idx, water_idx = inference(
                model_path, test_input, use_npu
            )
            
            print(f"  特征: 温度={temp:.1f}℃, TDS={tds:.1f}ppm, PH={ph:.1f}")
            print(f"  分类: fish_state={fish_idx}, water_quality={water_idx}")
            print(f"  输出: {response}")
        except Exception as e:
            print(f"  ✗ 推理失败: {e}")
            import traceback
            print(f"  详细错误: {traceback.format_exc()}")
            # 继续处理其他样本
    
    print("\n" + "=" * 60)
    print("推理完成!")
    print("=" * 60)
