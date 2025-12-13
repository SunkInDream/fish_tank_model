"""
Atlas 200dk 推理脚本 - MindSpore版分类器
支持MindIR文件，NPU加速推理，模板填充输出
"""
import mindspore as ms
from mindspore import nn, ops, Tensor
import numpy as np
import re
import os

# MindSpore版分类器（与PyTorch结构完全一致）
class FishTankClassifierMS(nn.Cell):
    def __init__(self, input_dim=3, hidden_dim=64):
        super(FishTankClassifierMS, self).__init__()
        self.fc1 = nn.Dense(input_dim, hidden_dim)
        self.fc2 = nn.Dense(hidden_dim, hidden_dim)
        
        # 两个分类头
        self.fish_state = nn.Dense(hidden_dim, 3)
        self.water_quality = nn.Dense(hidden_dim, 3)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)  # 推理时不生效
    
    def construct(self, x):
        # x: [batch, 3] - 温度、TDS、PH（归一化）
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        
        fish = self.fish_state(x)
        water = self.water_quality(x)
        
        return fish, water

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

def inference(model_path, user_input, use_npu=True):
    """推理函数 - 支持MindIR和Checkpoint两种格式"""
    
    # 设置推理设备
    if use_npu:
        # Atlas 200dk NPU推理
        ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=0)
        print("使用NPU推理")
    else:
        # 本地CPU测试
        ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")
        print("使用CPU推理")
    
    # 提取特征
    temp, tds, ph = extract_features(user_input)
    feat = normalize_features(temp, tds, ph)
    x = Tensor([feat], dtype=ms.float32)
    
    # 判断模型格式并加载
    if model_path.endswith('.mindir'):
        # 加载MindIR文件（推荐用于生产环境）
        print(f"加载MindIR模型: {model_path}")
        graph = ms.load(model_path)
        model = nn.GraphCell(graph)
        
        # MindIR推理
        fish_pred, water_pred = model(x)
    else:
        # 加载Checkpoint文件
        print(f"加载Checkpoint模型: {model_path}")
        model = FishTankClassifierMS(input_dim=3, hidden_dim=64)
        param_dict = ms.load_checkpoint(model_path)
        ms.load_param_into_net(model, param_dict)
        model.set_train(False)
        
        # Checkpoint推理
        fish_pred, water_pred = model(x)
    
    # 获取预测类别
    fish_idx = int(ops.argmax(fish_pred, axis=1).asnumpy()[0])
    water_idx = int(ops.argmax(water_pred, axis=1).asnumpy()[0])
    
    # 生成回答（传入传感器数据）
    response = generate_response(fish_idx, water_idx, temp, tds, ph)
    
    return response, temp, tds, ph, fish_idx, water_idx

if __name__ == '__main__':
    import sys
    
    print("=" * 60)
    print("Atlas 200dk 分类器推理测试")
    print("=" * 60)
    
    # 检查模型文件
    mindir_path = 'models/classifier_model.mindir'
    ckpt_path = 'models/classifier_model.ckpt'
    
    if os.path.exists(mindir_path):
        model_path = mindir_path
        print(f"✓ 找到MindIR模型: {mindir_path}")
    elif os.path.exists(ckpt_path):
        model_path = ckpt_path
        print(f"✓ 找到Checkpoint模型: {ckpt_path}")
    else:
        print(f"✗ 错误: 模型文件不存在")
        print(f"   请先运行 convert_classifier.py 生成模型")
        sys.exit(1)
    
    # 检测是否在Atlas 200dk上（判断NPU设备）
    try:
        ms.set_context(device_target="Ascend")
        use_npu = True
        print("✓ 检测到NPU设备，使用NPU推理")
    except:
        use_npu = False
        print("⚠ 未检测到NPU设备，使用CPU推理")
    
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
        
        response, temp, tds, ph, fish_idx, water_idx = inference(model_path, test_input, use_npu)
        
        print(f"  特征: 温度={temp:.1f}℃, TDS={tds:.1f}ppm, PH={ph:.1f}")
        print(f"  分类: fish_state={fish_idx}, water_quality={water_idx}")
        print(f"  输出: {response}")
    
    print("\n" + "=" * 60)
    print("推理完成!")
    print("=" * 60)
