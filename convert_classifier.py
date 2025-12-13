"""
PyTorch模型转MindSpore格式 - 分类器版本
支持Checkpoint和MindIR两种格式
"""
import torch
import mindspore as ms
from mindspore import nn, Tensor, export
import numpy as np
import os

# MindSpore版分类器（与PyTorch完全一致）
class FishTankClassifierMS(nn.Cell):
    def __init__(self, input_dim=3, hidden_dim=64):
        super(FishTankClassifierMS, self).__init__()
        self.fc1 = nn.Dense(input_dim, hidden_dim)
        self.fc2 = nn.Dense(hidden_dim, hidden_dim)
        
        # 两个分类头
        self.fish_state = nn.Dense(hidden_dim, 3)
        self.water_quality = nn.Dense(hidden_dim, 3)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
    
    def construct(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        
        fish = self.fish_state(x)
        water = self.water_quality(x)
        
        return fish, water

def convert_classifier():
    """转换分类器模型为Checkpoint和MindIR格式"""
    print("=" * 60)
    print("PyTorch → MindSpore 转换")
    print("=" * 60)
    
    # 加载PyTorch模型
    print("\n1. 加载PyTorch模型...")
    torch_ckpt = torch.load('models/classifier_model.pth', map_location='cpu')
    torch_state = torch_ckpt['model_state_dict']
    
    print("PyTorch模型参数:")
    for k, v in torch_state.items():
        print(f"  {k}: {v.shape}")
    
    # 转换参数为MindSpore Checkpoint
    print("\n2. 转换为MindSpore Checkpoint...")
    ms_params = []
    
    for name, param in torch_state.items():
        # PyTorch → NumPy → MindSpore
        np_param = param.cpu().detach().numpy()
        
        # MindSpore参数格式
        ms_param = {
            'name': name,
            'data': Tensor(np_param, dtype=ms.float32)
        }
        ms_params.append(ms_param)
        print(f"  ✓ {name}")
    
    # 保存为MindSpore checkpoint
    print("\n3. 保存MindSpore Checkpoint...")
    ms.save_checkpoint(ms_params, 'models/classifier_model.ckpt')
    print("✓ 已保存到 models/classifier_model.ckpt")
    
    # 导出MindIR格式（用于Atlas NPU推理）
    print("\n4. 导出MindIR格式...")
    ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")
    
    # 创建模型并加载参数
    model = FishTankClassifierMS(input_dim=3, hidden_dim=64)
    param_dict = ms.load_checkpoint('models/classifier_model.ckpt')
    ms.load_param_into_net(model, param_dict)
    model.set_train(False)
    
    # 定义输入shape
    input_shape = Tensor(np.zeros([1, 3], dtype=np.float32))
    
    # 导出MindIR
    export(model, input_shape, file_name='models/classifier_model', file_format='MINDIR')
    print("✓ 已保存到 models/classifier_model.mindir")
    
    # 验证
    print("\n5. 验证转换结果...")
    
    # 验证Checkpoint
    loaded_ckpt = ms.load_checkpoint('models/classifier_model.ckpt')
    print(f"✓ Checkpoint包含 {len(loaded_ckpt)} 个参数")
    
    # 验证MindIR
    if os.path.exists('models/classifier_model.mindir'):
        file_size = os.path.getsize('models/classifier_model.mindir') / 1024
        print(f"✓ MindIR文件大小: {file_size:.2f} KB")
    
    print("\n" + "=" * 60)
    print("转换完成！")
    print("=" * 60)
    print("\n生成的文件:")
    print("1. models/classifier_model.ckpt   - Checkpoint格式（调试用）")
    print("2. models/classifier_model.mindir - MindIR格式（生产推荐）")
    print("\n下一步:")
    print("1. 上传到 Atlas 200dk:")
    print("   scp models/classifier_model.mindir HwHiAiUser@<atlas_ip>:~/models/")
    print("   scp inference_classifier.py HwHiAiUser@<atlas_ip>:~/")
    print("2. 在 Atlas 上运行:")
    print("   python inference_classifier.py")

if __name__ == '__main__':
    convert_classifier()
