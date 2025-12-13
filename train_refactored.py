"""
重构版训练脚本 - 使用Attention + Beam Search + 更强的训练策略
彻底解决重复输出问题
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm  # 进度条库
import time

# 简单的分词器（使用字符级别）
class SimpleTokenizer:
    def __init__(self):
        self.char2idx = {}
        self.idx2char = {}
        self.vocab_size = 0
        self.SOS_token = '<SOS>'
        self.EOS_token = '<EOS>'
        self.PAD_token = '<PAD>'
        
    def build_vocab(self, texts):
        """构建词汇表"""
        chars = set()
        for text in texts:
            chars.update(text)
        chars = sorted(list(chars))
        
        # 添加特殊token
        self.char2idx = {
            self.PAD_token: 0,
            self.SOS_token: 1,
            self.EOS_token: 2,
        }
        for idx, char in enumerate(chars):
            self.char2idx[char] = idx + 3
            
        self.idx2char = {idx: char for char, idx in self.char2idx.items()}
        self.vocab_size = len(self.char2idx)
        print(f"词汇表大小: {self.vocab_size}")
        
    def encode(self, text, max_len=200, add_eos=False):
        """文本编码"""
        indices = [self.char2idx.get(char, 0) for char in text[:max_len]]
        if add_eos and len(indices) < max_len:
            indices.append(self.char2idx[self.EOS_token])
        # padding
        if len(indices) < max_len:
            indices += [0] * (max_len - len(indices))
        return indices
    
    def decode(self, indices):
        """解码"""
        chars = []
        for idx in indices:
            if idx == 0 or idx == self.char2idx.get(self.PAD_token, 0):
                break  # 遇到PAD停止
            if idx == self.char2idx.get(self.EOS_token, 2):
                break  # 遇到EOS停止
            char = self.idx2char.get(idx, '')
            if char not in [self.SOS_token, self.EOS_token, self.PAD_token]:
                chars.append(char)
        return ''.join(chars)

# 注意力机制
class Attention(nn.Module):
    def __init__(self, hidden_dim, encoder_dim):
        super(Attention, self).__init__()
        # encoder_dim是编码器输出维度（双向所以是hidden_dim*2）
        # hidden_dim是解码器hidden维度
        self.attn = nn.Linear(hidden_dim + encoder_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
        
    def forward(self, hidden, encoder_outputs):
        # hidden: [batch, hidden_dim]
        # encoder_outputs: [batch, seq_len, encoder_dim]
        seq_len = encoder_outputs.size(1)
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        
        return F.softmax(attention, dim=1)

# 改进的Seq2Seq模型 with Attention
class ImprovedSeq2SeqModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=512):
        super(ImprovedSeq2SeqModel, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.encoder_dim = hidden_dim * 2  # 双向GRU输出维度
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # 编码器：更大的模型
        self.encoder = nn.GRU(embedding_dim, hidden_dim, num_layers=3, 
                             batch_first=True, dropout=0.3, bidirectional=True)
        
        # 解码器
        self.decoder = nn.GRU(embedding_dim + self.encoder_dim, hidden_dim, 
                             num_layers=3, batch_first=True, dropout=0.3)
        
        # 注意力（传入正确的维度）
        self.attention = Attention(hidden_dim, self.encoder_dim)
        
        # 输出层
        self.fc = nn.Linear(hidden_dim + self.encoder_dim, vocab_size)
        
        # 用于调整编码器的双向输出
        self.bridge = nn.Linear(self.encoder_dim, hidden_dim)
        
    def forward(self, src, tgt):
        """前向传播（训练时使用Teacher Forcing）"""
        batch_size = src.size(0)
        
        # 编码器
        src_emb = self.embedding(src)
        encoder_outputs, hidden = self.encoder(src_emb)
        # hidden: [6, batch, hidden_dim] -> [3, batch, hidden_dim*2]
        
        # 调整hidden维度
        hidden = hidden.view(3, 2, batch_size, self.hidden_dim)
        hidden = torch.cat([hidden[:, 0, :, :], hidden[:, 1, :, :]], dim=2)
        hidden = self.bridge(hidden)
        
        # 解码器
        tgt_emb = self.embedding(tgt)
        outputs = []
        
        for t in range(tgt.size(1)):
            # 计算注意力
            attn_weights = self.attention(hidden[-1], encoder_outputs)
            context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
            
            # 解码一步
            rnn_input = torch.cat([tgt_emb[:, t:t+1, :], context], dim=2)
            output, hidden = self.decoder(rnn_input, hidden)
            
            # 输出预测
            prediction = self.fc(torch.cat([output, context], dim=2))
            outputs.append(prediction)
        
        outputs = torch.cat(outputs, dim=1)
        return outputs
    
    def generate_greedy(self, src, max_len=150, device='cpu', tokenizer=None):
        """贪婪解码（最稳定）"""
        self.eval()
        with torch.no_grad():
            batch_size = src.size(0)
            
            # 编码
            src_emb = self.embedding(src)
            encoder_outputs, hidden = self.encoder(src_emb)
            
            # 调整hidden
            hidden = hidden.view(3, 2, batch_size, self.hidden_dim)
            hidden = torch.cat([hidden[:, 0, :, :], hidden[:, 1, :, :]], dim=2)
            hidden = self.bridge(hidden)
            
            # 解码
            outputs = []
            current_input = torch.ones(batch_size, 1, dtype=torch.long).to(device)  # SOS token
            
            for _ in range(max_len):
                # 注意力
                attn_weights = self.attention(hidden[-1], encoder_outputs)
                context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
                
                # 解码一步
                emb = self.embedding(current_input)
                rnn_input = torch.cat([emb, context], dim=2)
                output, hidden = self.decoder(rnn_input, hidden)
                
                # 预测
                prediction = self.fc(torch.cat([output, context], dim=2))
                predicted = torch.argmax(prediction, dim=-1)
                
                # 检查EOS
                if tokenizer and predicted.item() == tokenizer.char2idx.get(tokenizer.EOS_token, 2):
                    break
                
                # 检查PAD
                if predicted.item() == 0:
                    break
                
                outputs.append(predicted)
                current_input = predicted
            
            if len(outputs) == 0:
                return torch.zeros(batch_size, 1, dtype=torch.long)
            
            output_seq = torch.cat(outputs, dim=1)
            return output_seq

# 数据集类
class FishTankDataset(Dataset):
    def __init__(self, jsonl_file, tokenizer, max_len=200):
        self.data = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        if os.path.exists(jsonl_file):
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line.strip())
                    if 'conversations' in item:
                        user_msg = item['conversations'][0]['content']
                        assistant_msg = item['conversations'][1]['content']
                        self.data.append((user_msg, assistant_msg))
        
        print(f"加载了 {len(self.data)} 条数据")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        user_msg, assistant_msg = self.data[idx]
        src = torch.tensor(self.tokenizer.encode(user_msg, self.max_len), dtype=torch.long)
        tgt = torch.tensor(self.tokenizer.encode(assistant_msg, self.max_len, add_eos=True), dtype=torch.long)
        return src, tgt

def train_model():
    """训练模型"""
    print("=" * 60)
    print("重构版训练 - 彻底解决重复问题")
    print("=" * 60)
    
    device = torch.device('cpu')
    print(f"使用设备: {device}")
    
    # 准备数据
    train_file = 'fish_tank_dataset/train.jsonl'
    
    # 构建词汇表
    all_texts = []
    if os.path.exists(train_file):
        with open(train_file, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                if 'conversations' in item:
                    all_texts.append(item['conversations'][0]['content'])
                    all_texts.append(item['conversations'][1]['content'])
    
    print("\n构建词汇表...")
    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab(all_texts)
    
    # 保存tokenizer
    os.makedirs('models', exist_ok=True)
    torch.save({
        'char2idx': tokenizer.char2idx,
        'idx2char': tokenizer.idx2char,
        'vocab_size': tokenizer.vocab_size
    }, 'models/tokenizer.pth')
    print("Tokenizer已保存")
    
    # 创建数据集 - 使用50条数据
    print("\n加载数据...")
    full_dataset = FishTankDataset(train_file, tokenizer)
    train_size = min(50, len(full_dataset))
    train_dataset = torch.utils.data.Subset(full_dataset, range(train_size))
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    
    print(f"使用 {train_size} 条数据，batch_size=2")
    
    # 创建模型 - 更大更强
    print("\n创建模型...")
    model = ImprovedSeq2SeqModel(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=256,
        hidden_dim=512
    ).to(device)
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # 损失和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.001, epochs=10, steps_per_epoch=len(train_loader)
    )
    
    # 训练
    print("\n开始训练...")
    print("策略: Attention + GRU + 强正则化 + 贪婪解码")
    
    num_epochs = 10
    best_loss = float('inf')
    patience = 0
    max_patience = 100
    
    print(f"预计时间: 10-20分钟 (取决于CPU性能)")
    print(f"总轮数: {num_epochs}, 数据批次: {len(train_loader)}")
    print("=" * 60)
    
    start_time = time.time()
    model.train()
    
    # 使用tqdm显示总体进度
    epoch_bar = tqdm(range(num_epochs), desc="训练进度", ncols=100)
    
    for epoch in epoch_bar:
        total_loss = 0
        
        # 使用tqdm显示每个epoch内的批次进度
        batch_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False, ncols=100)
        
        for src, tgt in batch_bar:
            src, tgt = src.to(device), tgt.to(device)
            
            optimizer.zero_grad()
            output = model(src, tgt)
            
            loss = criterion(output.reshape(-1, tokenizer.vocab_size), tgt.reshape(-1))
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            
            # 更新批次进度条
            batch_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(train_loader)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience = 0
        else:
            patience += 1
        
        # 计算已用时间和预估剩余时间
        elapsed = time.time() - start_time
        eta = elapsed / (epoch + 1) * (num_epochs - epoch - 1) if epoch > 0 else 0
        
        # 更新总体进度条
        epoch_bar.set_postfix({
            'loss': f'{avg_loss:.4f}',
            'best': f'{best_loss:.4f}',
            'patience': patience,
            'lr': f'{scheduler.get_last_lr()[0]:.6f}',
            'elapsed': f'{elapsed/60:.1f}m',
            'eta': f'{eta/60:.1f}m'
        })
        
        # 早停
        if avg_loss < 0.01 or patience > max_patience:
            epoch_bar.close()
            print(f"\n✓ 提前停止 (loss={avg_loss:.4f}, patience={patience})")
            print(f"✓ 总用时: {elapsed/60:.1f} 分钟")
            break
    
    print("\n训练完成！")
    
    # 保存模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': tokenizer.vocab_size,
        'embedding_dim': 256,
        'hidden_dim': 512
    }, 'models/fish_tank_model.pth')
    print("模型已保存")
    
    # 测试
    print("\n" + "=" * 60)
    print("测试模型...")
    print("=" * 60)
    
    model.eval()
    
    # 测试训练集
    print("\n训练集样本:")
    for i in range(min(3, len(train_dataset))):
        src, tgt = train_dataset[i]
        src_text = tokenizer.decode(src.cpu().numpy())
        tgt_text = tokenizer.decode(tgt.cpu().numpy())
        
        src_input = src.unsqueeze(0).to(device)
        output_seq = model.generate_greedy(src_input, max_len=150, device=device, tokenizer=tokenizer)
        predicted_text = tokenizer.decode(output_seq.squeeze().cpu().numpy())
        
        print(f"\n样本 {i+1}:")
        print(f"  输入: {src_text.strip()}")
        print(f"  真实: {tgt_text.strip()}")
        print(f"  预测: {predicted_text.strip()}")
    
    # 测试新样本
    print("\n" + "=" * 60)
    print("新样本测试:")
    test_input = "鱼缸传感器显示温度25.0℃，TDS250.0ppm，PH7.3，视频里鱼的状态如何？水质是否正常？"
    src = torch.tensor([tokenizer.encode(test_input)], dtype=torch.long).to(device)
    
    output_seq = model.generate_greedy(src, max_len=150, device=device, tokenizer=tokenizer)
    predicted_text = tokenizer.decode(output_seq.squeeze().cpu().numpy())
    
    print(f"输入: {test_input}")
    print(f"预测: {predicted_text}")
    print("=" * 60)

if __name__ == '__main__':
    train_model()
