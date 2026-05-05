# coding:utf8
import os
import torch
import torch.nn as nn
import numpy as np
# import matplotlib.pyplot as plt
import random
from torch.utils.data import Dataset, DataLoader

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
五维判断：包含“你”的句子是一个5维向量，向量中"你"所在的位置就输出哪一维下标

"""

# ─── 超参数 ────────────────────────────────────────────────
SEED        = 42
N_SAMPLES   = 4000  # 样本数量
MAXLEN      = 5     # 句子固定长度为5（对应5维向量）
EMBED_DIM   = 64    # 每个字的维度
HIDDEN_DIM  = 64    # 隐藏层维度
LR          = 1e-3  # 学习率
BATCH_SIZE  = 64    # 批次大小
EPOCHS      = 20    # 训练轮数
TRAIN_RATIO = 0.8   # 训练集比例

random.seed(SEED)
torch.manual_seed(SEED)

# 数据生成：生成5字句子，“你”在随机位置0-4，标签为位置索引
def make_sample():
    words = ['好', '棒', '赞', '喜', '满', '这', '家', '设', '让', '的', '务', '态', '验', '非', '常', '物', '觉', '极', '服', '体', '感']
    sent = [random.choice(words) for _ in range(5)]  # 生成5字句子
    pos = random.randint(0, 4)  # “你”的位置
    sent[pos] = '你'  # 替换为“你”
    sent = ''.join(sent)  # 转为字符串
    return sent, pos

def build_dataset(n=N_SAMPLES):
    data = []
    for _ in range(n):
        data.append(make_sample())
    random.shuffle(data)
    return data

# ─── 2. 词表构建与编码 ──────────────────────────────────────
def build_vocab(data):
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for sent, _ in data:
        for ch in sent:
            if ch not in vocab:
                vocab[ch] = len(vocab)
    return vocab

def encode(sent, vocab, maxlen=MAXLEN):
    ids = [vocab.get(ch, 1) for ch in sent]
    ids = ids[:maxlen]
    ids += [0] * (maxlen - len(ids))
    return ids

# ─── 3. Dataset / DataLoader ────────────────────────────────
class TextDataset(Dataset):
    def __init__(self, data, vocab):
        self.X = [encode(s, vocab) for s, _ in data]
        self.y = [lb for _, lb in data]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (
            torch.tensor(self.X[i], dtype=torch.long),
            torch.tensor(self.y[i], dtype=torch.long),
        )

# ─── 4. 模型定义 ────────────────────────────────────────────
class PositionRNN(nn.Module):
    """
    位置预测模型（RNN + MaxPooling 版）
    架构：Embedding → RNN → MaxPool → BN → Dropout → Linear → Softmax
    输出5维概率分布，对应“你”的位置0-4
    """
    def __init__(self, vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn       = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.bn        = nn.BatchNorm1d(hidden_dim)
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_dim, 5)  # 输出5维

    def forward(self, x):
        # x: (batch, seq_len)
        e, _ = self.rnn(self.embedding(x))  # (B, L, hidden_dim)
        pooled = e.max(dim=1)[0]            # (B, hidden_dim)  对序列做 max pooling
        pooled = self.dropout(self.bn(pooled))
        out = self.fc(pooled)  # (B, 5)  logits
        return out

# ─── 5. 训练与评估 ──────────────────────────────────────────
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X, y in loader:
            logits = model(X)
            pred = torch.argmax(logits, dim=1)
            correct += (pred == y).sum().item()
            total += len(y)
    return correct / total

def train():
    print("生成数据集...")
    data = build_dataset(N_SAMPLES)
    vocab = build_vocab(data)
    print(f"  样本数：{len(data)}，词表大小：{len(vocab)}")

    split = int(len(data) * TRAIN_RATIO)
    train_data = data[:split]
    val_data = data[split:]

    train_loader = DataLoader(TextDataset(train_data, vocab), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TextDataset(val_data, vocab), batch_size=BATCH_SIZE)

    model = PositionRNN(vocab_size=len(vocab))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量：{total_params:,}\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for X, y in train_loader:
            logits = model(X)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_acc = evaluate(model, val_loader)
        print(f"Epoch {epoch:2d}/{EPOCHS}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}")

    print(f"\n最终验证准确率：{evaluate(model, val_loader):.4f}")

    print("\n--- 推理示例 ---")
    model.eval()
    test_sents = [
        '你好棒赞喜',
        '好你棒赞喜',
        '好棒你赞喜',
        '好棒赞你喜',
        '好棒赞喜你',
    ]
    with torch.no_grad():
        for sent in test_sents:
            ids = torch.tensor([encode(sent, vocab)], dtype=torch.long)
            logits = model(ids)
            pred_pos = torch.argmax(logits, dim=1).item()
            print(f"  句子：{sent}  → 预测“你”位置：{pred_pos}")

if __name__ == '__main__':
    train()
