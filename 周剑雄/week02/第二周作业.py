#尝试完成一个多分类任务的训练:一个随机5维向量，哪一维数字最大就属于第几类。


import numpy as np
import torch
import torch.nn as nn

# 生成随机5维向量样本
def build_sample():
    x = np.random.random(5)
    max_numbers = np.argmax(x)#判断最大值处于第几维返回索引
    return x, max_numbers

def build_dataset(total_sample_num):
    X = []
    Y = []  
    
    for i in range(total_sample_num):
        x, max_numbers = build_sample()
        X.append(x)
        Y.append(max_numbers)
    # print(X)
    # print(Y)
    return torch.FloatTensor(X), torch.LongTensor(Y)

# 测试
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            if torch.argmax(y_p) == int(y_t):
                correct += 1
            else:
                wrong += 1

    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


# 定义简单模型
class SimpleClassifier(nn.Module):
    def __init__(self, input_size):
        super(SimpleClassifier, self).__init__()
        self.linear = nn.Linear(input_size, 5)  # 线性层
        self.loss = nn.functional.cross_entropy # # loss函数采用交叉熵损失 nn.CrossEntropyLoss()内部已包含 softmax + cross entropy 计算

    def forward(self, x, y=None):
        y_pred = self.linear(x)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return torch.softmax(y_pred, dim=1)

# 参数
vector_dim = 5  # 向量维度
num_classes = vector_dim  # 类别数等于维度
num_samples = 5000  # 增加样本数
batch_size = 20 # 每次训练样本个数
epochs = 20  # 增加训练轮数
learning_rate = 0.01  # 学习率

# 生成数据
train_x, train_y = build_dataset(num_samples)



# 模型、损失函数、优化器
#建立模型
model = SimpleClassifier(vector_dim)
# 选择优化器
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
log = []
# 训练
for epoch in range(epochs):
    model.train()
    watch_loss = []
    for batch_index in range( num_samples// batch_size):
        
        x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
        y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
        loss = model(x, y)  # 计算loss model.forward(x, y)
        loss.backward()# 计算梯度
        optimizer.step()# 更新权重
        optimizer.zero_grad()# 梯度归零
        watch_loss.append(loss.item())
    print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
    acc = evaluate(model)  # 测试本轮模型结果
    log.append([acc, float(np.mean(watch_loss))])
torch.save(model.state_dict(), 'model.pt')

# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    model = SimpleClassifier(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res in zip(input_vec, result):
        print("输入：%s, 预测类别：%s, 概率值：%s" % (vec, torch.argmax(res), res))  # 打印结果
