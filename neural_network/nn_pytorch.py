import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# 定义网络结构
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 2)  # 输入层→隐藏层
        self.fc2 = nn.Linear(2, 1)  # 隐藏层→输出层

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


# 准备数据
X = torch.tensor([[-2, -1], [25, 6], [17, 4], [-15, -6]], dtype=torch.float32)
# 对应的真实标签(1表示女性，0表示男性)
y = torch.tensor([[1], [0], [0], [1]], dtype=torch.float32)

# 训练模型
# criterion: 损失函数（回归任务常用MSE）
# optimizer: 优化器，model.parameters()自动获取所有可训练参数
# lr=0.1: 学习率（控制参数更新步长）
model = Net()
print(model.fc1)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# 训练循环
losses = []
for epoch in range(1000):
    optimizer.zero_grad()  # 清空梯度
    outputs = model(X)  # 前向传播
    loss = criterion(outputs, y)  # 计算损失
    loss.backward()  # 反向传播计算梯度
    optimizer.step()  # 更新参数
    losses.append(loss.item())  # 记录损失值

# 可视化
plt.figure(figsize=(10, 5))
plt.plot(losses, label='Training Loss', color='blue', linewidth=2)
plt.title('Loss Curve', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('MSE Loss', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()  # 关键！必须调用show()显示图像