import torch
import torch.nn as nn

# 定义线性层：输入维度=3，输出维度=2
linear_layer = nn.Linear(3, 2, bias=True)

# 手动初始化权重（实际训练时会自动初始化）
linear_layer.weight.data = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
linear_layer.bias.data = torch.tensor([0.1, 0.2])

# 输入张量（形状=(batch_size, in_features)）
x = torch.tensor([[1., 2., 3.]])  # 形状=(1, 3)

# 前向计算
y = linear_layer(x)
print(y)

