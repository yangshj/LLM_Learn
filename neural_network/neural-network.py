import numpy as np
import matplotlib.pyplot as plt


# Sigmoid激活函数: f(x) = 1 / (1 + e^(-x))
# 将任意输入映射到(0,1)区间，常用于二分类问题
def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
  # sigmoid的导数: f'(x) = f(x) * (1 - f(x))
  fx = sigmoid(x)
  return fx * (1 - fx)

# 损失函数：均方误差
# y_true - y_pred：计算每个样本的误差（支持广播机制）。
# ** 2：对误差平方（放大大误差，忽略正负）。
# .mean()：求所有样本平方误差的平均值（即MSE）mean squared error。
def mse_loss(y_true, y_pred):
  #   均方误差损失函数
  #     计算预测值与真实值的平均平方差
  #     参数:
  #         y_true: 真实值数组
  #         y_pred: 预测值数组
  #     返回:
  #         平均平方误差
  return ((y_true - y_pred) ** 2).mean()

# 可视化损失值
losses = []

class OurNeuralNetwork:
  '''
  一个简单的神经网络实现:
    - 2个输入神经元
    - 1个隐藏层(2个神经元h1,h2)
    - 1个输出层(1个神经元o1)
    注意: 此代码为教学目的而简化，实际神经网络实现要复杂得多
  '''
  def __init__(self):
    # 输入层到隐藏层的权重
    self.w1 = np.random.normal()  # 输入1到h1的权重
    self.w2 = np.random.normal()  # 输入2到h1的权重
    self.w3 = np.random.normal()  # 输入1到h2的权重
    self.w4 = np.random.normal()  # 输入2到h2的权重

    # 隐藏层到输出层的权重
    self.w5 = np.random.normal()  # h1到o1的权重
    self.w6 = np.random.normal()  # h2到o1的权重

    # Biases
    self.b1 = np.random.normal()
    self.b2 = np.random.normal()
    self.b3 = np.random.normal()


  def feedforward(self, x):
    '''
        前向传播计算
        参数:
            x: 包含2个特征的输入数组
        返回:
            网络的预测输出(0到1之间的值)
    '''
    # 计算隐藏层神经元的加权和与激活值
    h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
    h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
    o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
    return o1

  def train(self, data, all_y_trues):
    '''
      训练神经网络
      参数:
          data: (n×2)的numpy数组，n个样本的特征数据
          all_y_trues: 包含n个元素的数组，对应样本的真实标签
    '''
    learn_rate = 0.1   # 学习率(梯度下降的步长)
    epochs = 1000      # 训练轮数(整个数据集的迭代次数)

    for epoch in range(epochs):
      for x, y_true in zip(data, all_y_trues):
        # --- 前向传播(保留中间结果用于反向传播) ---
        # 计算隐藏层
        sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
        h1 = sigmoid(sum_h1)

        sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
        h2 = sigmoid(sum_h2)

        sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
        o1 = sigmoid(sum_o1)
        y_pred = o1 # 预测值数组

        # --- 反向传播: 计算各参数的梯度 ---
        # 损失函数对预测值的导数
        d_L_d_ypred = -2 * (y_true - y_pred)

        # 输出层(o1)各参数的梯度
        d_ypred_d_w5 = h1 * deriv_sigmoid(sum_o1)  # w5的梯度(预测值对w5的偏导数)
        d_ypred_d_w6 = h2 * deriv_sigmoid(sum_o1)  # w6的梯度(预测值对w6的偏导数)
        d_ypred_d_b3 = deriv_sigmoid(sum_o1)       # b3的梯度(预测值对b3的偏导数)

        # 隐藏层对预测值的影响
        d_ypred_d_h1 = self.w5 * deriv_sigmoid(sum_o1)  # h1对输出的影响(预测值对h1的偏导数)
        d_ypred_d_h2 = self.w6 * deriv_sigmoid(sum_o1)  # h2对输出的影响(预测值对h2的偏导数)

        # 隐藏层(h1)各参数的梯度
        d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1)  # w1的梯度(h1对w1偏导数)
        d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1)  # w2的梯度(h1对w2偏导数)
        d_h1_d_b1 = deriv_sigmoid(sum_h1)  # b1的梯度(h1对b1的偏导数)

        # 隐藏层(h2)各参数的梯度
        d_h2_d_w3 = x[0] * deriv_sigmoid(sum_h2)  # w3的梯度(h2对w3偏导数)
        d_h2_d_w4 = x[1] * deriv_sigmoid(sum_h2)  # w4的梯度(h2对w4偏导数)
        d_h2_d_b2 = deriv_sigmoid(sum_h2)  # b2的梯度(h2对b2偏导数)

        # --- 更新权重和偏置(梯度下降) ---
        # 更新h1的参数：如果（d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1）为正数则减小w1的值，否则增加w1的值。
        self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1 #损失值对w1的偏导数
        self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2 #损失值对w2的偏导数
        self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1 #损失值对b1的偏导数

        # 更新h2的参数
        self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3 #损失值对w3的偏导数
        self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4 #损失值对w4的偏导数
        self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2 #损失值对b2的偏导数

        # 更新o1的参数
        self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_w5
        self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_w6
        self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_b3


      # 每10轮输出一次损失值
      if epoch % 10 == 0:
        # 计算当前所有样本的预测值
        # np.apply_along_axis() 是 NumPy 中的一个函数，用于沿着数组的指定轴应用一个函数
        # 对 data 数组的每一行（因为 axis=1）应用 self.feedforward 函数
        # 将每一行的数据作为参数传递给 self.feedforward
        # 将所有结果收集起来并返回
        y_preds = np.apply_along_axis(self.feedforward, 1, data)
        # 计算并输出损失
        loss = mse_loss(all_y_trues, y_preds)
        print("训练轮次 %d 损失值: %.3f" % (epoch, loss))
        losses.append(loss.item())  # 记录损失值


#这里使用的数据集是 4 个样本，根据体重和身高预测性别（1 代表女性，0 代表男性），体重是数值 135 的偏差，身高是数值 66 的偏差。
# Define dataset
# 1 英寸(inch) = 2.54 厘米(cm)
# 1 斤 ≈ 1.102 磅
data = np.array([
  [-2, -1],  # Alice
  [25, 6],   # Bob
  [17, 4],   # Charlie
  [-15, -6], # Diana
])

# 对应的真实标签(1表示女性，0表示男性)
all_y_trues = np.array([
  1, # Alice
  0, # Bob
  0, # Charlie
  1, # Diana
])

# 创建并训练神经网络
network = OurNeuralNetwork()
network.train(data, all_y_trues)

# 对新样本进行预测
emily = np.array([-7, -3])  # 测试样本1
frank = np.array([20, 2])   # 测试样本2
print("Emily预测值: %.3f" % network.feedforward(emily))  # 应接近1(女性)
print("Frank预测值: %.3f" % network.feedforward(frank))  # 应接近0(男性)


# 可视化
plt.figure(figsize=(10, 5))
plt.plot(losses, label='Training Loss', color='blue', linewidth=2)
plt.title('Loss Curve', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('MSE Loss', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()  # 关键！必须调用show()显示图像