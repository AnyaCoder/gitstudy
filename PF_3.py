import torch
import matplotlib.pyplot as plt
import numpy as np

file = open('DailyLifeGoods.txt', "r")

x = []
num = []
weight = []
line = file.readline()
for i in range(9000):
    # print(line, end="")
    elems = line.split(" ")

    x.append(int(elems[0]))
    num.append(int(elems[1]))
    weight.append(float(elems[2]))

    line = file.readline()

file.close()

N_SAMPLES = 40  # 样本点个数
N_HIDDEN = 100  # 隐藏层神经元

_x = np.random.choice(a=x, size=N_SAMPLES, replace=False)
_x = sorted(_x)
y = []
for i in _x:
    y.append(num[i])

# train数据
x = torch.unsqueeze(torch.linspace(1, N_SAMPLES, N_SAMPLES), 1)  # 一列数
y = np.array(y)  # 转换成 numpy.array 类型
y = torch.from_numpy(y).unsqueeze(1)  # 转换成torch.tensor类型并且把行向量变成列向量
y = y + 3 * torch.normal(torch.zeros(N_SAMPLES, 1), torch.ones(N_SAMPLES, 1))  # 添加正态分布扰动
# y = x + 0.3 * torch.normal(torch.zeros(N_SAMPLES, 1), torch.ones(N_SAMPLES, 1))  # 一列数
# test数据
test_x = torch.unsqueeze(torch.linspace(1, N_SAMPLES, N_SAMPLES), 1)
test_y = y + 2 * torch.normal(torch.zeros(N_SAMPLES, 1), torch.ones(N_SAMPLES, 1))
# 可视化
plt.scatter(x.data.numpy(), y.data.numpy(), c='magenta', s=50, alpha=0.5, label='train')
plt.scatter(test_x.data.numpy(), test_y.data.numpy(), c='cyan', s=50, alpha=0.5, label='test')
plt.legend(loc='upper left')
plt.ylim((0, 200))
plt.show()
# 网络一，未使用dropout正规化
net_overfitting = torch.nn.Sequential(
 torch.nn.Linear(1, N_HIDDEN),
 torch.nn.ReLU(),
 torch.nn.Linear(N_HIDDEN, N_HIDDEN),
 torch.nn.ReLU(),
 torch.nn.Linear(N_HIDDEN, 1),
)
# 网络二，使用dropout正规化
net_dropped = torch.nn.Sequential(
 torch.nn.Linear(1, N_HIDDEN),
 torch.nn.Dropout(0.5),  # 随机屏蔽50%的网络连接
 torch.nn.ReLU(),
 torch.nn.Linear(N_HIDDEN, N_HIDDEN),
 torch.nn.Dropout(0.5),  # 随机屏蔽50%的网络连接
 torch.nn.ReLU(),
 torch.nn.Linear(N_HIDDEN, 1),
)
# 选择优化器
optimizer_ofit = torch.optim.Adam(net_overfitting.parameters(), lr=0.001)
optimizer_drop = torch.optim.Adam(net_dropped.parameters(), lr=0.001)
# 选择计算误差的工具
loss_func = torch.nn.MSELoss()
plt.ion()
for t in range(1000):
    # 神经网络训练数据的固定过程
    pred_ofit = net_overfitting(x)  # 预测结果
    pred_drop = net_dropped(x)
    loss_ofit = loss_func(pred_ofit, y)  # 计算损失
    loss_drop = loss_func(pred_drop, y)
    optimizer_ofit.zero_grad()  # 梯度清理，防止梯度累加对后继计算带来的影响
    optimizer_drop.zero_grad()
    loss_ofit.backward()  # 损失函数反向传播，梯度增加
    loss_drop.backward()
    optimizer_ofit.step()  # 优化器，根据梯度来对参数进行调整
    optimizer_drop.step()
    if t % 10 == 0:
        # 脱离训练模式，这里便于展示神经网络的变化过程
        net_overfitting.eval()
        net_dropped.eval()
        # 可视化
        plt.cla()
        test_pred_ofit = net_overfitting(test_x)
        test_pred_drop = net_dropped(test_x)
        plt.scatter(x.data.numpy(), y.data.numpy(), c='magenta', s=50, alpha=0.3, label='train')
        plt.scatter(test_x.data.numpy(), test_y.data.numpy(), c='cyan', s=50, alpha=0.3, label='test')
        plt.plot(test_x.data.numpy(), test_pred_ofit.data.numpy(), 'r-', lw=3, label='overfitting')
        plt.plot(test_x.data.numpy(), test_pred_drop.data.numpy(), 'b--', lw=3, label='dropout(50%)')
        plt.text(0, 35, 'overfitting loss=%.4f' % loss_func(test_pred_ofit, test_y).data.numpy(), fontdict={'size': 18, 'color':  'red'})
        plt.text(0, 10, 'dropout loss=%.4f' % loss_func(test_pred_drop, test_y).data.numpy(), fontdict={'size': 18, 'color': 'blue'})
        plt.legend(loc='upper left')
        plt.ylim((0, 200))
        plt.pause(0.1)

        # 重新进入训练模式，并继续上次训练
        net_overfitting.train()
        net_dropped.train()
plt.ioff()
plt.show()