import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F


x = torch.linspace(-3, 3, 100).reshape(100, 1)
# 建立从-1到1之间一百个点并且改变他的阶数（从【1,100】变到【100,1】）
# y = x.pow(2)
y = torch.sin(x) + torch.cos(2 * x) + torch.sin(3 * x)
# 建立x与y之间的关系y=x^2
y_real = torch.normal(y, 0.05)
# 在实际过程中由于不可避免的因素存在会有误差发生但是围绕实际值上下波动



class Net(torch.nn.Module):

    def __init__(self, n_input, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_input, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x


model = Net(1, 50, 1)  # 假设一个输入，隐藏层里有50个神经元，和1个输出

criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-6, momentum=0.9)

global y_pred
for t in range(30000):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x)

    # Compute and print loss
    loss = criterion(y_pred, y)
    if t % 5000 == 0:
        print("epoch:{},mse:{}".format(t + 1, loss.item()))
        plt.plot(x.tolist(), y.tolist(), color="blue")
        plt.plot(x.tolist(), y_pred.tolist(), color="red")
        plt.show()
    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


plt.plot(x.tolist(), y.tolist(), color="blue")
plt.plot(x.tolist(), y_pred.tolist(), color="red")
plt.show()
