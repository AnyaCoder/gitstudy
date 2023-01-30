import torch
import math
import matplotlib.pyplot as plt


class Fitting_polynomial(torch.nn.Module):
    def __init__(self):
        super(Fitting_polynomial, self).__init__()
        self.a = torch.nn.Parameter(torch.randn(()))
        self.b = torch.nn.Parameter(torch.randn(()))
        self.c = torch.nn.Parameter(torch.randn(()))
        self.d = torch.nn.Parameter(torch.randn(()))

    def forward(self, x):

        y = self.a + self.b * x + self.c * x ** 2 + self.d * x ** 3

        return y

    def string(self):

        return f'y = {self.a.item()} + {self.b.item()} x + {self.c.item()} x^2 + {self.d.item()} x^3'

    def plot_poly(self, x):
        fig = plt.figure(figsize=(14, 8))
        y = self.a + self.b * x + self.c * x ** 2 + self.d * x ** 3
        y = y.detach().numpy()
        plt.plot(x, y, label="fitting")
        plt.legend()


# Create Tensors to hold input and outputs.
x = torch.linspace(-math.pi, math.pi, 100)
y = torch.sin(x)

# Construct our model by instantiating the class defined above
model = Fitting_polynomial()

criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-6, momentum=0.9)

for t in range(20000):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x)

    # Compute and print loss
    loss = criterion(y_pred, y)
    if t % 2000 == 1999:
        print("epoch:{},mse:{}".format(t + 1, loss.item()))
        print(f'Result: {model.string()}')
        plt.plot(x, y, label="raw")
        plt.legend()
        model.plot_poly(x)

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

plt.show()
