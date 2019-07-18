import torch
import torch.nn as nn
import torch.optim as optim

"""
equal to optimize problem: "find x when y = x and target = 2"

"""
class Net(nn.Module):  
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 1, bias=False)
    def forward(self, x):
        x = self.fc1(x)
        return x

net = Net()
inp = torch.tensor([[1.0]])
target = torch.tensor([[2.0]])

criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)

iter = 0
# every iter
while iter<500:
    output = net(inp) # compute the output
    loss = criterion(output, target) # compute the loss
    optimizer.zero_grad() # zero the gradient buffers
    loss.backward() # backprop the loss
    optimizer.step() # update the parameters
    iter += 1
print(list(net.parameters()))    #result = 2

