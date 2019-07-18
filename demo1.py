import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3) # 1:input channel; 6:output channel; 3:fliter size=3x3
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16*6*6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2) # equal to (2,2) if windows is square
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:] # except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()

# fake input and target
inp = torch.randn(1, 1, 32, 32)
target = torch.randn([1,10])

# define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)

iter = 0
# every iter
while iter<100:
    output = net(inp) # compute the output
    loss = criterion(output, target) # compute the loss
    optimizer.zero_grad() # zero the gradient buffers
    loss.backward() # backprop the loss
    optimizer.step() # update the parameters
    iter += 1