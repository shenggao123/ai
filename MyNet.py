import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch as t


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 5表示卷积核
        self.conv1 = nn.Conv2d(3, 6, 5)
        # 卷积层
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 仿射层/全连接层， y = Wx + b
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 卷积-》激活-》池化
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # reshape, '-1'表示自适应
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# net = Net()
# print(net)
# params = list(net.parameters())
# # print(len(params))
# # for name, parameters in net.named_parameters():
# #     print(name, ':', parameters.size())
#
# input = Variable(t.randn(1,1,32,32))
# out = net(input)
# # print(out.size())
# net.zero_grad()
# out.backward(Variable(t.ones(1,10)))
# # 损失函数
# output = net(input)
# target = Variable(t.arange(0, 10))
# criterion = nn.MSELoss()
# loss = criterion(output, target.float())
# net.zero_grad()
# print('反向传播之前conv1.bias的梯度')
# print(net.conv1.bias.grad)
# loss.backward()
# print('反向传播之后conv1.bias的梯度')
# print(net.conv1.bias.grad)
#
# learning_rate = 0.01
# for f in net.parameters():
#     f.data.sub_(f.grad.data * learning_rate)
#
# import torch.optim as optim
# # 新建一个优化器， 指定要调整的参数和学习率
# optimizer = optim.SGD(net.parameters(), lr = 0.01)
#
# # 在训练过程中
# # 先梯度清零(与net.zero_grad()效果一样)
# optimizer.zero_grad()
#
# # 计算损失
# output = net(input)
# loss = criterion(output, target.float())
# loss.backward()
# # 更新参数
# optimizer.step()
# print(net.conv1.bias.grad)