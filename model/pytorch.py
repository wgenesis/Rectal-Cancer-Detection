import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from torchsummary import summary

device = torch.device('cuda:0')
def loadtraindata():
    train_path = r'E:\Python\Medical big data\data\train'
    trainset = torchvision.datasets.ImageFolder(train_path,transform=transforms.Compose([transforms.Resize((125, 125)),transforms.ToTensor()]))

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=10,shuffle=True, num_workers=2)
    return trainloader


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(50176, 120)
        self.fc2 = nn.Linear(120, 60)
        self.fc3 = nn.Linear(60, 2)
        conv_output_size=None

    def forward(self, x):  # 前向传播
        #x=x.cuda(device)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        conv_output_size=x.size()[1]*x.size()[2]*x.size()[3]
        x = x.view(-1, conv_output_size)  # .view( )是一个tensor的方法，使得tensor改变size但是元素的总数是不变的。

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def loadtestdata():
    test_path = r'E:\Python\Medical big data\data\valid'
    testset = torchvision.datasets.ImageFolder(test_path,transform=transforms.Compose([transforms.Resize((125, 125)),transforms.ToTensor()]))
    testloader = torch.utils.data.DataLoader(testset, batch_size=10,shuffle=True, num_workers=2)
    return testloader

def trainandsave():
    trainloader = loadtraindata()
    # 神经网络结构
    net = Net()
    summary(net.cuda(), input_size=(3, 125, 125))
    net.to(device)
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    # 训练部分
    for epoch in range(20):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):

            #data=data.cuda(device)
            inputs, labels = data

            inputs, labels = Variable(inputs), Variable(labels)
            inputs=inputs.cuda(device)
            labels=labels.cuda(device)
            optimizer.zero_grad()  # 梯度置零，因为反向传播过程中梯度会累加上一次循环的梯度

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 200 == 199:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0

    print('Finished Training')
    # 保存神经网络
    torch.save(net, 'net.pkl')  # 保存整个神经网络的结构和模型参数
    torch.save(net.state_dict(), 'net_params.pkl')  # 只保存神经网络的模型参数

if __name__ == '__main__':
    #print(torch.cuda.is_available())
    trainandsave()


