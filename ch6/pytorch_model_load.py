import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# 数据准备
training_data = datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,
    transform=ToTensor()
)
test_data = datasets.FashionMNIST(
    root='data',
    train=False,
    download=True,
    transform=ToTensor()
)

batch_size = 64
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# 构建模型
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Net(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.flatten = nn.Flatten()  # 将多维张量拉伸成1维张量
        # 定义网络拓扑
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    # 前向传播
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = Net().to(device)  # 创建一个神经网络实体
print(model)

# 损失函数和优化器
loss_func = nn.CrossEntropyLoss(reduction='mean')
opt = torch.optim.SGD(model.parameters(), lr=1e-3)


# 训练函数
def train(dataloader, model, loss_func, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)  # 计算预测值
        loss = loss_func(pred, y)  # 计算损失函数
        opt.zero_grad()  # 梯度归零
        loss.backward()  # 误差反向传播
        opt.step()  # 更新参数

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:7f} [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_func):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)  # 计算预测值
            test_loss += loss_func(pred, y).item()  # 计算误差
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    correct /= size  # 预测分类正确率
    test_loss /= num_batches  # 预测数据平均误差
    print('Accuracy is {}, Average loss is {}'.format(correct, test_loss))


# 训练和测试
# epochs = 5
# for t in range(epochs):
#     print(f"Epoch {t+1}\n-------------------------------")
#     train(train_dataloader, model, loss_func, opt)
#     test(test_dataloader, model, loss_func)
# print("Done!")

# 访问模型参数
model.state_dict()
model.state_dict()['linear_relu_stack.4.bias']

# 保存模型和模型参数
torch.save(model, 'model')  # 保存整个模型
torch.save(model.state_dict(), 'model_parameter.pth')  # 保存模型参数

# 重载模型和模型参数
model1 = torch.load('model')  # 直接重载整个模型，包括网络拓扑和参数
model2 = Net()  # 只重载参数需要先创建一个相同网络拓扑的初始模型
model2.load_state_dict(torch.load('model_parameter.pth'))  # 重载模型参数

with torch.no_grad():
    model.to(device)
    model1.to(device)
    model2.to(device)
    for X, y in train_dataloader:
        X = X.to(device)
        pred = model(X)
        pred1 = model1(X)
        pred2 = model2(X)
        print(pred[0])
        print(pred1[0])
        print(pred2[0])
        break
"""
问题：
RuntimeError: Tensor for argument #2 'mat1' is on CPU, 
but expected it to be on GPU (while checking arguments for addmm)

解决方法：
说明model和数据不在同一个设备上，需要把model和数据都放在gpu上。写法如下：
model.to(device)
X = X.to(device)

https://zhuanlan.zhihu.com/p/609101410
"""

# 预训练模型加载
import torchvision.models as models

model_vgg16 = models.vgg16(pretrained=True)  # 创建一个已经训练好的Vgg16网络
torch.save(model_vgg16.state_dict(), 'model_vgg16_parameter')  # 保存模型参数

model_vgg16_1 = models.vgg16()  # 创建一个未训练得vgg16网络
model_vgg16_1.load_state_dict(torch.load('model_vgg16_parameter'))  # 加载模型参数

model_vgg16.eval()  # 评估模型
model_vgg16_1.eval()  # 评估模型

'''
预训练模型，是在大规模数据集上进行的一种先验训练，旨在训练一个通用的模型，以便在后续任务中进行微调或迁移学习。
预训练模型通常包括两个阶段：
    1. 无监督预训练阶段：模型通常在一个大规模、未标记的数据集上训练，以学习通用的特征表示。
    2. 有监督微调阶段：模型使用标记数据集进行微调，适应具体的任务。
'''
