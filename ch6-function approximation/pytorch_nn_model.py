import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# 数据准备
training_data = datasets.FashionMNIST(
    root='data',
    train=True,
    download=False,
    transform=ToTensor()
)
'''
transform = ToTensor():
    Converts a PIL Image or numpy.ndarray (H x W x C) in the range                     
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    
    [H,W,C] 表示图片的高度、宽度、通道数
    图片的读取的通道顺序需要注意：
        PIL图像在转换为numpy.ndarray后，格式为(h,w,c)，像素顺序为RGB；
        OpenCV在cv2.imread()后数据类型为numpy.ndarray，格式为(h,w,c)，像素顺序为BGR。
        PyTorch和caffe为(c,h,w)
'''

test_data = datasets.FashionMNIST(
    root='data',
    train=False,
    download=False,
    transform=ToTensor()
)
batch_size = 64
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# 构建模型
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # gpu还是cpu训练


class NeuNet(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.flatten = nn.Flatten()  # 将多维张量拉成1维张量
        # 定义网络拓扑
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    # 前向传播函数
    def forward(self, x):
        x = self.flatten(x)  # 因为FashionMNIST数据集的每张图片都转化成了一个28x28的张量，
        # 需要先将其展平为长度784的向量输入，然后再输入神经网络。
        logits = self.linear_relu_stack(x)
        return logits


model = NeuNet().to(device)  # 创建一个神经网络实体
print(model)
'''
调用NeuNet()实例时，forward方法会被自动执行
'''

'''
交叉熵（Cross Entropy）是Shannon信息论中一个重要概念，主要用于度量两个概率分布间的差异性信息
从名字上来看，Cross(交叉)主要是用于描述这是两个事件之间的相互关系，对自己求交叉熵等于熵。

熵的意义是对A事件中的随机变量进行编码所需的最小字节数。
KL散度的意义是“额外所需的编码长度”如果我们用B的编码来表示A。
交叉熵指的是当你用B作为密码本,来表示A时所需要的“平均的编码长度”。
'''
# 损失函数和优化器
loss_fn = nn.CrossEntropyLoss(reduction='mean')
opt = torch.optim.SGD(model.parameters(), lr=1e-3)


# 训练函数
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # model.train()         # 声明以下是训练环境
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)  # 计算预测值
        loss = loss_fn(pred, y)  # 计算损失函数
        opt.zero_grad()  # 梯度归零
        loss.backward()  # 误差反向传播
        opt.step()  # 参数更新

        if batch % 100 == 0:  # 每隔100批次打印训练精度
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}   [{current:>5d}/{size:>5d}]")


# 测试函数
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    # model.eval()                        # 声明模型评估状态
    test_loss, correct = 0, 0
    with torch.no_grad():  # 测试不需要计算梯度
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)  # 计算预测值
            test_loss += loss_fn(pred, y).item()  # 计算误差
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    correct /= size  # 预测分类正确率
    test_loss /= num_batches  # 测试数据平均误差
    print('Accuracy is {}, Average loss is {}'.format(correct, test_loss))


# In[训练和测试]
epochs = 5
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, opt)
    test(test_dataloader, model, loss_fn)
print("Done!")
