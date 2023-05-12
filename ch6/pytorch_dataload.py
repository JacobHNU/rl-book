'''
下载速度很慢时的处理方法：(前提是已经有data/FashionMNIST/raw文件夹了)
1. 直接下载文件
train-data:  http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
train-labels:  http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz

test-data: http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
test-labels: http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz

2. 下载后文件全部放在data/FashionMNIST/raw/中

3. 然后保持download=True, 再运行程序，就能直接识别已经下载好的文件
'''

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

# 下载数据集
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
    transform=ToTensor()   # 制定特征features和标签labels的数据格式转换，
    # 将(H, W, C)的nump.ndarray或img转为shape为(C, H, W)，转为tensor张量
)
'''

'''

# 查看数据结构
print(training_data.data.shape, training_data.targets.shape)
'''
torch.Size([60000, 28, 28]) torch.Size([60000])        60000张28x28的灰度图片
'''
print(test_data.data.shape, test_data.data.shape)
'''
torch.Size([10000, 28, 28]) torch.Size([10000, 28, 28])   10000张28x28的灰度图片
'''
print(training_data.data.dtype, training_data.targets.dtype)
'''
torch.uint8  torch.int64
'''
print(training_data.classes)
'''
10种衣物
['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
'''

# 画图
labels = training_data.classes
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels[label])
    plt.axis('off')
    plt.imshow(img.squeeze(), cmap='gray')
plt.show()

# 训练数据加载
batch_size = 32
training_data_loader = DataLoader(training_data, batch_size, shuffle=True)
test_data_loader = DataLoader(test_data, batch_size, shuffle=True)

for X, y in training_data_loader:
    print('Shape of X is', X.shape)
    print('Shape of y is', y.shape)
    break
'''
Shape of X is torch.Size([32, 1, 28, 28])
Shape of y is torch.Size([32])
'''