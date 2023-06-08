'''
   pytorch中的tensor和Numpy中的ndarray的类似，
   主要区别在于tensor可以部署在GPU和CPU上进行计算，ndarray只能在CPU中进行计算。
   tensor和ndarray可以相互转换
   将tensor转换为ndarray： tensor.numpy()
   将ndarray转换为tensor:  torch.from_numpy(ndarray)

   tensor 具有广播机制，类似于MATLAB里面的 repmat函数

    pytorch 的计算图是动态图，在前向传播时构建，梯度计算完毕后释放。
    因此，若再次执行backward()则会报错
    训练函数
    训练得主要任务时误差反向传播和调整参数
'''

import torch
import numpy as np

# torch.manual_seed(1)

# tensor 数据类型
a = torch.rand((3,))
# print(a.dtype)
# print('a={}'.format(a))

# 指定tensor数据类型
b = torch.randint(1, 10, (3, 3), dtype=torch.float64)  # low= 1 , high = 10, size= (3,)表示规格大小
# print(b)
'''
IN: torch.randint(1, 10, (3,), dtype=torch.float64)
OUT: tensor([9., 8., 9.], dtype=torch.float64)

IN: torch.randint(1, 10, (3,3), dtype=torch.float64)
OUT: tensor([[8., 7., 6.],
        [6., 6., 5.],
        [3., 9., 2.]], dtype=torch.float64)
'''
# In[Tensor数据类型转换]
a = torch.rand((3,))
# print(a, a.dtype)
b = a.int()
# print(b, b.dtype)
c = a.float()
# print(c, c.dtype)
# In[ndarray转Tensor]
a = np.array([1., 2., 3.])  # 默认数据类型为 float64
tensor_a1 = torch.from_numpy(a)  # 此处数据类型还是float64
tensor_a2 = torch.FloatTensor(a)  # = torch.from_numpy(a).float()  默认数据类型都是float32
# print(a, a.dtype)                   #  [1. 2. 3.] float64
# print(tensor_a1, tensor_a1.dtype)   #  tensor([1., 2., 3.], dtype=torch.float64) torch.float64
# print(tensor_a2, tensor_a2.dtype)   #  tensor([1., 2., 3.]) torch.float32

# In[Tensor转ndarray]
t = torch.rand((3,))
array_t1 = t.numpy()
array_t2 = np.array(t)
# print(t, t.dtype)  # tensor([0.5627, 0.8395, 0.9160]) torch.float32
# print(array_t1, array_t1.dtype)  # [0.56271327 0.83949006 0.916016  ] float32
# print(array_t2, array_t2.dtype)  # [0.56271327 0.83949006 0.916016  ] float32

# [张量的维度]
t = torch.Tensor(np.arange(24)).reshape((2, 3, 4))
# print(t)
'''
tensor([[[ 0.,  1.,  2.,  3.],
         [ 4.,  5.,  6.,  7.],
         [ 8.,  9., 10., 11.]],

        [[12., 13., 14., 15.],
         [16., 17., 18., 19.],
         [20., 21., 22., 23.]]])
'''
# print(t.sum(dim=0))  # print(t.sum(axis=0))  # 将原来三维数组转换为二维数组，将三维数组的两个二维数组按照每行对应位置元素相加
'''
tensor([[12., 14., 16., 18.],
        [20., 22., 24., 26.],
        [28., 30., 32., 34.]])
'''
# print(t[0])
'''
tensor([[ 0.,  1.,  2.,  3.],
        [ 4.,  5.,  6.,  7.],
        [ 8.,  9., 10., 11.]])
'''
# print(t[0][0])
'''
tensor([0., 1., 2., 3.])
'''
# print(t.sum(dim=1))  # 将原三维数组转换成一个二维数组，将三维数组的两个二维数组每个数组各自的列元素相加，然后再两个行数组拼接成二维数组
### 每个二维数组的列元素求和
'''
tensor([[12., 15., 18., 21.],
        [48., 51., 54., 57.]])
'''
# print(t.sum(dim=2))  # 将原三维数组转变为一个二维数组，将每个二维数组各自的行元素相加求和后组成一行，然后两个行数组拼接成二维数组
#### 数组元素为原数组每行元素的求和，
'''
tensor([[ 6., 22., 38.],
        [54., 70., 86.]])
'''

# [张量的维度重组]
t = torch.Tensor(np.arange(24)).reshape((2, 3, 4))
t1 = t.view(2, 2, 6)
t2 = t.reshape(8, -1)
# print(t)
# print(t1)
# print(t2)

# [维的添加和压缩]
t = torch.Tensor(np.arange(6)).reshape((2, 3))
'''
tensor([[0., 1., 2.],
        [3., 4., 5.]])
'''
t1 = t.unsqueeze(dim=0)  # 在行维度上增加
'''
tensor([[[0., 1., 2.],
         [3., 4., 5.]]])
'''
t2 = t1.unsqueeze(dim=3)  # 在二维数组维度上增加
'''
tensor([[[[0.],
          [1.],
          [2.]],

         [[3.],
          [4.],
          [5.]]]])
'''
t3 = t1.squeeze()  # 返回到维度扩充前，维度压缩
'''
tensor([[0., 1., 2.],
        [3., 4., 5.]])
'''
t4 = t2.squeeze()  # 返回维度扩充前，维度压缩
'''
tensor([[0., 1., 2.],
        [3., 4., 5.]])
'''
t5 = t2.squeeze(dim=3)  # 只在二维数组维度上压缩维度，但是维度还是在行维度上多了一个
'''
tensor([[[0., 1., 2.],
         [3., 4., 5.]]])
'''
# print(t5)
# print(t)
# print(t1)
# print(t2)
# print(t3)
# print(t4)

# [张量转置]
t = torch.Tensor(np.arange(6)).reshape((2, 3))
t_t = t.t()  # 类似于矩阵转置
'''
tensor([[0., 3.],
        [1., 4.],
        [2., 5.]]) torch.Size([3, 2])
'''
# print(t_t,t_t.shape)

t = torch.Tensor(np.arange(24)).reshape((2, 3, 4))
'''
tensor([[[ 0.,  1.,  2.,  3.],
         [ 4.,  5.,  6.,  7.],
         [ 8.,  9., 10., 11.]],

        [[12., 13., 14., 15.],
         [16., 17., 18., 19.],
         [20., 21., 22., 23.]]]) torch.Size([2, 3, 4])
'''
t_trans = t.transpose(0, 1)  # 第0和1维转置    针对三维张量的转置
'''
tensor([[[ 0.,  1.,  2.,  3.],
         [12., 13., 14., 15.]],

        [[ 4.,  5.,  6.,  7.],
         [16., 17., 18., 19.]],

        [[ 8.,  9., 10., 11.],
         [20., 21., 22., 23.]]]) torch.Size([3, 2, 4])
'''
# print(t, t.shape)
# print(t_trans, t_trans.shape)

t_perm = t.permute((1, 0, 2))  # 将维度按照1,0,2方式转置，即t_perm.shape为（3,2,4）

# [张量的广播]
t1 = torch.Tensor(np.arange(6)).reshape((3, 2))
t2 = torch.cat((t1, t1), dim=0)
'''
tensor([[0., 1.],
        [2., 3.],
        [4., 5.],
        [0., 1.],
        [2., 3.],
        [4., 5.]])
'''
## 将两个张量（tensor）按指定维度拼接在一起
t3 = torch.cat((t1, t1), dim=1)
'''
tensor([[0., 1., 0., 1.],
        [2., 3., 2., 3.],
        [4., 5., 4., 5.]])
'''
# print(t2)
##    torch.split()作用将tensor分成块结构。
t4 = torch.split(t1, [1, 2])  # 分割成了tuple，默认按照dim=0的维度分割，但是要求分割的维度=列表里面的维度和，即3=1+2
'''
(tensor([[0., 1.]]), tensor([[2., 3.],
        [4., 5.]]))
'''
t5 = torch.split(t1, [1, 1], dim=1)  # 按照dim=1的维度分割，2=1+1
'''
(tensor([[0.],
        [2.],
        [4.]]), tensor([[1.],
        [3.],
        [5.]]))
'''
# print(t4)
# print(t5)
## torch.hsplit函数是PyTorch库中的一个用于将张量沿着指定维度进行水平分割的函数。
# t6 = torch.hsplit(t1, 2)  # 使用torch.hsplit函数将张量水平分割为两个3x1的张量
# print(t6)
t7 = torch.hstack((t1,t1))
'''
tensor([[0., 1., 0., 1.],
        [2., 3., 2., 3.],
        [4., 5., 4., 5.]])
'''
# print(t7)
# t8 = torch.vsplit(t1,(1,2))
t9 = torch.vstack((t1,t1))
'''
tensor([[0., 1.],
        [2., 3.],
        [4., 5.],
        [0., 1.],
        [2., 3.],
        [4., 5.]])
'''
print(t9)