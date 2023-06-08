import torch

# [搭建计算图]
x = torch.ones((5,), requires_grad=True)  # 变量，需要计算梯度
W = torch.ones((5, 5), requires_grad=False)  # 参数，不需要计算梯度
b = torch.ones((5,))  # 参数， 默认不需要计算梯度
Q = torch.matmul(torch.matmul(x, W), x)  # 二次项，中间结果
L = torch.matmul(b, x)  # 一次项，中间结果
C = torch.matmul(b, b)  # 常数项
y = Q + L + C  # 前向传播，建立计算图

# 查看需要求梯度的量
# print(x.requires_grad,Q.requires_grad,C.requires_grad,y.requires_grad)
'''
True True False True
'''
# print(y.grad_fn) # 对最终结果y的梯度函数
# print(Q.grad_fn) # 对中间结果Q的梯度函数
# print(C.grad_fn) # 对中间结果C的梯度函数
# print(L.grad_fn) # 对中间结果L的梯度函数
'''
<AddBackward0 object at 0x000002C37E6560D0>
<DotBackward object at 0x000002C37E6560D0>
None -> C没有x，所以无梯度函数
<DotBackward object at 0x00000234FBDF6670>
'''

# 梯度计算和查看
y.backward(retain_graph=True)
# print(x.grad)
# print(b.grad)
# print(Q.grad)
'''
tensor([11., 11., 11., 11., 11.])   ==  x.grad
None
None

只有y关于x的梯度输出了有效值，
y关于b的梯度为none，因为require_grad=False。
y关于Q的梯度为none，因为Q是中间变量，pytorch的计算梯度是中间变量的梯度不能获取。
pytorch中的计算图是动态图，在前向传播时构建，梯度计算完毕后释放
'''
# y.backward()
'''
静态图与动态图的区别，
如果是静态图，则计算图一直保存，再次反向传播计算梯度，仍然可以计算。
此处是用的pytorch，动态图，在第一次反向传播计算完梯度后，动态图就释放了。如果仍需计算，则需要在第一次进行反向传播时修改属性值retain_graph=True
'''
# y.backward()
# print(x.grad)
'''
tensor([22., 22., 22., 22., 22.])    

y关于x的梯度变成了原来的2倍，这是因为梯度是累积的， 再次计算梯度时，会将求出的梯度累加到第一次计算的梯度上。
为了避免这种情况，每次求梯度之前需要使用zero_()将函数梯度归零。
'''
x.grad.zero_()
# print(x.grad)
y.backward()
# print(x.grad)
'''
tensor([0., 0., 0., 0., 0.])
tensor([11., 11., 11., 11., 11.])

如果想要多次利用动态图反向传播计算梯度，则需要除了在第一次backward()时添加retain_graph=True,其余多次也需要添加retain_graph=True
'''

# 关闭自动梯度计算
'''
自动梯度计算通常用在模型训练中，但在前向传播过程中并不需要计算梯度，这时关闭自动梯度计算效率会更高。
两种方式关闭自动梯度计算 
1. with torch.no_grad()
2. y.detach() 
'''
y = torch.matmul(b, x)
print(y.requires_grad)
'''
True
'''
# with torch.no_grad():
#     y1 = torch.matmul(b,x)
# print(y1.requires_grad)
'''
False
'''
y2 = y.detach()
print(y2.requires_grad)
'''
False
'''
