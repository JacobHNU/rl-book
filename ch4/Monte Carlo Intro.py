# monte carlo 法——  free model ，无明确的状态转移概率，无法用动态规划求解。

## MC法求\pi, 即利用概率分布求\pi——单位正方形内切圆，圆内随机点的数量k与正方形内随机点的数量n之比=圆与正方形的面积之比
# k/n = \pi * 1/4 / 1  =>  \pi = 4k / n
import numpy as np
import math
np.random.seed(0)  #选取随机种子，为了保证每次产生的随机数都是一样的
n = 1000000
k = 0

for _ in range(n):
    x1,x2 = np.random.random(), np.random.random()   # 随机生成一个点
    if math.sqrt(x1*x1+x2*x2) < 1:                   # 落在圆内
        k += 1;
print("\pi={}".format(4*k/n))

