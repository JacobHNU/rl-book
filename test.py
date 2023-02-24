import numpy as np
a = np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
b = a[::-1, :]      # a[::-1]) ### 取从后向前（相反）的元素 [[5 6 7 8 9] [0 1 2 3 4]]
b = a[-1, :]        # a[-1, :] ## 取最后一个元素里面的全部内容 [5 6 7 8 9]
b = a[:,-1]         # a[:,-1]  ## 取所有元素里面的的最后一个[4 9]
print(b)

