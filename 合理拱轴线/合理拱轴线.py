import elvet

def equation(x, y, dy, d2y):
    return d2y - (20/7207) * y - (350/7207)

bcs = (
    elvet.BC(0, lambda x, y, dy, d2y: y ),
    elvet.BC(0, lambda x, y, dy, d2y: dy ),
)

domain = elvet.box((0, 25, 100))

result = elvet.solver(equation,bcs, domain, model=elvet.nn(1, 10, 1), epochs=50000)



# 作图部分
import numpy as np
import elvet.plotting


# y_truth  = lambda x: 1.5 * (np.cosh(np.sqrt(2) * x) - 1)
y_truth  = lambda x: 17.5 * (np.cosh(np.sqrt(20/7207) * x) - 1)

print('*'*1000)
result.model.summary()
# Param = （输入数据维度+1）* 神经元个数
# 之所以要加1，是考虑到每个神经元都有一个Bias。
print('*'*1000)

# 控制台的显示设置
import pandas as pd
pd.set_option('display.max_rows', 500)#显示的最大行数
pd.set_option('display.max_columns', 100)#显示的最大列数
pd.set_option('display.width', 200)#横向最多显示字符数

#获取权重与偏置
print('获取权重与偏置')
weights = result.model.get_weights() #获取整个网络模型的全部参数
print(weights[0].shape)  #第一层的w
print(weights[1].shape)  #第一层的b
print(weights[2].shape)  #第二层的w
print(weights[3].shape)  #第二层的b
print('打印权重与偏置')
print(weights[0])  #第一层的w
print(weights[1])  #第一层的b
print(weights[2])  #第二层的w
print(weights[3])  #第二层的b

elvet.plotting.plot_losses(result)
elvet.plotting.plot_prediction(result, true_function=y_truth)
elvet.plotting.plot_loss_density(result)

