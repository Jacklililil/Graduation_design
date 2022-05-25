import elvet
import tensorflow as tf
import elvet.plotting

#判断是使用cpu还是gpu
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

'''
已知条件：
对测地线问题进行简化，假定光滑曲面为平面
假设A点坐标为（0,1），B点坐标为（1,0）。
'''

#elvet.box用于生成训练点集：在0到1之间生成300个等距离点，主要用于计算目标泛函中的积分
domain = elvet.box((0, 1, 300))
domain = elvet.box((0, 1, 300))

# 构建神经网络模型，可以在创建对象时就指定层数，这样就无需用add方法
model = tf.keras.Sequential(
    [  # 第一层：Dense隐藏层，自定义单元个数为10,输入数据为3维,且用relu作为激活
        tf.keras.layers.Dense(10, input_shape=(1,), activation="sigmoid"),
        # 第二层：Dense输出层，单元个数为1(因为output：sale本身就是1维)
        tf.keras.layers.Dense(1)
    ]
)
info = model.summary()  # 可查看当前模型信息
print(info)

# 定义损失函数
def geodesic(y_1, y_2):
    def loss(x, y, dy_dx):
        # 对dy_dx张量进行了整形:dy_dx张量表示y对x的导数，使其与x和y具有相同的形状，便于接下来的运算
        dy_dx = dy_dx[:, 0]
        # 计算悬链线能量，默认采用simpson方法计算
        length = elvet.math.integral((1 + dy_dx**2)**0.5, x)
        # 确定边界条件
        bc_loss = (y[0] - y_1)**2 + (y[-1] - y_2)**2
        # 返回损失函数
        return length + 1e2 * bc_loss

    # 使用elvet.minimizer训练神经网络模型
    result = elvet.minimizer(loss, domain, batch=True, model=model, metrics=None, epochs=50000, verbose=True)
    '''
    elvet.minimizer参数解读：
    functional：需要最小化的泛函
    domain：训练点集
    order：函数中导数的最大阶。
    combinator：默认情况下，一种结合方程、边界和约束、恒等式的方法
    batch：如果为False，functional将接收域中的一个点以及函数和导数。
        否则，它将接收由定义域上的所有点和函数的值组成这些点的导数。
    model:问题采用的解决方案
    dtype:域和输入层的类型
    optimizer：要使用的优化器
    metrics：在拟合过程中监控参数的工具
    epoch:对模型进行指定纪元数的训练
    verbose：如果为Ture为输出进度条记录，否则不输出
    '''
    return result
result = geodesic(1, 0)

# 使用min函数返回计算结果中损失函数loss最小的项
min_loss = min(result.losses)
result.losses = [loss - min_loss + 1e-5 for loss in result.losses]
print('loss function最小为 %.8f' % min_loss)

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

min_loss = min(result.losses)
print('最小能量 %.8f' % min_loss)

# 绘制损失变化图
print("绘制损失函数变化图")
result.losses = [loss - min_loss + 1e-5 for loss in result.losses]
elvet.plotting.plot_losses(result)

#绘制预测曲线与真实曲线图
print("绘制预测曲线与真实曲线图")
x = domain.numpy()
y1=-x+1
elvet.plotting.plot_double(x,y1,result)

# 输出预测曲线长度
y, dy_dx = result.derivatives()
x = domain.numpy()
print('长度',elvet.math.integral((1 + dy_dx[:, 0]**2)**0.5, x).numpy().item())

#作平方误差图
print("绘制平方误差图")
elvet.plotting.plot_squared_error(x,y,y1)
