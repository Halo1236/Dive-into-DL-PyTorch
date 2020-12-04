# 数学基础
## 最小二乘法公式
最小二乘法公式是一个数学的公式，在数学上称为曲线拟合，此处所讲最小二乘法，专指线性回归方程！最小二乘法公式为a=y(平均)-b*x（平均）。

[最小二乘法的本质是什么](https://www.zhihu.com/question/37031188)、

[逆矩阵](https://baike.baidu.com/item/%E9%80%86%E7%9F%A9%E9%98%B5/10481136?fr=aladdin)

## 线性代数

### 标量，向量，矩阵与张量
#### 标量

一个标量就是一个单独的数，一般用小写的的变量名称表示。

#### 向量

一个向量就是一列数，这些数是有序排列的。用过次序中的索引，我们可以确定每个单独的数。通常会赋予向量粗体的小写名称

#### 矩阵

矩阵是二维数组

#### 张量

几何代数中定义的张量是基于向量和矩阵的推广，通俗一点理解的话，我们可以将标量视为零阶张量，矢量视为一阶张量，那么矩阵就是二阶张量。

PyTorch还支持一些线性函数

函数 |	功能
---|---
trace |	对角线元素之和(矩阵的迹)
diag |	对角线元素
triu/tril |	矩阵的上三角/下三角，可指定偏移量
mm/bmm	| 矩阵乘法，batch的矩阵乘法
addmm/addbmm/addmv/addr/baddbmm.. |	矩阵运算
t |	转置
dot/cross	| 内积/外积
inverse |	求逆矩阵
svd |	奇异值分解

PyTorch中的Tensor支持超过一百种操作，包括转置、索引、切片、数学运算、线性代数、随机数等等


## 内存开销
### Tensor和NumPy相互转换

我们很容易用numpy()和from_numpy()将Tensor和NumPy中的数组相互转换。但是需要注意的一点是： **这两个函数所产生的的Tensor和NumPy中的数组共享相同的内存（所以他们之间的转换很快），改变其中一个时另一个也会改变！！！**

==还有一个常用的将NumPy中的array转换成Tensor的方法就是torch.tensor(), 需要注意的是，此方法总是会进行数据拷贝（就会消耗更多的时间和空间），所以返回的Tensor和原来的数据不再共享内存。==

### torch.rand和torch.randn有什么区别

均匀分布

torch.rand(*sizes, out=None) → Tensor

返回一个张量，包含了从区间[0, 1)的均匀分布中抽取的一组随机数。张量的形状由参数sizes定义。

标准正态分布

torch.randn(*sizes, out=None) → Tensor

返回一个张量，包含了从标准正态分布（均值为0，方差为1，即高斯白噪声）中抽取的一组随机数。张量的形状由参数sizes定义。

==x[:,n]表示在全部数组（维）中取第n个数据，直观来说，x[:,n]就是取所有集合的第n个数据==

==x[n,:]表示在n个数组（维）中取全部数据，直观来说，x[n,:]就是取第n集合的所有数据==

### numpy.random.normal(loc=0.0, scale=1.0, size=None)  
loc:float

概率分布的均值，对应着整个分布的中心center

scale:float

概率分布的标准差，对应于分布的宽度，scale越大越矮胖，scale越小，越瘦高

size:int or tuple of ints

输出的shape，默认为None，只输出一个值

### torch.mul()和torch.mm()的区别
torch.mul(a, b)是矩阵a和b对应位相乘，a和b的维度必须相等，比如a的维度是(1, 2)，b的维度是(1, 2)，返回的仍是(1, 2)的矩阵

torch.mm(a, b)是矩阵a和b矩阵相乘，比如a的维度是(1, 2)，b的维度是(2, 3)，返回的就是(1, 3)的矩阵

# 置信度
[置信度](https://baijiahao.baidu.com/s?id=1596169784713150436&wfr=spider&for=pc)

== 如果我们抽取一个样本，得到了 63％，那么我们可以说我们 95％ 确信实际比例在 60％（63-3）和 66％（63 + 3）之间。这就是置信区间，区间为 63 + -3，置信度为 95％。==

mean() 函数定义：
numpy.mean(a, axis, dtype, out，keepdims )

##### mean()函数功能：求取均值
经常操作的参数为axis，以m * n矩阵举例：

- axis 不设置值，对 m*n 个数求均值，返回一个实数
- axis = 0：压缩行，对各列求均值，返回 1* n 矩阵
- axis =1 ：压缩列，对各行求均值，返回 m *1 矩阵


## nn.Sequential来更加方便地搭建网络，Sequential是一个有序的容器，网络层将按照在传入Sequential的顺序依次被添加到计算图中。

# 写法一
```
net = nn.Sequential(
    nn.Linear(num_inputs, 1)
    # 此处还可以传入其他层
    )
```
# 写法二
```
net = nn.Sequential()
net.add_module('linea', nn.Linear(num_inputs, 1))
# net.add_module ......
```

# 写法三
```
from collections import OrderedDict
net = nn.Sequential(OrderedDict([
          ('linear', nn.Linear(num_inputs, 1))
          # ......
        ]))


print(net)
print(net[0])
```
ModuleList仅仅是一个储存各种模块的列表，这些模块之间没有联系也没有顺序（所以不用保证相邻层的输入输出维度匹配），而且没有实现forward功能需要自己实现，所以上面执行net(torch.zeros(1, 784))会报NotImplementedError；而Sequential内的模块需要按照顺序排列，要保证相邻层的输入输出大小相匹配，内部forward功能已经实现。

# 使用TensorBoard
```python
from torch.utils.tensorboard import SummaryWriter

# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('runs/fashion_mnist_experiment_1')

# cmd
# 开始进入正轨，执行tensorboard命令，来生成网址。
# tensorboard --logdir=path --host=127.0.0.1

```