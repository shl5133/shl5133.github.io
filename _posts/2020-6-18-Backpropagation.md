---
layout: post
title:  "Backpropagation"
date:   2020-06-18 17:30:00 -0600
tags:
   - Machine Learning Fundation
category: Machine Learning
created: 2020/06/18 17:30:00
last modified: 2020/06/19 15:20:00
---

This Blog is about Backpropagation.

### Contents

* [背景及定义](#a)
* [一个简单的例子](#b)
* [一点小技巧](#c)
* [复杂一点呢](#d)

<a name='a'></a>

### 1 背景及定义

前向传播以及反向传播是神经网络的基础，前向传播较简单（即按照表达式计算即可），反向传播相对来说较为复杂，本文为在学习了\[[1](<https://cs231n.github.io/optimization-2/#grad>)\]的基础上结合自己的理解所写。

反向传播中最核心的其实就是**链式法则**（chain rule）（一种用于求解复合函数导数的常用方法，详见\[[2](<https://en.wikipedia.org/wiki/Chain_rule>)\]），通过其来计算输出相对于输入变量的偏导数（partial derivative）（也就是相对于输入变量的梯度），进而在梯度下降算法中作为梯度来使用。

**对每个变量的偏导数告诉我们整个表达式对其值的敏感性。**如对于f(x, y) = xy表达式来说，当x = -2，y = 3时，对于x的偏导数为3，表示x增加一个极小值&Delta;时整个表达式将会减小2&Delta;倍，同理y增加一个极小值&Delta;时整个表达式将会增加3&Delta;倍。

<a name='b'></a>

### 2 一个简单的例子

我们可以将表达式转化为电路（circuit）的形式，这样不仅直观而且便于理解。将表达式：f(x, y, z) = (x + y) z，拆分为q = x + y，f = q &times; z两个表达式，我们可以将其转换为如下图所示的形式（其中圆圈代表运算符，箭头连接变量与运算符，绿色数字代表正向传播时各变量的值，红色数字代表反向传播时各变量相对于整个表达式的偏导数）。

<a name='pic1'></a>

![circuit1](/images/backpropagation1.png)

设x = -2，y = 5，z = -4；

前向传播：

q = x + y = -2 + 5 = 3

f = q &times; z = 3 &times; -4 = -12

反向传播：

<img src="https://latex.codecogs.com/png.latex?\frac{\partial&space;f}{\partial&space;f}&space;=&space;1" title="\frac{\partial f}{\partial f} = 1" />

<img src="https://latex.codecogs.com/png.latex?\frac{\partial&space;f}{\partial&space;z}&space;=&space;\frac{\partial&space;f}{\partial&space;f}\frac{\partial&space;f}{\partial&space;z}&space;=&space;1&space;\times&space;q&space;=&space;(x&space;&plus;&space;y)&space;=&space;3" title="\frac{\partial f}{\partial z} = \frac{\partial f}{\partial f}\frac{\partial f}{\partial z} = 1 \times q = (x + y) = 3" />

<img src="https://latex.codecogs.com/png.latex?\frac{\partial&space;f}{\partial&space;q}&space;=&space;\frac{\partial&space;f}{\partial&space;f}\frac{\partial&space;f}{\partial&space;q}&space;=&space;1&space;\times&space;z&space;=&space;-4" title="\frac{\partial f}{\partial q} = \frac{\partial f}{\partial f}\frac{\partial f}{\partial q} = 1 \times z = -4" />

<img src="https://latex.codecogs.com/png.latex?\frac{\partial&space;f}{\partial&space;x}&space;=&space;\frac{\partial&space;f}{\partial&space;q}\frac{\partial&space;q}{\partial&space;x}&space;=&space;z\times&space;1&space;=&space;-4" title="\frac{\partial f}{\partial x} = \frac{\partial f}{\partial q}\frac{\partial q}{\partial x} = z\times 1 = -4" />

<img src="https://latex.codecogs.com/png.latex?\frac{\partial&space;f}{\partial&space;y}&space;=&space;\frac{\partial&space;f}{\partial&space;q}\frac{\partial&space;q}{\partial&space;y}&space;=&space;z&space;\times&space;1&space;=&space;-4" title="\frac{\partial f}{\partial y} = \frac{\partial f}{\partial q}\frac{\partial q}{\partial y} = z \times 1 = -4" />

我们所关注的主要是输入对于输出的偏导数，即第2，4，5式。

可以看到，将表达式转换为电路形式后计算前向和反向传播十分直观。链式法则在电路图中的具体体现为：输出对于某一变量（如本例中的x，y，z，q）的偏导数，**实际上等于节点”输入和输出“的偏导数的乘积**。具体地说如对于本例中的q，将其左边的所有电路都抽象为一个节点，看作输入，对其求偏导数（即偏f/偏q = 偏qz/偏q = z），之后乘以其右边电路已经计算出的偏导数（即偏f/偏f = 1）（如下图所示）。

![circuit2](/images/backpropagation2.png)

这样做的好处是：我们只需要计算每个节点的偏导数（即输入输出乘积），而不需要去关心整个表达式。这样，无论多么复杂的表达式只要拆分合理，都可以对其轻松求导。就好像反向传播中的“传播”二字一样，在电路中进行偏导数值的传播。

注：2，3式之所以多乘了一个“偏f/偏f”是为了便于说明以上观点。



<a name='c'></a>

### 3 一点小技巧

#### 3.1 电路压缩

在实际应用中，转化电路不一定要每一个运算符都拆分。当已知整体表达式中部分表达式偏导数形式时，可以对电路图进行“压缩”，这样便于计算。

比如：在逻辑回归算法中，sigmod函数我们已知其导数为如下公式，所以我们可以将这一部分进行压缩。
<img src="https://latex.codecogs.com/png.latex?\sigma&space;(x)&space;=&space;\frac{1}{1&space;&plus;&space;e^{-x}}" title="\sigma (x) = \frac{1}{1 + e^{-x}}" />
<img src="https://latex.codecogs.com/png.latex?\frac{d\sigma&space;(x)}{dx}&space;=&space;(\frac{1&space;&plus;&space;e^{-x}&space;-1}{1&space;&plus;&space;e^{-x}})(\frac{1}{1&space;&plus;&space;e^{-x}})&space;=&space;(1&space;-&space;\sigma&space;(x))\sigma&space;(x)" title="\frac{d\sigma (x)}{dx} = (\frac{1 + e^{-x} -1}{1 + e^{-x}})(\frac{1}{1 + e^{-x}}) = (1 - \sigma (x))\sigma (x)" />

对于如下逻辑回归表达式：

<img src="https://latex.codecogs.com/png.latex?f(\omega,x)&space;=&space;\frac{1}{1&space;&plus;&space;e^{-(\omega_{0}x_{0}&space;&plus;&space;\omega_{1}x_{1}&space;&plus;&space;\omega_{2})}}" title="f(\omega,x) = \frac{1}{1 + e^{-(\omega_{0}x_{0} + \omega_{1}x_{1} + \omega_{2})}}" />

压缩前后的对比如下图所示：

![circuit4](/images/backpropagation4.png)

可以看到，我们直接用一个节点代替了sigmod运算中的四个节点，从而压缩了模型，提高了计算的效率。

#### 3.2 常用电路

其实在电路拆解中，最常用的节点无非就是加节点和乘节点。

* 加节点所做的就是将**输出的偏导数原封不动的传递给其所有节点**。如第一节中的[电路](#pic1)：对于x+y，x的偏导数 = y的偏导数 = 输出的偏导数 = -4。

* 乘节点所做的就是将**将两节点的值互换再乘以输出的偏导数**。如第一节中的[电路](#pic1)：对于q &times; z，q的偏导数 = z的值 &times; 输出的偏导数 = -4 &times; 1 = -4，z的偏导数 = q的值 &times; 输出的偏导数 = 3 &times; 1 = 3。

<a name='d'></a>

### 4 复杂一点呢

下面我们来计算一个复杂的例子：

<img src="https://latex.codecogs.com/png.latex?f(x,y)&space;=&space;\frac{x&space;&plus;&space;\sigma&space;(y)}{\sigma&space;(x)&space;&plus;&space;(x&space;&plus;&space;y)^{2}}" title="f(x,y) = \frac{x + \sigma (y)}{\sigma (x) + (x + y)^{2}}" />

转化为电路后如下图所示（前向传播的值已经给出）：

![circuit5](/images/backpropagation5.png)

开始反向传播：

![circuit6](/images/backpropagation6.png)

![circuit7](/images/backpropagation7.png)

![circuit8](/images/backpropagation8.png)

![circuit9](/images/backpropagation9.png)

![circuit10](/images/backpropagation10.png)

![circuit11](/images/backpropagation11.png)

![circuit12](/images/backpropagation12.png)

![circuit13](/images/backpropagation13.png)

![circuit14](/images/backpropagation14.png)

注：x，y的偏导数是所有分支偏导数的加和。

