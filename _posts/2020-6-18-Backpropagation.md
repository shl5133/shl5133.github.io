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
* [复杂一点呢](#c)

<a name='a'></a>

### 1 背景及定义

前向传播以及反向传播是神经网络的基础，前向传播较简单（即按照表达式计算即可），反向传播相对来说较为复杂，本文为在学习了\[[1](<https://cs231n.github.io/optimization-2/#grad>)\]的基础上结合自己的理解所写。

反向传播中最核心的其实就是**链式法则**（chain rule）（一种用于求解复合函数导数的常用方法，详见\[[2](<https://en.wikipedia.org/wiki/Chain_rule>)\]），通过其来计算输出相对于输入变量的偏导数（partial derivative）（也就是相对于输入变量的梯度），进而在梯度下降算法中作为梯度来使用。

**对每个变量的偏导数告诉我们整个表达式对其值的敏感性。**如对于f(x, y) = xy表达式来说，当x = -2，y = 3时，对于x的偏导数为3，表示x增加一个极小值&Delta;时整个表达式将会减小2&Delta;倍，同理y增加一个极小值&Delta;时整个表达式将会增加3&Delta;倍。

<a name='b'></a>

### 2 一个简单的例子

我们可以将表达式转化为电路（circuit）的形式，这样不仅直观而且便于理解。将表达式：f(x, y, z) = (x + y) z，拆分为q = x + y，f = q &times; z两个表达式，我们可以将其转换为如下图所示的形式（其中圆圈代表运算符，箭头连接变量与运算符，绿色数字代表正向传播时各变量的值，红色数字代表反向传播时各变量相对于整个表达式的偏导数）。

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

可以看到，将表达式转换为电路形式后计算前向和反向传播十分直观。链式法则在电路图中的具体体现为：输出对于某一变量（如本例中的x，y，z，q）的偏导数，**实际上等于节点”输入和输出“的偏导数的乘积**。具体地说如对于本例中的q，将其左边的所有电路都抽象为一个节点，看作输入，对其求偏导数（即偏f/偏q），之后乘以其右边电路已经计算出的偏导数（即偏f/偏f）（如下图所示）。

![circuit2](/images/backpropagation2.png)

这样做的好处是：我们只需要计算每个节点的偏导数（即输入输出乘积），而不需要去关心整个表达式。这样，无论多么复杂的表达式只要拆分合理，都可以对其轻松求导。就好像反向传播中的“传播”二字一样，在电路中进行偏导数值的传播。

注：2，3式之所以多乘了一个“偏f/偏f”是为了便于说明以上观点。



<a name='c'></a>

### 3 复杂一点呢

