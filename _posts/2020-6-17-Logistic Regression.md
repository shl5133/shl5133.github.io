---
layout: post
title:  "Logistic Regression"
date:   2020-06-17 16:28:00 -0600
tags:
   - Machine Learning Algorithm
category: Machine Learning
created: 2020/06/17 16:30:00
last modified: 2020/06/17 16:30:00
---

#  Logistic Regression (逻辑回归)

## 1 模型原理

逻辑回归使用sigmod函数（$ \frac{1}{1 + e^{-x}} $）（如下图）对样本进行回归，之后设定阈值将正负样本分开，实际是一种分类算法。
![sigmod function](https://github.com/shl5133/shl5133.github.io/blob/master/_posts/sigmod function.jpg)

之所以选用sigmod函数，是由于逻辑回归可以看作利用sigmod函数对后验概率P(y=1 | x)的逼近（具体请见[1](https://blog.csdn.net/qq_19645269/article/details/79551576)）。

## 2 问题背景定义

对于给定的m个样本（X~i~, y~i~）（<a href="https://www.codecogs.com/eqnedit.php?latex=i&space;\in&space;[1,m]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?i&space;\in&space;[1,m]" title="i \in [1,m]" /></a>，X为n维向量）进行二分类：y = 0为负例，y = 1为正例。

## 3 公式推导

### 3.1 假设函数（hypothesis function）

$h_{\theta }(x) = \frac{1}{1+e^{-(W^{T}X+b)}} = \frac{1}{1+e^{-\theta ^{T}x}}​$ 	          (1)

注：其中$\theta$表示模型的参数，即w，b；$W^{T}x = w_{1}x_{1} + w_{2}x_{2} + ... + w_{n}x_{n}$，$\theta^{T}x = \theta_{0} + \theta_{1}x_{1} + \theta_{2}x_{2} + ... + \theta_{n}x_{n}$；

### 3.2 预测正确的概率

我们将$h_{\theta }(x)$所给出的结果看作概率（因为sigmod可以将变量从（$-\infty，+\infty$）映射到（0，1）之间），即有：

1）p(y = 1 | w, x) = $h_{\theta }(x)​$	        	   (2)

2）p(y = 0 | w, x) = 1 - $h_{\theta }(x)​$	   	   (3)

3）p(correct) = $h_{\theta }(x)^{y}(1-h_{\theta }(x))^{1-y}​$       (4)

注：1) p(correct)表示模型预测正确的概率：当y = 1即样本为正例时，p(correct) = $h_{\theta }(x)$；当y = 0即样本为负例时，p(correct) = 1 - $h_{\theta }(x)$。

​       2)$h_{\theta }(x)$的意义：例如对于某患者是否为患病，$h_{\theta }(x)$输出结果为0.7，则表示患者未患病的概率为70%，患病的概率为30%。

### 3.3 最大似然估计

由于我们的目标是找到合适的$\theta$使得所有样本的p(correct)最大，即使得模型预测正确所有样本的概率最大，所以我们采用最大似然估计：

$L (\theta) = \prod_{i = 1}^{m}p(correct)^{i} = \prod_{i = 1}^{m}h_{\theta }(x^{i})^{y^{i}}(1-h_{\theta }(x^{i}))^{1-y^{i}}$							    (5)

连乘不好计算，所以我们对(5)式取对数：

$l(\theta) = logL (\theta) = \sum_{i = 1}^{m}log(p(correct)^{i}) = \sum_{i = 1}^{m}(y^{i}log(h_{\theta }(x^{i})) + (1-y^{i})log(1-h_{\theta }(x^{i})))​$	(6)

最优化任务时习惯上我们希望得到函数的最小值，所以对(6)式取负，即是求解最小值，并得到我们最终的损失函数（交叉熵损失函数）$J(\theta)​$：

$J(\theta) = -l(\theta)​$	            			 (7)

### 3.4 最佳$\theta$值计算

我们使用梯度下降计算损失函数的最小值，具体算法见[2](https://en.wikipedia.org/wiki/Gradient_descent)，求解梯度（即损失函数偏导）的公式为：

$\frac{\partial J(\theta)}{\partial \theta} = \frac{\partial (-\sum_{i = 1}^{m}(y^{i}log(h_{\theta }(x^{i})) + (1-y^{i})log(1-h_{\theta }(x^{i}))))}{\partial \theta}​$

​         $= -\sum_{i = 1}^{m}(y^{i}\frac{1}{h_{\theta }(x^{i})}\frac{\partial h_{\theta }(x^{i})}{\partial \theta} - (1 - y^{i})\frac{1}{1 - h_{\theta }(x^{i})}\frac{\partial h_{\theta }(x^{i})}{\partial \theta})​$

​         $= -\sum_{i = 1}^{m}(y^{i}\frac{1}{h_{\theta }(x^{i})} - (1 - y^{i})\frac{1}{1 - h_{\theta }(x^{i})})\frac{\partial \frac{1}{1+e^{-\theta ^{T}x^{i}}}}{\partial \theta}​$

​         $= -\sum_{i = 1}^{m}(y^{i}\frac{1}{h_{\theta }(x^{i})} - (1 - y^{i})\frac{1}{1 - h_{\theta }(x^{i})})\frac{x^{i}e^{- \theta ^{T}x^{i}}}{(1 + e^{-\theta ^{T}x^{i}})^{2}}​$

​         $= -\sum_{i = 1}^{m}(y^{i}\frac{1}{h_{\theta }(x^{i})} - (1 - y^{i})\frac{1}{1 - h_{\theta }(x^{i})})x^{i}h_{\theta }(x^{i})^{2}(\frac{1}{h_{\theta }(x^{i})}-1)$

​        $= -\sum_{i = 1}^{m}(y^{i}\frac{1}{h_{\theta }(x^{i})} - (1 - y^{i})\frac{1}{1 - h_{\theta }(x^{i})})x^{i}h_{\theta }(x^{i})(1 - h_{\theta }(x^{i}))$

​        $= -\sum_{i = 1}^{m}(y^{i}(1 - h_{\theta }(x^{i})) - (1 - y^{i})h_{\theta }(x^{i}))x^{i}$

​        $= -\sum_{i = 1}^{m}( y^{i} - h_{\theta }(x^{i}))x^{i}​$

​        $= \sum_{i = 1}^{m}(h_{\theta }(x^{i}) - y^{i})x^{i}​$

其形式与最小二乘法梯度一致，区别在于最小二乘法中$h_{\theta }(x) = W^{T}x​$，也即是线性回归。
