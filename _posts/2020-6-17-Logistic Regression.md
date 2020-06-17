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

This Blog is about Logistic Regression Algorithm.



## 1 模型原理

逻辑回归使用sigmod函数（如下图）对样本进行回归，之后设定阈值将正负样本分开，实际是一种分类算法。
<img src="https://image.baidu.com/search/detail?ct=503316480&z=0&ipn=d&word=sigmod%E5%87%BD%E6%95%B0&step_word=&hs=0&pn=1&spn=0&di=48070&pi=0&rn=1&tn=baiduimagedetail&is=0%2C0&istype=0&ie=utf-8&oe=utf-8&in=&cl=2&lm=-1&st=undefined&cs=3943895517%2C2299808605&os=262934790%2C152820750&simid=0%2C0&adpicid=0&lpn=0&ln=922&fr=&fmq=1592391098635_R&fm=&ic=undefined&s=undefined&hd=undefined&latest=undefined&copyright=undefined&se=&sme=&tab=0&width=undefined&height=undefined&face=undefined&ist=&jit=&cg=&bdtype=0&oriquery=&objurl=http%3A%2F%2F5b0988e595225.cdn.sohucs.com%2Fimages%2F20171108%2F821406a6c51d45179b0a0cb095d8f99b.jpeg&fromurl=ippr_z2C%24qAzdH3FAzdH3Fooo_z%26e3Bf5i7_z%26e3Bv54AzdH3FwAzdH3Fdananal0b_8aaa8ca0n&gsm=2&rpstart=0&rpnum=0&islist=&querylist=&force=undefined" style="width:150px height:250px" />

之所以选用sigmod函数，是由于逻辑回归可以看作是利用sigmod函数对后验概率P(y=1\|x)的逼近（具体请见[1](https://blog.csdn.net/qq_19645269/article/details/79551576)）。

## 2 问题背景定义

对于给定的m个样本（X<sub>i</sub>, y<sub>i</sub>）（i &in; [1,m]，X为n维向量）进行二分类：y = 0为负例，y = 1为正例。

## 3 公式推导

### 3.1 假设函数（hypothesis function）
<img src="https://latex.codecogs.com/gif.latex?h_{\theta&space;}(x)&space;=&space;\frac{1}{1&plus;e^{-(W^{T}X&plus;b)}}&space;=&space;\frac{1}{1&plus;e^{-\theta&space;^{T}x}}" title="h_{\theta }(x) = \frac{1}{1+e^{-(W^{T}X+b)}} = \frac{1}{1+e^{-\theta ^{T}x}}" />


注：1) 其中&theta;表示模型的参数，即w，b；
<img src="https://latex.codecogs.com/gif.latex?W^{T}x&space;=&space;w_{1}x_{1}&space;&plus;&space;w_{2}x_{2}&space;&plus;&space;...&space;&plus;&space;w_{n}x_{n}" title="W^{T}x = w_{1}x_{1} + w_{2}x_{2} + ... + w_{n}x_{n}" />
<img src="https://latex.codecogs.com/gif.latex?\theta^{T}x&space;=&space;\theta_{0}&space;&plus;&space;\theta_{1}x_{1}&space;&plus;&space;\theta_{2}x_{2}&space;&plus;&space;...&space;&plus;&space;\theta_{n}x_{n}" title="\theta^{T}x = \theta_{0} + \theta_{1}x_{1} + \theta_{2}x_{2} + ... + \theta_{n}x_{n}" />

### 3.2 预测正确的概率

我们将h<sub>&theta;</sub>(x)所给出的结果看作概率（因为sigmod可以将变量从（-&infin;，+&infin;）映射到（0，1）之间），即有：

<img src="https://latex.codecogs.com/gif.latex?p(y&space;=&space;1&space;|&space;w,&space;x)&space;=&space;h_{\theta&space;}(x)" title="p(y = 1 | w, x) = h_{\theta }(x)" />

<img src="https://latex.codecogs.com/gif.latex?p(y&space;=&space;0&space;|&space;w,&space;x)&space;=&space;1&space;-&space;h_{\theta&space;}(x)" title="p(y = 0 | w, x) = 1 - h_{\theta }(x)" />

<img src="https://latex.codecogs.com/gif.latex?p(correct)&space;=&space;h_{\theta&space;}(x)^{y}(1-h_{\theta&space;}(x))^{1-y}" title="p(correct) = h_{\theta }(x)^{y}(1-h_{\theta }(x))^{1-y}" />

注：1) p(correct)表示模型预测正确的概率：当y = 1即样本为正例时，p(correct) = h<sub>&theta;</sub>(x)；当y = 0即样本为负例时，p(correct) = 1 - h<sub>&theta;</sub>(x)。

2)h<sub>&theta;</sub>(x)的意义：例如对于某患者是否为患病，h<sub>&theta;</sub>(x)输出结果为0.7，则表示患者未患病的概率为70%，患病的概率为30%。

### 3.3 最大似然估计

由于我们的目标是找到合适的$\theta$使得所有样本的p(correct)最大，即使得模型预测正确所有样本的概率最大，所以我们采用最大似然估计：

<img src="https://latex.codecogs.com/gif.latex?L&space;(\theta)&space;=&space;\prod_{i&space;=&space;1}^{m}p(correct)^{i}&space;=&space;\prod_{i&space;=&space;1}^{m}h_{\theta&space;}(x^{i})^{y^{i}}(1-h_{\theta&space;}(x^{i}))^{1-y^{i}}" title="L (\theta) = \prod_{i = 1}^{m}p(correct)^{i} = \prod_{i = 1}^{m}h_{\theta }(x^{i})^{y^{i}}(1-h_{\theta }(x^{i}))^{1-y^{i}}" />

连乘不好计算，所以我们对上式取对数：

<img src="https://latex.codecogs.com/png.latex?l(\theta)&space;=&space;logL&space;(\theta)&space;=&space;\sum_{i&space;=&space;1}^{m}log(p(correct)^{i})&space;=&space;\sum_{i&space;=&space;1}^{m}(y^{i}log(h_{\theta&space;}(x^{i}))&space;&plus;&space;(1-y^{i})log(1-h_{\theta&space;}(x^{i})))" title="l(\theta) = logL (\theta) = \sum_{i = 1}^{m}log(p(correct)^{i}) = \sum_{i = 1}^{m}(y^{i}log(h_{\theta }(x^{i})) + (1-y^{i})log(1-h_{\theta }(x^{i})))" />

最优化任务时习惯上我们希望得到函数的最小值，所以对上式取负，即是求解最小值，并得到我们最终的损失函数（交叉熵损失函数）J(&theta;)：

<img src="https://latex.codecogs.com/png.latex?J(\theta)&space;=&space;-l(\theta)" title="J(\theta) = -l(\theta)" />

### 3.4 最佳&theta;值计算

我们使用梯度下降计算损失函数的最小值，具体算法见[2](https://en.wikipedia.org/wiki/Gradient_descent)，求解梯度（即损失函数偏导）的公式为：

<img src="https://latex.codecogs.com/png.latex?\frac{\partial&space;J(\theta)}{\partial&space;\theta}&space;=&space;\frac{\partial&space;(-\sum_{i&space;=&space;1}^{m}(y^{i}log(h_{\theta&space;}(x^{i}))&space;&plus;&space;(1-y^{i})log(1-h_{\theta&space;}(x^{i}))))}{\partial&space;\theta}" title="\frac{\partial J(\theta)}{\partial \theta} = \frac{\partial (-\sum_{i = 1}^{m}(y^{i}log(h_{\theta }(x^{i})) + (1-y^{i})log(1-h_{\theta }(x^{i}))))}{\partial \theta}" />

&emsp;&emsp;&nbsp;<img src="https://latex.codecogs.com/png.latex?=&space;-\sum_{i&space;=&space;1}^{m}(y^{i}\frac{1}{h_{\theta&space;}(x^{i})}\frac{\partial&space;h_{\theta&space;}(x^{i})}{\partial&space;\theta}&space;-&space;(1&space;-&space;y^{i})\frac{1}{1&space;-&space;h_{\theta&space;}(x^{i})}\frac{\partial&space;h_{\theta&space;}(x^{i})}{\partial&space;\theta})" title="= -\sum_{i = 1}^{m}(y^{i}\frac{1}{h_{\theta }(x^{i})}\frac{\partial h_{\theta }(x^{i})}{\partial \theta} - (1 - y^{i})\frac{1}{1 - h_{\theta }(x^{i})}\frac{\partial h_{\theta }(x^{i})}{\partial \theta})" />

&emsp;&emsp;&nbsp;<img src="https://latex.codecogs.com/png.latex?=&space;-\sum_{i&space;=&space;1}^{m}(y^{i}\frac{1}{h_{\theta&space;}(x^{i})}&space;-&space;(1&space;-&space;y^{i})\frac{1}{1&space;-&space;h_{\theta&space;}(x^{i})})\frac{\partial&space;\frac{1}{1&plus;e^{-\theta&space;^{T}x^{i}}}}{\partial&space;\theta}" title="= -\sum_{i = 1}^{m}(y^{i}\frac{1}{h_{\theta }(x^{i})} - (1 - y^{i})\frac{1}{1 - h_{\theta }(x^{i})})\frac{\partial \frac{1}{1+e^{-\theta ^{T}x^{i}}}}{\partial \theta}" />

&emsp;&emsp;&nbsp;<img src="https://latex.codecogs.com/png.latex?=&space;-\sum_{i&space;=&space;1}^{m}(y^{i}\frac{1}{h_{\theta&space;}(x^{i})}&space;-&space;(1&space;-&space;y^{i})\frac{1}{1&space;-&space;h_{\theta&space;}(x^{i})})\frac{x^{i}e^{-&space;\theta&space;^{T}x^{i}}}{(1&space;&plus;&space;e^{-\theta&space;^{T}x^{i}})^{2}}" title="= -\sum_{i = 1}^{m}(y^{i}\frac{1}{h_{\theta }(x^{i})} - (1 - y^{i})\frac{1}{1 - h_{\theta }(x^{i})})\frac{x^{i}e^{- \theta ^{T}x^{i}}}{(1 + e^{-\theta ^{T}x^{i}})^{2}}" />

&emsp;&emsp;&nbsp;<img src="https://latex.codecogs.com/png.latex?=&space;-\sum_{i&space;=&space;1}^{m}(y^{i}\frac{1}{h_{\theta&space;}(x^{i})}&space;-&space;(1&space;-&space;y^{i})\frac{1}{1&space;-&space;h_{\theta&space;}(x^{i})})x^{i}h_{\theta&space;}(x^{i})^{2}(\frac{1}{h_{\theta&space;}(x^{i})}-1)" title="= -\sum_{i = 1}^{m}(y^{i}\frac{1}{h_{\theta }(x^{i})} - (1 - y^{i})\frac{1}{1 - h_{\theta }(x^{i})})x^{i}h_{\theta }(x^{i})^{2}(\frac{1}{h_{\theta }(x^{i})}-1)" />

&emsp;&emsp;&nbsp;<img src="https://latex.codecogs.com/png.latex?=&space;-\sum_{i&space;=&space;1}^{m}(y^{i}\frac{1}{h_{\theta&space;}(x^{i})}&space;-&space;(1&space;-&space;y^{i})\frac{1}{1&space;-&space;h_{\theta&space;}(x^{i})})x^{i}h_{\theta&space;}(x^{i})(1&space;-&space;h_{\theta&space;}(x^{i}))" title="= -\sum_{i = 1}^{m}(y^{i}\frac{1}{h_{\theta }(x^{i})} - (1 - y^{i})\frac{1}{1 - h_{\theta }(x^{i})})x^{i}h_{\theta }(x^{i})(1 - h_{\theta }(x^{i}))" />

&emsp;&emsp;&nbsp;<img src="https://latex.codecogs.com/png.latex?=&space;-\sum_{i&space;=&space;1}^{m}(y^{i}(1&space;-&space;h_{\theta&space;}(x^{i}))&space;-&space;(1&space;-&space;y^{i})h_{\theta&space;}(x^{i}))x^{i}" title="= -\sum_{i = 1}^{m}(y^{i}(1 - h_{\theta }(x^{i})) - (1 - y^{i})h_{\theta }(x^{i}))x^{i}" />

&emsp;&emsp;&nbsp;<img src="https://latex.codecogs.com/png.latex?=&space;-\sum_{i&space;=&space;1}^{m}(&space;y^{i}&space;-&space;h_{\theta&space;}(x^{i}))x^{i}" title="= -\sum_{i = 1}^{m}( y^{i} - h_{\theta }(x^{i}))x^{i}" />
<img src="https://latex.codecogs.com/png.latex?=&space;\sum_{i&space;=&space;1}^{m}(h_{\theta&space;}(x^{i})&space;-&space;y^{i})x^{i}" title="= \sum_{i = 1}^{m}(h_{\theta }(x^{i}) - y^{i})x^{i}" />

其形式与最小二乘法梯度一致，区别在于最小二乘法中h<sub>&theta;</sub>(x) = W<sup>T</sup>x，也即是线性回归。
