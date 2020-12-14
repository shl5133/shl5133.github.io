---
layout: post
title:  "Unbiased Estimate"
date:   2020-12-13 16:28:00 -0600
tags:
   - Machine Learning Fundation
category: Machine Learning
created: 2020/12/13 16:30:00
last modified: 2020/12/13 16:30:00
---

This Blog is about Unbiased Estimate.

### Contents

- [什么是无偏估计](#a)
- [为什么要采用无偏估计](#b)
- [如何进行无偏估计](#c)
- [总体均值及方差的无偏估计及证明](#d)

<a name='a'></a>

### 1 什么是无偏估计
很多时候我们要对全体样本的统计量（如:平均数，众数等）进行计算时，我们有可能无法得到所有样本（如要计算全国人的平均身高），所以只能够采用部分样本对这些指标进行估计，这就会涉及到无偏估计以及有偏估计。估计值的数学期望等于真实值则认为此估计值是无偏的，否则就认为是有偏的。简单地说无偏估计指的是多次估计后估计值的均值与真实值相等（也就是估计值基本上在真实值附近）。

<a name='b'></a>

### 2 为什么要采用无偏估计

以射箭靶为例，靶心就是我们要求的真实值，我们的目标是要射中靶心，当然靶心我们是无法知晓的（如果可以得到所有数据那么靶心就是可知道，我们可以直接计算真实值而不需要进行估计）。我们只能通过在总体数据中抽样出样本，并采用某种算法对真实值进行估计进而得到估计值。无偏估计可以认为我们的是瞄向靶心的（没有系统误差），而有偏估计则并没有瞄准靶心（存在系统误差）。

直觉上我们认为无偏估计是好的，通常我们也采用无偏估计，但是无偏估计不一定就是好估计。判断一个估计量的好坏有三个性质：无偏性，有效性以及一致性。

- 有效性：指估计量与总体参数的离散程度，离散程度用方差来衡量。如果两个估计量都是无偏的，那么离散程度较小的估计量相对来说是有效的。估计量越靠近目标，我们认为效果越“好”，也即是有效性越高。
- 一致性：样本数目越大，估计量就越接近总体参数的真实值，也即是估计量的序列在概率上收敛于真实值（如下式所示）。
![1](/images/unbiased_estimate/unbiased_1.png)

满足三个性质的估计值是一个好的估计值，但是现实中找到满足全部三个性质的估计值并不容易，所以我们需要根据实际情况进行取舍。如下图中，右图虽然有偏但是其有效性更好，也许是一个更好的选择。
![2](/images/unbiased_estimate/unbiased_2.png)

<a name='c'></a>

### 3 如何进行无偏估计
要想得到无偏估计值，那么在各个环节都需要保证无偏。

1.怎么采样：采集形式是否无偏（如问卷的设计有无歧义等）；

2.选择哪些样本：样本选择方式是否无偏（如没有选择到某一类样本或者没有对某一性质没有加以区分）；

...

<a name='d'></a>

### 4 总体均值及方差的无偏估计及证明
<a name='mean'></a>
1.样本均值期望等于总体均值，所以样本均值是对总体均值的无偏估计。

设有随机变量X，其期望为μ。抽样n个样本X={X<sub>1</sub>,X<sub>2</sub>,..., X<sub>n</sub>}
![3](/images/unbiased_estimate/unbiased_3.png)
2.样本方差小于总体方差

设有随机变量X，其方差为σ<sup>2</sup>。抽样n个样本X={X<sub>1</sub>,X<sub>2</sub>,..., X<sub>n</sub>}
![4](/images/unbiased_estimate/unbiased_4.png)
3.样本方差期望小于总体方差，所以样本均值是对总体均值的有偏估计，但是1/(n-1)倍的样本方差则是无偏估计。
总体方差σ<sup>2</sup>可由下式计算：
![5](/images/unbiased_estimate/unbiased_5.png)
按照上式计算总体方差是需要得知X的具体分布或者所有数据，但实际中我们是无法得知的。所以需要用样本去估计方差，S<sup>2</sup>就是对σ<sup>2</sup>的一个近似（下式[1](#f1)），但是S<sup>2</sup>中的期望μ实际上我们也无法得到，所以需要用样本均值去估计总体均值（如[1](#mean)所述）（下式[2](#f2)）。
<a name='f1'></a>
![6](/images/unbiased_estimate/unbiased_6.png)
<a name='f2'></a>
![7](/images/unbiased_estimate/unbiased_7.png)
下面我们来验证S<sup>2</sup>是对σ<sup>2</sup>的有偏估计还是无偏估计：
<img src="https://latex.codecogs.com/gif.latex?E(S^{2})=E[\frac{1}{n}\sum_{n}^{i=1}(X_{i}-\bar{X})^{2}]" title="E(S^{2})=E[\frac{1}{n}\sum_{n}^{i=1}(X_{i}-\bar{X})^{2}]" />
<img src="https://latex.codecogs.com/gif.latex?=E[\frac{1}{n}\sum_{n}^{i=1}((X_{i}-\mu&space;)-(\bar{X}-\mu))^{2}]" title="=E[\frac{1}{n}\sum_{n}^{i=1}((X_{i}-\mu )-(\bar{X}-\mu))^{2}]" />
<img src="https://latex.codecogs.com/gif.latex?=E[\frac{1}{n}\sum_{n}^{i=1}((X_{i}-\mu&space;)^{2}-2(X_{i}-\mu&space;)(\bar{X}-\mu)&plus;(\bar{X}-\mu)^{2})]" title="=E[\frac{1}{n}\sum_{n}^{i=1}((X_{i}-\mu )^{2}-2(X_{i}-\mu )(\bar{X}-\mu)+(\bar{X}-\mu)^{2})]" />
<img src="https://latex.codecogs.com/gif.latex?=E[\frac{1}{n}\sum_{n}^{i=1}(X_{i}-\mu&space;)^{2}-\frac{2}{n}\sum_{n}^{i=1}(X_{i}-\mu&space;)(\bar{X}-\mu)&plus;\frac{1}{n}\sum_{n}^{i=1}(\bar{X}-\mu)^{2}]" title="=E[\frac{1}{n}\sum_{n}^{i=1}(X_{i}-\mu )^{2}-\frac{2}{n}\sum_{n}^{i=1}(X_{i}-\mu )(\bar{X}-\mu)+\frac{1}{n}\sum_{n}^{i=1}(\bar{X}-\mu)^{2}]" />
<img src="https://latex.codecogs.com/gif.latex?=E[\frac{1}{n}\sum_{n}^{i=1}(X_{i}-\mu&space;)^{2}-\frac{2}{n}\sum_{n}^{i=1}(X_{i}-\mu&space;)(\bar{X}-\mu)&plus;\frac{1}{n}(\bar{X}-\mu)^{2}\sum_{n}^{i=1}1]" title="=E[\frac{1}{n}\sum_{n}^{i=1}(X_{i}-\mu )^{2}-\frac{2}{n}\sum_{n}^{i=1}(X_{i}-\mu )(\bar{X}-\mu)+\frac{1}{n}(\bar{X}-\mu)^{2}\sum_{n}^{i=1}1]" />
<img src="https://latex.codecogs.com/gif.latex?=E[\frac{1}{n}\sum_{n}^{i=1}(X_{i}-\mu&space;)^{2}-\frac{2}{n}\sum_{n}^{i=1}(X_{i}-\mu&space;)(\bar{X}-\mu)&plus;\frac{1}{n}(\bar{X}-\mu)^{2}n]" title="=E[\frac{1}{n}\sum_{n}^{i=1}(X_{i}-\mu )^{2}-\frac{2}{n}\sum_{n}^{i=1}(X_{i}-\mu )(\bar{X}-\mu)+\frac{1}{n}(\bar{X}-\mu)^{2}n]" />
<img src="https://latex.codecogs.com/gif.latex?=E[\frac{1}{n}\sum_{n}^{i=1}(X_{i}-\mu&space;)^{2}-\frac{2}{n}(\bar{X}-\mu)\sum_{n}^{i=1}(X_{i}-\mu&space;)&plus;(\bar{X}-\mu)^{2}]" title="=E[\frac{1}{n}\sum_{n}^{i=1}(X_{i}-\mu )^{2}-\frac{2}{n}(\bar{X}-\mu)\sum_{n}^{i=1}(X_{i}-\mu )+(\bar{X}-\mu)^{2}]" />
<img src="https://latex.codecogs.com/gif.latex?and, \bar{X}-\mu=\frac{1}{n}\sum_{n}^{i=1}(X_{i}-\mu&space;)" title="and, \bar{X}-\mu=\frac{1}{n}\sum_{n}^{i=1}(X_{i}-\mu )" />
<img src="https://latex.codecogs.com/gif.latex?E(S^{2})=E[\frac{1}{n}\sum_{n}^{i=1}(X_{i}-\mu&space;)^{2}-\frac{2}{n}(\bar{X}-\mu)n(\bar{X}-\mu)&plus;(\bar{X}-\mu)^{2}]" title="E(S^{2})=E[\frac{1}{n}\sum_{n}^{i=1}(X_{i}-\mu )^{2}-\frac{2}{n}(\bar{X}-\mu)n(\bar{X}-\mu)+(\bar{X}-\mu)^{2}]" />
<img src="https://latex.codecogs.com/gif.latex?=E[\frac{1}{n}\sum_{n}^{i=1}(X_{i}-\mu&space;)^{2}-2(\bar{X}-\mu)^{2}&plus;(\bar{X}-\mu)^{2}]" title="=E[\frac{1}{n}\sum_{n}^{i=1}(X_{i}-\mu )^{2}-2(\bar{X}-\mu)^{2}+(\bar{X}-\mu)^{2}]" />
<img src="https://latex.codecogs.com/gif.latex?=E[\frac{1}{n}\sum_{n}^{i=1}(X_{i}-\mu&space;)^{2}-(\bar{X}-\mu)^{2}]" title="=E[\frac{1}{n}\sum_{n}^{i=1}(X_{i}-\mu )^{2}-(\bar{X}-\mu)^{2}]" />
<img src="https://latex.codecogs.com/gif.latex?=\sigma&space;^{2}-E[(\bar{X}-\mu)^{2}]" title="=\sigma ^{2}-E[(\bar{X}-\mu)^{2}]" />
<img src="https://latex.codecogs.com/gif.latex?=\sigma&space;^{2}-E[(\bar{X}-E[\bar{X}])^{2}]" title="=\sigma ^{2}-E[(\bar{X}-E[\bar{X}])^{2}]" />
<img src="https://latex.codecogs.com/gif.latex?=\sigma&space;^{2}-D(\bar{X})" title="=\sigma ^{2}-D(\bar{X})" />
<img src="https://latex.codecogs.com/gif.latex?=\sigma&space;^{2}-\frac{1}{n}\sigma&space;^{2}" title="=\sigma ^{2}-\frac{1}{n}\sigma ^{2}" />
<img src="https://latex.codecogs.com/gif.latex?=\frac{n-1}{n}\sigma&space;^{2}\neq&space;\sigma&space;^{2}" title="=\frac{n-1}{n}\sigma ^{2}\neq \sigma ^{2}" />
从上式可以看出S<sup>2</sup>是对σ<sup>2</sup>的有偏估计。但是我们可以将其修正为无偏估计。
<img src="https://latex.codecogs.com/gif.latex?\sigma^{2}=&space;\frac{n}{n-1}E(S^{2})=\frac{n}{n-1}E(\frac{1}{n}\sum_{n}^{i=1}(X^{i}-\bar{X})^{2})=E(\frac{1}{n-1}\sum_{n}^{i=1}(X^{i}-\bar{X})^{2})=E(\frac{1}{n-1}S^{2})" title="\sigma^{2}= \frac{n}{n-1}E(S^{2})=\frac{n}{n-1}E(\frac{1}{n}\sum_{n}^{i=1}(X^{i}-\bar{X})^{2})=E(\frac{1}{n-1}\sum_{n}^{i=1}(X^{i}-\bar{X})^{2})=E(\frac{1}{n-1}S^{2})" />
由上式可得，修正后1/(n-1)倍的S<sup>2</sup>的期望对σ<sup>2</sup>是一种无偏估计。

### References
\[1\] [什么是无偏估计](https://www.zhihu.com/question/22983179/answer/404391738)

\[2\] [Unbiased in Statistics: Definition and Examples](https://www.statisticshowto.com/unbiased/)

\[3\] [为什么样本方差（sample variance）的分母是 n-1](https://www.zhihu.com/question/20099757)
