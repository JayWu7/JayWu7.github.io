---
layout: post
title: 简单线性回归模型
categories: Machine Learning
description: 基于单变量的线性回归模型
keywords: 线性回归, 机器学习
---

<head>
    <script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
    <script type="text/x-mathjax-config">
        MathJax.Hub.Config({
            tex2jax: {
            skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
            inlineMath: [['$','$']]
            }
        });
    </script>
</head>

#### 线性模型基本形式：

给定由d个属性描述的示例$x=(x_1;x_2;...;x_d)$，其中$x_i$是x在第i个属性上的取值，线性模型试图学得一个通过属性的线性组合来进行预测的函数，即

$$f(x)=w_1x_1+w_2x_2+...w_dx_d+b$$

一般用向量形式写成

$$f(x)=w^Tx + b$$

其中$w=(w_1;w_2;...;w_d)$.w和b学得之后，模型就得以确定.

线性模型形式简单、易于建模，但却蕴含着机器学习中一些重要的基本思想. 许多功能更为强大的非线性模型(nonlinear model)可在线性模型的基础上通过引入层级结构或高维映射而得. 此外，由于w直观表达了各属性在预测中的重要性，因此线性模型有很好的可解释性(comprehensibility).

#### 线性回归

给定数据集$D=\{(x_1,y_1),(x_2,y_2),...(x_m,y_m)\}$，其中$x_i=(x_{i1};x_{i2};...;x_{id}), y_i ∈ \mathbb R$. “线性回归”试图学得一个线性模型以尽可能准确地预测实值输出标记.

我们先考虑一种最简单的情形：输入属性的数目只有一个. 为便于讨论，此时我们忽略关于属性的下标，即$D={(x_i,y_i)}$. 对离散属性，若属性值间存在“序”(order)关系，可通过连续化将其转化为连续值，例如二值属性“身高”的取值“高”，“矮”可转化为{1.0,0.0}，三值属性“高度”的取值“高”“中”“低”可转化为{1.0,0.5,0.0}. 若属性值间不存在有序关系，假定有k个属性值，则通常转化为k维向量，例如属性“瓜类”的取值“西瓜”“南瓜”“黄瓜”可转化为(0,0,1),(0,1,0),(1,0,0).

线性回归试图学得：

$f(x_i)=Wx_i+b$，使得$f(x_i)\backsimeq y_i$

如何确定w和b呢？显然，关键在于如何衡量$f(x)$与$y$之间的差别.均方误差是回归任务中最常用的性能度量，因此我们可试图让均方误差最小化，即：

$$(w^*,b^*)= \mathop{\arg\min}_{(w,b)}\sum_{i=1}^m(wx_i+b-y_i)^2$$

均方误差有非常好的几何意义，它对应类常用的欧几里得距离或简称“欧式距离”(Euclidean distance). 基于均方误差最小化来进行模型求解的方法称为“最小二乘法”(least square method). 在线性回归中，最小二乘法就是试图找到一条直线，使得所有样本到直线上的欧式距离之和最小.

求解w和b使$E_{(a,b)}=\sum^m_{i=1}(y_i-wx_i-b)^2$最小化的过程，称为线性回归模型的最小二乘“参数估计”(parameter estimation). 我们可将$E_(w,b)$分别对$w$和$b$求导，得到：

$$\frac{\partial E_{(w,b)}}{\partial w}=2(w\sum_{i=1}^{m}x_i^2 - \sum^m_{i=1}(y_i-b)x_i$$

$$\frac{\partial E_{(w,b)}}{\partial b} = 2(mb - \sum_{i=1}^m(y_i-wx_i))$$

然后令上式为零，可得到w和b的最优解的闭式解：

$$w=\frac{\sum_{i=1}^my_i(x_i-\overline x)}{\sum^m_{i=1}x_i^2-\frac{1}{m}(\sum_{i=1}^{m}x_i)^2}$$

