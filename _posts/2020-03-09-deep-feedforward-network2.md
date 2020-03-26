---
layout: post
title: 深度前馈网络 part2
categories: [深度学习]
description: 简要介绍深度学习中的深度前馈神经网络
keywords: 深度学习
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

### 6.3 隐藏单元

隐藏单元的设计是一个非常活跃的研究领域，并且还没有许多明确的指导性理论原则。 **整流线性单元是隐藏单元极好的默认选择**
Relu： $g(z) = \max\{0,z\}$在z=0处不可微。这似乎使得$g$对于梯度的学习算法无效。然而在实践中，梯度下降对这些机器学习模型仍然表现得足够好。部分原因是神经网络训练算法通常不会达到代价函数的局部最小值，而是仅仅显著地减小它的值。

其它常用激活函数：
**sigmoid:**
$$g(z)=\sigma(z)$$
$$= \frac{1}{1+e^{-z}}$$

**tanh 双曲正切激活函数：**
$$g(z) = tanh(z)$$
$$= \frac{e^z - e^{-z}}{e^z + e^{-z}}$$

**其中：$tanh(z) = 2\sigma(2z) - 1$**


### 6.4 架构设计

神经网络设计的另一个关键点是确定它的架构：它应该具有多少单元，以及这些单元应该如何连接。

### 6.5 反向传播算法

当我们使用前馈神经网络接收输入$x$并产生输出$\hat{y}$时，信息通过网络向前流动。输入$x$提供初始信息，然后传播到每一层的隐藏单元，最终产生输出$\hat{y}$。这称之为 **前向传播(forward propagation)**。在训练过程中，前向传播可以持续向前直到它产生一个标量代价函数$J(\theta)$。**反向传播(back propagation)** 算法，允许来自代价函数的信息通过网络向后流动，以便于计算梯度。


#### 微积分中的链式法则

微积分中的链式法则用于计算复合函数的导数。反向传播是一种计算链式法则的算法，使用高效的特定运算顺序。设$x$是实数，$f$和$g$是从实数映射到实数的函数。假设$y=g(x)$并且$z=f(g(x))=f(y)$,那么：
$$\frac{dz}{dx}=\frac{dz}{dy}\frac{dy}{dx}$$
我们可以将这种标量情况进行扩展。假设$x∈ \mathbb R^m, y∈ \mathbb R^n$，$g$是从$\mathbb R^m$到$\mathbb R^n$的映射，$f$是从$\mathbb R^n$到$\mathbb R$的映射。如果$y=g(x), z=f(y)$,那么：
$$\frac{\partial z}{\partial x_i} = \sum_j \frac{\partial z}{\partial y_i}\frac{\partial y_i}{\partial x_i}$$
使用向量记法：
$$\nabla_{x}z = (\frac{\partial y}{\partial x})\nabla_{y}z$$
where $\frac{\partial y}{\partial x}$是$g$的$n × m$的Jacobian矩阵。