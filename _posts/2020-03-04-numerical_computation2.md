---
layout: post
title: 深度学习，数值计算要点 part1
categories: [深度学习]
description: 简要介绍一下深度学习中的数值计算常见注意事项和加速方法
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

### 4.3.1 Jacobian 和 Hessian 矩阵

有时我们需要计算输入和输出都为向量的函数的所有偏导数。 包含所有这样的偏导数的矩阵被称为 **Jaconian matrix**。具体来说，如果我们有一个函数：$f:\mathbb R^m \rightarrow R^n$，$f$的**Jacobian matrix** $J∈\mathbb R^{n×m}$ 定义为：
$$J_{i,j} = \frac{\partial}{\partial_{x_j}}f(x)_i$$
有时，我们也对 **二阶导数(second derivative)** 感兴趣. 二阶导数告诉我们， 一阶导数将如何随着输入的变化而变化。 它表示只基于梯度信息的梯度下降步骤是否会产生如我们预期的那样大的改善。   
当我们的函数具有多维输入时，二阶导数也有很多。我们可以将这些导数合并成一个矩阵，称为 **Hessian matrix**，定义为：
$$H(f)(x)_{i,j} = \frac{\partial ^2}{\partial_{x_i} \partial_{x_j}} f(x)$$
Hessian matrix 等价于梯度的 Jacobian matrix.

微分算子在任何二阶偏导连续的点处可交换，也就是它们的顺序可以互换：
$$\frac{\partial ^2}{\partial x_i \partial x_j}f(x)= \frac{\partial ^2}{\partial x_j \partial x_i}f(x)$$
这意味着 $H_{i,j} = H_{j,i}$，因此Hessian矩阵在这些点上是对称的。

二阶导数可以被用于确定一个临界点是否是局部极大点、局部极小点或鞍点。 回想一下，在临界点处 $f'(x) = 0$。 而$f''(x) > 0$意味着$f'(x)$会随着我们移向右边而增加，移向左边而减小。因此我们得出结论：

* 当$f'(x)=0且f''(x)>0$时，$x$是一个局部极小点
* 当$f'(x)=0且f''(x)<0$时，$x$是一个局部极大点

这就是**二阶导数测试(second derivative test)**。但是当$f''(x)=0$时测试是不确定的，在这种情况下，$x$可以是一个鞍点或平坦区域的一部分。

在多维情况下，我们需要检测函数的所有二阶导数。利用Hessian特征值分解，我们可以将二阶导数测试扩展到多维情况。在临界点处（$\nabla_xf(x) = 0$），我们通过检测Hessian的特征值来判断该临界点是一个局部极大点、局部极小点还是鞍点。当Hessian是正定的（所有特征值都是正的），则该临界点是局部极小点。因为方向二阶导数在任意方向都是正的。同样的，当 Hessian是负定的(所有特征值都是负的)，这个点就是局部极大点。

多维情况下，单个点处每个方向上的二阶导数是不同的。 Hessian的条件数衡量这些二阶导数的变化范围。当Hessian的条件数很差时，梯度下降法也会表现得很差。这是因为一个方向上的导数增加得很快，而在另一个方向上增加得很慢。 梯度下降不知道导数的这种变化，所以它不知道应该优先探索导数长期为负的方向。病态条件也导致很难选择合适的步长。步长必须足够小，以免冲过最小而向具有较强正曲率的方向上升。这通常意味着步长太小，以至于在其他较小的曲率方向上进展不明显。
我们可以使用Hessian矩阵的信息来指导搜索，以解决这个问题。其中最简单的方法是牛顿法（Newton's method），牛顿法基于一个二阶泰勒展开来近似$x^{(0)}$附近的$f(x)$:
$$f(x)≈f(x^{(0)}) + (x-x^{(0)})^T\nabla_x f(x^{(0)}) + \frac{1}{2}(x-x^{(0)})^T H(f)(x^{(0)})(x-x^{(0)})$$
接着通过计算，我们可以得到这个函数的临界点：
$$x^* = x^{(0)} - H(f)(x^{(0)})^{-1}\nabla_xf(x^{(0)})$$
当$f$是一个正定二次函数时，牛顿法只要应用一次上式就能跳到函数的最小点。

仅使用梯度信息的优化算法被称为**一阶优化算法**，如梯度下降。使用Hessian矩阵的优化算法被称为**二阶最优化算法**，如牛顿法。

