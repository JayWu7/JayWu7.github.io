---
layout: post
title: 深度学习中的线性代数 Part3
categories: [深度学习, 线性代数]
description: 简单介绍深度学习中需要用到的线性代数知识
keywords: 深度学习, 线性代数
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

### 2.8 Singular Value Decomposition

另外一种分解矩阵的方法，被称为奇异值分解（singular value decomposition，SVD），将矩阵分解为奇异向量（singular vector）和奇异值（singular value）。每个实数矩阵都有一个奇异值分解，但不一定都有特征分解。例如，非方阵的矩阵没有特征分解，这时我们只能使用奇异值分解。
我们将矩阵$A$分解成三个矩阵的乘积：
$$A = UDV^T$$
假设$A$上一个$m × n$的矩阵，那么$U$是一个$m × m$的矩阵，$D$是一个$m × n$的矩阵，$V$是一个$n × n$的矩阵。   
这些矩阵中的每一个经定义后都拥有特殊的结构。矩阵$U$和$V$都定义为正交矩阵，而矩阵$D$定义为对角矩阵。注意，矩阵$D$不一定是方阵。
对角矩阵$D$对角线上的元素被称为矩阵$A$的**奇异值（singular value）**。矩阵$U$的列向量被称为**左奇异向量（left singular vector）**，矩阵$V$的列向量被称为**右奇异向量（right singular vector）**。   
$A$的左奇异向量(left singular vector)是$AA^T$的特征向量。$A$的 右奇异向量(right singular vector)是$A^TA$的特征向量。A的非零奇异值是$A^TA$特征值的平方根，同时也是$AA^T$特征值的平方根。

SVD最有用的一个性质可能是拓展矩阵求逆到非方矩阵上。

### 2.9 The Moore-Penrose Pseudoinverse

Matrix inversion is not defined for matrices that are not square. The **Moore-Penrose pseudoinverse** enables us to make some headway in these cases. The pseudoinverse of $A$ is defined as a matrix:
$$
A^+ = \lim_{\alpha \searrow0}(A^TA+\alpha I)^{-1}A^T
$$
Practical algorithms for computing the pseudoinverse are based not on this definition above, but rather on the formular:
$$A^+ = VD^+U^T$$
where $U,D$ and $V$ are the sigular value decomposition of $A$, and the pseudoinverse $D^+$ of a diagonal matrix $D$ is obtained by taking the reciprocal of its nonzero elements then taking the transpose of the resulting matrix.

### The Trace Operator

The trace operator gives the sum of all the diagonal entries of a matrix:
$$Tr(A) = \sum_iA_{i,i}$$
The trace operator is useful for a variety of reasons. Some operations that are difficult to specify without resorting to summation notation can be specified using matrix products and the trace operator. For example, the trace operator provides an alternative way of writing the Frobenius norm of a matrix:
$$||A||_F = \sqrt{Tr(AA^T)}$$

Some useful equations of trace operator:
$$Tr(A) = Tr(A^T)$$
$$Tr(ABC)=Tr(CAB)=Tr(BCA)$$
$$Tr(AB)=Tr(BA)$$
$$a=TR(a)$$
where a is a scalar.


### 2.11 The Determinant（行列式）

**The determinant of a square matrix, denoted det(A), is a function that maps matrices to real scalars. The determinant is equal to the product of all the eigenvalues of the matrix. The absolute value of the determinant can be thought of as a measure of how much multiplication by the matrix expands or contracts space. If the determinant is 0, then space is contracted completely along at least one dimension, causing it to lose all its volume. If the determinant is 1, then the transformation preserves volume.**

