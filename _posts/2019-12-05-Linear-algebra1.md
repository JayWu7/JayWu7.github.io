---
layout: post
title: 深度学习中的线性代数 Part1
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

### 2.1 Scalars、Vectors、Matrices、Tensors

**Scalars：** A scalar is just a single number.

**Vectors:** A vector is an array of numbers. The numbers are arranges in order. 

**Matrices:** A matrix is a 2-D array of numbers, so each element is identified by two indices instead of just one.

**Tensors:** In some cases we will need an array with more than two axes. In the general case, an array of numbers arranged on a regular grid with a variable number of axes is known as a tensor.

**Transpose of matrix:** The transpose of a matrix is the mirror image of the matrix across a diagonal line, called the main diagonal, running down and to the right, starting from its upper left corner.

$$(A^T)_{i,j} = A_{j,i}$$


### 2.2 Multiplying Matrices and Vectors

The matrix product of matrices A and B is a third matrix C. In order for this product to be defined, A must have the same number of columns as B has rows. If A is of shape m × n and B is of shape n × p, then C is of shape m × p. We can write the matrix product just by placing two matrices together, for example:
$$C = AB$$
The product production is defined by:
$$C_{i,j}=\sum_{k}{A_{i,k}B_{k,j}}$$

**The matrix containing the product of the individual elements, such an operation exists and is called the element-wise product, or Hadamard product, and it's denoted as A⊙B**.


Matrix product operations have many useful properties that make mathematical analysis of matrices more convenient.

**distributive:**
$$A(B+C) = AB+AC$$

**associative:**
$$A(BC)=(AB)C$$

**matrix multiplication is not commutative(AB != BA)**

**However, the dot product between two vectors is commutative:**
$$x^{T}y = y^{T}x$$

**The transpose of a matrix product has a simple form:**
$$(AB)^T = B^TA^T$$


### 2.3 Identity and Inverse Matrices


#### Identity Matrix:

Identity Matrix is a matrix that does not change any vector when we multiply that vector by that matrix. We denote the identity matrix that preserves n-dimensional vectors as $I_n$. Formally, $I_n ∈ \mathbb{R^{n×n}}$,and:
$$\forall x∈\mathbb{R^{n}}, I_n x=x $$
**The structure of the identity matrix is simple: all the entries along the main diagonal are 1, while all the other entries are zero.**

The **matrix inverse** of A is denoted as $A^{-1}$, and it's defined as the matrix such that:
$$A^{-1}A=I_n$$


So, we can use this formula to compute the **linear equations**:

$Ax=b$, where A is a known matrix, b is a known vector, and $x∈ \mathbb{R^n}$ is a vector of unknown variables we would like to solve for.

$$
Ax = b
$$

$$
A^{-1}Ax = A^{-1}b
$$

$$
I_nx = A^{-1}b
$$

$$
x = A^{-1}b 
$$


### 2.4 Linear Dependence and Span(线性相关和生成子空间)

一组向量的线性组合，是指每个向量乘以对应标量系数之后的和，一组向量的生成子空间，是指原始向量线性组合后所能达到的点的集合.

如果一组向量中 的任意一个向量都不能表示成其他向量的线性组合，那么这组向量称为 线性无关 (linearly independent)

综上所述，这意味着该矩阵必须是一个 方阵(square)，即 m = n，并且所有列
向量都是线性无关的。一个列向量线性相关的方阵被称为 奇异的(singular)。 如果矩阵 A 不是一个方阵或者是一个奇异的方阵，该方程仍然可能有解。但是
我们不能使用矩阵逆去求解。
  目前为止，我们已经讨论了逆矩阵左乘。我们也可以定义逆矩阵右乘:
  $$AA^{-1}=I$$
 对于方阵而言，它的左逆和右逆是相等的。


### 2.5 Norms（范数）

Sometime we need to measure the size of a vector. In machine learning, we usually measure the size of vectors using a function called a **norm**. Formally, the $L^p$ norm is given by:
$$||x||_p = (\sum_i{|x_i|^p})^{\frac{1}{p}}$$
for $p∈\mathbb R, p≥1$.

On an intuitive level, the norm of a vector $x$ measures the distance from the origin(原点) to the point $x$. More rigorously, a norm is any function $f$ that satisfies the following properties:

* $f(x)=0 \Rightarrow x=0$
* $f(x+y) ≤ f(x) + f(y)$
* $\forall \alpha∈ \mathbb R, f(\alpha x)=|\alpha|f(x)$

The $L^2$ norm, with p=2, is known as the **Euclidean norm**, which is simply the Euclidean distance from the origin to the point identified by $x$. It's often denoted simply as $||x||$, 

**$L^1$ norm is commonly used when the difference between zero and nonzero elements is very important.**

$$L^1 = ||x||_1 = \sum_i |x_i|$$

We sometimes measure the size of the vector by counting its number of nonzero elements, some authors refer to this function as the "$L^0$ norm", but this is incorrect in terminology.

One other norm that commonly arises in machine learning is the $L^∞$ norm, also known as the **max norm**. This simplifies to the absolute value of the element with the largest magnitude in the vector.
$$||x||_∞ = \max_{i}|x_i|$$

Sometimes we may also wish to measure the size of a matrix. In the context of deep learning, the most common way to do this is with the otherwise obscure **Frobenius norm:**
$$||A||_F = \sqrt{\sum_{i,j}A^2_{i,j}}$$

**The dot product of two vectors can be rewritten in terms of norms. Specifically,**
$$x^Ty=||x||_2||y||_2cos\theta$$
where $\theta$ is the angle between x and y.

