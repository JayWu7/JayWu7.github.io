---
layout: post
title: 深度学习中的线性代数 Part2
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

### 2.6 Special Kinds of Matrices and Vectors

##### Diagonal matrix.  



Diagonal matrices consist mostly of zeros and have nonzero entries only along the main diagonal. Formally, a matrix D is diagonal if and only if $D_{i,j}=0$ for all i≠j. Identity matrix is one example of diagonal matrix.   



We write diag(v) to denote a square diagonal matrix whose diagonal entries are given by the entries of the vector $v$. Diagonal matrices are of interest in part because multiplying by a diagonal matrix is computationally efficient.   

**To compute $diag(v)x$, we only need to scale each element $x_i$ by $v_i$. In other words, $diag(v)x = v⊙x$**   
Inverting a square diagonal matrix is also efficient. The inverse exists only if every diagonal entry is nonzero, and in that case, **$diag(v)^{-1}=diag([1/v_1,...,1/v_n]^T) $**. 



##### Symmetric matrix.  



A symmetric matrix is any matrix that is equal to its own transpose:   

$$A = A^T$$
For example, if $A$ is a matrix of distance measurements, with $A_{i,j}$ giving the distance from point $i$ to point $j$, then $A_{i,j}=A_{j,i}$ because distance functions are symmetric.    



##### Unit vector:   



Unit vector is a vector with unit norm:   

$$||x||_2 = 1$$

##### Orthogonal:   



A vector $x$ and a vector $y$ are orthogonal to each other if $x^Ty=0$. If both vectors have nonzero norm, this means that they are at a 90 degree angle to each other. In $\mathbb R^n$, at most n vectors may ne mutually orthogonal with nonzero norm. If the vectors not only are orthogonal but also have unit norm, we call them orthonormal(标准正交).    



An **orthogonal matrix** is a square matrix whose rows are mutually orthonormal and whose columns are mutually orthonormal:   

$$A^TA = AA^T = I$$
This implies that:   

$$A^{-1} = A^T$$

So orthogonal matrices are of interest beacuse their inverse is very cheap to compute.   



### 2.7 Eigendecomposition(特征分解)

We can decompose matrices in ways that show us information about their functional properties that is not obvious from the representation of the matrix as an array of elements.   



One of the most widely used kinds of matrix decomposition is called **eigen-decomposition**, in which we decompose a matrix into a set of **eigenvectors** and **eigenvalues**.    



An **eigenvector** of a square matrix $A$ is a nonzero vector $v$ such that multi-plication by $A$ alters only the scale of $v$:   

$$Av = \lambda v $$   

The scalar $\lambda$ is known as the **eigenvalue** corresponding to this eigenvector.    

If $v$ is an eigenvec tor of $A$, then so is any rescaled vector $sv$ for $s∈\mathbb R, s≠0.$ Moreover, $sv$ still has the same eigenvalue. For this reason, we usually look only for unit eigenvectors.      



Suppose that a matrix $A$ has n linearly independent eigenvectors $\{v^{(1)},...,v^{(n)}\}$ with corresponding eigenvalues $\{\lambda_1,...,\lambda_n\}$. We may concatenate all the eigenvectors to form a matrix $V$ with one eigenvector per column: $V = [v^{(1)},...,v^{(n)}]$. Likewise, we can concatenate the eigenvalues to form a vector $\lambda = [\lambda_1,...,\lambda
_n]^T$. The eigendecomposition of $A$ is then given by:   

$$A=Vdiag(\lambda)V^{-1}$$

Not every matrix can be decomposed into eigenvalues and eigenvectors. In some cases, the decomposition exists but involves complex rather than real numbers. Fortunately, in this book, we ususally need to decompose only a specific class of matrices that have a simople decomposition. Specifically, every real symmetric matrix can be decomposed into an expression using only real-valued eigenvectors and    eigenvalues:   

$$A=Q\Lambda Q^T,$$   

Where $Q$ is an orthogonal matrix composed of eigenvectors of $A$, and $\Lambda$ is a diagonal matrix. The eigenvalue $\Lambda_{i,i}$ is associated with the eigenvector in column $i$ of $Q$, denoted as $Q:,i$. Because $Q$ is an orthogonal matrix, we can think of $A$ as scaling space by $\lambda_i$ in direction $v^{(i)} $  



While any real symmetric matrix $A$ is guaranteed to have an eigendecomposition, the tigendecomposition may not be unique. If any two or more eigenvectors share the same eigenvalue, then any set of orthogonal vectors lying in their span are also eigenvectors with that eigenvalue, and we could equivalently choose a $Q$ using those eigenvectors instead. By convention, we usually sort the entries of $\Lambda$ in descending order. Under this convention, the eigendecomposition is unique only if all the eigenvalues are unique.  

矩阵的特征分解给了我们很多关于矩阵的有用信息。矩阵是奇异的当且仅当含有零特征值。实对称矩阵的特征分解也可以用于优化二次方程 $f(x)=x^TAx$ , 其中限制 $||x||_2=1$.  当 $x$ 等于 $A$ 的某个特征向量时，$f$将返回对应的特征值。在限制条件下，函数 $f$ 的最大值是最大特征值，最小值是最小特征值。   

  


所有特征值都是正数的矩阵被称为 正定(positive definite);所有特征值都是非 负数的矩阵被称为 半正定(positive semidefinite)。同样地，所有特征值都是负数的 矩阵被称为 负定(negative definite);所有特征值都是非正数的矩阵被称为 半负定(negative semidefinite).  

**半正定矩阵受到关注是因为它们保证: $\forall x, x^T
Ax≥0$. 此外，正定矩阵还保证$x^TAx=0 \Rightarrow x=0$.   **



