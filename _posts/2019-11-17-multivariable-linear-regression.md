---
layout: post
title: 多变量线性回归模型
categories: 机器学习
description: 基于多变量的线性回归模型
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

更一般的情形是如本节开头的数据集D，样本有d个属性描述. 此时我们试图学得：

$f(x_i)=w^Tx_i+b$， 使得$f(x_i)\backsimeq y_i$，这称为“多元线性回归”(multivariate linear regression).

类似的，可利用最小二乘法来对w和b进行估计. 为便于讨论，我们把w和b吸收入向量形式：

$\hat w = (w;b)$，相应的，把数据集D表示为一个$m×(d+1)$大小的矩阵$X$,其中每行对应于一个示例，该行前d个元素对应于示例的d个属性值，最后一个元素恒置为1，即：

![image-20200219180748832](/images/posts/ml/image-20200219180748832.png)


![image-20200219180837374](/images/posts/ml/image-20200219180837374.png)

