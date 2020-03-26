---
layout: post
title: 深度学习中的概率论知识 part1
categories: [深度学习, 概率论]
description: 简单介绍深度学习中基本的概率论知识
keywords: 深度学习, 概率论
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

(前边部分太简单，直接跳过)

### 3.4 Marginal Probability

Sometimes we know the probability distribution over a set of variables and we want to know the probability distribution over just a subset of them. The probability over the subset is known as the **marginal probability distribution**.   

For example, suppose we have discrete random variables x and y, and we know P(x,y), we can find P(x) with the sum rule:   



$$\forall x∈X,P(x=x_i) = \sum_{y_j}P(x=x_i,y=y_j)$$   



For continuous variables, we need to use the integration instead of summation:   

$$p(x)=\int p(x,y)dy $$  



### 3.5 Conditional Probability

$$P(y=y_j | x=x_i) = \frac{P(y=y_j, x=x_i)}{P(x=x_i)}$$   

The conditional probability is only defined when $P(x=x_i)>0$.   



### 3.6 The Chain Rule of Conditional Probabilities.  



Any joint probability distribution over many random variables may be decomposed into conditional distributions over one variable:   

$$P(x^{(1)},...x^{(n)})=P(x^{(1)})\prod^n_{i=2}P(x^{(i)} | x^{(1)},...,x^{(i-1)})$$
For example:
$$P(a,b,c) = P(a|b,c)P(b,c)$$
$$P(b,c) = P(b|c)P(c)$$
$$P(a,b,c) = P(a|b,c)P(b|c)P(c)$$   



### 3.7 Independence and conditional independence

##### x an y are Independent:   



$$\forall x∈X, y∈Y, p(x,y)=p(x)p(y)$$
**compact notation:**    $x\bot y$     

##### x an y are conditional independent given a random variable z:   



$$\forall x∈X,y∈Y, p(x,y|z)=p(x|z)p(y|z)$$

**compact notation:**    $x\bot y | z$


### 3.8 Expectation, Variance and Covariance

#### Expectation:   



$$\mathbb E_{x\thicksim P}[f(x)] = \sum_xP(x)f(x)$$
While for continuous variables, it is computed with an integral:   

$$\mathbb E_{x\thicksim p}[f(x)] = \int p(x)f(x)dx$$ 

#### Expectation for linear equation:   



$$\mathbb E_x[\alpha f(x) + \beta g(x)] = \alpha \mathbb E_x[f(x)] + \beta \mathbb E_x[g(x)]$$
when $\alpha$ and $\beta$ are not depend on x.   



#### variance:   



$$Var(f(x)) = \mathbb E[(f(x) - \mathbb E[f(x)])^2]$$   



#### Covariance:   



$$Cov(f(x),g(y)) = \mathbb E[(f(x) - \mathbb E[f(x)])(g(y) - \mathbb E[g(y)])]$$.  



The **covariance matrix** of a random vector $x∈R^n$ is an $n×n$ matrix, such that:   

$$Cov(x)_{i,j}=Cov(x_i,x_j)$$
The diagonal elements of the covariance give the variance:   

$$Cov(x_i,x_i) = Var(x_i)$$