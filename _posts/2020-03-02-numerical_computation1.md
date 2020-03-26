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

### 4.1 Overflow and Underflow

##### Underflow:   

Underflow occurs when numbers near zero are rounded to zero.

##### Overflow:   

Overflow occurs when numbers with large magnitude are approximated as $\infty$ or -$\infty$. 

***

One example of a function that must be stabilized against underflow and overflow is the **softmax function**. The softmax function is often used to predict the probabilities associated with a multinoulli distribution. It is defined to be:   
$$softmax(x)_i = \frac{exp(x_i)}{\sum_{j=1}^nexp(x_j)}$$


### 4.2 Poor Conditioning

Conditionint refers to how rapidly a function changes with respect to small changes in its inputs. Functions that change rapidly when their inputs are perturbed slightly can be problematic for scientific computation because rounding errors in the inputs can result in large changes in the output.

### Gradient-Based Optimization

Optimization refers to the task of either minimizing or maximizing some function $f(x)$ by altering $x$. We usually phrase most otimization problems in terms of minimizing $f(x)$. Maximization may be accomplished via a minimization algorithm by minimizing $-f(x)$.   

The function we want to minimize or maximize is called the **objective function**, or **criterion**. When we are minimizing it, we may also call it the **cost function**, **loss function**, or **error function**.

We often denote the value that minimizes of maximizes a function with a superscript **\***. For example, we might say $x^*=\arg \min f(x)$.

Suppose we have a function $y=f(x)$, where both $x$ and $y$ are real numbers. The **derivative** of this function is denoted as $f'(x)$ or as $\frac{dy}{dx}$. The derivative of $f'(x)$ gives the slope of $f(x)$ at the point $x$. In other words, it specifies how to scale a small change in the input to obtain the corresponding change in the output:
$$f(x+\epsilon)≈ f(x) + \epsilon f'(x)$$

The derivative is therefore useful for minimizing a function because it tells us how to change $x$ in order to make a small improvement in $y$. For example, we know that $f(x-\epsilon * sign(f'(x)))$ is less than $f(x)$ for small enough $\epsilon$. We can thus reduce $f(x)$ by moving $x$ in small steps with the opposite sign of the derivative. This technique is called **gradient descent**

When $f'(x)=0$, the derivative provides no information about which direction to move. Points where $f'(x) = 0$ are known as **critical points(临界点)** or **stationary points(驻点)**. A **local minimum** is a point where $f(x)$ is lower than at all neighboring points, so it is no longer possible to decrease $f(x)$ by making infinitesimal steps.    
Some critical points are neigher maxima nor minima. These are known as **saddle points(鞍点)**.

![6b397b593b042d1d0a0a206d14d64d81](/images/posts/dl/neu1.png)
A point that obtains the absolute lowest value of $f(x)$ is a **global minimum.** There can be only one global minimum or nultiple global minimum of the function. It's also possible for there to be local minimum that are not globally optimal. In the context of deep learning, we optimize functions that may have many local minimum that are not optimal and many saddle points surrounded by very flat regions. All of this makes optimization difficult, especially when the input to the function is multidimensional. We therefore usually settle for finding a value of $f$ that is very low but not necessarily minimal in any formal sense. Below is an example:
![78e331f4995e5c90eea3faa670eeb6b0](/images/posts/dl/neu2.png)
We often minimize functions that have multiple inputs: $f: \mathbb R^n \rightarrow \mathbb R$. For the concept of "mininization" to make sense, **there must still be only one (scalar) output**.

For function with multiple inputs, we must make use of the concept of **partial derivatives(偏导数)**. The particial derivative $\frac{\partial}{\partial x_i}f(x)$ measures how $f$ changes as only the variable $x_i$ increases at point $x$.
The **gradient(梯度)** generalizes the notion of derivative to the case where the derivative is with respect to a vector: the gradient of $f$ is the vector containing all the partial derivatives, denoted as **$\nabla _xf(x)$**. Element $i$ of the gradient is the partial derivative of $f$ with respect to $x_i$. **In multiple dimensions, critical points are points where every element of the gradient is equal to zero.**

The **directional derivative** in direction $u$ (a unit vector) is the slope of the function $f$ in direction $u$. In other words, the directional derivative is the derivative of the function $f(x+\alpha u)$ with respect to $\alpha$, evaluated at $\alpha =0$. Using the chain rule, we can see that $\frac{\partial}{\partial \alpha}f(x + \alpha u)$ evaluates to $u^T \nabla _xf(x)$ when $\alpha = 0$.

To minimize $f$, we would like to find the direction in which $f$ decreases the fastest. We can do this using the directional derivative:
$$\min_{u,u^T u = 1}u^T \nabla _xf(x)$$
$$= \min_{u,u^T u = 1} ||u||_2||\nabla_x f(x)||_2 \cos \theta$$
where $\theta$ is the angle between $u$ and the gradient. Substituting in $||u||_2 = 1$ and ignoring factors that do not depend on $u$, this simplifies to $\min_u \cos \theta$. This is minimized when $u$ points in the opposite direction as the gradient. In other words, the gradient points directly uphill, and the negative gradient points directly downhill. We can decrease $f$ by moving in the direction of negative gradient. This is known as the **method of steepest descent,** or **gradient descent**.   

Steepest descent proposes a new point:
$$x' = x-\epsilon \nabla_x f(x)$$
where $\epsilon$ is the **learning rate**, a positive scalar determining the size of the step. We can choose $\epsilon$ in several diﬀerent ways. A popular approach is to set $\epsilon$ to a small constant. Sometimes, we can solve for the step size that makes the directional derivative vanish. Another approach is to evaluate $f(x-\epsilon \nabla_x f(x))$ for several values of $\epsilon$ and choose the one that results in the smallest objective function value.This last strategy is called a **line search.**   
Steepest descent converges when every element of the gradient is zero (or, inpractice, very close to zero). In some cases, we may be able to avoid running this iterative algorithm and just jump directly to the critical point by solving the equation $\nabla_x f(x) = 0$ for $x$.   

Although gradient descent is limited to optimization in continuous spaces, the general concept of repeatedly making a small move(that is approximately the bestsmall move) toward better conﬁgurations can be generalized to discrete spaces. Ascending an objective function of discrete parameters is called **hill climbing**.