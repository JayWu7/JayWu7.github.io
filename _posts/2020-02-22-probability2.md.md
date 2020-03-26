---
layout: post
title: 深度学习中的概率论知识 part2
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

### 3.9 Common Probability Distributions

直接在 prbability.cookbook 上看


### 3.10 Useful properties of common functions

#### Logistic sogmoid function:

$$\sigma(x)=\frac{1}{1+exp(-x)}$$
![c737a9bd6f3b7b61f93e8f2ef055d050](/images/posts/dl/prob1.png)

Another commonly encountered function is the **softplus function:**
$$\zeta (x) = log(1+exp(x))$$

The name of the softplus function comes from the fact that it is a smoothed, "softened" version of:
$$x^+ = max(0,x)$$
![1e5d68ce3b391c4703c3306d266e316a](/images/posts/dl/prob2.png)

#### Useful properties:

$$\sigma (x) = \frac{exp(x)}{exp(x) + exp(0)}$$
（上下都乘一个exp(x)）

$$\frac{d}{dx}\sigma (x) = \sigma (x)(1-\sigma (x))$$
$$1 - \sigma (x) = \sigma (-x)$$
$$\log \sigma (x) = -\zeta(-x)$$
$$\frac{d}{dx}\zeta(x) = \sigma(x)$$
$$\forall x ∈ (0,1), \sigma ^{-1} (x) = \log(\frac{x}{1-x})$$
$$\forall x >0, \zeta ^{-1}(x) = \log(exp(x) - 1)$$
$$\zeta(x) = \int_{-\infty}^x \sigma(y)dy$$
$$\zeta(x) - \zeta(-x) = x$$


### 3.11 Bayes' Rule

$$P(x|y) = \frac{P(x)P(y|x)}{p(y)}$$

$$P(y) = \sum_xP(y|x)P(x)$$

### 3.13 Information Theory

The basic intuition behind the information theory is that learning that an unlikely event has occurred is more informative than learning that a likely event has occurred. "今天早上有日食"包含的信息量就比“今天早上太阳出来了”要大得多.

We would like to quantify information in a way that formalizes this intuition.

* Likely events should have low information content, and in the extreme case,events that are guaranteed to happen should have no information contentwhatsoever.
* Less likely events should have higher information content.
* Independent events should have additive information. For example, ﬁnding out that a tossed coin has come up as heads twice should convey twice as much information as ﬁnding out that a tossed coin has come up as heads once.

To satisfy all three of these properties, we deﬁne the **self-information** of an event $x = x_i$ to be:
$$I(x) = -\log P(x)$$

Self-information deals only with a single outcome. We can quantify the amount of uncertainty in an entire probability distribution using the **Shannon entropy**(香农熵),
$$H(x) = \mathbb E_{x\sim P}[I(x)] = -\mathbb E_{x\sim P}[\log P(x)]$$
In other words, the Shannon entropy of a distribution is the expected amount of information in an event drawn from that distribution.


#### Kullback-Leibler (KL) divergence:

$$D_{KL}(P||Q) = \mathbb E_{x\sim P}[\log\frac{P(x)}{Q(x)}] = \mathbb E_{x\sim P}[\log P(x) - \log Q(x)]$$

The KL divergence is 0 iff $P$ and $Q$ are the same distribution in the case of discrete variables, or equal "almost everywhere" in the case of continuous variables.  It's often conceptualized as measuring some sort of "distance" between these distributions. Be careful that:
$$D_{KL}(P||Q) ≠ D_{KL}(Q||P)$$

A quantity that is closely related to the KL divergence is the **cross-entropy(交叉熵):**
$$H(P,Q) = -\mathbb E_{x\sim P}\log Q(x)$$
Minimizing the cross-entropy with respect to $Q$ is equivalent to minimizing the KL divergence, because $Q$ does not participate in the ommited term(left part).

**in the context of information theory, we treate 0 log0 as:**
$$\lim_{x\rightarrow0}\log x = 0$$