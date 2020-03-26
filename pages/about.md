---
layout: page
title: About
description: 芬兰阿尔托大学，机器学习硕士
keywords: Xiaobo Wu, 吴小波
comments: true
menu: 关于
permalink: /about/
---

我是JayWu，爱运动，爱生活。

坚信「人工智能将改变这个世界」。

在深度学习算法工程师这条路上努力着。

## 联系

{% for website in site.data.social %}
* {{ website.sitename }}：[@{{ website.name }}]({{ website.url }})
{% endfor %}

## Skill Keywords

{% for category in site.data.skills %}
### {{ category.name }}
<div class="btn-inline">
{% for keyword in category.keywords %}
<button class="btn btn-outline" type="button">{{ keyword }}</button>
{% endfor %}
</div>
{% endfor %}
