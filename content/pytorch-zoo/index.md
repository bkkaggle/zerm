+++
title="PyTorch Zoo"
description="A collection of useful modules and utilities in PyTorch"
date=2020-04-30

[taxonomies]
tags = ["ML"]
categories = ["cross-posts"]

[extra]
+++

I originally wrote this blog post for the PyTorch blog which is available [here](https://medium.com/pytorch/pytorch-zoo-a-collection-of-useful-modules-and-utilities-in-pytorch-c05ca4d500d8?source=friends_link&sk=9fcc0180af0abbc01d26d3680bdab83b) (Use this link to access my article with my friend link so you don't need to worry about Medium paywalling my article)

---

PyTorch Zoo is a collection of modules and utilities that I’ve found to be useful when working on machine learning projects and competitions. PyTorch Zoo contains several modules not available in PyTorch, like cyclical momentum and squeeze-and-excitation, as well as useful utilities like the ability to send notifications and set random seeds to get consistent results. PyTorch Zoo is meant to provide high-quality reference implementations of modules that don’t have official implementations in PyTorch and save you time that would have otherwise been spent searching for implementations on Github or coding the module yourself.

---

<img src='https://raw.githubusercontent.com/bilal2vec/pytorch_zoo/master/screenshot.png' width='100%'></img>

From: [https://github.com/bilal2vec/pytorch_zoo](https://github.com/bilal2vec/pytorch_zoo)

---

The library is open-source on [Github](https://github.com/bilal2vec/pytorch_zoo) and is available as a pip package. Just run:

```bash
pip install pytorch_zoo
```

to install it in your local development environment and check out the [documentation](https://github.com/bilal2vec/pytorch_zoo#documentation) for in-depth examples on all the library’s features. I’ve included quite a few modules in PyTorch Zoo, so I’ll try to focus only on some of the ones that I found to be the most interesting for this blog post.

---

# Cyclical Momentum

---

Cyclical momentum, which was first proposed in the same paper as cyclical learning rates [^1], is usually used together with cyclical learning rates. It decreases the amount of momentum while the learning rate increases and increases the amount of momentum while the learning rate decreases, stabilizing training and allowing for the use of higher learning rates. Here's an example of how you could use cyclical momentum just like a normal PyTorch scheduler:

{{ gist(url="https://gist.github.com/bilal2vec/30981747e186c406ea9c3213df9eb510") }}


---


# Squeeze and Excitation

---

Squeeze and Excitation modules [^2] [^3] can be easily integrated into existing models by just adding one of these modules after each convolutional block and improves the model’s performance without significantly impacting training time. All three variants of the squeeze-and-excitation block that were proposed in the original papers are available in PyTorch Zoo, see the [documentation](https://github.com/bilal2vec/pytorch_zoo#modules) for specific examples on how to use each one. Here's an example of how you could use SqueezeAndExcitation in a convolutional block

{{ gist(url="https://gist.github.com/bilal2vec/bd6e4d9a706f207235e193f85fedb8ec") }}

---

# Utilities

---

PyTorch Zoo also has a small range of utilities to make it easier to follow PyTorch best practices when doing things like saving a model to disk and setting random seeds, as well as easy to use one-liners to do things like sending push notifications when a training run ends.

Here’s an example of how you could use some of these utilities:

{{ gist(url="https://gist.github.com/bilal2vec/6b00aeff6a3aa9f3ea62b3b26a358fca") }}

---

# Conclusion

---

To learn more about PyTorch Zoo and its features, check out our [Github repository](https://github.com/bilal2vec/pytorch_zoo).

The project is still a work in progress, so if you find a bug, think there is something missing, or have any suggestions for new features or modules, feel free to open an issue or a pull request. Feel free to use the library or code from it in your own projects, and if you feel that some code used in this project hasn’t been properly accredited, please open an issue.

---

# References

---

[^1] https://arxiv.org/abs/1803.09820

[^2] https://arxiv.org/abs/1709.01507

[^3] https://arxiv.org/abs/1803.02579