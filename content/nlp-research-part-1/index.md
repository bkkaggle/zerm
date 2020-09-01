+++
title="NLP Reseach Project Part 1"
description="Best Practices for Finetuning Large Transformer Language models"
date=2020-06-22

[taxonomies]
tags = ["ML", "NLP"]
categories = ["deep-dives"]

[extra]
+++

---

> Part 1: Best Practices for Finetuning Large Transformer Language models
>
> [Part 2: How I (almost) replicated OpenAI's GPT-2 (124M version)](/blog/nlp-research-part-2)

## Background

---

A few months ago I started working on a research project on how to best finetune GPT2-like language models for text generation. Once I ran a few experiments on that, I wanted to expand the scope of the project and try to pretrain my own, more efficient language model from scratch. I got access to a 128-core TPUv3 pod from the Tensorflow Reseach Cloud and used it to pretrain GPT2-124M to a perplexity pretty close to OpenAI's results (my pretrained model used was trained for about $1/8$th of the number of iterations that OpenAI trained their model for and got $21$ ppl on OpenWebText compared to $17$ ppl for OpenAI's model), and then pretrained an ALBERT-style GPT2 (that I'm calling ALGPT2) language model with a factorized input embedding and parameter sharing that would reduce the number of paramters from 124M to around 12M.

Unfortunately, ALGPT2 doesn't generate coherent, natural sounding text as well as GPT2 (ALGPT2 gets $31$ ppl on OpenWebText compared to $21$ ppl for my pretrained GPT2 model), but I'm writing this series of blog posts to go through everything I've learned over the last few months.

I have a cleaned-up version of my codebase on Github [here](https://github.com/bkkaggle/lm-training-research-project), and my original codebase with all my notes [here](https://github.com/bkkaggle/lm-finetuning).

You can take a look at my Weights&Biases workspace with all my runs [here](https://app.wandb.ai/bkkaggle/lm-finetuning).

---

## Objectives

---

I don't usually have access to a lot of compute (I mostly just use Google Colab) so I started out by limiting the scope of my project to finetuning or running inference on GPT2. I wrote down a [few notes](https://github.com/bkkaggle/lm-finetuning/blob/master/Markdown/RESEARCH.md#objectives) on what I wanted to look into:

---

-   If I wanted to finetune a LM to generate text of a specific style/content, what good defaults would I choose?
-   Find best practices or good defaults for finetuning tranformer language models for text generation.
-   Understand the effect of context len, model, and dataset size on generating coherent text

---

## Data

---

I ran most of the finetuning experiments on WikiText-2, which was small enough (~10mb on disk with a total of ~2m words) that I could run experiments fast enough (usually within 5-10m) on the v100 or p100 that I usually got through Colab.

I also ran a few experiments using WikiText-103 (~500mb, 100m words) but these were a lot harder to do because the size of the dataset forced me to use smaller batch sizes which took too long.

Loading in larger datasets (like WikiText-103) into memory can become pretty inefficient because of Python's overhead. IIRC, if you want to load the _entire_ WikiText-103 train set into memory with Python and tokenize the whole thing in one go using Huggingface's Tokenizers library and the GPT2 byte-level BPE tokenizer, it takes about 10 minutes and uses up about 60GB of RAM (Most of the time is spent tokenizing and most of the RAM is used loading the file into memory). Using something like [Apache Arrow](https://arrow.apache.org/) like the [Huggingface NLP](https://github.com/huggingface/nlp) library should make this a whole lot more efficient.

The WikiText datasets are stored as a single text file, with one Wikipedia article per line. Another more efficient way of processing the data would be to load in the file line-by-line and tokenize each line in parallel using the Huggingface tokenizer library's `batch_encode_plus()` function. This is a lot faster and efficient (taking up only 2GB of RAM and 2 minutes) but has its own drawbacks. `batch_encode_plus()` truncates sequences beyond that have more than `context_len` tokens and leaves sequences that are smaller than `context_len` as is, which means that you need to zero-pad the sequences that are smaller than `context_len` and discard any portion of a sequence beyond the first `context_len` tokens. For datasets that are used to benchmark the performance of a wide range of language models, this can lead to your model being harder to compare against other models that follow the commonly used convention of just tokenizing the entire dataset and chunking it into sequences of `context_len` length.

I wanted to make sure the way that I preprocessed the data made sure that the models that I finetuned on WikiText-2 and WikiText-103 would be comparable to other models, so in my code, I load in the entire dataset, tokenize it, and split it into contigous sequences of length `context_len`. There are a few other preprocessing-related factors that can affect how comparable results between different models can be, I [wrote a post on the topic](/blog/2020/5/14/evaluating-language-models/) a while ago, check it out if you're interested.

---

## Frameworks

---

I originally wrote all my code in vanilla Pytorch. I wanted to try using Colab's free TPUv3 board that has 8 TPU cores each with 16GB of RAM, each of which is a little slower that a single V100. Using the entire TPU board should be _at least_ as fast as using a cluster of 8 V100s but at a much lower cost.

I tried using [Pytorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) to see if it would help make the Pytorch model run on the Colab TPU more easily, but after about a week of trying to use the library I switched over to having two training scripts; One in plain Pytorch and one in Tensorflow2.0 with Keras — Even though Pytorch Lightning was very well designed, the time and effort required to make sure the framework is working the way that I want wasn't worth it in the end.

This was my first time working with TF/Keras since around early 2018 when I switched to Pytorch (back in the 0.3 days when you still had to use `Variable`). TF2.0 is a lot better now than it used to be two or three years ago but still doesn't feel as intuitive or easy to use as Pytorch. The documentation looks pretty good at first glance but there were a lot of gaps in the documentation when I was trying to figure out how to write and decode TFRecord files and scale my Keras code to TPUs and TPU pods.

> Trying to get gradient accumulation to work with TPUs was especially hard, IIRC grad accumulation isn't natively supported in Keras but there are a lot of independent implementations that people have open-sourced on Github, but they didn't work well with TPUs.

I used the [Huggingface/Transformers](https://github.com/huggingface/transformers) repository for the GPT-2 model and [Weights&Biases](https://www.wandb.com/) to track all the experiments that I ran.

> Fun fact, I ran into a problem with Colab's TPU a couple of times where I was silently downgraded from a TPUv3 to a TPUv2 and as a result I was getting a lot of OOM errors for a model and batch size that was working perfectly just a few hours ago. Colab doesn't really advertise this and makes it almost impossible to know if you have been downgraded :(

Pytorch recently released [Pytorch/XLA](https://github.com/pytorch/xla) which is supposed to let you run Pytorch code on TPUs with only a few changes to your code. I spent quite a bit of time to try and make this work but using it is still a lot more complex than just using a GPU.

Pytorch/XLA is a lot slower on Colab, which probably has something to do with Colab's network connection to the TPUs being a lot slower. Using [some operations](https://github.com/pytorch/xla/issues/1777) that aren't supported by Pytorch/XLA can have a pretty drastic impact on the speed of your program, so if your code is running unusually slow on a TPU, unsupported ops are a common culprit. [For example](https://github.com/pytorch/xla/issues/1777), I was trying to use the memory-saving Adafactor optimizer on Pytorch/XLA but since I was using a non-Pytorch operation in one part of the code (using Python's `sqrt()` function instead of `torch.sqrt()`), a single iteration was taking ~10 seconds compared to 10 iterations/second for SGD.

TPU support for Pytorch works pretty differently from TPU support for Tensorflow. Each TPU has a powerful dedicated CPU and several 100GBs of RAM for data processing, so whenever you run TF code on a TPU, your data gets copied to each core's CPU (unless you use TFRecord files, in which case each core's CPU downloads and processes data directly from your cloud bucket) and gets processed there. By doing it in this way, you only need to rent a small cloud instance (like a n1-standard-1) and scaling your code from a single TPU board with 8 cores to a part of a TPU pod is (relatively) painless.

On the other hand, Pytorch/XLA can't [currently use](https://github.com/pytorch/xla/issues/1742) the TPU's CPU and instead has to replicate the data $8$ times on your own VM for an $8$ core TPU board. If you want to use Pytorch/XLA for a TPU pod, you have to create a VM group with one host VM for each $8$ core TPU board. This means that Pytorch/XLA isn't currently practical for large scale training, but it [looks like](https://github.com/pytorch/xla/issues/1858) the next version of TPUs will be a lot more optimized for Pytorch. It works alright for a small dataset like WikiText-2 but when I tried finetuning on WikiText-103 (~500mb, 100m words) I needed to upgrade my VM to have 80+ GB of RAM.

---

## Finetuning

---

I wasn't able to finetune GPT2-1.5b on a TPU with the AdamW optimizer even with the TPU's built in bfloat16 support, so most of the experiments that I ran were with the memory-saving Adafactor optimizer with `beta1` set to zero to disable momentum. Enabling momentum might increase the performance of the Adafactor optimizer, but would also require storing an extra momentum value for each parameter and would make it harder to train larger models.

_Fun fact: The AdamW optimizer implementation in Google's official BERT [repository](https://github.com/google-research/bert/blob/master/optimization.py#L65) excludes layernorm and bias parameters from weight decay and AFAICT is the only optimizer that does so. I tried running a few experiments with and without finetuning these parameters and didn't find any significant difference in performance._

Most of the GPU experiments I did were with NVidia's Apex library, with its $01$ mixed precision setting. I also tried running a few experiments on using only FP16, but the gradients would explode or vanish and the model wouldn't train.

I have a [forked version](https://github.com/bkkaggle/transformers/tree/grad-checkpointing) of Huggingface's Transformers repository where I've implemented gradient checkpointing for GPT-2. I haven't been maintaining it but you can see all the changes that I did to make it work (it's really only a few lines of code) [here](https://github.com/huggingface/transformers/compare/master...bkkaggle:grad-checkpointing). I tried training GPT2-XL with grad checkpointing which IIRC worked with a smaller context length of 256 but still threw OOM errors when finetuning at a context length of 1024.

For small datasets like WikiText-2, (WikiText-2 consists of about 2 million words, so it's actually on the larger size for datasets that you might collect yourself) the model usually overfits within the first 1-3 epochs, so most of the experiments that I did trained for a single epoch — there really is no performance benefit to finetuning for any longer. I set the learning rate for all of my finetuning experiments to $5e-5$ (This was just the first value I tried, no hyperparameter tuning involved) and linearly increased the learning rate from $0$ to $5e-5$ over the first 10% of the training iterations and then linearly decayed it to zero over the rest of the training iterations.

> Note: The Adafactor paper shows that warmup is strongly recommended to stabilize training, take a look at the [paper](https://arxiv.org/abs/1804.04235) for more details

If you want to finetune GPT2 on a dataset like WikiText-2, there's a relationship between the batch size, learning rate, and the number of training iterations that you need to adjust to train effectively and avoid overfitting or plateauing. There's a pretty important ratio that you need to keep constant between the batch size and the learning rate. A larger batch size means that there are fewer gradient updates performed if you keep the number of training iterations constant.

I have a [WandB report](https://app.wandb.ai/bkkaggle/lm-finetuning/reports/1-epoch-context-len--Vmlldzo3NTI4MA) showing a few different training runs on WikiText-2 with different sized GPT2 models and context lengths. The results aren't really that surprising, finetuning larger models at larger context lengths increases perplexity significantly.

I wrote a quick [Colab notebook](https://colab.research.google.com/drive/1Vxh91ASFvCPgBL0I6ui97SyxtoLBvY3I?usp=sharing) on how to finetune and evaluate on WikiText-2.