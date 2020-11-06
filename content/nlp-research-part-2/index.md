+++
title="NLP Reseach Project Part 2"
description="How I (almost) replicated OpenAI's GPT-2 (124M version)"
date=2020-07-17

[taxonomies]
tags = ["ML", "NLP"]
categories = ["deep-dives"]

[extra]
+++

---

> [Part 1: Best Practices for Finetuning Large Transformer Language models](/blog/nlp-research-part-1)
>
> Part 2: How I (almost) replicated OpenAI's GPT-2 (124M version)

---

# TL;DR

---

A few months ago I started working on a research project trying to pretrain my own, more efficient language model from scratch. I got access to a 128-core TPUv3 pod from the Tensorflow Reseach Cloud and used it to pretrain a $124$M parameter GPT-2 model to a perplexity pretty close to OpenAI's results (my pretrained model was trained for about $1/8$th of the number of iterations that OpenAI trained their model for and got $21$ ppl on OpenWebText compared to $17$ ppl for OpenAI's model), and then pretrained an ALBERT-style GPT-2 (that I'm calling ALGPT2) language model with a factorized input embedding and layer-wise parameter sharing that would reduce the number of paramters in the model from $124$M to around $12$M.

Unfortunately, ALGPT-2 doesn't perform as well as GPT-2 (ALGPT-2 gets $31$ ppl on OpenWebText compared to $21$ ppl for my pretrained GPT-2 model), but I'm writing this series of blog posts to go through everything I've learned over the last few months.

---

# The Idea

---

The main thing that I wanted to do from this sort-of "research project" that I was working on by myself this spring was to develop and train a more efficient version of the $124$M parameter version of [GPT-2](https://openai.com/blog/better-language-models/). I wanted to pretrain the $1.5$B parameter version of GPT-2 but since I only got access to the TPU pod for a week, I had to choose a model that would train in time. A $100$k iteration training run takes about $20$ hours to run which gave me plenty of time to run multiple experiments. In contrast, following OpenAI's training procedure exactly and training for the full $800$k iterations would take up almost an entire week and use up most of my quota.

I was able to almost replicate the $124$M parameter version of GPT-2 by pretraining it to a perplexity pretty close to OpenAI's results (my pretrained model used was trained for about $1/8$th of the number of iterations that OpenAI trained their model for and got $21$ perplexity (ppl) on the standard OpenWebText dataset compared to $17$ ppl for OpenAI's model),

My idea of making a more efficient transformer didn't really work out since my pretrained transformer ended up being about $20$ppl worse than an equivalent GPT-2 model, but I wanted to writeup what I learned over the two or three months that I was working on this anyway.

---

# Background

---

A little bit about myself: I'm an incoming software engineering student at the University of Waterloo and this post is supposed to be a writeup of a NLP research project that I was working on from around March to May of 2020 (right in the middle of the first Covid-19 lockdown of 2020, I'm currently writing this on July 15, 2020 while waiting for my Pix2PixHD model to train for a few hundred epochs on colab for a new project that I'm working on).

Over the last three or four years I've done a lot of machine learning related stuff. I started out back in early 2017 by going through the [Introduction to Machine Learning with Python](https://www.google.com/search?sxsrf=ALeKk010AvJn990-Vqb1MA50AAVbnMg8uw:1594828665718&q=Introduction+to+Machine+Learning+with+Python:+A+Guide+for+Data+Scientists&stick=H4sIAAAAAAAAAONgVuLVT9c3NEwyLzRLKs_Ke8RowS3w8sc9YSn9SWtOXmPU5OIKzsgvd80rySypFJLmYoOyBKX4uVB18ixi9fTMKynKTylNLsnMz1MoyVfwTUzOyMxLVfBJTSzKy8xLVyjPLMlQCKgsycjPs1JwVHAvzUxJVUjLL1JwSSxJVAhOzkwFGl9cUgwAxHs76ZgAAAA&biw=1920&bih=969) and [Hands-On Machine Learning with Scikit-Learn and TensorFlow](https://www.google.com/search?q=hands+on+machine+learning+with+scikit-learn+and+tensorflow&oq=hands+on+mac&aqs=chrome.0.0j69i57j46j0l5.1936j0j7&sourceid=chrome&ie=UTF-8) books. At the time, I didn't really understand all the math behind neural networks, but it got me hooked on ML and then I took the [deeplearning.ai](https://www.deeplearning.ai/) courses on Coursera and the original [fast.ai](https://www.fast.ai/) course (back in late 2017 when they hadn't switched over to Pytorch and still used Tensorflow and Keras).

I started competing on [Kaggle](https://kaggle.com) in early 2018 and kept on competing in competitions non-stop for about a year and a half, winning a few medals and becoming a competitions expert (At one point I was ranked in the top $100$ Kagglers on the competitions leaderboard). Kaggle was a really nice way to get a lot of experience using neural networks because of the wide range of competitions and datasets that I had access to. I started out by doing a few semantic segmentation competitions then moved onto competing in NLP competitions. Since around mid 2019, I've been working on a bunch of different projects in ML and lower-level CS stuff. I worked on making a PyTorch-style machine learning library in [C++](https://github.com/bilal2vec/L2/tree/c%2B%2B) and more recently in [Rust](https://github.com/bilal2vec/L2), and for the last few months I've also been trying to keep up with all the new machine learning (esp. NLP) papers on arXiv.

I was pretty lucky that I started learning NLP right before transformers exploded in popularity, I remember when [word2vec](https://arxiv.org/abs/1301.3781) and LSTMs were still SOTA on a lot of NLP tasks, and it has been really interesting to see how much the field of NLP has changed in just a few years, going from when LSTMs with only a a handful of layers and somewhere on the order of $512$ units were considered to be large networks and computationally expensive to train, to training LSTMs with [attention](https://arxiv.org/abs/1409.0473) layers on top, to the original [transformer encoder/decoder networks](https://arxiv.org/abs/1706.03762), to [ULMFIT](https://arxiv.org/abs/1801.06146) and [ELMO](https://arxiv.org/abs/1802.05365), then [BERT](https://arxiv.org/abs/1810.04805), [RoBERTa](https://arxiv.org/abs/1907.11692), [GPT-2](https://openai.com/blog/better-language-models/), and [T5](https://arxiv.org/abs/1910.10683), to just a few months ago with the explosion of new, more efficient replacements for self-attention like the [Sparse Transformer](https://openai.com/blog/sparse-transformer/), the [Reformer](https://arxiv.org/abs/2001.04451), and [Synthesizers](https://arxiv.org/abs/2005.00743), and now [GPT-3](https://arxiv.org/abs/2005.14165), which IMO has the potential to really change the whole field of NLP.

Just a few years ago we trained shallow recurrent networks on datasets, then pretrained large transformer language models on large datasets and finetuned on task-specific datasets. Now the whole idea of just training a gigantic language model on a huge dataset, then conditioning the model in a form of few-shot learning by prepending a few examples of a certain task to an input feels like it can really make NLP models a lot more accessible and easier to productionize as well as making human-chatbot interactions a lot more realistic than they are today.

I've rambled on for long enough, lets get to the main topic of this post.


---

# GPT-2 and ALBERT

---

[GPT-2](https://openai.com/blog/better-language-models/) is a transformer decoder. 


The embedding layer at the root of the model maps a one-hot vector of a given token's index (all the GPT-2 models use a vocabulary size of $50257$) to a $768$ dimensional vector (all GPT-2 numbers in this blog post will be for the $124$m parameter version of GPT-2).

The embedding matrix is followed by a stack of self-attention and feed-forward layers that each output a $768$ dimensional vector (keeping the number of outputs for each layer constant), which makes up the main part of the transformer.

The stack of self-attention layers is then followed by an output embedding (the weights of the input and output embeddings are tied to make training easier) that maps the $768$ dimensional vector that is the output of the last layer of the transformer to the same $50257$ dimensional vector that represents the probability of each token in the vocabulary being the next token in the sequence.

Take a look at [The Illustrated GPT-2](http://jalammar.github.io/illustrated-gpt2/) for a more in-depth look into GPT-2.

[ALBERT](https://arxiv.org/abs/1909.11942) (A Lite BERT) is a paper that takes a look at [BERT](https://arxiv.org/abs/1810.04805) and identifies some ways in which to make it more efficient and reduce the number of parameters in the model in four major ways: a factorized embedding, layer-wise parameter sharing, a sentence-order-prediction auxillary loss, and removing dropout.

---

### Factorized embedding

---

GPT-2's embedding has a lot of parameters. It's really just a matrix of dimensions $50257 \times 768$. That means that the input embedding alone uses up almost $50257 \times 768 = \space \sim 38,000,000$ parameters which is a pretty big chunk of the $128$M total parameters in the model.

The ALBERT authors propose a factorized embedding with an intermediate embedding size of $128$: one embedding of size $50257 \times 128$ and another embedding of size $128 \times 768$. By breaking up the large embedding matrix into two smaller matrices, the total number of parameters used in the embedding goes from about $38$M to about $6$M.

$50257 \times 128 = \sim 6,000,000$

$128 \times 768 = \sim 100,000$

The authors try different intermediate embedding sizes and settle on $128$ as a good tradeoff betweeen parameters and performance.

---

### Layer-wise parameter sharing

---

In a normal transformer model, the transformer layers are created something like this:

```python
class BERT(nn.Module):
    def __init__(self, n_layers):
        super().__init__()
        // ...
        self.blocks = nn.ModuleList([Block() for _ in range(n_layers)])
        // ...
    def forward(self, x):
        // ...
        for block in self.blocks:
            x = block(x)
        // ...
```

ALBERT shares all parameters across the transformer layers something like this:

```python
class ALBERT(nn.Module):
    def __init__(self, n_layers):
        super().__init__()
        // ...

        self.n_layers = n_layers
        self.block = Block()
        // ...
    def forward(self, x):
        // ...
        for _ in self.n_layers:
            x = block(x)
        // ...
```

By only defining one transformer block and looping around it `n_layers` times, ALBERT saves the GPU memory that would be used to store the parameters for all the layers.

Since we normally use $32$ bit floats to store parameters on the GPU, storing the $1.5$B parameter GPT-2 on the GPU will use up about $6$GB of the GPU's memory — that's a pretty big chunk of the $16$GB of memory that's on a normal V100 GPU already being used up before taking into account the memory needed to store the model's activations as well as any momentum parameters needed by the optimizer. In contrast, if you share parameters across all transformer layers in the $1.5$B parameter GPT-2, the resulting model will only have about $37$M parameters, the parameter-sharing version would only use up around $148$MB of GPU memory.

The authors try applying parameter sharing to BERT and see that it reduces performance but makes it easier to train larger and larger models.

> In a machine learning framework like JAX, which by default unrolls and inlines loops when it's compiling your code with XLA, the size of the unrolled and inlined loop would make the computation graph really large and take a long time to compile. This is why you're recommended to use somehting like [`lax.scan()`](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scan.html) in these situations.

---

### Sentence-order-prediction auxillary loss

The ALBERT authors add an auxillary loss to help training. Since language modelling is usually done autoregressively, I didn't use this for my custom model.

---

### Removing dropout

The ALBERT authors remove all dropout from BERT and see that it significantly improves performance.

---

That's pretty much what my idea was: Take GPT-2, add a factorized embedding, share parameters across all transformer layers, remove dropout (I actually missed the part about ALBERT removing dropout until I was pretty far into my work, but I did run one or two runs without dropout to see how that works), and pretrain on a large dataset for a few hundred thousand iterations.

There's no way that I could pretrain something like GPT-2 by myself, so I applied to the [Tensorflow Research Cloud](https://www.tensorflow.org/tfrc) (TFRC).

> The TFRC puts an emphasis on wanting to help researchers from non-traditional backgrounds which makes it an amazing resource for anyone who isn't a "traditional" machine learning researcher. They were willing to give me, a 17 year old with no formal education or credentials (not even a high school diploma :/), access to an extremely powerful cluster of TPUs at no cost. Being able to be a part of this program was really helpful to me, especially since I don't have access to a dedicated GPU and usually rely on Colab's free GPU to train my models.

I emailed the TFRC team to ask if I could get upgraded from $5$ separate individual TPUv3's (with 8 cores each) to a TPU pod to pretrain a large language model. The very next day (!) I got an email back saying that I could get access to a preemptible 128-core TPUv3 Pod for 7 days which unfortunately wasn't long enough for me to pretrain the $1.5$B parameter model but was enough to train a few runs on the $124$M model.

---

# Setup

---

So for setup I'll be going through all the steps that I took to setup my VM and TPU Pod and preprocess the dataset as well.

When I was working on this project, I set up two VMs; One with a lot of RAM and CPU cores to process the data quickly and another small instance to run the TPU training script. _One of the nice things about training on TPUs and TPU pods is that as long as your data has been preprocessed as a set of TFRecord files, you don't need a really powerful VM instance which saves you a lot of money/compute credits._

You can look at [this](https://github.com/bilal2vec/lm-finetuning/blob/master/Markdown/CLOUD.md) for a full list of every command that I used to setup the VM and preprocess the dataset.

---

## OpenWebText

---

I used a `n-1-standard-16` instance with TF2.1 to process the OpenWebText dataset. Make sure that you use an instance with a SSD instead of the default HDD because processing the dataset involves processing a lot of very small text files and is mostly limited by your drive's io speed. _I made the mistake of using a HDD and just extracting the dataset's TAR archives took about 7 hours._ I put all the data in a folder at `~/data/openwebtext/` so modify it if you want to download the data elsewhere.

> TIL: most common linux utilities (like `ls`, `mv`, and `cat`) aren't really that optimized for working with almost 10 million files like in OpenWebText. Just counting the number of text files in the dataset could take several minutes._

Download the [OpenWebText](https://skylion007.github.io/OpenWebTextCorpus/) dataset (which is really just a tar archive of a bunch of tar archives that contain a lot of text files) and extract it:

```bash
gdown https://drive.google.com/uc?id=1EA5V0oetDCOke7afsktL_JDQ-ETtNOvx
tar -xf openwebtext.tar.xz
cat *.xz | tar -J -xf - -i
```

The dataset is about 12GB compressed and 53GB uncompressed and has just about 8 million text files.

I moved the first $100,000$ files in the dataset to a separate directory to create a validation set:

```bash
ls -f | head -100000 | xargs -i mv {} ../openwebtext-valid/
```

---

## Tokenization

---

I trained a Byte-level BPE tokenizer with a vocabulary size of $50,257$ (The same as GPT-2) on a $1$M file subset of the training set (I'm not sure if GPT-2 trains the tokenizer on the entire dataset or on just a subset, but I know that the [CTRL](https://arxiv.org/abs/1909.05858) paper trains their tokenizer on a 5% split of their training set.). I used Hugginface's fast Rust-based [Tokenizers](https://github.com/huggingface/tokenizers) library and their `ByteLevelBPETokenizer` tokenizer.

You can use my script [here](https://github.com/bilal2vec/lm-finetuning/blob/master/train_tokenizer.py) and run

```python
python3 train_tokenizer.py --train_path ./data/openwebtext/ --save_path ./tokenizer/ \
    --vocab_size 50257 --n_files 1000000
```

to train the tokenizer, or just take a look at this for the main details (It just trains a tokenizer and saves it as well as a configuration file to disk):

```python
import os
import glob
import json

from tokenizers import ByteLevelBPETokenizer

paths = glob.glob(os.path.join('./data/openwebtext', '*'))[:1000000]

tok = ByteLevelBPETokenizer()
tok.train(files=paths, vocab_size=args.vocab_size, special_tokens=args.control_codes)
tok.save('./tokenizer/')

tokenizer_config = {
    "max_len": 1024
}

with open(os.path.join('./tokenizer/', "tokenizer_config.json"), 'w') as fp:
    json.dump(tokenizer_config, fp)
```

---

## TFRecords

---

TPU Pods expect your data to be available as a set of TFRecord files in a GCP cloud bucket that get downloaded to each of your TPU board's built in powerful VM that will take care of de-serializing the files and feeding it to the TPU chips. Make sure that your GCP bucket and your TPU pod are in the same compute zone, otherwise you'll quickly rack up a lot of charges by transferring hundreds of GBs of data across compute zones.

> Here's a thing that's not very well documented when working with TPU Pods (this doesn't really apply to TPUs as much): TPU Pods create a lot (100s of GBs) of logs that get sent to Stackdriver, where you get charged about 50 cents for each GiB of logs ingested beyond a certain limit (I think it's around 50GiB/month). In just a few days of training, I ended up being charged about a \$$100$ IIRC. Luckily, I still had most of the free GCP credits so this didn't end up being a major problem for me, but make sure to turn off ingesting logs for TPUs.
> 
> I ran into a problem early on when I got access to the TPU pod where my code would work perfectly on a single TPU, but would throw an `Out of range: End of sequence` [error](https://gist.github.com/bilal2vec/ee63a04cd86c5fd45c41dc0b7ce109eb) when running it on a TPU pod. I struggled with this for a pretty long time until I took a look at [this](https://www.kaggle.com/c/flower-classification-with-tpus/discussion/130199) Kaggle discussion post that says that TPUs expect each TPU board (8 cores) to get its own TFrecord file (until that point, I was splitting the train set into 8 TFRecord files where I should've been splitting it into 16 (128 cores / 8 cores per board) TFRecord files.
> 
> TPUs are awesome for scaling to huge models and huge datasets, but there are a lot of TPU-specific information (especially for TPU Pods) that you need to know that's not covered in the documentation and isn't easy to find._**

You can use my script [here](https://github.com/bilal2vec/lm-finetuning/blob/master/make_tfrecords.py) and run

```bash
python3 make_tfrecords.py --path ./data/openwebtext/ --save_path ./train/ --files_per_tfrecord 500000 \
    --use_control_codes --seq_len 1024 --min_seq_len --tokenizer ./tokenizer/
```

```bash
python3 make_tfrecords.py --path ./data/openwebtext-valid/ --save_path ./val/ --files_per_tfrecord 50000 \
    --use_control_codes --seq_len 1024 --min_seq_len --tokenizer ./tokenizer/
```

to convert the raw text files from the train and validation splits into two sets of $16$ TFRecord files.

I ran a quick analysis on the average lengths of text fields in the dataset, $67$% of files have less than $1024$ tokens, $35$% of files have less than $512$ tokens, and only $10$% of files have less than $256$ tokens. This means that if I wanted to make the dataset as clean as possible and have each input sequence to the model be of a single contigous stream of $1024$ tokens, the dataset's size would be a lot smaller. For this reason, everyone prepends a token like `<|endoftext|>` to the beginning of each sequence and concatenates together sequences with lengths smaller than $1024$. The specifics of how exactly you do that (e.g. do you treat the dataset as single stream of tokens and just break it up into sequences of length $1024$, or do you keep track of sequences smaller that $1024$ and just concatenate them together into a single sequence) really shouldn't make too big of a difference in your model's performance, but you can take a look at my implementation [here](https://github.com/bilal2vec/lm-finetuning/blob/master/make_tfrecords.py).

My version doesn't take full advantage of the fast, multithreaded `batch_encode_plus()` way to tokenize large datasets in parallel since it only keeps the first `context_len` tokens in each line of the files which makes dealing with files with more or less than $1024$ tokens harder. Because of this, tokenizing the dataset takes about $8$ hours, which is something I want to improve.

The train set comes out to about $26$GB and consists of about $8$M text files that have been transformed into just under $7$M tfrecord examples, each with $1024$ tokens (same as GPT-2). The validation set comes out to about $300$MB and consists of about $100$K text files that have been transformed into just about $90$K tfrecord examples, each with $1024$ tokens (also the same as GPT-2).

---

  # Code

---

Since I'm using TPUs, the only real library that you can practically use right now would be Tensorflow. I didn't want to have to go through the learning curve of learning how to make custom training loops and stuff in TF2 so I just stuck to using Keras. You can take a look at my training script (It's pretty short) [here](https://github.com/bilal2vec/lm-finetuning/blob/master/train_tfrecords.py). It's pretty simple so I'm not going to copy over the entire training script, but I will talk about a few small code snippets.

I usually like to add a ptvsd breakpoint to my script so I can debug my training script locally with vscode before pushing it up to my VM

```python
if args.debug:
    import ptvsd
    ptvsd.enable_attach(address=('localhost', 5678),
                        redirect_output=True)
    ptvsd.wait_for_attach()
    breakpoint()
```

I'm using [Weights&Biases](https://www.wandb.com/) to keep track of my experiments and save checkpoints.

```python
    wandb.login()
    wandb.init(project='lm-finetuning', config=args, tags=args.tags)

    ...

    wandb_callback = WandbCallback(save_model=False)
```

Usually when you're using a TPU with Keras, you pass in the IP address and port of the TPU to `TPUClusterResolver`, but you pass the name of the TPU itself to the resolver when using a TPU Pod.

```python
resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=args.tpu)
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
```

---

# Replicating GPT-2

---

I tried to use as many of the original hyperparameters that OpenAI used when I was replicating their $124$M parameter version of GPT-2, but I had to modify a few things so I could train everything in time.

> Note: For some reason, the authors of the GPT-2 paper don't state exactly what learning rates they used for training their models and instead just state "The learning rate of each model was manually tuned for the best perplexity on a 5% held-out sample of WebText".

OpenAI trains their models for a total of $800$K iterations at a batch size of $512$ (Which comes out to around a total of $60$ epochs through the training set).

I trained my GPT-2 model for $1/8th$ the number of iterations that OpenAI trained theirs for (a total of around $100$K iterations) since each $100$K iteration training run took about $20$ hours to run on my 128-core TPU Pod. If I wanted to train GPT-2 for the same number of iterations as OpenAI, a single training run would have used up most of my one week of access to the pod.

Since my TPU pod was preemptible and resets every $24$ hours I usually had to resume my training run at least once and is the reason why all of these graphs usually have two or more training runs on them.

---

## Replicating GPT-2

---

So here's my model that came really close to replicating GPT-2, the training perplexity is about $21.5$ at the end of the almost $90$K training iterations. For comparison, GPT-2 gets a training perplexity about $17.5$ ppl after about $800$K training iterations, so a difference of only about $4$ ppl.

I made a [colab notebook](https://colab.research.google.com/drive/19Q0M9lMI4FqE7sosepkNVeIvan39SVFI?usp=sharing) showing how to use my pretrained GPT-2 model to generate text

<iframe
    title='Replicating GPT-2'
    src='https://app.wandb.ai/bilal2vec/lm-finetuning/reports/Replicating-GPT-2-(124M)--VmlldzoxNzE3Mzc'
    height='600px'
    width='100%'
> </iframe>

---

## AdamW vs Adafactor

---

I wanted to use the memory-saving [Adafactor](https://arxiv.org/abs/1804.04235) optimizer to make it easier to train larger language models but all of my Adafactor training runs were a lot (~5ppl IIRC) worse than using AdamW (This may be due to not using Adafactor's momentum parameter or relative update scale, so this is something I want to look into more soon).

---

## Learning rates

---

I started out with using Adam's default learning rate of $1e-4$ but I quickly figured out that I could train my models a lot faster by using a higher learning rate like $1e-3$.

> Section 2 of the [GPT-3](https://arxiv.org/pdf/2005.14165.pdf) paper lists the learning rates the OpenAI team used for different sized models when training GPT-3. They use a learning rate of $6e-4$ for the $124$M version of their model and decrease the learning rate with model size.

You can take a look at [this](https://app.wandb.ai/bilal2vec/lm-finetuning/reports/adamw-1e-4-vs-1e-3--VmlldzoxNzE3NDc) partial training run to see the difference between training at different learning rates.

---

# Pretraining ALGPT-2

---

Since I was using the Huggingface [Transformers](https://github.com/huggingface/transformers) repository's implementations of GPT-2 and ALBERT, I just [forked](https://github.com/bilal2vec/transformers/tree/albert-style) the repository and modified a few files to implement my ALGPT-2 model. You can take a look at all the changes that I had to make [here](https://github.com/bilal2vec/transformers/compare/master...bilal2vec:albert-style), most of the changes are only to make ALGPT-2 compatible with the /Transformers library and to be able to use the useful abstractions that it gives you, but most of the important code is in the [`modelling_algpt2.py` file](https://github.com/bilal2vec/transformers/blob/0f7c7c11e7b8bc8a275f3d16865b8a873c271571/src/transformers/modeling_algpt2.py) in which I just copied over the contents of `modelling_gpt2.py` and changed a few parts of the code. I'm only showing the changes that I made to the Pytorch version of ALGPT-2 here, the changes in the TF version are pretty similar to the Pytorch version and can be seen [here](https://github.com/bilal2vec/transformers/blob/0f7c7c11e7b8bc8a275f3d16865b8a873c271571/src/transformers/modeling_tf_algpt2.py).

---

## Implementing parameter sharing

---

Implementing parameter sharing only involves changing a few lines of code:

```diff
class ALGPT2Model(ALGPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        ...

-       self.h = nn.ModuleList([Block(config.n_ctx, config, scale=True)
-           for _ in range(config.n_layer)])
+       self.h = Block(config.n_ctx, config, scale=True)

        ...

    def forward(self, ...):

        ...

        if past is None:
            past_length = 0
-           past = [None] * len(self.h)
+           past = [None] * self.config.n_layer

        ...

-       for i, (block, layer_past) in enumerate(zip(self.h, past)):
+       for i in range(self.config.n_layer):

            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states.view(*output_shape),)
-
-           outputs = block(
+           outputs = self.h(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask[i],
                use_cache=use_cache,
            )
        ...

```

---

## Implementing a factorized embedding

---

Adding a factorized embedding is a little more work:

In the `config.json` that you use for your ALGPT-2 model, you need to specify that you want to use the ALGPT-2 and you need to specify the dimension of the factorized embedding that you want to use:

```diff
{
+	"architectures": ["ALGPT2LMHeadModel"],
	"attn_pdrop": 0.1,
	"bos_token_id": 50256,
	"embd_pdrop": 0.1,
	"eos_token_id": 50256,
	"initializer_range": 0.02,
	"layer_norm_epsilon": 1e-5,
+	"model_type": "algpt2",
	"n_ctx": 1024,
	"n_embd": 768,
	"n_head": 12,
	"n_layer": 12,
	"n_positions": 1024,
	"resid_pdrop": 0.1,
	"summary_activation": null,
	"summary_first_dropout": 0.1,
	"summary_proj_to_labels": true,
	"summary_type": "cls_index",
	"summary_use_proj": true,
	"vocab_size": 50257,
+	"embedding_size": 128
}
```

Back in `modelling_algpt2.py`, define the two factorized embedding matrices (the first second matrix that is really just a simple linear layer)

```diff
class ALGPT2Model(ALGPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        ...

-       self.wte = nn.Embedding(config.vocab_size, config.n_embd)
-       self.wpe = nn.Embedding(config.n_positions, config.n_embd)
+       self.wte = nn.Embedding(config.vocab_size, config.embedding_size)
+       self.wpe = nn.Embedding(config.n_positions, config.embedding_size)

+       self.projection_layer = nn.Linear(config.embedding_size, config.n_embd)


        ...

    def forward(self, ...):

        ...

        hidden_states = inputs_embeds + position_embeds + token_type_embeds

+       hidden_states = self.projection_layer(hidden_states)

        ...


class ALGPT2LMHeadModel(ALGPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        ...

-        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
+        self.dense = nn.Linear(config.n_embd, config.embedding_size)
+        self.lm_head = nn.Linear(config.embedding_size, config.vocab_size, bias=False)

    def forward(self, ...):

        ...

-       lm_logits = self.lm_head(hidden_states)
+       dense = self.dense(hidden_states)
+       lm_logits = self.lm_head(dense)
        ...
```

---

## Effect of layer-wise parameter sharing

---

This version of ALGPT-2 has about $47$M parameters while GPT-2 has $124$M. This ALGPT-2 model with parameter sharing trains a lot faster than GPT-2 ($9$ hours vs $20$ hours for a $90$K iteration training run), but is consistently about $10$ ppl worse than GPT-2 ($31$ vs $21$ ppl).

This difference is quite a bit larger than the difference between ALBERT and BERT, but might be explained by masked language modelling being an easier task than autoregressive language modelling. Increasing the size of the ALGPT-2 model might make it more competitive with GPT-2.

<iframe
    title='Effect of layer-wise parameter sharing'
    src='https://app.wandb.ai/bilal2vec/lm-finetuning/reports/Effect-of-layer-wise-parameter-sharing--VmlldzoxNzI0NzU'
    height='600px'
    width='100%'
> </iframe>

---

## Effect of removing dropout

---

I ran a [partial training run](https://app.wandb.ai/bilal2vec/lm-finetuning/reports/Effect-of-removing-dropout--VmlldzoxNzI0Nzk) on removing dropout from ALGPT-2. I didn't run it for very long, but it looks like removing dropout gives you a slight improvement (~3ppl).

---

## Effect of factorized embeddings

---

I ran three experiments for $90$K iterations with three different values for the factorized embedding ($128$, $256$, and $512$) as well as the baseline version without a factorized embedding.


| Model      | ALGPT-2 | ALGPT-2 512 | ALGPT-2 256 | ALGPT-2 128 |
| ---------- | ------- | ----------- | ----------- | ----------- |
| Parameters | 47M     | 34M         | 20M         | 13M         |
| Time       | ~9H     | ~9H         | ~9H         | ~9H         |
| Perplexity | 31      | 31          | 34          | 38          |

There was practically no difference in the loss curves between the baseline and the $512$ run since the change in the number of parameters wasn't that great. However, the training runs with factorized embeddings of sizes $256$ and $128$ were significantly worse than the baseline: $34$ and $38$ ppl respectively, a pretty big difference from the baseline of $31$ ppl.

<iframe
    title='Effect of factorized embeddings'
    src='https://app.wandb.ai/bilal2vec/lm-finetuning/reports/Effect-of-factorized-embeddings--VmlldzoxNzI0ODU'
    height='600px'
    width='100%'
> </iframe>

---

## Effect of model size

---

I only had the time to run one more full training run with ALGPT-2-medium (this one is comparable to the $345$M version of GPT-2). ALGPT-2-medium has about $66$M parameters and took twice as long as ALGPT-2 to train (a little more than $20$ hours). The larger model size made quite a big difference in performance, the training perplexity decreased $5$ppl from $31$ to $26$ ppl.

<iframe
    title='Effect of model size'
    src='https://app.wandb.ai/bilal2vec/lm-finetuning/reports/Effect-of-model-size--VmlldzoxNzI0OTM'
    height='600px'
    width='100%'
> </iframe>

---

# Conclusion and next steps

---

Well that's pretty much everything that I did. After my TPU pod's quota was used up, I started working on a [few](https://github.com/bilal2vec/L2) [other](/)
[things](https://github.com/bilal2vec/raytracer) over the summer and just kept delaying writing up what I did for a couple of months until now.

There are a lot of things that I still want to work on or look into:

-   Training larger versions of ALGPT-2
-   Removing or replacing the normalization layers in transformers
-   Working on distilling/shrinking language models with billions of parameters to make them more accessible
-   Apply something like [PPLM](https://arxiv.org/abs/1912.02164) to condition language models for few-shot inference (kinda like what GPT-3 does).

Thanks for reading through all this. If you think there's any mistakes or inaccuracies in this post, please let me know.