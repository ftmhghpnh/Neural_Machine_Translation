#! /usr/bin/env python3
# -*- coding: utf-8 -*-
'''
This code is provided solely for the personal and private use of students
taking the CSC401H/2511H course at the University of Toronto. Copying for
purposes other than this use is expressly prohibited. All forms of
distribution of this code, including but not limited to public repositories on
GitHub, GitLab, Bitbucket, or any other online platform, whether as given or
with any changes, are expressly prohibited.

Author: Raeid Saqur <raeidsaqur@cs.toronto.edu>

Some building blocks modified from: http://nlp.seas.harvard.edu/annotated-transformer/
All of the files in this directory and all subdirectories are:
Copyright (c) 2023 University of Toronto
'''

import copy
import logging
import math
import time

import matplotlib.pyplot as plt
import seaborn
import torch
import torch.nn as nn
from torch.nn.functional import log_softmax
from torch.optim.lr_scheduler import LambdaLR

logging.basicConfig(level=logging.INFO)


class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return log_softmax(self.proj(x), dim=-1)

class DummyOptimizer(torch.optim.Optimizer):
    def __init__(self):
        self.param_groups = [{"lr": 0}]

    def step(self):
        pass

    def zero_grad(self, set_to_none=False):
        pass

class DummyScheduler:
    def step(self):
        pass

## Transformer building blocks ##
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.std(-1, keepdim=True)
        return self.a_2 * (x-mu)/(sigma + self.eps) + self.b_2

class Sublayer(nn.Module):
    """
    A residual connection followed by a layer norm.
    """
    def __init__(self, size, dropout):
        super(Sublayer, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """ Residual connection w/ dropout """
        return x + self.dropout(sublayer(self.norm(x)))

# MAIN MODEL BLOCKS =============================================================================#
class EncoderDecoder(nn.Module):
    """ A standard Encoder-Decoder architecture. """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

    def reset_parameters(self):
        for layer in self.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(Sublayer(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        x = self.sublayer[1](x, self.feed_forward) # Sublayer takes a fn as arg so lambda
        return x

class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        #print(f"In Decoder::{inspect.currentframe().f_code.co_name}")
        for i,layer in enumerate(self.layers):
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, cross-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, cross_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.cross_attn = cross_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(Sublayer(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))                            # self-attn
        x = self.sublayer[1](x, lambda x: self.cross_attn(query=x, key=m, value=m, mask=src_mask))      # cross-attn
        x = self.sublayer[2](x, self.feed_forward)                                                      # feed-forward
        return x

# MODEL BLOCKS END ===========================================================================#
class LossObjective:
    "A simple loss compute and train function."
    def __init__(self, generator, criterion):
        self.generator = generator
        self.criterion = criterion

    def __call__(self, x, y, norm):
        x = self.generator(x)
        sloss = ( self.criterion(x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)) / norm )
        sloss_data_norm = sloss.data * norm

        return sloss_data_norm, sloss

class LabelSmoothing(nn.Module):
    """Regularizer: Label Smoothing """
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())

class FeedForwardSubLayer(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForwardSubLayer, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        """ """
        return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        _pe = self.pe[:, : x.size(1)].requires_grad_(False)
        x = x + _pe     # Add the position encoding _pe to original vector x
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, opts=None):
        "Take in model size (d_model) and number of heads (h)"
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        # Assuming d_v == d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        nbatches = query.size(0)
        if mask is not None:
            "Same for all h heads"
            mask = mask.unsqueeze(1)
            if nbatches < mask.size(0):
                logging.debug("=== Buggy case now: === ")
                logging.debug(f"\tmask.shape = \t\t{mask.shape}")
                logging.debug(f"\tquery.shape =  \t\t{query.shape}")
                logging.debug(f"\tkey_shape =    \t\t{key.shape}")
                logging.debug(f"\tvalue.shape = \t\t{value.shape}")

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            lin(x).view( nbatches, -1, self.h, self.d_k).transpose(1,2)
                for lin, x in zip(self.linears, (query, key, value))
        ]
        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        # 3) "Concat" using a view and apply a final linear.
        x = (x.transpose(1,2).contiguous().view( nbatches, -1, self.h * self.d_k))

        del query
        del key
        del value
        x = self.linears[-1](x)
        return x

### Batches and Masking
class Batch:
    """Object for holding a batch of data with mask during training."""
    def __init__(self, src, tgt=None, pad=2, device:torch.device=torch.device("cpu")):  # 2 = <blank>
        self.src = src.to(device)
        tgt = tgt.to(device)
        self.src_mask = (src != pad).unsqueeze(-2).to(device)
        if tgt is not None:
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        future_words_mask = subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
        tgt_mask = tgt_mask & future_words_mask
        return tgt_mask

class TrainState:
    """Track number of steps, examples, and tokens processed"""
    step: int = 0           # Steps in the current epoch
    accum_step: int = 0     # Number of gradient accumulation steps
    samples: int = 0        # total # of examples used
    tokens: int = 0         # total # of tokens processed

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above. Extracted out in the new version."
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))

## Helper functions ##
DEFAULT_TRAIN_CONFIG = {
            "batch_size": 32,
            "word_embedding_size": 512,
            "distributed": False,
            "num_epochs": 6,
            "accum_iter": 10,
            "base_lr": 1.0,
            "max_padding": 72,
            "warmup": 3000,
            "file_prefix": "model_w_transformer-",
        }

def train_for_epoch(model, dataloader,
                    optimizer: torch.optim.Optimizer = DummyOptimizer(),
                    device: torch.device = torch.device("cpu"),
                    mode="train"):
    """ Train a single epoch """
    start = time.time()
    if mode == "train":
            assert model.training
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0
    pad_idx = 2
    d_model = DEFAULT_TRAIN_CONFIG["word_embedding_size"]   # 512
    warmup = DEFAULT_TRAIN_CONFIG["warmup"]                 # 3000
    accum_iter = DEFAULT_TRAIN_CONFIG["accum_iter"]         # accum_iter = 1
    module = model
    criterion = LabelSmoothing(size=dataloader.dataset.target_vocab_size, padding_idx=pad_idx, smoothing=0.1)
    criterion.to(device)
    if model.training:
        scheduler = LambdaLR(optimizer=optimizer,
                                lr_lambda=lambda step: rate(
                                    step, d_model, factor=1, warmup=warmup))
    else:
        assert mode == "eval"
        scheduler = DummyScheduler()

    loss_fn = LossObjective(module.generator, criterion)
    train_state = TrainState()
    for i, B in enumerate(dataloader):
        batch = Batch(B[0], B[2], pad_idx, device)
        out = model.forward(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
        loss, loss_node = loss_fn(out, batch.tgt_y, batch.ntokens)
        if mode == "train" or mode == "train+log":
            loss_node.backward()
            train_state.step += 1
            train_state.samples += batch.src.shape[0]
            train_state.tokens += batch.ntokens
            if i % accum_iter == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                n_accum += 1
                train_state.accum_step += 1
            scheduler.step()

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 40 == 1 and (mode == "train" or mode == "train+log"):
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            print(
                (
                    "Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f "
                    + "| Tokens / Sec: %7.1f | Learning Rate: %6.1e"
                )
                % (i, n_accum, loss / batch.ntokens, tokens / elapsed, lr)
            )
            start = time.time()
            tokens = 0
        del loss
        del loss_node
    return total_loss / total_tokens, train_state

def rate(step, model_size, factor, warmup):
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )

def clones(module, N):
    "Produce N identical layers"
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.zeros(1,1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len -1):
        out = model.decode(memory, src_mask, ys,
                           subsequent_mask(ys.size(1)).type_as(src.data))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, torch.zeros(1,1).type_as(src.data).fill_(next_word)], dim=1)
    return ys

def subsequent_mask(size):
    "Mask out subsequent positions"
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return subsequent_mask == 0

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


def plot_attn(model, sent:list, tgt_sent:list, decoder_only=False):
    def draw(data, x, y, ax):
        seaborn.heatmap(data,
                        xticklabels=x, square=True, yticklabels=y, vmin=0.0, vmax=1.0,
                        cbar=False, ax=ax)
    if not decoder_only:
        for layer in range(1, 6, 2):
            fig, axs = plt.subplots(1, 4, figsize=(20, 10))
            print("Encoder Layer", layer + 1)
            for h in range(4):
                draw(model.encoder.layers[layer].self_attn.attn[0, h].data,
                     sent, sent if h == 0 else [], ax=axs[h])
            plt.show()

    for layer in range(1, 6, 2):
        fig, axs = plt.subplots(1, 4, figsize=(20, 10))
        print("Decoder Self Layer", layer + 1)
        for h in range(4):
            draw(model.decoder.layers[layer].self_attn.attn[0, h].data[:len(tgt_sent), :len(tgt_sent)],
                 tgt_sent, tgt_sent if h == 0 else [], ax=axs[h])
        plt.show()
        print("Decoder Src Layer", layer + 1)
        fig, axs = plt.subplots(1, 4, figsize=(20, 10))
        for h in range(4):
            draw(model.decoder.layers[layer].self_attn.attn[0, h].data[:len(tgt_sent), :len(sent)],
                 sent, tgt_sent if h == 0 else [], ax=axs[h])
        plt.show()