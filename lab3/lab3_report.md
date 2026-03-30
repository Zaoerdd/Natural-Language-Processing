# Lab 3 Report: Word2Vec Data Preparation and Skip-Gram with Negative Sampling

## 1. 实验目标

本次 Lab 的目标是从零理解并实现一个最小可运行的 `word2vec` 流程，具体包括：

- 用滑动窗口把原始文本变成 `skip-gram` 训练样本。
- 为每个正样本生成负样本，理解 `negative sampling` 的作用。
- 用 PyTorch 实现 `SkipGram` 模型的前向传播和损失函数。
- 在《论语》前 20 篇的小语料上训练词向量，并导出文本格式的 embedding。
- 使用预训练英文词向量做 analogy 和 SVD 可视化，理解词向量的几何性质。


## 2. 本次完成内容

- 保留原模板 notebook 不动：
  [`lab3_w2v.ipynb`](/D:/OneDrive/Study/2026%20Spring/Natural%20Language%20Processing/lab3/lab3_w2v.ipynb)
- 新建并完成工作副本：
  [`lab3_w2v_completed.ipynb`](/D:/OneDrive/Study/2026%20Spring/Natural%20Language%20Processing/lab3/lab3_w2v_completed.ipynb)
- 导出训练得到的词向量：
  [`embeddings.txt`](/D:/OneDrive/Study/2026%20Spring/Natural%20Language%20Processing/lab3/embeddings.txt)
- 生成了 T4 的可视化图片：
  [`t4_svd.png`](/D:/OneDrive/Study/2026%20Spring/Natural%20Language%20Processing/lab3/t4_svd.png)


## 3. 实验结果摘要

- 语料：`lunyu_20chapters.txt`
- token 数：19,874
- 词表大小：1,352
- 滑窗正样本数：113,100
- embedding 维度：100
- window size：3
- negative samples：5
- batch size：64
- optimizer：`SparseAdam`
- epochs：5

训练 loss 结果：

- Epoch 1: 3.1472
- Epoch 2: 2.8216
- Epoch 3: 2.5710
- Epoch 4: 2.4037
- Epoch 5: 2.3069

可以看到 loss 持续下降，说明实现的训练流程是正常工作的。


## 4. 核心知识点理解

### 4.1 为什么需要词向量

One-hot 向量只能表示“这是哪个词”，不能表达“两个词有多像”。  
例如 `king` 和 `queen` 的 one-hot 向量依然几乎完全正交，但它们语义上显然很接近。

词向量的目标是把每个词映射到一个稠密的实数向量空间中，使得：

- 语义相近的词更靠近。
- 某些关系可以用向量运算表达。
- 下游模型可以直接利用这种连续表示。


### 4.2 Skip-Gram 在做什么

`skip-gram` 的思想是：

- 选中一个中心词 `c`
- 用它去预测窗口里的上下文词 `o`

如果一个词经常出现在另一个词附近，那么它们的向量应该更相似。

本次 lab 中，给定一句话：

```text
学 而 时 习 之
```

如果中心词是 `学`，窗口大小是 3，那么 outside words 就是：

- `而`
- `时`
- `习`

因此会生成多个正样本：

- `(学, 而)`
- `(学, 时)`
- `(学, 习)`


### 4.3 为什么要负采样

原始 softmax 要把中心词和整个词表中的所有词都算一遍概率，代价非常高。  
如果词表很大，训练会非常慢。

负采样的思路是把问题改成很多个二分类：

- 正样本：这个 outside word 真的出现在中心词窗口里，标签为 1
- 负样本：随机采几个不在窗口里的词，标签为 0

这样就不需要对整个词表做归一化，只需要：

- 1 次正样本打分
- `k` 次负样本打分

所以训练会快很多。


### 4.4 为什么有两套 embedding

在 word2vec 中，每个词通常有两套向量：

- `v`：当它作为中心词时使用
- `u`：当它作为上下文词或负样本时使用

在代码里对应：

- `self.emb_v`
- `self.emb_u`

这是 word2vec 的标准写法。训练结束后，一般把 `emb_v` 作为最终词向量导出。


### 4.5 负采样中的 0.75 次幂是什么意思

`utils.py` 里构造负采样表时，不是直接按词频采样，而是按：

```text
freq(w)^0.75
```

这样做的原因是：

- 高频词应该更容易被采到，因为它们在真实语言中也很常见
- 但如果完全按原始词频采样，像标点、功能词会被采得过多
- 用 `0.75` 次幂会压低超高频词的权重，同时保留“高频词更常出现”的趋势

这是 Mikolov 等人 word2vec 中经典的经验设置。


## 5. 代码实现说明

### 5.1 `generate_data`

这个函数完成了训练样本生成，流程是：

1. 把词列表转成 `word id`
2. 枚举每个位置作为中心词
3. 取左右窗口里的词作为 outside words
4. 对每个 `(center, outside)` 正样本调用 `corpus.getNegatives(center, k)`
5. 用 `yield` 逐条输出样本

输出格式是：

```python
(center_id, outside_id, negative_ids)
```

其中 `negative_ids` 是一个长度为 `k` 的数组。


### 5.2 `batchify`

这个函数把上一步的样本流打包成 batch。

每个 batch 输出三个张量：

- `center`: `(B,)`
- `outside`: `(B,)`
- `negative`: `(B, k)`

数据类型都设成了 `torch.long`，因为 embedding lookup 需要整数索引。


### 5.3 `SkipGram.forward`

前向传播里做了三件事：

1. 取出中心词向量 `v_c`
2. 取出正样本 outside 向量 `u_o`
3. 取出负样本向量 `u_n`

对应张量形状：

- `v_c`: `(B, D)`
- `u_o`: `(B, D)`
- `u_n`: `(B, K, D)`

然后计算：

```text
positive score = v_c · u_o
negative score = v_c · u_k
```

损失函数为：

```text
-log σ(v_c · u_o) - Σ log σ(-v_c · u_k)
```

代码里使用 `F.logsigmoid` 实现，并用 `torch.clamp(..., min=-10, max=10)` 防止数值太大或太小。


### 5.4 `train`

训练函数主要做了这些工作：

- 遍历每个 batch
- 调用 `model(center, outside, negative)`
- 对 batch loss 求平均
- `backward()`
- `optimizer.step()`
- 统计每个 epoch 的平均 loss

这里使用 `SparseAdam` 的原因是 embedding 层设置了 `sparse=True`，这样只会更新当前 batch 用到的词向量，效率更高。


## 6. 关键公式解读

对一个中心词 `c`、一个正样本上下文词 `o` 和 `K` 个负样本 `u_1 ... u_K`，目标是：

```text
J = log σ(v_c · u_o) + Σ log σ(-v_c · u_k)
```

训练时通常最小化负号后的 loss：

```text
L = -log σ(v_c · u_o) - Σ log σ(-v_c · u_k)
```

这两个部分的含义是：

- 第一项希望 `v_c · u_o` 变大
  也就是让真实共现的词更接近
- 第二项希望 `v_c · u_k` 变小
  也就是让随机负样本和中心词更远

所以整个训练过程，就是在向量空间中“拉近真邻居，推远假邻居”。


## 7. 结果分析

### 7.1 本地训练结果

loss 从 `3.1472` 下降到 `2.3069`，说明：

- 数据生成逻辑正确
- 损失函数实现正确
- 参数更新有效

但要注意，这里使用的是《论语》前 20 篇，而且还是按字切分：

- 语料规模很小
- 中文字级别表示的语义能力有限
- 导出的 embedding 更适合教学演示，不适合作为高质量通用词向量


### 7.2 T4 预训练词向量结果

T4 使用的是 `glove-wiki-gigaword-200`，词表大小为 400,000。

类比结果如下：

- `man : grandfather :: woman : ?`
  最优答案是 `grandmother`
- `china : beijing :: japan : ?`
  最优答案是 `tokyo`
- `france : paris :: italy : ?`
  最优答案是 `rome`
- `man : king :: woman : ?`
  最优答案是 `queen`

这些结果说明预训练词向量已经学到了一部分关系结构：

- 国家和首都关系
- 性别关系
- 家庭关系

这也是词向量最经典的性质之一：  
语义关系在向量空间里可以近似表示成“方向”。


### 7.3 SVD 可视化能说明什么

我选了这些词来做二维投影：

- `man`
- `woman`
- `king`
- `queen`
- `prince`
- `princess`

虽然二维投影会损失很多信息，但通常还能看到：

- 性别相关词比较接近
- 皇室相关词比较接近
- `king/queen` 和 `prince/princess` 有一定群聚关系

这说明高维词向量里确实编码了语义结构。


## 8. 遇到的问题与解决

### 8.1 `SparseAdam` 不能直接做 gradient clipping

最开始我在训练里加了：

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
```

但因为 embedding 是 `sparse=True`，梯度是稀疏张量，PyTorch 在这里会报错。  
所以最终把这一步移除了。


### 8.2 `gensim` 在 Python 3.14 下安装失败

课程虚拟环境使用的是 Python 3.14，而 `gensim` 在当前 Windows + Python 3.14 环境下编译失败。  
因此：

- `T1-T3` 使用课程环境完成
- `T4` 使用本机 Python 3.12 验证

这不是 notebook 逻辑错误，而是包兼容性问题。


## 9. 我对这节 Lab 的理解

这节 lab 的重点不只是“把代码补完”，而是理解 word2vec 背后的训练逻辑：

- 先从语料中构造 `(center, outside)` 共现关系
- 再用负采样把“预测整个词表”改成“区分真邻居和假邻居”
- 最后通过梯度更新，把有语义关系的词放到更接近的位置

如果用一句话总结：

> Skip-gram with negative sampling 本质上是在用局部上下文共现信号，学习一个能表达语义相似性和语义关系的向量空间。


## 10. 复现方式

`T1-T3` 运行环境：

```powershell
& 'D:\OneDrive\Study\2026 Spring\Natural Language Processing\nlp\Scripts\python.exe'
```

`T4` 运行环境：

```powershell
py -3.12
```

如果要重新运行 notebook，建议：

- 先在完成版 notebook 中跑到 `T3`
- 再切到支持 `gensim` 的 Python 3.12 环境运行 `T4`

