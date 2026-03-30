# Lab 4 Report: Transformer Basics

## 1. Lab Goal

This lab walks through the core building blocks of a **decoder-only Transformer** for language modeling.

本次实验的目标不是直接训练一个完整的大模型，而是先把 GPT-style Transformer 里最关键的几个模块拆开理解：

- Self-Attention
- Causal Mask / Future Masking
- Multi-Head Attention
- FeedForward Network
- LayerNorm
- Residual Connection
- GPT forward pipeline

完成这节 lab 之后，应该能回答两个问题：

- **原理上**：Transformer 为什么能建模序列，并且为什么 GPT 只能看见当前位置之前的 token。
- **实现上**：这些公式在 PyTorch 里到底怎么落到 `matmul`、`softmax`、`view`、`permute` 和 `masked_fill` 上。

## 2. T1: Single-Head Self-Attention

### 2.1 Input Representation

实验里首先构造输入矩阵：

\[
X \in \mathbb{R}^{N \times d}
\]

其中：

- `N` 是 token 数量
- `d` 是 embedding dimension

在 notebook 里，`X = torch.randn(N, d)` 只是随机造一个 embedding 矩阵，用来演示 attention 计算流程。

### 2.2 Q, K, V 的含义

Transformer 不直接拿 `X` 去做注意力，而是先线性映射成三个空间：

\[
Q = XW^Q,\quad K = XW^K,\quad V = XW^V
\]

这里：

- `Q` = Query，表示“我当前想找什么信息”
- `K` = Key，表示“我能提供什么信息”
- `V` = Value，表示“真正被加权汇总的内容”

在代码里，对应：

```python
W_q = torch.empty(d, d_k)
W_k = torch.empty(d, d_k)
W_v = torch.empty(d, d_k)

nn.init.xavier_normal_(W_q)
nn.init.xavier_normal_(W_k)
nn.init.xavier_normal_(W_v)

Q = torch.matmul(X, W_q)
K = torch.matmul(X, W_k)
V = torch.matmul(X, W_v)
```

这里使用 `xavier_normal_` 是为了让线性层初始方差更稳定，避免一开始数值过大或过小。

### 2.3 Attention Score

attention 的核心是先计算 token 之间的相关性：

\[
\text{score} = \frac{QK^\top}{\sqrt{d_k}}
\]

这里 `QK^T` 的 shape 是：

- `Q`: `(N, d_k)`
- `K^T`: `(d_k, N)`
- 结果: `(N, N)`

所以 `score[i, j]` 表示第 `i` 个 token 对第 `j` 个 token 的关注程度。

除以 `\sqrt{d_k}` 是 **scaled dot-product attention** 的关键。  
如果不缩放，维度很大时点积会很大，`softmax` 会过于尖锐，训练不稳定。

### 2.4 Attention Weight and Output

\[
\alpha = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)
\]

\[
\text{output} = \alpha V
\]

代码实现：

```python
alpha = F.softmax(torch.matmul(Q, K.T) / np.sqrt(d_k), dim=-1)
output = torch.matmul(alpha, V)
```

这里最重要的是 `softmax(dim=-1)`：

- `alpha` 是 `(N, N)`
- 对最后一维做 softmax，表示“对于每个 query token，把它看向所有 key token 的权重归一化”
- 每一行加起来等于 1

如果 softmax 维度写错，attention 的语义就会彻底错掉。

## 3. T2: Causal Mask / Future Masking

### 3.1 为什么要 Mask Future Tokens

GPT 是 **causal language model**。  
它在预测当前位置 token 时，不能偷看未来单词，否则训练和推理不一致。

比如序列是：

`I love NLP`

当模型在预测 `love` 后面的词时，可以看 `I` 和 `love`，但不能提前看 `NLP`。

### 3.2 Lower-Triangular Mask

实验里构造了一个下三角矩阵：

\[
M =
\begin{bmatrix}
1 & 0 & 0 \\
1 & 1 & 0 \\
1 & 1 & 1
\end{bmatrix}
\]

对应代码：

```python
mask = torch.tril(torch.ones(N, N))
```

它的含义是：

- 当前 token 可以看自己和过去
- 不能看未来

### 3.3 为什么在 Softmax 前填 `-inf`

mask 不是直接乘在 `alpha` 上，而是先作用在 score matrix 上：

```python
masked_scores = torch.matmul(Q, K.T) / np.sqrt(d_k)
masked_scores = masked_scores.masked_fill(mask == 0, -np.inf)
new_alpha = F.softmax(masked_scores, dim=-1)
```

原因是：

- 被屏蔽的位置填成 `-inf`
- 经过 softmax 后，这些位置的概率会变成 0

这比 softmax 之后再强行置零更自然，因为归一化过程会自动只在合法位置之间分配概率。

### 3.4 Masked Output

最后重新做加权和：

```python
new_output = torch.matmul(new_alpha, V)
```

这一步说明：**mask 改变的不是 `V`，而是“从哪些 token 收集信息”**。

## 4. T3: Multi-Head Attention

### 4.1 为什么要多头

Single-head attention 只能学一种“关注方式”。  
Multi-head attention 允许模型在不同子空间并行关注不同关系，例如：

- 一个 head 关注局部邻近词
- 一个 head 关注主谓关系
- 一个 head 关注长距离依赖

### 4.2 Shape 变化是本题关键

实验里先投影：

\[
Q, K, V \in \mathbb{R}^{N \times d}
\]

再拆成多头：

\[
Q_ \in \mathbb{R}^{h \times N \times d_k}
,\quad d_k = d / h
\]

代码：

```python
Q_ = Q.view(N, self.n_heads, self.d_k).permute(1, 0, 2)
K_ = K.view(N, self.n_heads, self.d_k).permute(1, 0, 2)
V_ = V.view(N, self.n_heads, self.d_k).permute(1, 0, 2)
```

这里要特别理解两个操作：

- `view(N, h, d_k)`：把最后一个 embedding 维度切成多个 head
- `permute(1, 0, 2)`：把维度顺序改成 `(h, N, d_k)`，方便每个 head 单独做矩阵乘法

### 4.3 Head-wise Attention

每个 head 单独计算：

```python
alpha = torch.matmul(Q_, K_.permute(0, 2, 1)) / np.sqrt(self.d_k)
```

shape 变化：

- `Q_`: `(h, N, d_k)`
- `K_.permute(0, 2, 1)`: `(h, d_k, N)`
- 输出 `alpha`: `(h, N, N)`

这意味着每个 head 都有自己的一张 attention matrix。

### 4.4 Multi-Head Mask

mask 也要复制到每个 head：

```python
mask = torch.tril(torch.ones(N, N, device=X.device)).unsqueeze(0).repeat(self.n_heads, 1, 1)
alpha = alpha.masked_fill(mask == 0, -np.inf)
alpha = F.softmax(alpha, dim=-1)
```

其中：

- `unsqueeze(0)` 把 `(N, N)` 变成 `(1, N, N)`
- `repeat(self.n_heads, 1, 1)` 复制成 `(h, N, N)`

### 4.5 合并 Heads

每个 head 算完之后，再拼回原来的 embedding 维度：

```python
output = torch.matmul(alpha, V_).permute(1, 0, 2).contiguous().view(N, -1)
```

这一步里 `contiguous()` 很重要。  
因为 `permute()` 只改变视图顺序，不保证内存连续；如果直接 `view()`，可能报错或得到错误结果。  
所以常见安全写法是：

```python
permute(...).contiguous().view(...)
```

### 4.6 为什么 `n_heads = 1` 要退化回单头

当 `n_heads = 1` 时：

- `d_k = d`
- 只有一个 head
- reshape 和拼接不会真正改变信息结构

所以理论上 multi-head 实现应与前面的 single-head masked attention 一致。  
实验里也确实验证了这一点，输出与 `new_output` 对齐。

## 5. T4: FeedForward, GELU, LayerNorm, Residual

### 5.1 FeedForward Network

Transformer block 里不只有 attention，还有一个 position-wise MLP：

```python
nn.Linear(emb_dim, 4 * emb_dim)
GELU()
nn.Linear(4 * emb_dim, emb_dim)
```

它的作用是：

- 对每个 token 的表示做非线性变换
- 提升模型表达能力

这里“position-wise”表示每个 token 都用同一个两层网络处理，但不同 token 之间不在这一步交互。

### 5.2 Why GELU Instead of ReLU

GELU 相比 ReLU 更平滑。  
ReLU 是硬截断，小于 0 直接变 0；GELU 会保留一部分负值信息，曲线更连续。

这也是 GPT-2 的经典配置之一。

### 5.3 LayerNorm

LayerNorm 的公式：

\[
\text{LN}(x) = \gamma \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
\]

其中：

- `mean` 和 `var` 沿最后一个维度算
- `scale = γ`
- `shift = β`

在代码里：

```python
mean = x.mean(dim=-1, keepdim=True)
var = x.var(dim=-1, keepdim=True, unbiased=False)
norm_x = (x - mean) / torch.sqrt(var + self.eps)
return self.scale * norm_x + self.shift
```

`keepdim=True` 的作用是保留维度，方便后面做 broadcast。

### 5.4 Residual Connection

shortcut / residual connection 的形式：

\[
x \leftarrow x + \text{submodule}(x)
\]

作用：

- 缓解 vanishing gradient
- 让深层网络更容易训练
- 允许模型在“保留原信息”和“学习新变换”之间折中

实验里通过比较有无 shortcut 的梯度大小，直观展示了 residual 对梯度传播的帮助。

## 6. TransformerBlock and GPTModel

### 6.1 TransformerBlock Structure

本 lab 里的 `TransformerBlock` 流程是：

1. `LayerNorm`
2. `MultiHeadAttention`
3. `Dropout`
4. `Residual Add`
5. `LayerNorm`
6. `FeedForward`
7. `Dropout`
8. `Residual Add`

这是典型的 **pre-norm Transformer block**。

### 6.2 GPT Forward Pipeline

`GPTModel` 的前向传播可以概括为：

1. token ids -> token embeddings
2. 加上 positional embeddings
3. 通过多个 Transformer blocks
4. final LayerNorm
5. linear output head -> logits over vocabulary

公式上可以理解为：

\[
\text{logits} = W_{out} \cdot \text{Transformer}(\text{tok\_emb} + \text{pos\_emb})
\]

输出 shape 为：

\[
(\text{batch\_size}, \text{seq\_len}, \text{vocab\_size})
\]

实验最后得到的 `torch.Size([2, 4, 50257])` 正是这个含义：

- batch 中有 2 个句子
- 每个句子长度为 4
- 每个位置都要预测一个大小为 `50257` 的词表分布

## 7. Common Pitfalls

这节 lab 最容易错的点主要有下面几个：

### 7.1 `softmax(dim=-1)` 写错

attention 的归一化应该沿 key 维度做，也就是每个 query 对所有 key 的分布。  
如果 `dim` 写错，attention matrix 的含义会错。

### 7.2 `QK^T` 的 shape 不清楚

单头时：

- `Q`: `(N, d_k)`
- `K.T`: `(d_k, N)`
- `QK^T`: `(N, N)`

多头时：

- `Q_`: `(h, N, d_k)`
- `K_.permute(0, 2, 1)`: `(h, d_k, N)`
- 输出: `(h, N, N)`

### 7.3 Mask 放错位置

mask 应该放在 softmax 之前，而不是之后。  
否则概率分布不会正确归一化。

### 7.4 `view()` 和 `permute()` 的顺序

多头实现里常见模板是：

```python
view(...)
permute(...)
...
permute(...)
contiguous()
view(...)
```

这个顺序不能随便改。

### 7.5 忽略 `device`

mask 最好写成：

```python
torch.ones(..., device=X.device)
```

否则模型在 GPU 上运行时，可能出现 tensor 不在同一 device 的错误。

## 8. What This Lab Really Teaches

这节 lab 的核心收获可以总结成一句话：

**Transformer is mostly linear algebra with careful tensor reshaping and masking.**

更具体地说，你应该建立下面这条理解链：

- embedding 是输入表示
- `Q/K/V` 是对输入的三种线性投影
- `QK^T` 建立 token-token 相关性
- `softmax` 把相关性变成概率权重
- `alpha @ V` 生成上下文表示
- multi-head 让模型在多个子空间并行建模
- LayerNorm + Residual + FeedForward 让深层 Transformer 可以稳定训练
- 多层堆叠后，再加一个 LM head，就形成 GPT-like language model

## 9. Reproducibility

- Notebook template: `lab4_trm.ipynb`
- Completed notebook: `lab4_trm_completed.ipynb`
- Helper module: `utils.py`
- Environment: repo-local `.venv` with Python 3.12

建议复习顺序：

1. 先手推 single-head attention 的 shape
2. 再理解 causal mask 为什么是下三角
3. 再看 multi-head 里的 `view/permute/contiguous`
4. 最后从 `TransformerBlock` 一路串到 `GPTModel`

如果这四步都能自己复述出来，这节 lab 就算真正掌握了。
